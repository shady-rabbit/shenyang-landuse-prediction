"""CLCD 原始 9 类 Markov 基线实验脚本。

这个脚本用于 CA-Markov 实验的第一步：先只做 Markov 链基线。
它回答的是“各土地利用类型数量如何转移”，暂时不处理“变化发生在哪里”。
后续真正的 CA-Markov 会在这个基础上加入邻域、约束和适宜性因子。

脚本保持 CLCD 原始类别编码不变：
1 Cropland, 2 Forest, 3 Shrub, 4 Grassland, 5 Water,
6 Snow/Ice, 7 Barren, 8 Impervious, 9 Wetland.

主要输出：
- 相邻年份转移数量矩阵
- 相邻年份转移概率矩阵
- 滑动验证的预测 GeoTIFF
- 混淆矩阵、各类精度、面积预测表、总体验证表
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。

    rasterio/GDAL 在 Windows 上有时用解释器绝对路径运行时拿不到
    GDAL_DATA 和 PROJ_LIB。这里在导入 rasterio 之前尝试自动设置，
    避免反复出现 GDAL_DATA is not defined 之类的警告。
    """
    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"

    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_geospatial_data_paths()

# 注意：rasterio 依赖 GDAL，所以要放在 configure_geospatial_data_paths()
# 之后导入，保证 GDAL/PROJ 数据路径已经准备好。
import rasterio


# 默认使用沈阳已经裁剪好的 6 期 CLCD 数据。
DEFAULT_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]

# CLCD 原始 9 类编码。NoData 为 0，不参与转移矩阵和精度评价。
CLASS_CODES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
CLASS_NAMES = {
    1: "Cropland",
    2: "Forest",
    3: "Shrub",
    4: "Grassland",
    5: "Water",
    6: "Snow/Ice",
    7: "Barren",
    8: "Impervious",
    9: "Wetland",
}


@dataclass(frozen=True)
class RasterLayer:
    """保存单期土地利用栅格及其空间元数据。"""

    year: int
    path: Path
    array: np.ndarray
    profile: dict
    nodata: int | float | None


@dataclass(frozen=True)
class ValidationResult:
    """保存一次滑动验证的核心结果。"""

    fit_from: int
    base_year: int
    target_year: int
    output_raster: Path
    overall_accuracy: float
    kappa: float
    total_pixels: int


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    常用方式：
        python scripts/markov_baseline.py --city shenyang
    """
    parser = argparse.ArgumentParser(
        description="CLCD original 9-class Markov baseline for CA-Markov setup."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("E:/CA-Markov"),
        help="Project root directory.",
    )
    parser.add_argument(
        "--city",
        default="shenyang",
        help="City name used in processed land-use filenames.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Ordered CLCD years to use.",
    )
    parser.add_argument(
        "--landuse-dir",
        type=Path,
        default=None,
        help="Directory containing {city}_clcd_v01_{year}_original.tif.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for predicted GeoTIFF outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for non-spatial Markov allocation.",
    )
    parser.add_argument(
        "--no-rasters",
        action="store_true",
        help="Only write CSV tables; skip predicted GeoTIFF outputs.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录和城市名补齐输入/输出路径。"""

    root = args.project_root
    if args.landuse_dir is None:
        args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    if args.tables_dir is None:
        args.tables_dir = root / "tables"
    if args.output_dir is None:
        args.output_dir = root / "output" / "markov_baseline" / args.city
    return args


def raster_path(landuse_dir: Path, city: str, year: int) -> Path:
    """拼出某城市某年份的已裁剪 CLCD 栅格路径。"""

    return landuse_dir / f"{city}_clcd_v01_{year}_original.tif"


def load_raster(path: Path, year: int) -> RasterLayer:
    """读取单个土地利用栅格到内存。

    当前沈阳裁剪结果体量不大，一次读入内存便于矩阵计算。
    如果后续换成更大区域，可以再改成分块处理。
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing land-use raster: {path}")

    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    return RasterLayer(year=year, path=path, array=array, profile=profile, nodata=nodata)


def load_layers(args: argparse.Namespace) -> dict[int, RasterLayer]:
    """读取所有年份栅格，并检查它们是否完全对齐。"""

    layers = {
        year: load_raster(raster_path(args.landuse_dir, args.city, year), year)
        for year in args.years
    }
    validate_alignment(layers)
    return layers


def alignment_signature(layer: RasterLayer) -> tuple:
    """提取用于判断栅格对齐的一组关键信息。"""

    profile = layer.profile
    transform = profile["transform"]
    return (
        profile["width"],
        profile["height"],
        str(profile["crs"]),
        tuple(round(v, 9) for v in transform),
    )


def validate_alignment(layers: dict[int, RasterLayer]) -> None:
    """检查所有年份的尺寸、CRS 和仿射变换是否一致。

    Markov 和后续 CA 都要求同一像元位置在不同年份代表同一地理位置。
    如果这里不一致，必须先重新裁剪/重采样，不能直接计算转移。
    """

    first_year = min(layers)
    first_signature = alignment_signature(layers[first_year])
    for year, layer in layers.items():
        if alignment_signature(layer) != first_signature:
            raise ValueError(
                f"Raster alignment differs: {first_year} and {year}. "
                "Run preprocessing before Markov analysis."
            )


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """返回所有输入数组中都属于 CLCD 1-9 类的像元位置。

    NoData=0 和其他异常值会在这里被排除，不参与转移矩阵或精度计算。
    """

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isin(array, CLASS_CODES)
    return mask


def transition_counts(from_array: np.ndarray, to_array: np.ndarray) -> np.ndarray:
    """计算两个年份之间的 9x9 转移数量矩阵。

    行表示起始年份类别，列表示目标年份类别。
    例如 matrix[0, 7] 表示 Cropland(1) 转为 Impervious(8) 的像元数。
    """

    mask = valid_mask(from_array, to_array)
    from_values = from_array[mask].astype(np.int16) - 1
    to_values = to_array[mask].astype(np.int16) - 1

    # 将二维的“from 类别 x to 类别”组合编码成一维索引，
    # 然后用 bincount 一次性统计，速度比逐像元循环快很多。
    flat_index = from_values * len(CLASS_CODES) + to_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    """把转移数量矩阵按行归一化为转移概率矩阵。

    每一行的和约等于 1，表示某一类土地下一期转为各类的概率。
    如果某一类在起始年份完全不存在，就让它保持自身，避免除零。
    """

    probabilities = np.zeros(counts.shape, dtype=np.float64)
    row_totals = counts.sum(axis=1)
    for row_index, total in enumerate(row_totals):
        if total > 0:
            probabilities[row_index] = counts[row_index] / total
        else:
            probabilities[row_index, row_index] = 1.0
    return probabilities


def matrix_csv_path(
    tables_dir: Path,
    city: str,
    kind: str,
    from_year: int,
    to_year: int,
) -> Path:
    """统一生成矩阵 CSV 的文件名。"""

    return tables_dir / f"{city}_markov_{kind}_{from_year}_{to_year}.csv"


def write_matrix_csv(path: Path, matrix: np.ndarray, value_format: str) -> None:
    """把转移数量矩阵或概率矩阵写成 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["from_code", "from_name"] + [f"to_{code}" for code in CLASS_CODES]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, from_code in enumerate(CLASS_CODES):
            row = {
                "from_code": int(from_code),
                "from_name": CLASS_NAMES[int(from_code)],
            }
            for col_index, to_code in enumerate(CLASS_CODES):
                value = matrix[row_index, col_index]
                row[f"to_{int(to_code)}"] = value_format.format(value)
            writer.writerow(row)


def pixel_area_km2(profile: dict) -> float:
    """根据栅格仿射变换计算单个像元面积，单位为平方公里。"""

    transform = profile["transform"]
    return abs(transform.a * transform.e - transform.b * transform.d) / 1_000_000


def class_counts(array: np.ndarray) -> np.ndarray:
    """统计单期栅格中 CLCD 1-9 类的像元数。"""

    counts = np.zeros(len(CLASS_CODES), dtype=np.int64)
    mask = np.isin(array, CLASS_CODES)
    values, value_counts = np.unique(array[mask], return_counts=True)
    for value, count in zip(values, value_counts, strict=True):
        counts[int(value) - 1] = int(count)
    return counts


def rounded_flow_counts(source_count: int, probabilities: np.ndarray) -> np.ndarray:
    """把某一类的转移概率换算成整数像元数。

    Markov 概率乘以像元数后通常会得到小数。这里先向下取整，
    再把剩余像元补给小数部分最大的类别，保证总数不丢失。
    """

    expected = probabilities * source_count
    counts = np.floor(expected).astype(np.int64)
    remainder = source_count - int(counts.sum())
    if remainder > 0:
        fractions = expected - counts
        order = np.argsort(-fractions, kind="stable")
        counts[order[:remainder]] += 1
    return counts


def projected_counts(base_array: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """根据基期面积和转移概率预测目标期各类像元总数。"""

    base_counts = class_counts(base_array)
    projected = np.zeros(len(CLASS_CODES), dtype=np.int64)
    for row_index, source_count in enumerate(base_counts):
        projected += rounded_flow_counts(int(source_count), probabilities[row_index])
    return projected


def allocate_markov_map(
    base_array: np.ndarray,
    probabilities: np.ndarray,
    nodata: int,
    seed: int,
) -> np.ndarray:
    """生成非空间 Markov 预测图。

    重要说明：这是 Markov 基线，不是 CA。
    它只按照转移概率控制“各类变化多少”，然后在同一来源类别的像元中
    随机分配目标类别；它没有考虑邻域、道路、水系、坡度等空间驱动因素。
    后续 CA-Markov 会替换/强化这一步的空间分配逻辑。
    """

    rng = np.random.default_rng(seed)
    prediction = np.full(base_array.shape, nodata, dtype=np.uint8)
    base_flat = base_array.reshape(-1)
    pred_flat = prediction.reshape(-1)

    for row_index, source_code in enumerate(CLASS_CODES):
        source_indices = np.flatnonzero(base_flat == source_code)
        if len(source_indices) == 0:
            continue

        # 在同一来源类别内部随机打乱像元顺序，使分配结果可复现但不含空间规则。
        rng.shuffle(source_indices)
        target_counts = rounded_flow_counts(len(source_indices), probabilities[row_index])

        start = 0
        for col_index, target_code in enumerate(CLASS_CODES):
            count = int(target_counts[col_index])
            if count == 0:
                continue
            end = start + count
            pred_flat[source_indices[start:end]] = target_code
            start = end

    return prediction


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """计算混淆矩阵。

    行表示真实类别，列表示预测类别。
    """

    mask = valid_mask(actual, predicted)
    actual_values = actual[mask].astype(np.int16) - 1
    predicted_values = predicted[mask].astype(np.int16) - 1
    flat_index = actual_values * len(CLASS_CODES) + predicted_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def accuracy_summary(confusion: np.ndarray) -> tuple[float, float, int]:
    """根据混淆矩阵计算总体精度 OA 和 Kappa 系数。"""

    total = int(confusion.sum())
    if total == 0:
        return 0.0, 0.0, 0

    correct = float(np.trace(confusion))
    overall = correct / total
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)

    # Kappa = (实际一致率 - 随机一致率) / (1 - 随机一致率)。
    expected = float(np.sum(row_totals * col_totals)) / float(total * total)
    kappa = (overall - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return overall, kappa, total


def write_confusion_csv(path: Path, confusion: np.ndarray) -> None:
    """把混淆矩阵写成 CSV，方便复制到论文表格或继续分析。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["actual_code", "actual_name"] + [
        f"predicted_{code}" for code in CLASS_CODES
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, actual_code in enumerate(CLASS_CODES):
            row = {
                "actual_code": int(actual_code),
                "actual_name": CLASS_NAMES[int(actual_code)],
            }
            for col_index, predicted_code in enumerate(CLASS_CODES):
                row[f"predicted_{int(predicted_code)}"] = int(
                    confusion[row_index, col_index]
                )
            writer.writerow(row)


def write_per_class_accuracy_csv(path: Path, confusion: np.ndarray) -> None:
    """输出每个类别的制图精度、用户精度和 F1 分数。

    producer_accuracy：真实为该类的像元中被正确预测的比例。
    user_accuracy：预测为该类的像元中真实也为该类的比例。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    actual_totals = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_code",
            "class_name",
            "actual_pixels",
            "predicted_pixels",
            "correct_pixels",
            "producer_accuracy",
            "user_accuracy",
            "f1_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, class_code in enumerate(CLASS_CODES):
            correct = int(confusion[index, index])
            actual = int(actual_totals[index])
            predicted = int(predicted_totals[index])
            producer = correct / actual if actual else 0.0
            user = correct / predicted if predicted else 0.0
            f1 = (
                2.0 * producer * user / (producer + user)
                if producer + user > 0
                else 0.0
            )
            writer.writerow(
                {
                    "class_code": int(class_code),
                    "class_name": CLASS_NAMES[int(class_code)],
                    "actual_pixels": actual,
                    "predicted_pixels": predicted,
                    "correct_pixels": correct,
                    "producer_accuracy": f"{producer:.6f}",
                    "user_accuracy": f"{user:.6f}",
                    "f1_score": f"{f1:.6f}",
                }
            )


def write_area_projection_csv(
    path: Path,
    base_array: np.ndarray,
    predicted_array: np.ndarray,
    actual_array: np.ndarray,
    markov_counts: np.ndarray,
    area_km2: float,
) -> None:
    """输出面积预测对比表。

    这个表用于检查 Markov 是否把各类面积总量预测合理。
    它不评价空间位置，只比较各类面积/像元数的变化。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    base_counts = class_counts(base_array)
    predicted_counts = class_counts(predicted_array)
    actual_counts = class_counts(actual_array)

    fieldnames = [
        "class_code",
        "class_name",
        "base_pixels",
        "markov_projected_pixels",
        "predicted_raster_pixels",
        "actual_pixels",
        "markov_projected_area_km2",
        "actual_area_km2",
        "area_difference_km2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, class_code in enumerate(CLASS_CODES):
            projected_area = float(markov_counts[index]) * area_km2
            actual_area = float(actual_counts[index]) * area_km2
            writer.writerow(
                {
                    "class_code": int(class_code),
                    "class_name": CLASS_NAMES[int(class_code)],
                    "base_pixels": int(base_counts[index]),
                    "markov_projected_pixels": int(markov_counts[index]),
                    "predicted_raster_pixels": int(predicted_counts[index]),
                    "actual_pixels": int(actual_counts[index]),
                    "markov_projected_area_km2": f"{projected_area:.6f}",
                    "actual_area_km2": f"{actual_area:.6f}",
                    "area_difference_km2": f"{projected_area - actual_area:.6f}",
                }
            )


def write_prediction_raster(path: Path, layer: RasterLayer, array: np.ndarray) -> None:
    """用基期栅格的空间参考写出预测 GeoTIFF。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = layer.profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def run_validation(
    args: argparse.Namespace,
    layers: dict[int, RasterLayer],
    fit_from: int,
    base_year: int,
    target_year: int,
) -> ValidationResult:
    """执行一次滑动窗口验证。

    例子：fit_from=2015, base_year=2020, target_year=2025
    表示用 2015->2020 的转移概率，从 2020 年图预测 2025 年图，
    然后与真实 2025 年 CLCD 进行精度评价。
    """

    # 1. 用历史相邻两期估计 Markov 转移概率。
    fit_counts = transition_counts(layers[fit_from].array, layers[base_year].array)
    probabilities = transition_probabilities(fit_counts)
    base_layer = layers[base_year]
    actual_layer = layers[target_year]

    # 2. 从基期土地利用图生成目标期预测图。
    predicted = allocate_markov_map(
        base_array=base_layer.array,
        probabilities=probabilities,
        nodata=0,
        seed=args.seed + target_year,
    )

    # 3. 分别评价面积数量和空间像元分类结果。
    markov_counts = projected_counts(base_layer.array, probabilities)
    confusion = confusion_matrix(actual_layer.array, predicted)
    overall, kappa, total = accuracy_summary(confusion)

    stem = f"{args.city}_markov_fit_{fit_from}_{base_year}_predict_{target_year}"
    raster_path_out = args.output_dir / f"{stem}.tif"
    if not args.no_rasters:
        write_prediction_raster(raster_path_out, base_layer, predicted)

    write_confusion_csv(args.tables_dir / f"{stem}_confusion_matrix.csv", confusion)
    write_per_class_accuracy_csv(
        args.tables_dir / f"{stem}_per_class_accuracy.csv", confusion
    )
    write_area_projection_csv(
        args.tables_dir / f"{stem}_area_projection.csv",
        base_array=base_layer.array,
        predicted_array=predicted,
        actual_array=actual_layer.array,
        markov_counts=markov_counts,
        area_km2=pixel_area_km2(base_layer.profile),
    )

    return ValidationResult(
        fit_from=fit_from,
        base_year=base_year,
        target_year=target_year,
        output_raster=raster_path_out,
        overall_accuracy=overall,
        kappa=kappa,
        total_pixels=total,
    )


def write_validation_summary(path: Path, results: list[ValidationResult]) -> None:
    """汇总所有滑动验证窗口的 OA/Kappa。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fit_from",
        "base_year",
        "target_year",
        "overall_accuracy",
        "kappa",
        "total_pixels",
        "output_raster",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "fit_from": result.fit_from,
                    "base_year": result.base_year,
                    "target_year": result.target_year,
                    "overall_accuracy": f"{result.overall_accuracy:.6f}",
                    "kappa": f"{result.kappa:.6f}",
                    "total_pixels": result.total_pixels,
                    "output_raster": str(result.output_raster),
                }
            )


def main() -> None:
    """脚本入口：读取数据、输出转移矩阵、执行滑动验证。"""

    args = resolve_paths(parse_args())
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = load_layers(args)

    print(f"City: {args.city}")
    print(f"Land-use rasters: {args.landuse_dir}")
    print(f"Years: {', '.join(str(year) for year in args.years)}")
    print(f"Tables: {args.tables_dir}")
    print(f"Output rasters: {args.output_dir}")
    print("")

    # 第一部分：输出每两个相邻年份之间的转移数量和转移概率。
    for from_year, to_year in zip(args.years[:-1], args.years[1:], strict=True):
        counts = transition_counts(layers[from_year].array, layers[to_year].array)
        probabilities = transition_probabilities(counts)
        counts_path = matrix_csv_path(
            args.tables_dir, args.city, "transition_counts", from_year, to_year
        )
        probabilities_path = matrix_csv_path(
            args.tables_dir, args.city, "transition_probabilities", from_year, to_year
        )
        write_matrix_csv(counts_path, counts, "{:.0f}")
        write_matrix_csv(probabilities_path, probabilities, "{:.8f}")
        print(f"Transition matrix: {from_year}->{to_year}")
        print(f"  {counts_path}")
        print(f"  {probabilities_path}")

    # 第二部分：滑动窗口验证。每次用前一段转移概率预测下一期。
    results: list[ValidationResult] = []
    for fit_from, base_year, target_year in zip(
        args.years[:-2], args.years[1:-1], args.years[2:], strict=True
    ):
        result = run_validation(args, layers, fit_from, base_year, target_year)
        results.append(result)
        print(
            f"Validation: fit {fit_from}->{base_year}, predict {target_year}; "
            f"OA={result.overall_accuracy:.4f}, Kappa={result.kappa:.4f}"
        )

    summary_path = args.tables_dir / f"{args.city}_markov_validation_summary.csv"
    write_validation_summary(summary_path, results)
    print("")
    print(f"Validation summary: {summary_path}")


if __name__ == "__main__":
    main()
