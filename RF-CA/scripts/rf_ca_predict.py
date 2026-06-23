"""RF 概率预测 + CA 空间分配脚本。

本脚本把已经训练好的随机森林模型用于 RF-CA 全图预测：

1. 读取 RF 模型和基期土地利用图；
2. 按模型训练时的 feature_names 为全图分块构建特征；
3. 用 RF 输出每个有效像元转为各土地利用类别的概率；
4. 用 Markov 转移矩阵控制目标年份各类别需求量；
5. 用 CA 邻域吸引力 + RF 概率进行空间分配；
6. 如果目标年份真实 CLCD 存在，则输出 OA、Kappa、per-class F1 和混淆矩阵。

典型任务：

- 2015->2020 训练的 RF，基于 2020 预测 2025，并与真实 2025 验证；
- 2020->2025 训练的 RF，基于 2025 预测 2030，输出未来预测图。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.ndimage import convolve
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


def configure_console_encoding() -> None:
    """尽量让 Windows 终端正确显示中文提示。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。"""

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"
    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_console_encoding()
configure_geospatial_data_paths()

# rasterio 依赖 GDAL，因此放在路径配置之后导入。
import rasterio
from rasterio.windows import Window


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

LANDUSE_COLORMAP = {
    0: (255, 255, 255, 0),
    1: (244, 224, 77, 255),
    2: (38, 115, 0, 255),
    3: (109, 187, 117, 255),
    4: (156, 195, 107, 255),
    5: (75, 156, 211, 255),
    6: (247, 251, 255, 255),
    7: (189, 189, 189, 255),
    8: (215, 25, 28, 255),
    9: (65, 182, 196, 255),
}


@dataclass(frozen=True)
class RasterLayer:
    """保存单期土地利用栅格和空间元数据。"""

    year: int
    path: Path
    array: np.ndarray
    profile: dict
    nodata: int | float | None


@dataclass(frozen=True)
class DriverPlan:
    """保存预测阶段使用的驱动因子路径。"""

    driver_year: int | None
    paths: dict[str, Path]
    warnings: list[str]


@dataclass(frozen=True)
class AccuracyResult:
    """保存全图验证精度。"""

    overall_accuracy: float
    kappa: float
    total_pixels: int


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="RF probability prediction + CA allocation.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/RF-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument(
        "--fit-from",
        type=int,
        default=2015,
        help="用于估计 Markov 需求和查找 RF 模型的训练起始年份。",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="CA 模拟起始年份，也是 RF 模型训练目标年份。",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2025,
        help="预测目标年份。若该年份 CLCD 存在，则自动做验证。",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="RF 模型 .joblib 路径。为空时按 fit-from/base-year 自动查找。",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="自动查找模型时匹配 trainseed。",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="自动查找模型时匹配 sampseed。",
    )
    parser.add_argument(
        "--driver-year",
        type=int,
        default=None,
        help="预测阶段使用的驱动因子年份。默认优先使用 base-year。",
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=5,
        help="CA 邻域窗口大小，必须是 >=3 的奇数。",
    )
    parser.add_argument("--iterations", type=int, default=5, help="CA 分配迭代次数。")
    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.7,
        help="候选像元评分中 RF 概率权重。",
    )
    parser.add_argument(
        "--neighbor-weight",
        type=float,
        default=0.3,
        help="候选像元评分中 CA 邻域吸引力权重。",
    )
    parser.add_argument(
        "--random-weight",
        type=float,
        default=0.02,
        help="候选像元评分中的随机扰动权重。",
    )
    parser.add_argument(
        "--probability-floor",
        type=float,
        default=1e-6,
        help="RF 未输出某类别概率时使用的极小概率，避免该类完全不可分配。",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=128,
        help="RF 全图概率预测时每个分块的行数。",
    )
    parser.add_argument("--seed", type=int, default=42, help="CA 随机种子。")
    parser.add_argument(
        "--landuse-dir",
        type=Path,
        default=None,
        help="土地利用栅格目录，默认 data/processed/landuse/{city}。",
    )
    parser.add_argument(
        "--driver-dir",
        type=Path,
        default=None,
        help="驱动因子栅格目录，默认 data/processed/drivers/{city}。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="预测 GeoTIFF 输出目录，默认 output/rf_ca/{city}。",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="评价表输出目录，默认 tables/rf_ca/{city}。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="概率 memmap 临时缓存目录，默认 output/rf_ca/{city}/_cache。",
    )
    parser.add_argument(
        "--keep-probability-cache",
        action="store_true",
        help="保留内部概率 memmap 文件，便于调试。默认运行结束后删除。",
    )
    parser.add_argument(
        "--write-probability-raster",
        action="store_true",
        help="额外写出 9 波段 RF 概率 GeoTIFF，文件较大。",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名输出。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查模型、输入栅格、驱动因子和输出路径，不做全图预测。",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录补齐路径。"""

    root = args.project_root
    if args.landuse_dir is None:
        args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    if args.driver_dir is None:
        args.driver_dir = root / "data" / "processed" / "drivers" / args.city
    if args.output_dir is None:
        args.output_dir = root / "output" / "rf_ca" / args.city
    if args.tables_dir is None:
        args.tables_dir = root / "tables" / "rf_ca" / args.city
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "_cache"
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查参数合法性。"""

    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        raise ValueError("--neighborhood-size 必须是 >=3 的奇数。")
    if args.iterations < 1:
        raise ValueError("--iterations 必须 >= 1。")
    if args.rf_weight < 0 or args.neighbor_weight < 0:
        raise ValueError("--rf-weight 和 --neighbor-weight 不能为负数。")
    if args.rf_weight + args.neighbor_weight <= 0:
        raise ValueError("RF 权重和邻域权重至少有一个必须大于 0。")
    if args.random_weight < 0:
        raise ValueError("--random-weight 不能为负数。")
    if args.probability_floor < 0:
        raise ValueError("--probability-floor 不能为负数。")
    if args.block_rows < 16:
        raise ValueError("--block-rows 建议 >= 16。")


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    """拼接某年份土地利用栅格路径。"""

    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def driver_path(args: argparse.Namespace, name: str, year: int) -> Path:
    """拼接某年份驱动因子路径。"""

    return args.driver_dir / f"{args.city}_{name}_{year}.tif"


def find_model_file(args: argparse.Namespace) -> Path:
    """自动查找 RF 模型文件。"""

    if args.model_file is not None:
        if not args.model_file.exists():
            raise FileNotFoundError(f"RF 模型不存在：{args.model_file}")
        return args.model_file

    model_dir = args.project_root / "models" / "random_forest" / args.city
    pattern = (
        f"{args.city}_rf_fit_{args.fit_from}_{args.base_year}_*"
        f"_sampseed{args.sample_seed}_*"
        f"_trainseed{args.model_seed}.joblib"
    )
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"未找到 RF 模型：{model_dir / pattern}\n"
            "请先运行 scripts/train_rf.py，或用 --model-file 指定模型。"
        )
    if len(matches) > 1:
        raise ValueError(
            "找到多个候选 RF 模型，请用 --model-file 明确指定：\n"
            + "\n".join(str(path) for path in matches)
        )
    return matches[0]


def load_model_artifact(path: Path) -> dict[str, Any]:
    """读取训练脚本保存的模型包。"""

    artifact = joblib.load(path)
    if "model" not in artifact or "feature_names" not in artifact:
        raise KeyError(f"模型包缺少 model 或 feature_names：{path}")
    return artifact


def read_raster(path: Path, year: int) -> RasterLayer:
    """读取单波段土地利用栅格到内存。"""

    if not path.exists():
        raise FileNotFoundError(f"土地利用栅格不存在：{path}")
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    return RasterLayer(year=year, path=path, array=array, profile=profile, nodata=nodata)


def inspect_raster(path: Path) -> dict[str, Any]:
    """只读取栅格元数据。"""

    if not path.exists():
        raise FileNotFoundError(f"栅格不存在：{path}")
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": tuple(round(value, 9) for value in src.transform),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def alignment_signature(profile: dict) -> tuple:
    """提取用于检查栅格对齐的信息。"""

    transform = profile["transform"]
    return (
        profile["width"],
        profile["height"],
        str(profile["crs"]),
        tuple(round(value, 9) for value in transform),
    )


def validate_alignment(reference: RasterLayer, others: list[RasterLayer]) -> None:
    """检查所有土地利用栅格是否完全对齐。"""

    signature = alignment_signature(reference.profile)
    for layer in others:
        if alignment_signature(layer.profile) != signature:
            raise ValueError(f"栅格未对齐：{reference.path} vs {layer.path}")


def validate_driver_alignment(reference_profile: dict, path: Path) -> None:
    """检查驱动因子是否与土地利用图对齐。"""

    meta = inspect_raster(path)
    signature = (
        reference_profile["width"],
        reference_profile["height"],
        str(reference_profile["crs"]),
        tuple(round(value, 9) for value in reference_profile["transform"]),
    )
    driver_signature = (
        meta["width"],
        meta["height"],
        meta["crs"],
        meta["transform"],
    )
    if driver_signature != signature:
        raise ValueError(f"驱动因子未与土地利用图对齐：{path}")


def feature_driver_names(feature_names: list[str]) -> list[str]:
    """从模型特征名中解析所需驱动因子名称。"""

    names = []
    for feature in feature_names:
        if not feature.startswith("driver_"):
            continue
        body = feature.removeprefix("driver_")
        match = re.match(r"(.+)_\d{4}$", body)
        name = match.group(1) if match else body
        if name not in names:
            names.append(name)
    return names


def resolve_driver_plan(
    args: argparse.Namespace,
    feature_names: list[str],
    sample_metadata: dict[str, Any],
) -> DriverPlan:
    """确定预测阶段使用哪一年驱动因子。"""

    needed = feature_driver_names(feature_names)
    if not needed:
        return DriverPlan(driver_year=None, paths={}, warnings=[])

    warnings: list[str] = []
    trained_driver_year = sample_metadata.get("driver_year")
    candidate_years = (
        [args.driver_year]
        if args.driver_year is not None
        else [args.base_year, trained_driver_year]
    )
    candidate_years = [int(year) for year in candidate_years if year is not None]

    selected: int | None = None
    for year in candidate_years:
        if all(driver_path(args, name, year).exists() for name in needed):
            selected = year
            break
    if selected is None:
        raise FileNotFoundError(
            "未找到完整预测驱动因子。需要：\n"
            + "\n".join(f"- {name}" for name in needed)
            + "\n可用 --driver-year 指定年份，或先复制/生成相应驱动因子。"
        )

    if args.driver_year is None and selected != args.base_year:
        warnings.append(
            f"未找到 {args.base_year} 年完整驱动因子，已改用训练样本中的 {selected} 年驱动因子。"
        )
    return DriverPlan(
        driver_year=selected,
        paths={name: driver_path(args, name, selected) for name in needed},
        warnings=warnings,
    )


def valid_landuse_mask(*arrays: np.ndarray) -> np.ndarray:
    """返回所有数组中均为 CLCD 1-9 类的位置。"""

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isin(array, CLASS_CODES)
    return mask


def compute_driver_valid_mask(paths: dict[str, Path], shape: tuple[int, int]) -> np.ndarray:
    """计算所有驱动因子都有有效值的位置。"""

    mask = np.ones(shape, dtype=bool)
    for path in paths.values():
        with rasterio.open(path) as src:
            data = src.read(1, masked=True).astype(np.float32)
            valid = ~np.ma.getmaskarray(data)
            values = np.asarray(data.filled(np.nan), dtype=np.float32)
            valid &= np.isfinite(values)
            if src.nodata is not None:
                valid &= values != float(src.nodata)
            mask &= valid
    return mask


def transition_counts(
    from_array: np.ndarray,
    to_array: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """计算 9x9 Markov 转移数量矩阵。"""

    mask = valid & valid_landuse_mask(from_array, to_array)
    from_values = from_array[mask].astype(np.int16) - 1
    to_values = to_array[mask].astype(np.int16) - 1
    flat_index = from_values * len(CLASS_CODES) + to_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    """把转移数量矩阵按行归一化为转移概率矩阵。"""

    probabilities = np.zeros(counts.shape, dtype=np.float64)
    row_totals = counts.sum(axis=1)
    for row_index, total in enumerate(row_totals):
        if total > 0:
            probabilities[row_index] = counts[row_index] / total
        else:
            probabilities[row_index, row_index] = 1.0
    return probabilities


def class_counts(array: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """统计 CLCD 1-9 类像元数。"""

    if valid is None:
        valid = np.isin(array, CLASS_CODES)
    counts = np.zeros(len(CLASS_CODES), dtype=np.int64)
    values, value_counts = np.unique(array[valid], return_counts=True)
    for value, count in zip(values, value_counts, strict=True):
        if int(value) in CLASS_NAMES:
            counts[int(value) - 1] = int(count)
    return counts


def rounded_flow_counts(source_count: int, probabilities: np.ndarray) -> np.ndarray:
    """把概率转成整数像元数，并保证总数不丢失。"""

    expected = probabilities * source_count
    counts = np.floor(expected).astype(np.int64)
    remainder = source_count - int(counts.sum())
    if remainder > 0:
        fractions = expected - counts
        order = np.argsort(-fractions, kind="stable")
        counts[order[:remainder]] += 1
    return counts


def markov_demand(
    base_array: np.ndarray,
    valid: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """根据基期图和 Markov 概率计算目标期类别需求量。"""

    base_counts = class_counts(base_array, valid)
    demand = np.zeros(len(CLASS_CODES), dtype=np.int64)
    for source_index, source_count in enumerate(base_counts):
        demand += rounded_flow_counts(int(source_count), probabilities[source_index])
    return demand


def format_weight(value: float) -> str:
    """把权重转为适合文件名的字符串。"""

    percent = value * 100
    rounded = round(percent)
    if abs(percent - rounded) < 1e-8:
        return f"{int(rounded):03d}"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def model_tag(model_file: Path) -> str:
    """从模型文件名中提取 RF 参数标签。"""

    match = re.search(r"_rf\d+_depth[^_]+_leaf\d+", model_file.stem)
    return match.group(0).lstrip("_") if match else "rfmodel"


def output_stem(args: argparse.Namespace, model_file: Path, driver_year: int | None) -> str:
    """生成 RF-CA 输出文件名前缀。"""

    driver_part = f"driver{driver_year}" if driver_year is not None else "nodriver"
    return (
        f"{args.city}_rf_ca_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_{driver_part}_n{args.neighborhood_size}"
        f"_i{args.iterations}_{model_tag(model_file)}"
        f"_rw{format_weight(args.rf_weight)}_nw{format_weight(args.neighbor_weight)}"
        f"_seed{args.seed}"
    )


def output_paths(args: argparse.Namespace, stem: str, has_observed_target: bool) -> dict[str, Path]:
    """生成输出路径。"""

    paths = {
        "prediction": args.output_dir / f"{stem}.tif",
        "summary": args.tables_dir / f"{stem}_summary.csv",
        "area": args.tables_dir / f"{stem}_area_projection.csv",
        "simulation_log": args.tables_dir / f"{stem}_simulation_log.csv",
        "metadata": args.tables_dir / f"{stem}_metadata.json",
        "probability_cache": args.cache_dir / f"{stem}_probabilities.dat",
    }
    if has_observed_target:
        paths["confusion"] = args.tables_dir / f"{stem}_confusion_matrix.csv"
        paths["per_class"] = args.tables_dir / f"{stem}_per_class_accuracy.csv"
    if args.write_probability_raster:
        paths["probability_raster"] = args.output_dir / f"{stem}_rf_probabilities.tif"
    return paths


def ensure_outputs_available(paths: dict[str, Path], overwrite: bool) -> None:
    """防止输出文件被意外覆盖。"""

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "输出文件已存在，为避免覆盖请调整参数或使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )


def window_counts_full(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """用卷积计算全图邻域数量。"""

    return convolve(mask.astype(np.float32), kernel, mode="constant", cval=0.0)


def neighbor_fraction(
    array: np.ndarray,
    target_code: int,
    valid: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    """计算全图中目标类别的邻域比例。"""

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)
    kernel[neighborhood_size // 2, neighborhood_size // 2] = 0.0
    target_neighbors = window_counts_full(array == target_code, kernel)
    valid_neighbors = window_counts_full(valid, kernel)
    return np.divide(
        target_neighbors,
        valid_neighbors,
        out=np.zeros(array.shape, dtype=np.float32),
        where=valid_neighbors > 0,
    )


def block_neighbor_features(
    base_array: np.ndarray,
    valid: np.ndarray,
    row_start: int,
    row_stop: int,
    neighborhood_size: int,
) -> dict[int, np.ndarray]:
    """为一个行分块计算邻域 9 类比例。"""

    radius = neighborhood_size // 2
    height = base_array.shape[0]
    halo_top = max(row_start - radius, 0)
    halo_bottom = min(row_stop + radius, height)
    local_start = row_start - halo_top
    local_stop = local_start + (row_stop - row_start)

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)
    kernel[radius, radius] = 0.0
    valid_halo = valid[halo_top:halo_bottom, :]
    valid_neighbors = convolve(
        valid_halo.astype(np.float32),
        kernel,
        mode="constant",
        cval=0.0,
    )[local_start:local_stop, :]
    valid_neighbors = np.maximum(valid_neighbors, 1.0)

    features: dict[int, np.ndarray] = {}
    base_halo = base_array[halo_top:halo_bottom, :]
    for code in CLASS_CODES:
        counts = convolve(
            (base_halo == code).astype(np.float32),
            kernel,
            mode="constant",
            cval=0.0,
        )[local_start:local_stop, :]
        features[int(code)] = np.clip(counts / valid_neighbors, 0.0, 1.0).astype(
            np.float32
        )
    return features


def parse_feature_driver_name(feature: str) -> str:
    """从 driver_* 特征名中解析驱动因子名称，忽略训练年份后缀。"""

    body = feature.removeprefix("driver_")
    match = re.match(r"(.+)_\d{4}$", body)
    return match.group(1) if match else body


def build_block_features(
    feature_names: list[str],
    base_array: np.ndarray,
    valid: np.ndarray,
    driver_blocks: dict[str, np.ndarray],
    row_start: int,
    row_stop: int,
    neighborhood_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按模型特征名顺序为一个分块构建特征矩阵。"""

    block = base_array[row_start:row_stop, :]
    block_valid = valid[row_start:row_stop, :]
    rows_local, cols = np.nonzero(block_valid)
    if len(rows_local) == 0:
        return (
            np.empty((0, len(feature_names)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    rows = rows_local + row_start
    current = block[rows_local, cols].astype(np.uint8)
    neighbor = block_neighbor_features(
        base_array,
        valid,
        row_start=row_start,
        row_stop=row_stop,
        neighborhood_size=neighborhood_size,
    )

    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    height, width = base_array.shape
    for col_index, feature in enumerate(feature_names):
        if feature == "current_class_code":
            x[:, col_index] = current.astype(np.float32)
        elif feature.startswith("current_is_"):
            code = int(feature.split("_")[2])
            x[:, col_index] = (current == code).astype(np.float32)
        elif feature.startswith("neighbor_frac_"):
            code = int(feature.split("_")[2])
            x[:, col_index] = neighbor[code][rows_local, cols]
        elif feature.startswith("driver_"):
            driver_name = parse_feature_driver_name(feature)
            if driver_name not in driver_blocks:
                raise KeyError(f"缺少驱动因子分块：{driver_name}")
            x[:, col_index] = driver_blocks[driver_name][rows_local, cols].astype(
                np.float32
            )
        elif feature == "row_norm":
            x[:, col_index] = rows.astype(np.float32) / max(height - 1, 1)
        elif feature == "col_norm":
            x[:, col_index] = cols.astype(np.float32) / max(width - 1, 1)
        else:
            raise ValueError(f"无法识别模型特征：{feature}")
    return x, rows.astype(np.int32), cols.astype(np.int32)


def predict_probability_maps(
    args: argparse.Namespace,
    model: Any,
    feature_names: list[str],
    base_layer: RasterLayer,
    driver_plan: DriverPlan,
    valid: np.ndarray,
    cache_path: Path,
) -> np.memmap:
    """分块预测 RF 概率，并写入 memmap。"""

    height, width = base_layer.array.shape
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.memmap(
        cache_path,
        dtype="float32",
        mode="w+",
        shape=(len(CLASS_CODES), height, width),
    )
    probabilities[:] = np.float32(args.probability_floor)
    for band_index in range(len(CLASS_CODES)):
        probabilities[band_index][~valid] = 0.0

    class_to_band = {int(code): index for index, code in enumerate(CLASS_CODES)}
    model_classes = [int(code) for code in model.classes_]

    with ExitStack() as stack:
        driver_sources = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in driver_plan.paths.items()
        }
        for row_start in range(0, height, args.block_rows):
            row_stop = min(row_start + args.block_rows, height)
            window = Window(0, row_start, width, row_stop - row_start)
            driver_blocks = {
                name: src.read(1, window=window).astype(np.float32)
                for name, src in driver_sources.items()
            }
            x, rows, cols = build_block_features(
                feature_names=feature_names,
                base_array=base_layer.array,
                valid=valid,
                driver_blocks=driver_blocks,
                row_start=row_start,
                row_stop=row_stop,
                neighborhood_size=args.neighborhood_size,
            )
            if len(rows) == 0:
                continue

            block_probs = model.predict_proba(x)
            for model_col, class_code in enumerate(model_classes):
                if class_code not in class_to_band:
                    continue
                probabilities[class_to_band[class_code], rows, cols] = block_probs[
                    :, model_col
                ].astype(np.float32)
            print(f"RF probability block: rows {row_start}-{row_stop}")

    probabilities.flush()
    return probabilities


def apportion_with_caps(total: int, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """按权重分配整数数量，同时不超过供给上限。"""

    total = int(min(total, int(caps.sum())))
    allocation = np.zeros(len(weights), dtype=np.int64)
    if total <= 0:
        return allocation

    remaining = total
    remaining_caps = caps.astype(np.int64).copy()
    weights = weights.astype(np.float64).copy()
    while remaining > 0 and remaining_caps.sum() > 0:
        active = remaining_caps > 0
        active_weights = np.where(active, weights, 0.0)
        if active_weights.sum() <= 0:
            active_weights = remaining_caps.astype(np.float64)

        expected = active_weights / active_weights.sum() * remaining
        proposal = np.floor(expected).astype(np.int64)
        proposal = np.minimum(proposal, remaining_caps)
        if proposal.sum() == 0:
            order = np.argsort(-np.where(active, expected, -1.0), kind="stable")
            for index in order:
                if remaining <= 0:
                    break
                if remaining_caps[index] <= 0:
                    continue
                proposal[index] += 1
                remaining_caps[index] -= 1
                remaining -= 1
            allocation += proposal
            continue

        allocation += proposal
        remaining_caps -= proposal
        remaining -= int(proposal.sum())
    return allocation


def select_best_source_cells(
    prediction_flat: np.ndarray,
    source_code: int,
    score_flat: np.ndarray,
    count: int,
) -> np.ndarray:
    """在某个来源类别中选择综合评分最高的像元。"""

    if count <= 0:
        return np.array([], dtype=np.int64)
    positions = np.flatnonzero(prediction_flat == source_code)
    if len(positions) == 0:
        return np.array([], dtype=np.int64)
    if count >= len(positions):
        return positions
    scores = score_flat[positions]
    top_local = np.argpartition(scores, -count)[-count:]
    return positions[top_local]


def allocate_target_class(
    prediction: np.ndarray,
    target_index: int,
    target_need: int,
    demand: np.ndarray,
    transition_probs: np.ndarray,
    rf_probabilities: np.memmap,
    valid: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    """为某个目标类别分配本轮新增像元。"""

    if target_need <= 0:
        return 0

    current_counts = class_counts(prediction, valid)
    surplus = np.maximum(current_counts - demand, 0)
    source_indices = [
        index
        for index in range(len(CLASS_CODES))
        if index != target_index and surplus[index] > 0
    ]
    if not source_indices:
        return 0

    caps = surplus[source_indices]
    weights = caps.astype(np.float64) * transition_probs[source_indices, target_index]
    if weights.sum() <= 0:
        weights = caps.astype(np.float64)
    source_take = apportion_with_caps(target_need, weights, caps)
    if source_take.sum() <= 0:
        return 0

    target_code = int(CLASS_CODES[target_index])
    neighbor = neighbor_fraction(
        prediction,
        target_code=target_code,
        valid=valid,
        neighborhood_size=args.neighborhood_size,
    )
    weight_total = args.rf_weight + args.neighbor_weight
    score = (
        args.rf_weight / weight_total * rf_probabilities[target_index]
        + args.neighbor_weight / weight_total * neighbor
    ).astype(np.float32)
    if args.random_weight > 0:
        score += rng.random(score.shape, dtype=np.float32) * np.float32(
            args.random_weight
        )
    score[~valid] = -1.0

    prediction_flat = prediction.reshape(-1)
    score_flat = score.reshape(-1)
    changed = 0
    for source_index, take in zip(source_indices, source_take, strict=True):
        take = int(take)
        if take <= 0:
            continue
        source_code = int(CLASS_CODES[source_index])
        selected = select_best_source_cells(
            prediction_flat=prediction_flat,
            source_code=source_code,
            score_flat=score_flat,
            count=take,
        )
        if len(selected) == 0:
            continue
        prediction_flat[selected] = target_code
        changed += int(len(selected))
    return changed


def simulation_log_rows(
    iteration: int,
    changed_pixels: int,
    counts: np.ndarray,
    demand: np.ndarray,
) -> list[dict[str, Any]]:
    """整理每轮 CA 模拟后各类数量。"""

    rows = []
    for index, code in enumerate(CLASS_CODES):
        rows.append(
            {
                "iteration": iteration,
                "changed_pixels": int(changed_pixels),
                "class_code": int(code),
                "class_name": CLASS_NAMES[int(code)],
                "predicted_pixels": int(counts[index]),
                "markov_demand_pixels": int(demand[index]),
                "difference_pixels": int(counts[index] - demand[index]),
            }
        )
    return rows


def repair_remaining_demand(
    prediction: np.ndarray,
    demand: np.ndarray,
    transition_probs: np.ndarray,
    rf_probabilities: np.memmap,
    valid: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    """修补迭代结束后仍未满足的需求。"""

    changed_total = 0
    for _ in range(len(CLASS_CODES)):
        current_counts = class_counts(prediction, valid)
        deficits = np.maximum(demand - current_counts, 0)
        if deficits.sum() == 0:
            break
        changed_round = 0
        for target_index in np.argsort(-deficits, kind="stable"):
            current_counts = class_counts(prediction, valid)
            need = max(int(demand[target_index] - current_counts[target_index]), 0)
            if need <= 0:
                continue
            changed_round += allocate_target_class(
                prediction=prediction,
                target_index=int(target_index),
                target_need=need,
                demand=demand,
                transition_probs=transition_probs,
                rf_probabilities=rf_probabilities,
                valid=valid,
                args=args,
                rng=rng,
            )
        changed_total += changed_round
        if changed_round == 0:
            break
    return changed_total


def simulate_rf_ca(
    base_array: np.ndarray,
    demand: np.ndarray,
    transition_probs: np.ndarray,
    rf_probabilities: np.memmap,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """执行 RF-CA 空间分配。"""

    rng = np.random.default_rng(args.seed)
    prediction = np.full(base_array.shape, 0, dtype=np.uint8)
    prediction[valid] = base_array[valid]
    log_rows: list[dict[str, Any]] = []

    for iteration in range(1, args.iterations + 1):
        current_counts = class_counts(prediction, valid)
        deficits = np.maximum(demand - current_counts, 0)
        if deficits.sum() == 0:
            break

        remaining_iterations = args.iterations - iteration + 1
        step_needs = np.ceil(deficits / remaining_iterations).astype(np.int64)
        changed_this_iteration = 0
        for target_index in np.argsort(-step_needs, kind="stable"):
            current_counts = class_counts(prediction, valid)
            remaining_need = max(int(demand[target_index] - current_counts[target_index]), 0)
            target_need = min(int(step_needs[target_index]), remaining_need)
            if target_need <= 0:
                continue
            changed_this_iteration += allocate_target_class(
                prediction=prediction,
                target_index=int(target_index),
                target_need=target_need,
                demand=demand,
                transition_probs=transition_probs,
                rf_probabilities=rf_probabilities,
                valid=valid,
                args=args,
                rng=rng,
            )

        after_counts = class_counts(prediction, valid)
        log_rows.extend(
            simulation_log_rows(
                iteration=iteration,
                changed_pixels=changed_this_iteration,
                counts=after_counts,
                demand=demand,
            )
        )
        print(f"CA iteration {iteration}: changed={changed_this_iteration}")
        if changed_this_iteration == 0:
            break

    final_changed = repair_remaining_demand(
        prediction=prediction,
        demand=demand,
        transition_probs=transition_probs,
        rf_probabilities=rf_probabilities,
        valid=valid,
        args=args,
        rng=rng,
    )
    if final_changed > 0:
        after_counts = class_counts(prediction, valid)
        log_rows.extend(
            simulation_log_rows(
                iteration=args.iterations + 1,
                changed_pixels=final_changed,
                counts=after_counts,
                demand=demand,
            )
        )
        print(f"CA repair: changed={final_changed}")
    return prediction, log_rows


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """计算全图混淆矩阵，行是真实类别，列是预测类别。"""

    mask = valid & valid_landuse_mask(actual, predicted)
    actual_values = actual[mask].astype(np.int16) - 1
    predicted_values = predicted[mask].astype(np.int16) - 1
    flat_index = actual_values * len(CLASS_CODES) + predicted_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def accuracy_summary(confusion: np.ndarray) -> AccuracyResult:
    """根据混淆矩阵计算 OA 和 Kappa。"""

    total = int(confusion.sum())
    if total == 0:
        return AccuracyResult(0.0, 0.0, 0)
    overall = float(np.trace(confusion)) / float(total)
    y_true = np.repeat(CLASS_CODES, confusion.sum(axis=1).astype(np.int64))
    y_pred_parts = []
    for row_index in range(len(CLASS_CODES)):
        y_pred_parts.append(
            np.repeat(CLASS_CODES, confusion[row_index].astype(np.int64))
        )
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.uint8)
    kappa = cohen_kappa_score(y_true, y_pred, labels=CLASS_CODES) if total else 0.0
    return AccuracyResult(overall_accuracy=overall, kappa=float(kappa), total_pixels=total)


def pixel_area_km2(profile: dict) -> float:
    """计算单个像元面积，单位平方公里。"""

    transform = profile["transform"]
    return abs(transform.a * transform.e - transform.b * transform.d) / 1_000_000


def write_prediction_raster(path: Path, reference: RasterLayer, prediction: np.ndarray) -> None:
    """写出 RF-CA 预测图。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
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
        dst.write(prediction.astype(np.uint8), 1)
        dst.write_colormap(1, LANDUSE_COLORMAP)


def write_probability_raster(path: Path, reference: RasterLayer, probabilities: np.memmap) -> None:
    """写出 9 波段 RF 概率 GeoTIFF。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(
        {
            "count": len(CLASS_CODES),
            "dtype": "float32",
            "nodata": 0.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        for band_index in range(len(CLASS_CODES)):
            dst.write(probabilities[band_index].astype(np.float32), band_index + 1)
            dst.set_band_description(
                band_index + 1,
                f"prob_{int(CLASS_CODES[band_index])}_{CLASS_NAMES[int(CLASS_CODES[band_index])]}",
            )


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    """写出混淆矩阵 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["actual_code", "actual_name"] + [
        f"predicted_{int(code)}" for code in CLASS_CODES
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, code in enumerate(CLASS_CODES):
            row = {"actual_code": int(code), "actual_name": CLASS_NAMES[int(code)]}
            for col_index, pred_code in enumerate(CLASS_CODES):
                row[f"predicted_{int(pred_code)}"] = int(matrix[row_index, col_index])
            writer.writerow(row)


def write_per_class_csv(path: Path, matrix: np.ndarray) -> None:
    """写出各类别 precision、recall、F1。"""

    y_true = np.repeat(CLASS_CODES, matrix.sum(axis=1).astype(np.int64))
    y_pred_parts = []
    for row_index in range(len(CLASS_CODES)):
        y_pred_parts.append(np.repeat(CLASS_CODES, matrix[row_index].astype(np.int64)))
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.uint8)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        zero_division=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_code",
            "class_name",
            "actual_pixels",
            "predicted_pixels",
            "correct_pixels",
            "precision",
            "recall",
            "f1_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        predicted_totals = matrix.sum(axis=0)
        for index, code in enumerate(CLASS_CODES):
            writer.writerow(
                {
                    "class_code": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "actual_pixels": int(support[index]),
                    "predicted_pixels": int(predicted_totals[index]),
                    "correct_pixels": int(matrix[index, index]),
                    "precision": f"{float(precision[index]):.6f}",
                    "recall": f"{float(recall[index]):.6f}",
                    "f1_score": f"{float(f1[index]):.6f}",
                }
            )


def write_area_projection_csv(
    path: Path,
    base: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray | None,
    valid: np.ndarray,
    demand: np.ndarray,
    area_km2: float,
) -> None:
    """写出面积/数量对比表。"""

    base_counts = class_counts(base, valid)
    pred_counts = class_counts(prediction, valid)
    target_counts = class_counts(target, valid) if target is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_code",
            "class_name",
            "base_pixels",
            "markov_demand_pixels",
            "rf_ca_predicted_pixels",
            "actual_pixels",
            "base_area_km2",
            "markov_demand_area_km2",
            "rf_ca_predicted_area_km2",
            "actual_area_km2",
            "rf_ca_area_difference_km2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, code in enumerate(CLASS_CODES):
            actual = int(target_counts[index]) if target_counts is not None else None
            pred_area = float(pred_counts[index]) * area_km2
            actual_area = float(actual) * area_km2 if actual is not None else None
            writer.writerow(
                {
                    "class_code": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "base_pixels": int(base_counts[index]),
                    "markov_demand_pixels": int(demand[index]),
                    "rf_ca_predicted_pixels": int(pred_counts[index]),
                    "actual_pixels": actual if actual is not None else "",
                    "base_area_km2": f"{float(base_counts[index]) * area_km2:.6f}",
                    "markov_demand_area_km2": f"{float(demand[index]) * area_km2:.6f}",
                    "rf_ca_predicted_area_km2": f"{pred_area:.6f}",
                    "actual_area_km2": f"{actual_area:.6f}" if actual_area is not None else "",
                    "rf_ca_area_difference_km2": (
                        f"{pred_area - actual_area:.6f}" if actual_area is not None else ""
                    ),
                }
            )


def write_simulation_log_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CA 分配日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "changed_pixels",
        "class_code",
        "class_name",
        "predicted_pixels",
        "markov_demand_pixels",
        "difference_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    model_file: Path,
    driver_plan: DriverPlan,
    accuracy: AccuracyResult | None,
    demand_error: int,
    prediction_path: Path,
) -> None:
    """写出本次 RF-CA 实验总体摘要。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "city",
            "mode",
            "has_observed_target",
            "fit_from",
            "base_year",
            "target_year",
            "driver_year",
            "neighborhood_size",
            "iterations",
            "rf_weight",
            "neighbor_weight",
            "random_weight",
            "overall_accuracy",
            "kappa",
            "total_pixels",
            "max_abs_demand_error_pixels",
            "model_file",
            "prediction_raster",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "city": args.city,
                "mode": "validation" if accuracy is not None else "projection",
                "has_observed_target": "yes" if accuracy is not None else "no",
                "fit_from": args.fit_from,
                "base_year": args.base_year,
                "target_year": args.target_year,
                "driver_year": driver_plan.driver_year if driver_plan.driver_year is not None else "",
                "neighborhood_size": args.neighborhood_size,
                "iterations": args.iterations,
                "rf_weight": args.rf_weight,
                "neighbor_weight": args.neighbor_weight,
                "random_weight": args.random_weight,
                "overall_accuracy": f"{accuracy.overall_accuracy:.6f}" if accuracy else "",
                "kappa": f"{accuracy.kappa:.6f}" if accuracy else "",
                "total_pixels": accuracy.total_pixels if accuracy else "",
                "max_abs_demand_error_pixels": demand_error,
                "model_file": str(model_file),
                "prediction_raster": str(prediction_path),
            }
        )


def write_metadata_json(
    path: Path,
    args: argparse.Namespace,
    artifact: dict[str, Any],
    model_file: Path,
    driver_plan: DriverPlan,
    outputs: dict[str, Path],
) -> None:
    """写出 RF-CA 预测元数据。"""

    payload = {
        "script": "scripts/rf_ca_predict.py",
        "model_file": str(model_file),
        "model_sample_metadata": artifact.get("sample_metadata", {}),
        "model_feature_names": [str(name) for name in artifact["feature_names"]],
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "driver_year": driver_plan.driver_year,
        "driver_paths": {key: str(value) for key, value in driver_plan.paths.items()},
        "warnings": driver_plan.warnings,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_dry_run(
    args: argparse.Namespace,
    artifact: dict[str, Any],
    model_file: Path,
    driver_plan: DriverPlan,
    outputs: dict[str, Path],
) -> None:
    """输出 dry-run 检查信息。"""

    fit_meta = inspect_raster(landuse_path(args, args.fit_from))
    base_meta = inspect_raster(landuse_path(args, args.base_year))
    if (
        fit_meta["width"],
        fit_meta["height"],
        fit_meta["crs"],
        fit_meta["transform"],
    ) != (
        base_meta["width"],
        base_meta["height"],
        base_meta["crs"],
        base_meta["transform"],
    ):
        raise ValueError("fit_from 与 base_year 土地利用栅格未对齐。")

    target_path = landuse_path(args, args.target_year)
    has_target = target_path.exists()
    if has_target:
        target_meta = inspect_raster(target_path)
        if (
            base_meta["width"],
            base_meta["height"],
            base_meta["crs"],
            base_meta["transform"],
        ) != (
            target_meta["width"],
            target_meta["height"],
            target_meta["crs"],
            target_meta["transform"],
        ):
            raise ValueError("target_year 土地利用栅格未与 base_year 对齐。")

    for path in driver_plan.paths.values():
        validate_driver_alignment(
            {
                "width": base_meta["width"],
                "height": base_meta["height"],
                "crs": base_meta["crs"],
                "transform": base_meta["transform"],
            },
            path,
        )

    print("Dry-run OK")
    print(f"Model: {model_file}")
    print(f"RF classes: {[int(code) for code in artifact['model'].classes_]}")
    print(f"Feature count: {len(artifact['feature_names'])}")
    print(f"Fit transition: {args.fit_from}->{args.base_year}")
    print(f"Predict target: {args.base_year}->{args.target_year}")
    print(f"Observed target exists: {'yes' if has_target else 'no'}")
    print(f"Raster shape: {base_meta['width']} x {base_meta['height']}")
    print(f"Driver year: {driver_plan.driver_year if driver_plan.driver_year is not None else 'none'}")
    for warning in driver_plan.warnings:
        print(f"Warning: {warning}")
    for name, path in driver_plan.paths.items():
        print(f"Driver {name}: {path}")
    print("Planned outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


def main() -> None:
    """脚本入口。"""

    args = resolve_paths(parse_args())
    validate_args(args)
    model_file = find_model_file(args)
    artifact = load_model_artifact(model_file)
    feature_names = [str(name) for name in artifact["feature_names"]]
    driver_plan = resolve_driver_plan(args, feature_names, artifact.get("sample_metadata", {}))
    stem = output_stem(args, model_file, driver_plan.driver_year)
    has_observed_target = landuse_path(args, args.target_year).exists()
    outputs = output_paths(args, stem, has_observed_target=has_observed_target)

    if args.dry_run:
        print_dry_run(args, artifact, model_file, driver_plan, outputs)
        return

    ensure_outputs_available(outputs, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    fit_layer = read_raster(landuse_path(args, args.fit_from), args.fit_from)
    base_layer = read_raster(landuse_path(args, args.base_year), args.base_year)
    target_path = landuse_path(args, args.target_year)
    target_layer = (
        read_raster(target_path, args.target_year) if target_path.exists() else None
    )
    validate_alignment(
        fit_layer,
        [base_layer] + ([target_layer] if target_layer is not None else []),
    )
    for path in driver_plan.paths.values():
        validate_driver_alignment(base_layer.profile, path)

    valid = valid_landuse_mask(fit_layer.array, base_layer.array)
    if driver_plan.paths:
        valid &= compute_driver_valid_mask(driver_plan.paths, base_layer.array.shape)
    if int(valid.sum()) == 0:
        raise ValueError("没有有效像元可用于 RF-CA 预测。")

    counts = transition_counts(fit_layer.array, base_layer.array, valid)
    transition_probs = transition_probabilities(counts)
    demand = markov_demand(base_layer.array, valid, transition_probs)

    print(f"Model: {model_file}")
    print(f"Valid pixels: {int(valid.sum())}")
    print(f"Predicting RF probabilities...")
    rf_probabilities = predict_probability_maps(
        args=args,
        model=artifact["model"],
        feature_names=feature_names,
        base_layer=base_layer,
        driver_plan=driver_plan,
        valid=valid,
        cache_path=outputs["probability_cache"],
    )
    if args.write_probability_raster:
        write_probability_raster(
            outputs["probability_raster"],
            reference=base_layer,
            probabilities=rf_probabilities,
        )

    print("Running CA allocation...")
    prediction, log_rows = simulate_rf_ca(
        base_array=base_layer.array,
        demand=demand,
        transition_probs=transition_probs,
        rf_probabilities=rf_probabilities,
        valid=valid,
        args=args,
    )

    write_prediction_raster(outputs["prediction"], base_layer, prediction)
    write_simulation_log_csv(outputs["simulation_log"], log_rows)
    write_area_projection_csv(
        outputs["area"],
        base=base_layer.array,
        prediction=prediction,
        target=target_layer.array if target_layer is not None else None,
        valid=valid,
        demand=demand,
        area_km2=pixel_area_km2(base_layer.profile),
    )

    accuracy: AccuracyResult | None = None
    if target_layer is not None:
        matrix = confusion_matrix(target_layer.array, prediction, valid)
        accuracy = accuracy_summary(matrix)
        write_confusion_csv(outputs["confusion"], matrix)
        write_per_class_csv(outputs["per_class"], matrix)

    predicted_counts = class_counts(prediction, valid)
    demand_error = int(np.max(np.abs(predicted_counts - demand)))
    write_summary_csv(
        outputs["summary"],
        args=args,
        model_file=model_file,
        driver_plan=driver_plan,
        accuracy=accuracy,
        demand_error=demand_error,
        prediction_path=outputs["prediction"],
    )
    write_metadata_json(outputs["metadata"], args, artifact, model_file, driver_plan, outputs)

    print(f"Prediction raster: {outputs['prediction']}")
    print(f"Summary: {outputs['summary']}")
    if accuracy is not None:
        print(
            f"Validation: OA={accuracy.overall_accuracy:.4f}, "
            f"Kappa={accuracy.kappa:.4f}, pixels={accuracy.total_pixels}"
        )
    else:
        print("Validation skipped: observed target raster not found.")
    print(f"Max demand error: {demand_error} pixels")

    if not args.keep_probability_cache:
        cache_path = outputs["probability_cache"]
        del rf_probabilities
        if cache_path.exists():
            cache_path.unlink()


if __name__ == "__main__":
    main()
