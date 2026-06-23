"""RF-CA 训练样本构建脚本。

本脚本把两期 CLCD 土地利用图转换成随机森林可直接读取的样本包：

- X：当前土地利用 one-hot、邻域各类比例、道路/水系/夜光/地形驱动因子；
- y：下一期土地利用类别；
- rows/cols：样本在栅格中的行列号，便于回查空间位置；
- from_class/to_class：转移前后类别编码，便于统计和论文制表。

典型用途：

1. 2015 -> 2020 构建训练样本，用于 2020 -> 2025 验证；
2. 2020 -> 2025 构建训练样本，用于 2025 -> 2030 预测。

注意：
当前 RF-CA 项目中已复制的驱动因子只有 2020 和 2025 两期。
如果构建 2015 -> 2020 样本而没有 2015 驱动因子，脚本会默认使用 2020
驱动因子，并在输出元数据中明确记录 driver_year，避免后续混淆。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def configure_console_encoding() -> None:
    """尽量让 Windows 终端正确显示中文提示。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。

    在 Windows 上用解释器绝对路径运行脚本时，rasterio/GDAL 有时找不到
    GDAL_DATA 和 PROJ_LIB。这里在导入 rasterio 前自动补齐，减少环境噪声。
    """

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"

    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_console_encoding()
configure_geospatial_data_paths()

# rasterio 依赖 GDAL，因此必须放在 configure_geospatial_data_paths 之后导入。
import rasterio


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

DEFAULT_DRIVER_NAMES = [
    "road_closeness",
    "water_closeness",
    "nightlight",
    "elevation_norm",
    "low_slope",
]


@dataclass(frozen=True)
class RasterLayer:
    """保存一个栅格图层及其空间元数据。"""

    path: Path
    array: np.ndarray
    profile: dict
    nodata: int | float | None


@dataclass(frozen=True)
class DriverLayer:
    """保存一个驱动因子图层。"""

    name: str
    year: int
    path: Path
    array: np.ndarray
    nodata: int | float | None


@dataclass(frozen=True)
class DriverPlan:
    """保存驱动因子使用计划。"""

    driver_year: int | None
    paths: dict[str, Path]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Build stratified transition samples for RF-CA."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("E:/RF-CA"),
        help="RF-CA 项目根目录。",
    )
    parser.add_argument(
        "--city",
        default="shenyang",
        help="研究区名称，用于拼接输入文件名。",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2015,
        help="转移起始年份，例如 2015。",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2020,
        help="转移目标年份，例如 2020。",
    )
    parser.add_argument(
        "--driver-year",
        type=int,
        default=None,
        help=(
            "驱动因子年份。默认优先用 from-year；若不存在则尝试 to-year。"
            "文件名会记录 driver_year。"
        ),
    )
    parser.add_argument(
        "--drivers",
        nargs="+",
        default=DEFAULT_DRIVER_NAMES,
        help="参与样本构建的驱动因子名称。",
    )
    parser.add_argument(
        "--no-drivers",
        action="store_true",
        help="只使用当前类别和邻域特征，不读取外部驱动因子。",
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=5,
        help="邻域窗口大小，必须是 >=3 的奇数。",
    )
    parser.add_argument(
        "--max-change-samples-per-transition",
        type=int,
        default=8000,
        help="每一种变化转移 from!=to 最多抽样多少像元。",
    )
    parser.add_argument(
        "--max-stay-samples-per-class",
        type=int,
        default=8000,
        help="每一种稳定转移 from==to 最多抽样多少像元。",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="总样本数上限；默认不额外限制。",
    )
    parser.add_argument(
        "--include-current-code",
        action="store_true",
        help="额外加入 current_class_code 数值列。默认只加入 one-hot。",
    )
    parser.add_argument(
        "--include-coordinates",
        action="store_true",
        help="额外加入 row_norm 和 col_norm。默认不加入，避免过强空间位置记忆。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证抽样可复现。",
    )
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
        help="样本输出目录，默认 data/samples/{city}。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查输入文件和输出路径，不读取全量数组、不写样本。",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录补齐输入和输出路径。"""

    root = args.project_root
    if args.landuse_dir is None:
        args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    if args.driver_dir is None:
        args.driver_dir = root / "data" / "processed" / "drivers" / args.city
    if args.output_dir is None:
        args.output_dir = root / "data" / "samples" / args.city
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查参数是否合理。"""

    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        raise ValueError("--neighborhood-size 必须是 >=3 的奇数。")
    if args.from_year >= args.to_year:
        raise ValueError("--from-year 必须早于 --to-year。")
    if args.max_change_samples_per_transition < 0:
        raise ValueError("--max-change-samples-per-transition 不能为负数。")
    if args.max_stay_samples_per_class < 0:
        raise ValueError("--max-stay-samples-per-class 不能为负数。")
    if args.max_total_samples is not None and args.max_total_samples <= 0:
        raise ValueError("--max-total-samples 必须为正数。")


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    """拼接某一年 CLCD 土地利用栅格路径。"""

    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def driver_path(args: argparse.Namespace, name: str, year: int) -> Path:
    """拼接某一年某个驱动因子栅格路径。"""

    return args.driver_dir / f"{args.city}_{name}_{year}.tif"


def all_driver_paths_exist(args: argparse.Namespace, year: int) -> bool:
    """检查指定年份的全部驱动因子是否存在。"""

    return all(driver_path(args, name, year).exists() for name in args.drivers)


def resolve_driver_plan(args: argparse.Namespace) -> DriverPlan:
    """确定本次样本构建使用哪一年驱动因子。

    理想状态是使用 from_year 驱动因子，因为样本特征应该描述转移发生前的
    像元状态。如果 from_year 不存在，而 to_year 存在，则使用 to_year 并
    明确记录警告。
    """

    if args.no_drivers or not args.drivers:
        return DriverPlan(driver_year=None, paths={}, warnings=[])

    warnings: list[str] = []
    if args.driver_year is not None:
        candidate_years = [args.driver_year]
    else:
        candidate_years = [args.from_year, args.to_year]

    selected_year: int | None = None
    for year in candidate_years:
        if all_driver_paths_exist(args, year):
            selected_year = year
            break

    if selected_year is None:
        details = ", ".join(
            f"{name}: {driver_path(args, name, candidate_years[0])}"
            for name in args.drivers
        )
        raise FileNotFoundError(
            "未找到完整驱动因子。可补齐驱动因子、指定 --driver-year，"
            f"或使用 --no-drivers。检查示例：{details}"
        )

    if args.driver_year is None and selected_year != args.from_year:
        warnings.append(
            f"未找到 {args.from_year} 年完整驱动因子，已改用 {selected_year} 年驱动因子。"
        )

    paths = {name: driver_path(args, name, selected_year) for name in args.drivers}
    return DriverPlan(driver_year=selected_year, paths=paths, warnings=warnings)


def output_stem(args: argparse.Namespace, driver_year: int | None) -> str:
    """生成样本文件前缀，确保不同参数不会互相覆盖。"""

    driver_part = f"driver{driver_year}" if driver_year is not None else "nodriver"
    total_part = (
        f"_total{args.max_total_samples}" if args.max_total_samples is not None else ""
    )
    return (
        f"{args.city}_rf_samples_fit_{args.from_year}_{args.to_year}_{driver_part}"
        f"_n{args.neighborhood_size}"
        f"_chg{args.max_change_samples_per_transition}"
        f"_stay{args.max_stay_samples_per_class}"
        f"{total_part}"
        f"_seed{args.seed}"
    )


def alignment_signature(profile: dict) -> tuple:
    """提取判断栅格是否对齐所需的关键信息。"""

    transform = profile["transform"]
    return (
        profile["width"],
        profile["height"],
        str(profile["crs"]),
        tuple(round(value, 9) for value in transform),
    )


def read_raster(path: Path) -> RasterLayer:
    """读取单波段栅格到内存。"""

    if not path.exists():
        raise FileNotFoundError(f"输入栅格不存在：{path}")

    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    return RasterLayer(path=path, array=array, profile=profile, nodata=nodata)


def inspect_raster(path: Path) -> dict:
    """只读取栅格元数据，用于 dry-run。"""

    if not path.exists():
        raise FileNotFoundError(f"输入栅格不存在：{path}")

    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": tuple(round(value, 9) for value in src.transform),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def validate_alignment(reference: RasterLayer, others: Iterable[RasterLayer]) -> None:
    """检查所有输入栅格是否与参考土地利用图对齐。"""

    signature = alignment_signature(reference.profile)
    for layer in others:
        if alignment_signature(layer.profile) != signature:
            raise ValueError(f"栅格未对齐：{reference.path} vs {layer.path}")


def valid_landuse_mask(*arrays: np.ndarray) -> np.ndarray:
    """返回所有土地利用数组中都属于 CLCD 1-9 类的像元。"""

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isin(array, CLASS_CODES)
    return mask


def load_driver_layers(plan: DriverPlan, reference: RasterLayer) -> list[DriverLayer]:
    """读取驱动因子，并检查它们是否与土地利用图对齐。"""

    layers: list[DriverLayer] = []
    for name, path in plan.paths.items():
        raster = read_raster(path)
        validate_alignment(reference, [raster])
        array = raster.array.astype(np.float32, copy=False)
        layers.append(
            DriverLayer(
                name=name,
                year=int(plan.driver_year) if plan.driver_year is not None else -1,
                path=path,
                array=array,
                nodata=raster.nodata,
            )
        )
    return layers


def valid_driver_mask(driver_layers: list[DriverLayer]) -> np.ndarray | None:
    """计算所有驱动因子都有有效值的位置。"""

    if not driver_layers:
        return None

    mask = np.ones(driver_layers[0].array.shape, dtype=bool)
    for layer in driver_layers:
        values = layer.array
        layer_mask = np.isfinite(values)
        if layer.nodata is not None:
            layer_mask &= values != float(layer.nodata)
        mask &= layer_mask
    return mask


def transition_summary(from_array: np.ndarray, to_array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """统计 9x9 转移数量矩阵。"""

    from_values = from_array[valid].astype(np.int16) - 1
    to_values = to_array[valid].astype(np.int16) - 1
    flat_index = from_values * len(CLASS_CODES) + to_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def sample_flat_indices(
    from_array: np.ndarray,
    to_array: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    """按转移类型分层抽样，返回样本的一维像元索引和摘要记录。"""

    rng = np.random.default_rng(args.seed)
    width = from_array.shape[1]
    from_flat = from_array.reshape(-1)
    to_flat = to_array.reshape(-1)
    valid_indices = np.flatnonzero(valid.reshape(-1))

    transition_ids = (
        (from_flat[valid_indices].astype(np.int16) - 1) * len(CLASS_CODES)
        + (to_flat[valid_indices].astype(np.int16) - 1)
    )
    counts = np.bincount(transition_ids, minlength=len(CLASS_CODES) ** 2)

    selected_parts: list[np.ndarray] = []
    rows: list[dict] = []
    for transition_id, available in enumerate(counts):
        from_index = transition_id // len(CLASS_CODES)
        to_index = transition_id % len(CLASS_CODES)
        from_code = int(CLASS_CODES[from_index])
        to_code = int(CLASS_CODES[to_index])
        is_stay = from_code == to_code
        cap = (
            args.max_stay_samples_per_class
            if is_stay
            else args.max_change_samples_per_transition
        )

        if available == 0 or cap == 0:
            sampled = 0
            chosen = np.array([], dtype=np.int64)
        else:
            positions = np.flatnonzero(transition_ids == transition_id)
            if len(positions) <= cap:
                chosen = valid_indices[positions]
            else:
                chosen_positions = rng.choice(positions, size=cap, replace=False)
                chosen = valid_indices[chosen_positions]
            sampled = int(len(chosen))
            if sampled > 0:
                selected_parts.append(chosen.astype(np.int64, copy=False))

        rows.append(
            {
                "from_code": from_code,
                "from_name": CLASS_NAMES[from_code],
                "to_code": to_code,
                "to_name": CLASS_NAMES[to_code],
                "is_stay": "yes" if is_stay else "no",
                "available_pixels": int(available),
                "sampled_pixels": sampled,
            }
        )

    if selected_parts:
        selected = np.concatenate(selected_parts)
        if args.max_total_samples is not None and len(selected) > args.max_total_samples:
            selected = rng.choice(selected, size=args.max_total_samples, replace=False)
        rng.shuffle(selected)
    else:
        selected = np.array([], dtype=np.int64)

    # 如果设置了总样本数上限，上面的二次抽样会改变各转移类型的最终样本数，
    # 因此这里按最终 selected 重新统计 sampled_pixels。
    final_from = from_flat[selected].astype(np.int16)
    final_to = to_flat[selected].astype(np.int16)
    final_ids = (final_from - 1) * len(CLASS_CODES) + (final_to - 1)
    final_counts = np.bincount(final_ids, minlength=len(CLASS_CODES) ** 2)
    for transition_id, row in enumerate(rows):
        row["sampled_pixels"] = int(final_counts[transition_id])

    # 保留 width 的局部引用是为了提醒：flat_index 后续将用 width 还原为行列号。
    _ = width
    return selected, rows


def window_counts_at(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray, radius: int) -> np.ndarray:
    """用积分图计算样本点邻域窗口内的 True 数量。"""

    height, width = mask.shape
    top = np.maximum(rows - radius, 0)
    bottom = np.minimum(rows + radius + 1, height)
    left = np.maximum(cols - radius, 0)
    right = np.minimum(cols + radius + 1, width)

    # int32 足够容纳沈阳裁剪范围内任意窗口和全图累积计数。
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    integral[1:, 1:] = np.cumsum(
        np.cumsum(mask.astype(np.int32), axis=0, dtype=np.int32),
        axis=1,
        dtype=np.int32,
    )
    return (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def build_feature_matrix(
    from_array: np.ndarray,
    driver_layers: list[DriverLayer],
    flat_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """根据抽样像元构建随机森林特征矩阵。"""

    height, width = from_array.shape
    rows = (flat_indices // width).astype(np.int32)
    cols = (flat_indices % width).astype(np.int32)
    current_classes = from_array.reshape(-1)[flat_indices].astype(np.uint8)

    feature_parts: list[np.ndarray] = []
    feature_names: list[str] = []

    if args.include_current_code:
        feature_parts.append(current_classes.astype(np.float32).reshape(-1, 1))
        feature_names.append("current_class_code")

    # 当前土地利用使用 one-hot，避免把 1-9 类当成连续有序数值。
    current_one_hot = np.column_stack(
        [(current_classes == code).astype(np.float32) for code in CLASS_CODES]
    )
    feature_parts.append(current_one_hot)
    feature_names.extend(
        [f"current_is_{int(code)}_{CLASS_NAMES[int(code)]}" for code in CLASS_CODES]
    )

    radius = args.neighborhood_size // 2
    valid_mask = np.isin(from_array, CLASS_CODES)
    total_neighbors = window_counts_at(valid_mask, rows, cols, radius).astype(np.float32) - 1.0
    total_neighbors = np.maximum(total_neighbors, 1.0)

    neighbor_features = []
    for code in CLASS_CODES:
        class_mask = from_array == code
        counts = window_counts_at(class_mask, rows, cols, radius).astype(np.float32)
        # 中心像元不参与邻域比例统计。
        counts -= (current_classes == code).astype(np.float32)
        neighbor_features.append(np.clip(counts / total_neighbors, 0.0, 1.0))
        feature_names.append(f"neighbor_frac_{int(code)}_{CLASS_NAMES[int(code)]}")
    feature_parts.append(np.column_stack(neighbor_features).astype(np.float32))

    for layer in driver_layers:
        values = layer.array[rows, cols].astype(np.float32)
        feature_parts.append(values.reshape(-1, 1))
        feature_names.append(f"driver_{layer.name}_{layer.year}")

    if args.include_coordinates:
        row_norm = rows.astype(np.float32) / max(height - 1, 1)
        col_norm = cols.astype(np.float32) / max(width - 1, 1)
        feature_parts.append(np.column_stack([row_norm, col_norm]).astype(np.float32))
        feature_names.extend(["row_norm", "col_norm"])

    x = np.column_stack(feature_parts).astype(np.float32)
    return x, feature_names, rows, cols


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """写出按转移类型统计的样本摘要。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "from_code",
        "from_name",
        "to_code",
        "to_name",
        "is_stay",
        "available_pixels",
        "sampled_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_json(
    path: Path,
    args: argparse.Namespace,
    plan: DriverPlan,
    sample_path: Path,
    summary_path: Path,
    feature_names: list[str],
    sample_count: int,
    valid_pixel_count: int,
    warnings: list[str],
) -> None:
    """写出样本构建元数据，保证实验可追溯。"""

    metadata = {
        "city": args.city,
        "from_year": args.from_year,
        "to_year": args.to_year,
        "driver_year": plan.driver_year,
        "neighborhood_size": args.neighborhood_size,
        "max_change_samples_per_transition": args.max_change_samples_per_transition,
        "max_stay_samples_per_class": args.max_stay_samples_per_class,
        "max_total_samples": args.max_total_samples,
        "seed": args.seed,
        "include_current_code": args.include_current_code,
        "include_coordinates": args.include_coordinates,
        "feature_names": feature_names,
        "class_codes": [int(code) for code in CLASS_CODES],
        "class_names": {str(code): name for code, name in CLASS_NAMES.items()},
        "landuse_from": str(landuse_path(args, args.from_year)),
        "landuse_to": str(landuse_path(args, args.to_year)),
        "driver_paths": {name: str(path) for name, path in plan.paths.items()},
        "sample_count": int(sample_count),
        "valid_pixel_count": int(valid_pixel_count),
        "sample_npz": str(sample_path),
        "transition_summary_csv": str(summary_path),
        "warnings": warnings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sample_npz(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    from_class: np.ndarray,
    feature_names: list[str],
) -> None:
    """写出压缩样本包。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=x.astype(np.float32),
        y=y.astype(np.uint8),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        from_class=from_class.astype(np.uint8),
        to_class=y.astype(np.uint8),
        feature_names=np.array(feature_names, dtype="U128"),
        class_codes=CLASS_CODES.astype(np.uint8),
    )


def print_dry_run(args: argparse.Namespace, plan: DriverPlan) -> None:
    """输出 dry-run 检查结果。"""

    from_meta = inspect_raster(landuse_path(args, args.from_year))
    to_meta = inspect_raster(landuse_path(args, args.to_year))
    if (
        from_meta["width"],
        from_meta["height"],
        from_meta["crs"],
        from_meta["transform"],
    ) != (
        to_meta["width"],
        to_meta["height"],
        to_meta["crs"],
        to_meta["transform"],
    ):
        raise ValueError("from-year 与 to-year 土地利用栅格未对齐。")

    for path in plan.paths.values():
        driver_meta = inspect_raster(path)
        if (
            from_meta["width"],
            from_meta["height"],
            from_meta["crs"],
            from_meta["transform"],
        ) != (
            driver_meta["width"],
            driver_meta["height"],
            driver_meta["crs"],
            driver_meta["transform"],
        ):
            raise ValueError(f"驱动因子未与土地利用栅格对齐：{path}")

    stem = output_stem(args, plan.driver_year)
    print("Dry-run OK")
    print(f"Project root: {args.project_root}")
    print(f"Land-use from: {landuse_path(args, args.from_year)}")
    print(f"Land-use to:   {landuse_path(args, args.to_year)}")
    print(f"Raster shape:  {from_meta['width']} x {from_meta['height']}")
    print(f"Driver year:   {plan.driver_year if plan.driver_year is not None else 'none'}")
    for warning in plan.warnings:
        print(f"Warning: {warning}")
    for name, path in plan.paths.items():
        print(f"Driver {name}: {path}")
    print(f"Output stem:   {stem}")
    print(f"Output dir:    {args.output_dir}")


def main() -> None:
    """脚本入口。"""

    args = resolve_paths(parse_args())
    validate_args(args)
    plan = resolve_driver_plan(args)

    if args.dry_run:
        print_dry_run(args, plan)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(plan.warnings)

    print(f"City: {args.city}")
    print(f"Transition: {args.from_year}->{args.to_year}")
    print(f"Neighborhood: {args.neighborhood_size}x{args.neighborhood_size}")
    for warning in warnings:
        print(f"Warning: {warning}")

    from_layer = read_raster(landuse_path(args, args.from_year))
    to_layer = read_raster(landuse_path(args, args.to_year))
    validate_alignment(from_layer, [to_layer])

    driver_layers = load_driver_layers(plan, from_layer)
    driver_mask = valid_driver_mask(driver_layers)

    valid = valid_landuse_mask(from_layer.array, to_layer.array)
    if driver_mask is not None:
        valid &= driver_mask
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("没有可用于抽样的有效像元。")

    flat_indices, summary_rows = sample_flat_indices(
        from_layer.array,
        to_layer.array,
        valid,
        args,
    )
    if len(flat_indices) == 0:
        raise ValueError("抽样结果为空，请检查抽样上限参数。")

    x, feature_names, rows, cols = build_feature_matrix(
        from_layer.array,
        driver_layers,
        flat_indices,
        args,
    )
    y = to_layer.array.reshape(-1)[flat_indices].astype(np.uint8)
    from_class = from_layer.array.reshape(-1)[flat_indices].astype(np.uint8)

    stem = output_stem(args, plan.driver_year)
    sample_path = args.output_dir / f"{stem}.npz"
    summary_path = args.output_dir / f"{stem}_transition_summary.csv"
    metadata_path = args.output_dir / f"{stem}_metadata.json"

    if sample_path.exists() or summary_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "输出文件已存在，为避免覆盖请调整参数或先手动处理旧文件："
            f"\n{sample_path}\n{summary_path}\n{metadata_path}"
        )

    write_sample_npz(
        sample_path,
        x=x,
        y=y,
        rows=rows,
        cols=cols,
        from_class=from_class,
        feature_names=feature_names,
    )
    write_summary_csv(summary_path, summary_rows)
    write_metadata_json(
        metadata_path,
        args=args,
        plan=plan,
        sample_path=sample_path,
        summary_path=summary_path,
        feature_names=feature_names,
        sample_count=len(flat_indices),
        valid_pixel_count=valid_count,
        warnings=warnings,
    )

    transition_counts = transition_summary(from_layer.array, to_layer.array, valid)
    changed_pixels = int(transition_counts.sum() - np.trace(transition_counts))
    print(f"Valid pixels: {valid_count}")
    print(f"Changed valid pixels: {changed_pixels}")
    print(f"Sample count: {len(flat_indices)}")
    print(f"Feature count: {x.shape[1]}")
    print(f"Sample NPZ: {sample_path}")
    print(f"Transition summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
