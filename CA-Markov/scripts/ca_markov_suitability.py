"""CLCD 原始 9 类：带适宜性因子的 CA-Markov 实验脚本。

这个脚本是 ca_markov.py 的增强版，用于测试“加入驱动/适宜性因子后，
预测精度是否提高”。它不会替换原来的无驱动 CA-Markov 脚本。

当前脚本会尝试使用项目中已有的因子：
- road_closeness：道路邻近性，来自 OSM roads 矢量；
- water_closeness：水系/水面邻近性，来自 OSM water/waterways 矢量；
- nightlight：夜光强度，优先读取已经处理好的栅格；如果 raw .tif.gz
  能被 GDAL 直接读取，也会尝试读取；
- low_slope：低坡度适宜性，如果能读取 DEM，会自动生成。

建模思想：
- Markov 仍然控制各类别的目标数量；
- CA 邻域控制空间聚集；
- 适宜性因子调整候选像元的转入优先级。

常用验证命令：
    python scripts/ca_markov_suitability.py --fit-from 2015 --base-year 2020 --target-year 2025

常用 2030 预测命令：
    python scripts/ca_markov_suitability.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from scipy.ndimage import convolve, distance_transform_edt

from ca_markov import (
    CLASS_CODES,
    CLASS_NAMES,
    AccuracyResult,
    RasterLayer,
    accuracy_summary,
    apportion_with_caps,
    class_counts,
    configure_geospatial_data_paths,
    confusion_matrix,
    load_raster,
    markov_demand,
    pixel_area_km2,
    raster_path,
    transition_counts,
    transition_probabilities,
    validate_alignment,
    valid_mask,
    write_area_projection_csv,
    write_confusion_csv,
    write_per_class_accuracy_csv,
    write_prediction_raster,
    write_simulation_log_csv,
    write_summary_csv,
)


configure_geospatial_data_paths()

# configure_geospatial_data_paths 必须先执行，再导入 rasterio。
import rasterio


@dataclass(frozen=True)
class FactorPaths:
    """保存自动因子构建时可能用到的原始数据路径。"""

    roads: Path
    water_polygon: Path
    water_line: Path
    dem_dir: Path
    nightlight_dir: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    默认任务是用 2020->2025 的转移规律，从 2025 年预测 2030 年。
    如果要检查适宜性因子是否提高 2025 验证精度，请显式传入：
        --fit-from 2015 --base-year 2020 --target-year 2025
    """

    parser = argparse.ArgumentParser(
        description="CA-Markov with optional suitability factors."
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
        "--fit-from",
        type=int,
        default=2015,
        help="Start year used to estimate Markov transition probabilities.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="Base year used as the CA-Markov simulation start.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2025,
        help="Observed target year for validation, or future prediction year.",
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=5,
        help="Odd CA neighborhood size, for example 3 or 5.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of CA allocation iterations.",
    )
    parser.add_argument(
        "--neighbor-weight",
        type=float,
        default=0.9,
        help="Weight of CA neighborhood attraction in candidate scoring.",
    )
    parser.add_argument(
        "--suitability-weight",
        type=float,
        default=0.1,
        help="Weight of external suitability factors in candidate scoring.",
    )
    parser.add_argument(
        "--random-weight",
        type=float,
        default=0.03,
        help="Small stochastic perturbation used when ranking candidate cells.",
    )
    parser.add_argument(
        "--road-decay-m",
        type=float,
        default=1500.0,
        help="Distance decay in meters for road closeness.",
    )
    parser.add_argument(
        "--water-decay-m",
        type=float,
        default=1000.0,
        help="Distance decay in meters for water closeness.",
    )
    parser.add_argument(
        "--skip-roads",
        action="store_true",
        help="Do not build/use the road closeness factor.",
    )
    parser.add_argument(
        "--skip-water",
        action="store_true",
        help="Do not build/use the water closeness factor.",
    )
    parser.add_argument(
        "--skip-nightlight",
        action="store_true",
        help="Do not build/use the nightlight factor.",
    )
    parser.add_argument(
        "--skip-dem",
        action="store_true",
        help="Do not build/use the low-slope factor from DEM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible CA allocation.",
    )
    parser.add_argument(
        "--landuse-dir",
        type=Path,
        default=None,
        help="Directory containing {city}_clcd_v01_{year}_original.tif.",
    )
    parser.add_argument(
        "--suitability-dir",
        type=Path,
        default=None,
        help="Directory for cached suitability rasters.",
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
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录补齐输入、输出和因子路径。"""

    root = args.project_root
    if args.landuse_dir is None:
        args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    if args.suitability_dir is None:
        args.suitability_dir = root / "data" / "processed" / "suitability" / args.city
    if args.tables_dir is None:
        args.tables_dir = root / "tables"
    if args.output_dir is None:
        args.output_dir = root / "output" / "ca_markov_suitability" / args.city

    args.factor_paths = FactorPaths(
        roads=root
        / "data"
        / "raw"
        / "roads"
        / "osm_liaoning"
        / "shp"
        / "gis_osm_roads_free_1.shp",
        water_polygon=root
        / "data"
        / "raw"
        / "water"
        / "osm_liaoning"
        / "gis_osm_water_a_free_1.shp",
        water_line=root
        / "data"
        / "raw"
        / "water"
        / "osm_liaoning"
        / "gis_osm_waterways_free_1.shp",
        dem_dir=root / "data" / "raw" / "dem",
        nightlight_dir=root / "data" / "raw" / "nightlight",
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查参数是否合理。"""

    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        raise ValueError("--neighborhood-size must be an odd integer >= 3.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.neighbor_weight < 0 or args.suitability_weight < 0:
        raise ValueError("Weights must be >= 0.")
    if args.neighbor_weight + args.suitability_weight <= 0:
        raise ValueError("At least one of neighbor/suitability weights must be > 0.")
    if args.random_weight < 0:
        raise ValueError("--random-weight must be >= 0.")


def output_stem(args: argparse.Namespace) -> str:
    """生成带适宜性因子的 CA-Markov 输出文件名前缀。"""

    return (
        f"{args.city}_ca_markov_suitability_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
        f"{weight_suffix(args)}"
    )


def format_weight_for_filename(value: float) -> str:
    """把权重转换成适合文件名的百分比字符串。

    例如 0.65 -> 065，0.2 -> 020，0.125 -> 12p5。
    """

    percent = value * 100
    rounded = round(percent)
    if abs(percent - rounded) < 1e-6:
        return f"{int(rounded):03d}"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def weight_suffix(args: argparse.Namespace) -> str:
    """生成权重后缀，避免不同权重实验互相覆盖。"""

    return (
        f"_nw{format_weight_for_filename(args.neighbor_weight)}"
        f"_sw{format_weight_for_filename(args.suitability_weight)}"
    )


def factor_raster_path(args: argparse.Namespace, name: str, reference_year: int) -> Path:
    """生成某个适宜性因子的缓存栅格路径。"""

    return args.suitability_dir / f"{args.city}_{name}_{reference_year}.tif"


def write_factor_raster(path: Path, reference: RasterLayer, array: np.ndarray) -> None:
    """把 0-1 适宜性因子写成和基期土地利用一致的 GeoTIFF。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": -9999.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def read_factor_raster(path: Path, reference: RasterLayer) -> np.ndarray:
    """读取已经缓存的适宜性因子，并确认网格和基期图一致。"""

    with rasterio.open(path) as src:
        if (
            src.width != reference.profile["width"]
            or src.height != reference.profile["height"]
            or src.transform != reference.profile["transform"]
            or str(src.crs) != str(reference.profile["crs"])
        ):
            raise ValueError(f"Suitability raster is not aligned with land use: {path}")
        data = src.read(1).astype(np.float32)
    return np.clip(data, 0.0, 1.0)


def normalize_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """用 2%-98% 分位数把连续因子归一化到 0-1。"""

    result = np.zeros(values.shape, dtype=np.float32)
    sample = values[valid & np.isfinite(values)]
    if sample.size == 0:
        return result

    low, high = np.nanpercentile(sample, [2, 98])
    if high <= low:
        return result

    result = (values - low) / (high - low)
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    result[~valid] = 0.0
    return result


def vector_closeness_factor(
    vector_path: Path,
    reference: RasterLayer,
    valid: np.ndarray,
    decay_m: float,
    output_path: Path,
    label: str,
) -> np.ndarray | None:
    """从矢量数据生成“距离越近越适宜”的 0-1 栅格。

    道路和水系都可以用这个函数处理：
    先把矢量栅格化为 0/1，再计算每个像元到最近要素的距离，
    最后用 exp(-distance / decay) 转成邻近性。
    """

    if output_path.exists():
        print(f"Using cached factor: {output_path}")
        return read_factor_raster(output_path, reference)

    if not vector_path.exists():
        print(f"Skip {label}: source vector not found: {vector_path}")
        return None

    print(f"Building {label} factor from: {vector_path}")
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        print(f"Skip {label}: source vector is empty.")
        return None

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        print(f"Skip {label}: no valid geometries.")
        return None

    gdf = gdf.to_crs(reference.profile["crs"])
    bounds = rasterio.transform.array_bounds(
        reference.profile["height"],
        reference.profile["width"],
        reference.profile["transform"],
    )
    minx, miny, maxx, maxy = bounds
    gdf = gdf.cx[minx:maxx, miny:maxy]
    if gdf.empty:
        print(f"Skip {label}: no features intersect the study area.")
        return None

    burned = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(reference.profile["height"], reference.profile["width"]),
        transform=reference.profile["transform"],
        fill=0,
        dtype="uint8",
    )

    if burned.max() == 0:
        print(f"Skip {label}: rasterized factor is empty.")
        return None

    pixel_size = abs(reference.profile["transform"].a)
    distance_m = distance_transform_edt(burned == 0).astype(np.float32) * pixel_size
    closeness = np.exp(-distance_m / float(decay_m)).astype(np.float32)
    closeness[~valid] = 0.0
    write_factor_raster(output_path, reference, closeness)
    return closeness


def find_nightlight_source(args: argparse.Namespace, year: int) -> Path | None:
    """寻找与基期年份最接近的夜光数据。

    当前目录中通常有 2015、2020、2025 VIIRS，以及 2000/2005/2010 DMSP。
    """

    candidates = sorted(args.factor_paths.nightlight_dir.rglob(f"*{year}*.tif"))
    candidates += sorted(args.factor_paths.nightlight_dir.rglob(f"*{year}*.tif.gz"))
    if candidates:
        return candidates[0]
    return None


def maybe_decompress_gzip(source: Path, cache_dir: Path) -> Path:
    """如果夜光数据是 .gz，解压到缓存目录再读取。

    这一步可能占用一些磁盘空间，但只会做一次；后续会复用缓存文件。
    """

    if source.suffix.lower() != ".gz":
        return source

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / source.name.removesuffix(".gz")
    if output.exists():
        return output

    print(f"Decompressing nightlight raster to cache: {output}")
    with gzip.open(source, "rb") as src, output.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output


def raster_factor_from_source(
    source_path: Path,
    reference: RasterLayer,
    valid: np.ndarray,
    output_path: Path,
    label: str,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray | None:
    """把外部栅格重投影/重采样到基期土地利用网格，并归一化到 0-1。"""

    if output_path.exists():
        print(f"Using cached factor: {output_path}")
        return read_factor_raster(output_path, reference)

    if not source_path.exists():
        print(f"Skip {label}: source raster not found: {source_path}")
        return None

    print(f"Building {label} factor from: {source_path}")
    try:
        with rasterio.open(source_path) as src:
            vrt_options = {
                "crs": reference.profile["crs"],
                "transform": reference.profile["transform"],
                "width": reference.profile["width"],
                "height": reference.profile["height"],
                "resampling": resampling,
            }
            with WarpedVRT(src, **vrt_options) as vrt:
                data = vrt.read(1, masked=True).astype(np.float32)
                filled = np.asarray(data.filled(np.nan), dtype=np.float32)
    except Exception as exc:
        print(f"Skip {label}: failed to read raster: {exc}")
        return None

    normalized = normalize_percentile(filled, valid)
    write_factor_raster(output_path, reference, normalized)
    return normalized


def dem_source(args: argparse.Namespace) -> Path | None:
    """寻找一个可用 DEM 源。

    优先使用已有的合并 DEM 高程T.img；找不到时再尝试其他 .img。
    """

    preferred = args.factor_paths.dem_dir / "existing_srtm_candidates" / "高程" / "高程T.img"
    if preferred.exists():
        return preferred

    candidates = sorted(args.factor_paths.dem_dir.rglob("*.img"))
    return candidates[0] if candidates else None


def slope_from_dem_factor(
    args: argparse.Namespace,
    reference: RasterLayer,
    valid: np.ndarray,
) -> np.ndarray | None:
    """从 DEM 生成低坡度适宜性因子。

    低坡度适宜性 = 1 - 归一化坡度。
    对建设用地、耕地和水域通常更有利。
    """

    output_path = factor_raster_path(args, "low_slope", args.base_year)
    if output_path.exists():
        print(f"Using cached factor: {output_path}")
        return read_factor_raster(output_path, reference)

    source = dem_source(args)
    if source is None:
        print("Skip low_slope: no DEM source found.")
        return None

    elevation_path = factor_raster_path(args, "elevation_norm", args.base_year)
    elevation = raster_factor_from_source(
        source,
        reference=reference,
        valid=valid,
        output_path=elevation_path,
        label="elevation",
        resampling=Resampling.bilinear,
    )
    if elevation is None:
        return None

    # 这里使用归一化高程近似计算梯度，目标是得到相对坡度因子，
    # 不是用于地形学分析的严格坡度角。
    dy, dx = np.gradient(elevation.astype(np.float32))
    slope = np.sqrt(dx * dx + dy * dy)
    slope_norm = normalize_percentile(slope, valid)
    low_slope = (1.0 - slope_norm).astype(np.float32)
    low_slope[~valid] = 0.0
    write_factor_raster(output_path, reference, low_slope)
    return low_slope


def build_factors(args: argparse.Namespace, reference: RasterLayer) -> dict[str, np.ndarray]:
    """构建或读取所有可用适宜性因子。"""

    valid = np.isin(reference.array, CLASS_CODES)
    factors: dict[str, np.ndarray] = {}
    args.suitability_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_roads:
        road = vector_closeness_factor(
            args.factor_paths.roads,
            reference=reference,
            valid=valid,
            decay_m=args.road_decay_m,
            output_path=factor_raster_path(args, "road_closeness", args.base_year),
            label="road_closeness",
        )
        if road is not None:
            factors["road_closeness"] = road
    else:
        print("Skip road_closeness: disabled by --skip-roads.")

    if not args.skip_water:
        water_sources = [args.factor_paths.water_polygon, args.factor_paths.water_line]
        water_parts = []
        for source in water_sources:
            part = vector_closeness_factor(
                source,
                reference=reference,
                valid=valid,
                decay_m=args.water_decay_m,
                output_path=factor_raster_path(
                    args,
                    f"water_closeness_{source.stem}",
                    args.base_year,
                ),
                label=f"water_closeness:{source.stem}",
            )
            if part is not None:
                water_parts.append(part)
        if water_parts:
            factors["water_closeness"] = np.maximum.reduce(water_parts).astype(np.float32)
            write_factor_raster(
                factor_raster_path(args, "water_closeness", args.base_year),
                reference,
                factors["water_closeness"],
            )
    else:
        print("Skip water_closeness: disabled by --skip-water.")

    if not args.skip_nightlight:
        nightlight_source = find_nightlight_source(args, args.base_year)
        if nightlight_source is not None:
            raster_source = maybe_decompress_gzip(
                nightlight_source,
                args.suitability_dir / "_cache",
            )
            nightlight = raster_factor_from_source(
                raster_source,
                reference=reference,
                valid=valid,
                output_path=factor_raster_path(args, "nightlight", args.base_year),
                label="nightlight",
                resampling=Resampling.bilinear,
            )
            if nightlight is not None:
                factors["nightlight"] = nightlight
        else:
            print(f"Skip nightlight: no source found for year {args.base_year}.")
    else:
        print("Skip nightlight: disabled by --skip-nightlight.")

    if not args.skip_dem:
        low_slope = slope_from_dem_factor(args, reference=reference, valid=valid)
        if low_slope is not None:
            factors["low_slope"] = low_slope
    else:
        print("Skip low_slope: disabled by --skip-dem.")

    return factors


def weighted_average(parts: list[tuple[np.ndarray, float]], shape: tuple[int, int]) -> np.ndarray:
    """对若干因子做加权平均；没有可用因子时返回 0.5 中性适宜性。"""

    numerator = np.zeros(shape, dtype=np.float32)
    denominator = 0.0
    for array, weight in parts:
        if weight <= 0:
            continue
        numerator += array.astype(np.float32) * float(weight)
        denominator += float(weight)

    if denominator <= 0:
        return np.full(shape, 0.5, dtype=np.float32)
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def target_suitability_maps(
    factors: dict[str, np.ndarray],
    reference: RasterLayer,
) -> dict[int, np.ndarray]:
    """根据可用因子组合出各目标类别的适宜性图。

    这里使用的是经验规则，目的是快速测试“适宜性因子是否改善精度”。
    后续如果需要更严谨，可以用历史转移样本训练 Logistic/随机森林模型。
    """

    shape = reference.array.shape
    neutral = np.full(shape, 0.5, dtype=np.float32)
    road = factors.get("road_closeness", neutral)
    water = factors.get("water_closeness", neutral)
    night = factors.get("nightlight", neutral)
    low_slope = factors.get("low_slope", neutral)

    low_development = 1.0 - np.maximum(night, road * 0.7)
    far_from_water = 1.0 - water

    suitability = {
        # 耕地通常偏好低坡度、低建设强度，不能太靠近水体本身。
        1: weighted_average(
            [(low_slope, 0.40), (low_development, 0.45), (far_from_water, 0.15)],
            shape,
        ),
        # 林地偏好低建设强度；地形因子在这里给较低权重。
        2: weighted_average([(low_development, 0.70), (1.0 - road, 0.30)], shape),
        # 灌木类别在沈阳样本极少，给中性规则，避免强行驱动。
        3: neutral.copy(),
        # 草地偏好低建设强度和低坡度。
        4: weighted_average([(low_development, 0.55), (low_slope, 0.45)], shape),
        # 水域主要靠近已有水体，低坡度作为辅助。
        5: weighted_average([(water, 0.80), (low_slope, 0.20)], shape),
        # 冰雪在本研究区几乎不存在，保持中性。
        6: neutral.copy(),
        # 裸地通常远离水域且建设强度较低，这里只作为弱约束。
        7: weighted_average([(far_from_water, 0.55), (low_development, 0.45)], shape),
        # 建设用地偏好道路、夜光和低坡度。
        8: weighted_average([(road, 0.35), (night, 0.45), (low_slope, 0.20)], shape),
        # 湿地样本极少，主要靠近水体。
        9: weighted_average([(water, 0.75), (low_development, 0.25)], shape),
    }

    valid = np.isin(reference.array, CLASS_CODES)
    for class_code, array in suitability.items():
        array = np.clip(array, 0.0, 1.0).astype(np.float32)
        array[~valid] = 0.0
        suitability[class_code] = array
    return suitability


def neighborhood_fraction(
    array: np.ndarray,
    target_code: int,
    valid: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    """计算每个像元邻域内目标类别的比例。"""

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)
    center = neighborhood_size // 2
    kernel[center, center] = 0.0
    target = (array == target_code).astype(np.float32)
    valid_float = valid.astype(np.float32)
    target_neighbors = convolve(target, kernel, mode="constant", cval=0.0)
    valid_neighbors = convolve(valid_float, kernel, mode="constant", cval=0.0)
    return np.divide(
        target_neighbors,
        valid_neighbors,
        out=np.zeros(array.shape, dtype=np.float32),
        where=valid_neighbors > 0,
    )


def select_best_source_cells(
    prediction_flat: np.ndarray,
    source_code: int,
    score_flat: np.ndarray,
    count: int,
) -> np.ndarray:
    """在某个来源类别中按综合适宜性分数选择待转换像元。"""

    if count <= 0:
        return np.array([], dtype=np.int64)
    source_positions = np.flatnonzero(prediction_flat == source_code)
    if len(source_positions) == 0:
        return np.array([], dtype=np.int64)
    if count >= len(source_positions):
        return source_positions

    scores = score_flat[source_positions]
    top_local = np.argpartition(scores, -count)[-count:]
    return source_positions[top_local]


def allocate_target_class(
    prediction: np.ndarray,
    target_index: int,
    target_need: int,
    demand: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
    suitability_maps: dict[int, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    """为一个目标类别分配本轮需要新增的像元。

    综合评分 = 邻域权重 * 邻域吸引力 + 适宜性权重 * 目标类别适宜性 + 随机扰动。
    """

    if target_need <= 0:
        return 0

    current_counts = class_counts(prediction)
    surplus = np.maximum(current_counts - demand, 0)
    target_code = int(CLASS_CODES[target_index])
    source_indices = [
        index
        for index in range(len(CLASS_CODES))
        if index != target_index and surplus[index] > 0
    ]
    if not source_indices:
        return 0

    caps = surplus[source_indices]
    weights = caps.astype(np.float64) * probabilities[source_indices, target_index]
    if weights.sum() <= 0:
        weights = caps.astype(np.float64)
    source_take = apportion_with_caps(target_need, weights, caps)
    if source_take.sum() <= 0:
        return 0

    neighborhood = neighborhood_fraction(
        prediction,
        target_code=target_code,
        valid=valid,
        neighborhood_size=args.neighborhood_size,
    )
    suitability = suitability_maps.get(target_code)
    if suitability is None:
        suitability = np.full(prediction.shape, 0.5, dtype=np.float32)

    weight_total = args.neighbor_weight + args.suitability_weight
    score = (
        args.neighbor_weight / weight_total * neighborhood
        + args.suitability_weight / weight_total * suitability
    ).astype(np.float32)
    if args.random_weight > 0:
        score += rng.random(score.shape, dtype=np.float32) * args.random_weight
    score[~valid] = -1.0

    prediction_flat = prediction.reshape(-1)
    score_flat = score.reshape(-1)
    changed = 0
    for source_index, take in zip(source_indices, source_take, strict=True):
        source_code = int(CLASS_CODES[source_index])
        selected = select_best_source_cells(
            prediction_flat=prediction_flat,
            source_code=source_code,
            score_flat=score_flat,
            count=int(take),
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
) -> list[dict]:
    """整理每轮模拟后各类别数量与 Markov 需求量的差异。"""

    rows = []
    for index, class_code in enumerate(CLASS_CODES):
        rows.append(
            {
                "iteration": iteration,
                "changed_pixels": int(changed_pixels),
                "class_code": int(class_code),
                "class_name": CLASS_NAMES[int(class_code)],
                "predicted_pixels": int(counts[index]),
                "markov_demand_pixels": int(demand[index]),
                "difference_pixels": int(counts[index] - demand[index]),
            }
        )
    return rows


def simulate_ca_markov_suitability(
    base_array: np.ndarray,
    probabilities: np.ndarray,
    demand: np.ndarray,
    suitability_maps: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    """执行带适宜性因子的 CA-Markov 空间分配。"""

    rng = np.random.default_rng(args.seed)
    prediction = base_array.copy()
    valid = np.isin(base_array, CLASS_CODES)
    log_rows: list[dict] = []

    for iteration in range(1, args.iterations + 1):
        current_counts = class_counts(prediction)
        deficits = np.maximum(demand - current_counts, 0)
        if deficits.sum() == 0:
            break

        remaining_iterations = args.iterations - iteration + 1
        step_needs = np.ceil(deficits / remaining_iterations).astype(np.int64)
        target_order = np.argsort(-step_needs, kind="stable")

        changed_this_iteration = 0
        for target_index in target_order:
            current_counts = class_counts(prediction)
            remaining_need = max(int(demand[target_index] - current_counts[target_index]), 0)
            target_need = min(int(step_needs[target_index]), remaining_need)
            if target_need <= 0:
                continue
            changed_this_iteration += allocate_target_class(
                prediction=prediction,
                target_index=int(target_index),
                target_need=target_need,
                demand=demand,
                probabilities=probabilities,
                valid=valid,
                suitability_maps=suitability_maps,
                args=args,
                rng=rng,
            )

        after_counts = class_counts(prediction)
        log_rows.extend(
            simulation_log_rows(
                iteration=iteration,
                changed_pixels=changed_this_iteration,
                counts=after_counts,
                demand=demand,
            )
        )
        if changed_this_iteration == 0:
            break

    return prediction, log_rows


def write_factor_summary(path: Path, factors: dict[str, np.ndarray]) -> None:
    """输出本次实际使用了哪些适宜性因子。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["factor", "min", "max", "mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, array in sorted(factors.items()):
            finite = array[np.isfinite(array)]
            writer.writerow(
                {
                    "factor": name,
                    "min": f"{float(np.min(finite)):.6f}" if finite.size else "",
                    "max": f"{float(np.max(finite)):.6f}" if finite.size else "",
                    "mean": f"{float(np.mean(finite)):.6f}" if finite.size else "",
                }
            )


def main() -> None:
    """脚本入口：构建适宜性因子、执行 CA-Markov、输出验证或预测结果。"""

    args = resolve_paths(parse_args())
    validate_args(args)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.suitability_dir.mkdir(parents=True, exist_ok=True)

    fit_from_layer = load_raster(
        raster_path(args.landuse_dir, args.city, args.fit_from), args.fit_from
    )
    base_layer = load_raster(
        raster_path(args.landuse_dir, args.city, args.base_year), args.base_year
    )
    target_path = raster_path(args.landuse_dir, args.city, args.target_year)
    target_layer = load_raster(target_path, args.target_year) if target_path.exists() else None
    if target_layer is not None:
        validate_alignment(fit_from_layer, base_layer, target_layer)
    else:
        validate_alignment(fit_from_layer, base_layer)

    counts = transition_counts(fit_from_layer.array, base_layer.array)
    probabilities = transition_probabilities(counts)
    demand = markov_demand(base_layer.array, probabilities)

    print(f"City: {args.city}")
    print(f"Fit transition: {args.fit_from}->{args.base_year}")
    print(f"Predict target: {args.base_year}->{args.target_year}")
    print(f"Mode: {'validation' if target_layer is not None else 'future projection'}")
    print(f"Neighborhood: {args.neighborhood_size}x{args.neighborhood_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Neighbor weight: {args.neighbor_weight}")
    print(f"Suitability weight: {args.suitability_weight}")
    print(f"Suitability cache: {args.suitability_dir}")
    print("")

    factors = build_factors(args, reference=base_layer)
    if not factors:
        print("Warning: no external suitability factor was built; result will be close to no-factor CA.")
    else:
        print("Active factors:")
        for name in sorted(factors):
            print(f"  - {name}")
    suitability_maps = target_suitability_maps(factors, reference=base_layer)

    prediction, log_rows = simulate_ca_markov_suitability(
        base_array=base_layer.array,
        probabilities=probabilities,
        demand=demand,
        suitability_maps=suitability_maps,
        args=args,
    )

    stem = output_stem(args)
    raster_out = args.output_dir / f"{stem}.tif"
    write_prediction_raster(raster_out, base_layer, prediction)

    accuracy: AccuracyResult | None = None
    if target_layer is not None:
        confusion = confusion_matrix(target_layer.array, prediction)
        accuracy = accuracy_summary(confusion)
        write_confusion_csv(args.tables_dir / f"{stem}_confusion_matrix.csv", confusion)
        write_per_class_accuracy_csv(
            args.tables_dir / f"{stem}_per_class_accuracy.csv", confusion
        )

    write_area_projection_csv(
        args.tables_dir / f"{stem}_area_projection.csv",
        base_array=base_layer.array,
        predicted_array=prediction,
        actual_array=target_layer.array if target_layer is not None else None,
        demand=demand,
        area_km2=pixel_area_km2(base_layer.profile),
    )
    write_simulation_log_csv(args.tables_dir / f"{stem}_simulation_log.csv", log_rows)
    write_factor_summary(args.tables_dir / f"{stem}_factor_summary.csv", factors)

    summary_path = args.tables_dir / f"{stem}_summary.csv"
    write_summary_csv(
        summary_path,
        args=args,
        accuracy=accuracy,
        output_raster=raster_out,
        demand=demand,
        predicted=prediction,
    )

    predicted_counts = class_counts(prediction)
    max_abs_demand_error = int(np.max(np.abs(predicted_counts - demand)))

    print("")
    print(f"Prediction raster: {raster_out}")
    print(f"Summary CSV: {summary_path}")
    if accuracy is not None:
        print(
            f"Validation: OA={accuracy.overall_accuracy:.4f}, "
            f"Kappa={accuracy.kappa:.4f}, pixels={accuracy.total_pixels}"
        )
    else:
        print("Validation skipped: no observed target raster is available.")
    print(f"Max demand error: {max_abs_demand_error} pixels")


if __name__ == "__main__":
    main()
