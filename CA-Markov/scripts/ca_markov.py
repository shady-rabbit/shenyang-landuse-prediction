"""CLCD 原始 9 类 CA-Markov 第一版实验脚本。

这个脚本在 markov_baseline.py 的基础上加入 CA 邻域效应：
- Markov 部分负责估计各土地利用类型的目标数量；
- CA 部分根据邻域内目标类别的比例，决定变化更可能发生在哪里。

当前版本只使用邻域信息，不使用 DEM、道路、水系、夜光等外部驱动因子。
这样可以先复现 IDRISI CA-Markov 的核心思想，再逐步加入适宜性因子。

脚本保持 CLCD 原始类别编码不变：
1 Cropland, 2 Forest, 3 Shrub, 4 Grassland, 5 Water,
6 Snow/Ice, 7 Barren, 8 Impervious, 9 Wetland.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。

    用解释器绝对路径运行脚本时，Windows 环境有时不会自动带上
    GDAL_DATA 和 PROJ_LIB。这里在导入 rasterio 前补齐，避免警告。
    """

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"

    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_geospatial_data_paths()

# rasterio 依赖 GDAL，因此要在 configure_geospatial_data_paths() 之后导入。
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

# 写入 GeoTIFF 的分类颜色表。这样在 GIS 软件里打开预测结果时，
# 不会因为单波段分类值 1-9 太小而显示成近乎全黑的灰度图。
LANDUSE_COLORMAP = {
    0: (255, 255, 255, 0),  # NoData
    1: (244, 224, 77, 255),  # Cropland
    2: (38, 115, 0, 255),  # Forest
    3: (109, 187, 117, 255),  # Shrub
    4: (156, 195, 107, 255),  # Grassland
    5: (75, 156, 211, 255),  # Water
    6: (247, 251, 255, 255),  # Snow/Ice
    7: (189, 189, 189, 255),  # Barren
    8: (215, 25, 28, 255),  # Impervious
    9: (65, 182, 196, 255),  # Wetland
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
class AccuracyResult:
    """保存一次 CA-Markov 验证的核心精度结果。"""

    overall_accuracy: float
    kappa: float
    total_pixels: int


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    常用运行方式：
        python scripts/ca_markov.py --city shenyang

    默认任务是用 2020->2025 的转移规律，从 2025 年预测 2030 年。
    如果要做 2025 年验证，请显式传入：
        --fit-from 2015 --base-year 2020 --target-year 2025
    """

    parser = argparse.ArgumentParser(
        description="CLCD original 9-class CA-Markov experiment."
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
        default=2020,
        help="Start year used to estimate Markov transition probabilities.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2025,
        help="Base year used as the CA-Markov simulation start.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2030,
        help="Observed target year used for validation.",
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
        "--random-weight",
        type=float,
        default=0.03,
        help="Small stochastic perturbation used when ranking candidate cells.",
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
    """根据项目根目录补齐输入和输出路径。"""

    root = args.project_root
    if args.landuse_dir is None:
        args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    if args.tables_dir is None:
        args.tables_dir = root / "tables"
    if args.output_dir is None:
        args.output_dir = root / "output" / "ca_markov" / args.city
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查 CA 参数是否合理。"""

    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        raise ValueError("--neighborhood-size must be an odd integer >= 3.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.random_weight < 0:
        raise ValueError("--random-weight must be >= 0.")


def raster_path(landuse_dir: Path, city: str, year: int) -> Path:
    """拼出某城市某年份的 CLCD 裁剪结果路径。"""

    return landuse_dir / f"{city}_clcd_v01_{year}_original.tif"


def load_raster(path: Path, year: int) -> RasterLayer:
    """读取单期土地利用栅格。"""

    if not path.exists():
        raise FileNotFoundError(f"Missing land-use raster: {path}")

    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    return RasterLayer(year=year, path=path, array=array, profile=profile, nodata=nodata)


def alignment_signature(layer: RasterLayer) -> tuple:
    """提取用于检查栅格对齐的空间信息。"""

    transform = layer.profile["transform"]
    return (
        layer.profile["width"],
        layer.profile["height"],
        str(layer.profile["crs"]),
        tuple(round(value, 9) for value in transform),
    )


def validate_alignment(*layers: RasterLayer) -> None:
    """检查所有输入栅格是否在同一网格上。"""

    first = layers[0]
    first_signature = alignment_signature(first)
    for layer in layers[1:]:
        if alignment_signature(layer) != first_signature:
            raise ValueError(
                f"Raster alignment differs: {first.year} and {layer.year}. "
                "Run preprocessing before CA-Markov simulation."
            )


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """返回所有数组中都属于 CLCD 1-9 类的位置。

    NoData=0 不参与 Markov 统计、CA 分配或精度评价。
    """

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isin(array, CLASS_CODES)
    return mask


def transition_counts(from_array: np.ndarray, to_array: np.ndarray) -> np.ndarray:
    """计算两个年份之间的 9x9 转移数量矩阵。

    行表示起始年份类别，列表示目标年份类别。
    """

    mask = valid_mask(from_array, to_array)
    from_values = from_array[mask].astype(np.int16) - 1
    to_values = to_array[mask].astype(np.int16) - 1
    flat_index = from_values * len(CLASS_CODES) + to_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    """把转移数量矩阵转换成按行归一化的转移概率矩阵。"""

    probabilities = np.zeros(counts.shape, dtype=np.float64)
    row_totals = counts.sum(axis=1)
    for row_index, total in enumerate(row_totals):
        if total > 0:
            probabilities[row_index] = counts[row_index] / total
        else:
            # 如果某类在起始年份不存在，让它保持自身，避免除零。
            probabilities[row_index, row_index] = 1.0
    return probabilities


def class_counts(array: np.ndarray) -> np.ndarray:
    """统计栅格中 CLCD 1-9 类的像元数。"""

    counts = np.zeros(len(CLASS_CODES), dtype=np.int64)
    mask = np.isin(array, CLASS_CODES)
    values, value_counts = np.unique(array[mask], return_counts=True)
    for value, count in zip(values, value_counts, strict=True):
        counts[int(value) - 1] = int(count)
    return counts


def rounded_flow_counts(source_count: int, probabilities: np.ndarray) -> np.ndarray:
    """把一类像元的转移概率换算成整数像元数。

    先向下取整，再把剩余像元补给小数部分最大的类别，保证总数不丢失。
    """

    expected = probabilities * source_count
    counts = np.floor(expected).astype(np.int64)
    remainder = source_count - int(counts.sum())
    if remainder > 0:
        fractions = expected - counts
        order = np.argsort(-fractions, kind="stable")
        counts[order[:remainder]] += 1
    return counts


def markov_demand(base_array: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """根据基期图和 Markov 概率计算目标期各类需求量。"""

    base_counts = class_counts(base_array)
    demand = np.zeros(len(CLASS_CODES), dtype=np.int64)
    for source_index, source_count in enumerate(base_counts):
        demand += rounded_flow_counts(int(source_count), probabilities[source_index])
    return demand


def apportion_with_caps(
    total: int,
    weights: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    """按权重分配整数数量，同时不超过每个来源类别的可供给上限。

    CA 分配时，一个来源类别如果已经达到 Markov 需求量，就不能继续
    被转换出去。caps 就是每个来源类别还可以贡献的像元数量。
    """

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

        # 当 remaining 很小时，floor 可能全为 0；这时逐个补给优先级最高的类。
        if proposal.sum() == 0:
            priority = np.where(active, expected, -1.0)
            order = np.argsort(-priority, kind="stable")
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


def neighborhood_fraction(
    array: np.ndarray,
    target_code: int,
    valid: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    """计算每个像元邻域内目标类别的比例。

    比例越高，说明该像元周围目标类别越集中，CA 中转为该类的吸引力越强。
    中心像元不参与邻域统计。
    """

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
    target_neighborhood_flat: np.ndarray,
    count: int,
    rng: np.random.Generator,
    random_weight: float,
) -> np.ndarray:
    """在某个来源类别中选择最适合转为目标类别的像元。

    评分主要由“邻域中目标类别比例”决定，并加入很小的随机扰动。
    当多个像元邻域条件相近时，随机扰动可以避免总是选择同一批位置。
    """

    if count <= 0:
        return np.array([], dtype=np.int64)

    source_positions = np.flatnonzero(prediction_flat == source_code)
    if len(source_positions) == 0:
        return np.array([], dtype=np.int64)
    if count >= len(source_positions):
        return source_positions

    scores = target_neighborhood_flat[source_positions].astype(np.float32, copy=True)
    if random_weight > 0:
        scores += rng.random(len(source_positions), dtype=np.float32) * random_weight

    # argpartition 只找前 count 个高分像元，比完整排序更省时间。
    top_local = np.argpartition(scores, -count)[-count:]
    return source_positions[top_local]


def allocate_target_class(
    prediction: np.ndarray,
    target_index: int,
    target_need: int,
    demand: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
    neighborhood_size: int,
    rng: np.random.Generator,
    random_weight: float,
) -> int:
    """为一个目标类别分配本轮需要新增的像元。

    先根据来源类别的剩余供给量和 Markov 转移概率确定从哪些类转入；
    再在每个来源类别内部，按邻域吸引力选择具体像元位置。
    """

    if target_need <= 0:
        return 0

    current_counts = class_counts(prediction)
    surplus = np.maximum(current_counts - demand, 0)
    target_code = int(CLASS_CODES[target_index])

    # 目标类别本身不能作为转入来源；只有超过需求量的类别才能贡献像元。
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
        # 如果历史转移概率全为 0，退化为按来源类别可供给量分配。
        weights = caps.astype(np.float64)

    source_take = apportion_with_caps(target_need, weights, caps)
    if source_take.sum() <= 0:
        return 0

    target_neighborhood = neighborhood_fraction(
        prediction,
        target_code=target_code,
        valid=valid,
        neighborhood_size=neighborhood_size,
    )
    prediction_flat = prediction.reshape(-1)
    neighborhood_flat = target_neighborhood.reshape(-1)

    changed = 0
    for source_index, take in zip(source_indices, source_take, strict=True):
        take = int(take)
        if take <= 0:
            continue

        source_code = int(CLASS_CODES[source_index])
        selected = select_best_source_cells(
            prediction_flat=prediction_flat,
            source_code=source_code,
            target_neighborhood_flat=neighborhood_flat,
            count=take,
            rng=rng,
            random_weight=random_weight,
        )
        if len(selected) == 0:
            continue

        prediction_flat[selected] = target_code
        changed += int(len(selected))

    return changed


def simulate_ca_markov(
    base_array: np.ndarray,
    probabilities: np.ndarray,
    demand: np.ndarray,
    neighborhood_size: int,
    iterations: int,
    seed: int,
    random_weight: float,
) -> tuple[np.ndarray, list[dict]]:
    """执行 CA-Markov 空间分配。

    模拟过程：
    1. 从基期图开始；
    2. 根据 Markov 目标需求量计算哪些类别需要增加、哪些类别需要减少；
    3. 每一轮只分配一部分变化量，并重新计算邻域；
    4. 迭代结束时尽量使预测图的各类数量达到 Markov 需求量。
    """

    rng = np.random.default_rng(seed)
    prediction = base_array.copy()
    valid = np.isin(base_array, CLASS_CODES)
    log_rows: list[dict] = []

    for iteration in range(1, iterations + 1):
        current_counts = class_counts(prediction)
        deficits = np.maximum(demand - current_counts, 0)
        if deficits.sum() == 0:
            break

        remaining_iterations = iterations - iteration + 1
        step_needs = np.ceil(deficits / remaining_iterations).astype(np.int64)
        target_order = np.argsort(-step_needs, kind="stable")

        changed_this_iteration = 0
        for target_index in target_order:
            current_counts = class_counts(prediction)
            remaining_need = max(int(demand[target_index] - current_counts[target_index]), 0)
            target_need = min(int(step_needs[target_index]), remaining_need)
            if target_need <= 0:
                continue

            changed = allocate_target_class(
                prediction=prediction,
                target_index=int(target_index),
                target_need=target_need,
                demand=demand,
                probabilities=probabilities,
                valid=valid,
                neighborhood_size=neighborhood_size,
                rng=rng,
                random_weight=random_weight,
            )
            changed_this_iteration += changed

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

    # 最后一轮后如果还有少量需求未满足，再做一次修补分配。
    final_changed = repair_remaining_demand(
        prediction=prediction,
        demand=demand,
        probabilities=probabilities,
        valid=valid,
        neighborhood_size=neighborhood_size,
        rng=rng,
        random_weight=random_weight,
    )
    if final_changed > 0:
        after_counts = class_counts(prediction)
        log_rows.extend(
            simulation_log_rows(
                iteration=iterations + 1,
                changed_pixels=final_changed,
                counts=after_counts,
                demand=demand,
            )
        )

    return prediction, log_rows


def repair_remaining_demand(
    prediction: np.ndarray,
    demand: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
    neighborhood_size: int,
    rng: np.random.Generator,
    random_weight: float,
) -> int:
    """修补迭代后仍未满足的类别需求。

    正常情况下前面的多轮分配已经接近 Markov 需求量。
    这个函数用于处理四舍五入或来源类别上限造成的少量残差。
    """

    changed_total = 0
    for _ in range(len(CLASS_CODES)):
        current_counts = class_counts(prediction)
        deficits = np.maximum(demand - current_counts, 0)
        if deficits.sum() == 0:
            break

        changed_round = 0
        target_order = np.argsort(-deficits, kind="stable")
        for target_index in target_order:
            current_counts = class_counts(prediction)
            target_need = max(int(demand[target_index] - current_counts[target_index]), 0)
            if target_need <= 0:
                continue
            changed_round += allocate_target_class(
                prediction=prediction,
                target_index=int(target_index),
                target_need=target_need,
                demand=demand,
                probabilities=probabilities,
                valid=valid,
                neighborhood_size=neighborhood_size,
                rng=rng,
                random_weight=random_weight,
            )

        changed_total += changed_round
        if changed_round == 0:
            break

    return changed_total


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


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """计算真实图与预测图之间的混淆矩阵。"""

    mask = valid_mask(actual, predicted)
    actual_values = actual[mask].astype(np.int16) - 1
    predicted_values = predicted[mask].astype(np.int16) - 1
    flat_index = actual_values * len(CLASS_CODES) + predicted_values
    return np.bincount(flat_index, minlength=len(CLASS_CODES) ** 2).reshape(
        (len(CLASS_CODES), len(CLASS_CODES))
    )


def accuracy_summary(confusion: np.ndarray) -> AccuracyResult:
    """根据混淆矩阵计算 OA 和 Kappa 系数。"""

    total = int(confusion.sum())
    if total == 0:
        return AccuracyResult(overall_accuracy=0.0, kappa=0.0, total_pixels=0)

    overall = float(np.trace(confusion)) / float(total)
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    expected = float(np.sum(row_totals * col_totals)) / float(total * total)
    kappa = (overall - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return AccuracyResult(
        overall_accuracy=overall,
        kappa=kappa,
        total_pixels=total,
    )


def pixel_area_km2(profile: dict) -> float:
    """根据栅格仿射变换计算单个像元面积，单位为平方公里。"""

    transform = profile["transform"]
    return abs(transform.a * transform.e - transform.b * transform.d) / 1_000_000


def write_prediction_raster(path: Path, layer: RasterLayer, array: np.ndarray) -> None:
    """写出 CA-Markov 预测 GeoTIFF。"""

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
        dst.write(array.astype(np.uint8), 1)
        dst.write_colormap(1, LANDUSE_COLORMAP)


def write_confusion_csv(path: Path, confusion: np.ndarray) -> None:
    """写出混淆矩阵 CSV。"""

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
    """写出各类别精度表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    actual_totals = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
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
    with path.open("w", newline="", encoding="utf-8") as f:
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
    actual_array: np.ndarray | None,
    demand: np.ndarray,
    area_km2: float,
) -> None:
    """写出 Markov 需求量、CA 预测量和真实面积对比表。

    当 target_year 是 2030 这类未来年份时，没有真实 CLCD 图，
    actual_array 传入 None，真实面积和面积差异字段留空。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    base_counts = class_counts(base_array)
    predicted_counts = class_counts(predicted_array)
    actual_counts = class_counts(actual_array) if actual_array is not None else None
    fieldnames = [
        "class_code",
        "class_name",
        "base_pixels",
        "markov_demand_pixels",
        "ca_predicted_pixels",
        "actual_pixels",
        "markov_demand_area_km2",
        "ca_predicted_area_km2",
        "actual_area_km2",
        "ca_area_difference_km2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, class_code in enumerate(CLASS_CODES):
            ca_area = float(predicted_counts[index]) * area_km2
            actual_area = (
                float(actual_counts[index]) * area_km2
                if actual_counts is not None
                else None
            )
            writer.writerow(
                {
                    "class_code": int(class_code),
                    "class_name": CLASS_NAMES[int(class_code)],
                    "base_pixels": int(base_counts[index]),
                    "markov_demand_pixels": int(demand[index]),
                    "ca_predicted_pixels": int(predicted_counts[index]),
                    "actual_pixels": (
                        int(actual_counts[index]) if actual_counts is not None else ""
                    ),
                    "markov_demand_area_km2": f"{float(demand[index]) * area_km2:.6f}",
                    "ca_predicted_area_km2": f"{ca_area:.6f}",
                    "actual_area_km2": (
                        f"{actual_area:.6f}" if actual_area is not None else ""
                    ),
                    "ca_area_difference_km2": (
                        f"{ca_area - actual_area:.6f}"
                        if actual_area is not None
                        else ""
                    ),
                }
            )


def write_simulation_log_csv(path: Path, rows: list[dict]) -> None:
    """写出每轮 CA 模拟后的类别数量记录。"""

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
    accuracy: AccuracyResult | None,
    output_raster: Path,
    demand: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """写出本次 CA-Markov 实验的总体结果。

    如果存在 target_year 的真实图，这是验证结果；
    如果 target_year 是未来年份，这是预测结果，OA/Kappa 留空。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    predicted_counts = class_counts(predicted)
    max_abs_demand_error = int(np.max(np.abs(predicted_counts - demand)))
    fieldnames = [
        "city",
        "mode",
        "has_observed_target",
        "fit_from",
        "base_year",
        "target_year",
        "neighborhood_size",
        "iterations",
        "random_weight",
        "overall_accuracy",
        "kappa",
        "total_pixels",
        "max_abs_demand_error_pixels",
        "output_raster",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
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
                "neighborhood_size": args.neighborhood_size,
                "iterations": args.iterations,
                "random_weight": args.random_weight,
                "overall_accuracy": (
                    f"{accuracy.overall_accuracy:.6f}" if accuracy is not None else ""
                ),
                "kappa": f"{accuracy.kappa:.6f}" if accuracy is not None else "",
                "total_pixels": accuracy.total_pixels if accuracy is not None else "",
                "max_abs_demand_error_pixels": max_abs_demand_error,
                "output_raster": str(output_raster),
            }
        )


def output_stem(args: argparse.Namespace) -> str:
    """统一生成 CA-Markov 输出文件名前缀。"""

    return (
        f"{args.city}_ca_markov_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
    )


def main() -> None:
    """脚本入口：计算 Markov 需求量，执行 CA 分配，并输出结果。

    两种典型用法：
    - 验证：2015->2020 预测 2025，因为 2025 真实图存在，可输出 OA/Kappa；
    - 预测：2020->2025 预测 2030，因为 2030 真实图不存在，只输出未来预测图。
    """

    args = resolve_paths(parse_args())
    validate_args(args)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    if target_layer is None:
        print(f"Observed target raster not found, validation will be skipped: {target_path}")
    print(f"Neighborhood: {args.neighborhood_size}x{args.neighborhood_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Land-use rasters: {args.landuse_dir}")
    print(f"Output rasters: {args.output_dir}")
    print(f"Tables: {args.tables_dir}")
    print("")

    prediction, log_rows = simulate_ca_markov(
        base_array=base_layer.array,
        probabilities=probabilities,
        demand=demand,
        neighborhood_size=args.neighborhood_size,
        iterations=args.iterations,
        seed=args.seed,
        random_weight=args.random_weight,
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
