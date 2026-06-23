"""Logistic-CA land-use prediction for Shenyang.

This script is the first Logistic-CA experiment for the Logistic-CA project.

Default experiment:
1. Use 2015 -> 2020 land-use change to train a multinomial Logistic model.
2. Start from the 2020 land-use map and predict 2025.
3. Compare the prediction with the real 2025 CLCD map.

Model idea:
- Logistic regression learns transition probability from:
  current land-use class, driving factors, and same-class neighborhood support.
- The CA allocation step adds target-class neighborhood attraction.
- Markov demand constrains the final number of pixels in each class, so the
  model is comparable with the previous Markov and CA-Markov experiments.

The implementation avoids storing a full 9-class probability cube for the
whole raster. It predicts one target class at a time to reduce memory use.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
FACTOR_NAMES = [
    "elevation_norm",
    "low_slope",
    "nightlight",
    "road_closeness",
    "water_closeness",
]


@dataclass
class RasterLayer:
    year: int
    path: Path
    array: np.ndarray
    profile: dict
    nodata: float | int | None


@dataclass
class TrainedModel:
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Logistic-CA land-use prediction.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/Logistic-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument("--fit-from", type=int, default=2015)
    parser.add_argument("--base-year", type=int, default=2020)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument(
        "--train-factor-year",
        type=int,
        default=None,
        help="Driver-factor year used for training samples. Default: base year.",
    )
    parser.add_argument(
        "--predict-factor-year",
        type=int,
        default=None,
        help="Driver-factor year used for CA prediction. Default: base year.",
    )
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=40000,
        help="Maximum randomly sampled training pixels for each target class.",
    )
    parser.add_argument(
        "--max-changed-samples",
        type=int,
        default=100000,
        help="Extra changed pixels added to training samples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Number of pixels predicted per sklearn chunk.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--solver", choices=["lbfgs", "saga"], default="lbfgs")
    parser.add_argument("--logistic-weight", type=float, default=0.65)
    parser.add_argument("--neighbor-weight", type=float, default=0.30)
    parser.add_argument("--transition-weight", type=float, default=0.05)
    parser.add_argument("--random-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--landuse-dir", type=Path, default=None)
    parser.add_argument("--factor-dir", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-raster", action="store_true")
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="Only check inputs and class counts, then exit without training.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.project_root
    args.landuse_dir = args.landuse_dir or root / "data" / "processed" / "landuse" / args.city
    args.factor_dir = args.factor_dir or root / "data" / "processed" / "suitability" / args.city
    args.tables_dir = args.tables_dir or root / "tables"
    args.output_dir = args.output_dir or root / "output" / "logistic_ca" / args.city
    args.train_factor_year = args.train_factor_year or args.base_year
    args.predict_factor_year = args.predict_factor_year or args.base_year
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        raise ValueError("--neighborhood-size must be an odd integer >= 3.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1.")
    if args.max_changed_samples < 0:
        raise ValueError("--max-changed-samples must be >= 0.")
    if args.chunk_size < 1000:
        raise ValueError("--chunk-size must be >= 1000.")
    for name in ["logistic_weight", "neighbor_weight", "transition_weight", "random_weight"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0.")


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def factor_path(args: argparse.Namespace, name: str, year: int) -> Path:
    return args.factor_dir / f"{args.city}_{name}_{year}.tif"


def output_stem(args: argparse.Namespace) -> str:
    return (
        f"{args.city}_logistic_ca_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
    )


def log(message: str, log_path: Path | None = None) -> None:
    """Print a progress message and optionally append it to a run log file."""

    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def load_landuse(path: Path, year: int) -> RasterLayer:
    if not path.exists():
        raise FileNotFoundError(f"Missing land-use raster: {path}")
    with rasterio.open(path) as src:
        return RasterLayer(
            year=year,
            path=path,
            array=src.read(1),
            profile=src.profile.copy(),
            nodata=src.nodata,
        )


def alignment_signature_from_profile(profile: dict) -> tuple:
    transform = profile["transform"]
    return (
        profile["width"],
        profile["height"],
        str(profile["crs"]),
        tuple(round(value, 9) for value in transform),
    )


def alignment_signature(layer: RasterLayer) -> tuple:
    return alignment_signature_from_profile(layer.profile)


def validate_alignment(*layers: RasterLayer) -> None:
    signatures = {alignment_signature(layer) for layer in layers}
    if len(signatures) != 1:
        detail = "\n".join(f"{layer.path}: {alignment_signature(layer)}" for layer in layers)
        raise ValueError(f"Land-use rasters are not aligned:\n{detail}")


def load_factor(path: Path, reference: RasterLayer) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing factor raster: {path}")
    with rasterio.open(path) as src:
        if alignment_signature_from_profile(src.profile) != alignment_signature(reference):
            raise ValueError(f"Factor raster is not aligned: {path}")
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        array[array == nodata] = np.nan
    array[~np.isfinite(array)] = 0.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def load_factors(args: argparse.Namespace, year: int, reference: RasterLayer) -> dict[str, np.ndarray]:
    return {name: load_factor(factor_path(args, name, year), reference) for name in FACTOR_NAMES}


def valid_landuse(array: np.ndarray, nodata: float | int | None = None) -> np.ndarray:
    valid = np.isin(array, CLASS_CODES)
    if nodata is not None:
        valid &= array != nodata
    return valid


def valid_mask(*layers: RasterLayer) -> np.ndarray:
    valid = np.ones(layers[0].array.shape, dtype=bool)
    for layer in layers:
        valid &= valid_landuse(layer.array, layer.nodata)
    return valid


def neighborhood_denominator(valid: np.ndarray, size: int) -> np.ndarray:
    """Count valid neighboring cells for every pixel."""

    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 0.0
    return ndimage.convolve(valid.astype(np.float32), kernel, mode="constant", cval=0.0)


def class_neighborhood_fraction(
    array: np.ndarray,
    valid: np.ndarray,
    denominator: np.ndarray,
    class_code: int,
    size: int,
) -> np.ndarray:
    """Fraction of neighboring cells belonging to one target class."""

    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 0.0
    class_cells = ((array == class_code) & valid).astype(np.float32)
    counts = ndimage.convolve(class_cells, kernel, mode="constant", cval=0.0)
    return np.divide(
        counts,
        denominator,
        out=np.zeros_like(counts, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32)


def same_class_neighborhood_map(array: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """For each pixel, store the neighbor fraction of its own current class."""

    denominator = neighborhood_denominator(valid, size)
    result = np.zeros(array.shape, dtype=np.float32)
    for code in CLASS_CODES:
        class_fraction = class_neighborhood_fraction(array, valid, denominator, int(code), size)
        result[(array == code) & valid] = class_fraction[(array == code) & valid]
    return result


def feature_names() -> list[str]:
    class_features = [f"from_{CLASS_NAMES[int(code)].lower().replace('/', '_')}" for code in CLASS_CODES]
    return class_features + FACTOR_NAMES + ["same_class_neighborhood"]


def build_feature_matrix(
    source_array: np.ndarray,
    factors: dict[str, np.ndarray],
    same_neigh: np.ndarray,
    flat_indices: np.ndarray,
) -> np.ndarray:
    """Build model features for selected flat pixel indices."""

    x = np.empty((len(flat_indices), len(feature_names())), dtype=np.float32)
    source_flat = source_array.ravel()[flat_indices]
    col = 0
    for code in CLASS_CODES:
        x[:, col] = source_flat == code
        col += 1
    for name in FACTOR_NAMES:
        x[:, col] = factors[name].ravel()[flat_indices]
        col += 1
    x[:, col] = same_neigh.ravel()[flat_indices]
    return x


def sample_training_indices(
    from_array: np.ndarray,
    to_array: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Balanced sampling by target class, with extra emphasis on changed pixels."""

    rng = np.random.default_rng(args.seed)
    flat_valid = valid.ravel()
    source_flat = from_array.ravel()
    target_flat = to_array.ravel()
    samples = []

    for code in CLASS_CODES:
        idx = np.flatnonzero(flat_valid & (target_flat == code))
        if len(idx) == 0:
            continue
        if len(idx) > args.samples_per_class:
            idx = rng.choice(idx, size=args.samples_per_class, replace=False)
        samples.append(idx)

    if args.max_changed_samples > 0:
        changed = np.flatnonzero(flat_valid & (source_flat != target_flat))
        if len(changed) > args.max_changed_samples:
            changed = rng.choice(changed, size=args.max_changed_samples, replace=False)
        samples.append(changed)

    if not samples:
        raise ValueError("No valid training samples were found.")
    selected = np.unique(np.concatenate(samples))
    rng.shuffle(selected)
    return selected


def train_model(
    fit_layer: RasterLayer,
    base_layer: RasterLayer,
    factors: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> TrainedModel:
    valid = valid_mask(fit_layer, base_layer)
    print("Sampling training pixels...", flush=True)
    sample_idx = sample_training_indices(fit_layer.array, base_layer.array, valid, args)
    print(f"Sampled pixels: {len(sample_idx):,}", flush=True)
    print("Building same-class neighborhood feature for training...", flush=True)
    same_neigh = same_class_neighborhood_map(fit_layer.array, valid, args.neighborhood_size)
    print("Building training feature matrix...", flush=True)
    x = build_feature_matrix(fit_layer.array, factors, same_neigh, sample_idx)
    y = base_layer.array.ravel()[sample_idx].astype(np.uint8)

    print("Scaling training features...", flush=True)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    print("Fitting multinomial LogisticRegression...", flush=True)
    model = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        solver=args.solver,
    )
    model.fit(x_scaled, y)
    print(f"Training samples: {len(sample_idx):,}")
    print("Model target classes:", ", ".join(str(int(code)) for code in model.classes_))
    return TrainedModel(scaler=scaler, model=model, feature_names=feature_names())


def transition_counts(from_array: np.ndarray, to_array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(CLASS_CODES), len(CLASS_CODES)), dtype=np.int64)
    from_flat = from_array.ravel()
    to_flat = to_array.ravel()
    valid_flat = valid.ravel()
    for i, source in enumerate(CLASS_CODES):
        source_mask = valid_flat & (from_flat == source)
        for j, target in enumerate(CLASS_CODES):
            counts[i, j] = np.count_nonzero(source_mask & (to_flat == target))
    return counts


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums > 0,
    )
    for i, row_sum in enumerate(row_sums[:, 0]):
        if row_sum == 0:
            probs[i, i] = 1.0
    return probs


def class_counts(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.array([np.count_nonzero(valid & (array == code)) for code in CLASS_CODES], dtype=np.int64)


def rounded_flow_counts(source_count: int, probabilities: np.ndarray) -> np.ndarray:
    raw = source_count * probabilities
    flows = np.floor(raw).astype(np.int64)
    remainder = int(source_count - flows.sum())
    if remainder > 0:
        order = np.argsort(raw - flows)[::-1]
        flows[order[:remainder]] += 1
    return flows


def markov_demand(base_array: np.ndarray, valid: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    demand = np.zeros(len(CLASS_CODES), dtype=np.int64)
    base_counts = class_counts(base_array, valid)
    for i, source_count in enumerate(base_counts):
        demand += rounded_flow_counts(int(source_count), probabilities[i])
    return demand


def predict_target_probability(
    trained: TrainedModel,
    source_array: np.ndarray,
    factors: dict[str, np.ndarray],
    same_neigh: np.ndarray,
    valid: np.ndarray,
    target_code: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Predict one target-class probability map to keep memory bounded."""

    scores = np.zeros(source_array.size, dtype=np.float32)
    class_list = [int(code) for code in trained.model.classes_]
    if target_code not in class_list:
        return scores.reshape(source_array.shape)

    target_col = class_list.index(target_code)
    valid_flat = valid.ravel()
    for start in range(0, source_array.size, args.chunk_size):
        stop = min(start + args.chunk_size, source_array.size)
        chunk_valid = valid_flat[start:stop]
        if not np.any(chunk_valid):
            continue
        chunk_idx = np.arange(start, stop, dtype=np.int64)[chunk_valid]
        x = build_feature_matrix(source_array, factors, same_neigh, chunk_idx)
        x_scaled = trained.scaler.transform(x)
        scores[chunk_idx] = trained.model.predict_proba(x_scaled)[:, target_col].astype(np.float32)
    return scores.reshape(source_array.shape)


def select_top_pixels(scores: np.ndarray, need: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if need <= 0 or len(finite) == 0:
        return np.array([], dtype=np.int64)
    need = min(int(need), len(finite))
    finite_scores = scores[finite]
    if need == len(finite):
        return finite
    selected = np.argpartition(finite_scores, -need)[-need:]
    return finite[selected]


def surplus_candidate_mask(
    current_flat: np.ndarray,
    valid_flat: np.ndarray,
    changed_flat: np.ndarray,
    target_code: int,
    counts: np.ndarray,
    demand: np.ndarray,
) -> np.ndarray:
    surplus = np.zeros_like(valid_flat, dtype=bool)
    for i, code in enumerate(CLASS_CODES):
        if counts[i] > demand[i]:
            surplus |= current_flat == code
    return valid_flat & (~changed_flat) & surplus & (current_flat != target_code)


def simulate_logistic_ca(
    base_layer: RasterLayer,
    factors: dict[str, np.ndarray],
    trained: TrainedModel,
    demand: np.ndarray,
    transition_probs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    """Iteratively allocate classes according to logistic score and CA attraction."""

    valid = valid_landuse(base_layer.array, base_layer.nodata)
    current = base_layer.array.copy()
    current_flat = current.ravel()
    valid_flat = valid.ravel()
    rng = np.random.default_rng(args.seed)
    log_rows = []

    for iteration in range(1, args.iterations + 1):
        counts_before = class_counts(current, valid)
        same_neigh = same_class_neighborhood_map(current, valid, args.neighborhood_size)
        denominator = neighborhood_denominator(valid, args.neighborhood_size)
        changed = np.zeros(current_flat.shape, dtype=bool)
        remaining_iterations = args.iterations - iteration + 1

        source_idx = np.zeros(current_flat.shape, dtype=np.int16)
        for i, code in enumerate(CLASS_CODES):
            source_idx[current_flat == code] = i

        for target_i, target_code in enumerate(CLASS_CODES):
            counts_now = class_counts(current, valid)
            deficit = int(demand[target_i] - counts_now[target_i])
            if deficit <= 0:
                continue

            step_need = int(math.ceil(deficit / remaining_iterations))
            candidate = surplus_candidate_mask(
                current_flat=current_flat,
                valid_flat=valid_flat,
                changed_flat=changed,
                target_code=int(target_code),
                counts=counts_now,
                demand=demand,
            )
            if not np.any(candidate):
                candidate = valid_flat & (~changed) & (current_flat != target_code)

            logistic_prob = predict_target_probability(
                trained, current, factors, same_neigh, valid, int(target_code), args
            ).ravel()
            target_neigh = class_neighborhood_fraction(
                current, valid, denominator, int(target_code), args.neighborhood_size
            ).ravel()
            transition_score = transition_probs[source_idx, target_i].astype(np.float32)
            scores = (
                args.logistic_weight * logistic_prob
                + args.neighbor_weight * target_neigh
                + args.transition_weight * transition_score
            )
            if args.random_weight > 0:
                scores += args.random_weight * rng.random(scores.shape, dtype=np.float32)
            scores[~candidate] = -np.inf

            selected = select_top_pixels(scores, step_need)
            if len(selected) > 0:
                current_flat[selected] = target_code
                changed[selected] = True

        counts_after = class_counts(current, valid)
        log_rows.append(
            {
                "iteration": iteration,
                "max_abs_demand_error_pixels": int(np.max(np.abs(counts_after - demand))),
                "changed_pixels": int(np.sum(np.abs(counts_after - counts_before))),
            }
        )

    repair_remaining_demand(current, valid, factors, trained, demand, args)
    return current, log_rows


def repair_remaining_demand(
    current: np.ndarray,
    valid: np.ndarray,
    factors: dict[str, np.ndarray],
    trained: TrainedModel,
    demand: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Final small correction so class counts match Markov demand as closely as possible."""

    current_flat = current.ravel()
    valid_flat = valid.ravel()
    same_neigh = same_class_neighborhood_map(current, valid, args.neighborhood_size)

    for target_i, target_code in enumerate(CLASS_CODES):
        counts = class_counts(current, valid)
        deficit = int(demand[target_i] - counts[target_i])
        if deficit <= 0:
            continue

        candidate = valid_flat & (current_flat != target_code)
        surplus = np.zeros_like(candidate, dtype=bool)
        for i, code in enumerate(CLASS_CODES):
            if counts[i] > demand[i]:
                surplus |= current_flat == code
        if np.any(candidate & surplus):
            candidate &= surplus

        scores = predict_target_probability(
            trained, current, factors, same_neigh, valid, int(target_code), args
        ).ravel()
        scores[~candidate] = -np.inf
        selected = select_top_pixels(scores, deficit)
        current_flat[selected] = target_code


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> np.ndarray:
    confusion = np.zeros((len(CLASS_CODES), len(CLASS_CODES)), dtype=np.int64)
    actual_flat = actual.ravel()
    predicted_flat = predicted.ravel()
    valid_flat = valid.ravel()
    for i, actual_code in enumerate(CLASS_CODES):
        actual_mask = valid_flat & (actual_flat == actual_code)
        for j, predicted_code in enumerate(CLASS_CODES):
            confusion[i, j] = np.count_nonzero(actual_mask & (predicted_flat == predicted_code))
    return confusion


def accuracy_summary(confusion: np.ndarray) -> tuple[float, float, int]:
    total = int(confusion.sum())
    if total == 0:
        return 0.0, 0.0, 0
    oa = float(np.trace(confusion) / total)
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    expected = float(np.dot(row_totals, col_totals) / (total * total))
    kappa = 0.0 if expected >= 1 else (oa - expected) / (1.0 - expected)
    return oa, kappa, total


def pixel_area_km2(profile: dict) -> float:
    transform = profile["transform"]
    return abs(float(transform.a) * float(transform.e)) / 1_000_000.0


def write_prediction_raster(path: Path, reference: RasterLayer, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(dtype="uint8", count=1, compress="lzw", nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)


def write_confusion_csv(path: Path, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + [CLASS_NAMES[int(code)] for code in CLASS_CODES])
        for i, code in enumerate(CLASS_CODES):
            writer.writerow([CLASS_NAMES[int(code)]] + confusion[i].tolist())


def write_per_class_accuracy_csv(path: Path, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_code",
            "class_name",
            "producer_accuracy",
            "user_accuracy",
            "f1_score",
            "support_pixels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, code in enumerate(CLASS_CODES):
            tp = confusion[i, i]
            recall = tp / row_totals[i] if row_totals[i] else 0.0
            precision = tp / col_totals[i] if col_totals[i] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            writer.writerow(
                {
                    "class_code": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "producer_accuracy": f"{recall:.6f}",
                    "user_accuracy": f"{precision:.6f}",
                    "f1_score": f"{f1:.6f}",
                    "support_pixels": int(row_totals[i]),
                }
            )


def write_area_projection_csv(
    path: Path,
    base_array: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    profile: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    area = pixel_area_km2(profile)
    base_counts = class_counts(base_array, valid)
    predicted_counts = class_counts(predicted, valid)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_code",
            "class_name",
            "base_pixels",
            "predicted_pixels",
            "base_area_km2",
            "predicted_area_km2",
            "change_area_km2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, code in enumerate(CLASS_CODES):
            base_area = base_counts[i] * area
            predicted_area = predicted_counts[i] * area
            writer.writerow(
                {
                    "class_code": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "base_pixels": int(base_counts[i]),
                    "predicted_pixels": int(predicted_counts[i]),
                    "base_area_km2": f"{base_area:.6f}",
                    "predicted_area_km2": f"{predicted_area:.6f}",
                    "change_area_km2": f"{predicted_area - base_area:.6f}",
                }
            )


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    mode: str,
    oa: float | None,
    kappa: float | None,
    total_pixels: int | None,
    max_demand_error: int,
    output_raster: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "city",
            "mode",
            "fit_from",
            "base_year",
            "target_year",
            "train_factor_year",
            "predict_factor_year",
            "neighborhood_size",
            "iterations",
            "samples_per_class",
            "max_changed_samples",
            "solver",
            "overall_accuracy",
            "kappa",
            "total_pixels",
            "max_abs_demand_error_pixels",
            "output_raster",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "city": args.city,
                "mode": mode,
                "fit_from": args.fit_from,
                "base_year": args.base_year,
                "target_year": args.target_year,
                "train_factor_year": args.train_factor_year,
                "predict_factor_year": args.predict_factor_year,
                "neighborhood_size": args.neighborhood_size,
                "iterations": args.iterations,
                "samples_per_class": args.samples_per_class,
                "max_changed_samples": args.max_changed_samples,
                "solver": args.solver,
                "overall_accuracy": "" if oa is None else f"{oa:.6f}",
                "kappa": "" if kappa is None else f"{kappa:.6f}",
                "total_pixels": "" if total_pixels is None else int(total_pixels),
                "max_abs_demand_error_pixels": int(max_demand_error),
                "output_raster": "" if output_raster is None else str(output_raster),
            }
        )


def write_simulation_log_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["iteration", "max_abs_demand_error_pixels", "changed_pixels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_coefficients_csv(path: Path, trained: TrainedModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["target_class_code", "target_class_name", "feature", "coefficient"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, class_code in enumerate(trained.model.classes_):
            for feature, coef in zip(trained.feature_names, trained.model.coef_[row]):
                writer.writerow(
                    {
                        "target_class_code": int(class_code),
                        "target_class_name": CLASS_NAMES[int(class_code)],
                        "feature": feature,
                        "coefficient": f"{float(coef):.8f}",
                    }
                )


def main() -> None:
    args = resolve_paths(parse_args())
    validate_args(args)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    run_log_path = args.tables_dir / f"{stem}_run_log.txt"
    if run_log_path.exists():
        run_log_path.unlink()

    log(f"City: {args.city}", run_log_path)
    log(f"Train transition: {args.fit_from}->{args.base_year}", run_log_path)
    log(f"Predict target: {args.base_year}->{args.target_year}", run_log_path)
    log(f"Train factor year: {args.train_factor_year}", run_log_path)
    log(f"Predict factor year: {args.predict_factor_year}", run_log_path)
    log(f"Neighborhood: {args.neighborhood_size}x{args.neighborhood_size}", run_log_path)
    log(f"Iterations: {args.iterations}", run_log_path)

    log("Loading fit/base land-use rasters...", run_log_path)
    fit_layer = load_landuse(landuse_path(args, args.fit_from), args.fit_from)
    base_layer = load_landuse(landuse_path(args, args.base_year), args.base_year)
    validate_alignment(fit_layer, base_layer)
    log("Loaded and aligned fit/base land-use rasters.", run_log_path)

    observed_layer = None
    observed_path = landuse_path(args, args.target_year)
    if observed_path.exists():
        log(f"Loading observed target raster: {observed_path}", run_log_path)
        observed_layer = load_landuse(observed_path, args.target_year)
        validate_alignment(fit_layer, base_layer, observed_layer)
        log("Loaded observed target raster.", run_log_path)
    else:
        log(f"Observed target raster not found: {observed_path}", run_log_path)

    log(f"Loading training factors for {args.train_factor_year}...", run_log_path)
    train_factors = load_factors(args, args.train_factor_year, fit_layer)
    log("Loaded training factors.", run_log_path)

    if args.precheck_only:
        train_valid = valid_mask(fit_layer, base_layer)
        fit_counts = class_counts(fit_layer.array, train_valid)
        base_counts = class_counts(base_layer.array, train_valid)
        log("Precheck class counts:", run_log_path)
        for i, code in enumerate(CLASS_CODES):
            log(
                f"  {int(code)} {CLASS_NAMES[int(code)]}: "
                f"fit={int(fit_counts[i])}, base={int(base_counts[i])}",
                run_log_path,
            )
        log("Precheck complete. Training was skipped.", run_log_path)
        return

    log("Training Logistic model...", run_log_path)
    trained = train_model(fit_layer, base_layer, train_factors, args)
    log("Finished Logistic model training.", run_log_path)
    del train_factors

    log("Estimating Markov demand from training transition matrix...", run_log_path)
    train_valid = valid_mask(fit_layer, base_layer)
    trans_counts = transition_counts(fit_layer.array, base_layer.array, train_valid)
    trans_probs = transition_probabilities(trans_counts)

    base_valid = valid_landuse(base_layer.array, base_layer.nodata)
    demand = markov_demand(base_layer.array, base_valid, trans_probs)
    log("Finished Markov demand estimation.", run_log_path)

    log(f"Loading prediction factors for {args.predict_factor_year}...", run_log_path)
    predict_factors = load_factors(args, args.predict_factor_year, base_layer)
    log("Loaded prediction factors.", run_log_path)
    log("Running Logistic-CA simulation...", run_log_path)
    predicted, log_rows = simulate_logistic_ca(
        base_layer=base_layer,
        factors=predict_factors,
        trained=trained,
        demand=demand,
        transition_probs=trans_probs,
        args=args,
    )
    log("Finished Logistic-CA simulation.", run_log_path)

    output_raster = args.output_dir / f"{stem}.tif"
    if args.no_raster:
        output_raster_for_summary = None
    else:
        write_prediction_raster(output_raster, base_layer, predicted)
        output_raster_for_summary = output_raster
        log(f"Prediction raster: {output_raster}", run_log_path)

    max_demand_error = int(np.max(np.abs(class_counts(predicted, base_valid) - demand)))
    oa = None
    kappa = None
    total_pixels = None
    mode = "prediction"

    if observed_layer is not None:
        mode = "validation"
        validation_valid = valid_mask(base_layer, observed_layer)
        confusion = confusion_matrix(observed_layer.array, predicted, validation_valid)
        oa, kappa, total_pixels = accuracy_summary(confusion)
        write_confusion_csv(args.tables_dir / f"{stem}_confusion_matrix.csv", confusion)
        write_per_class_accuracy_csv(args.tables_dir / f"{stem}_per_class_accuracy.csv", confusion)
        log(f"Validation: OA={oa:.6f}, Kappa={kappa:.6f}, pixels={total_pixels:,}", run_log_path)
    else:
        log("Prediction mode: validation metrics were skipped.", run_log_path)

    write_area_projection_csv(
        args.tables_dir / f"{stem}_area_projection.csv",
        base_layer.array,
        predicted,
        base_valid,
        base_layer.profile,
    )
    write_summary_csv(
        args.tables_dir / f"{stem}_summary.csv",
        args,
        mode,
        oa,
        kappa,
        total_pixels,
        max_demand_error,
        output_raster_for_summary,
    )
    write_simulation_log_csv(args.tables_dir / f"{stem}_simulation_log.csv", log_rows)
    write_coefficients_csv(args.tables_dir / f"{stem}_coefficients.csv", trained)
    log(f"Summary CSV: {args.tables_dir / f'{stem}_summary.csv'}", run_log_path)
    log(f"Run log: {run_log_path}", run_log_path)
    log(f"Max demand error: {max_demand_error} pixels", run_log_path)


if __name__ == "__main__":
    main()
