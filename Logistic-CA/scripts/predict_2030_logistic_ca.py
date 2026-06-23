"""Dedicated Logistic-CA 2030 prediction script.

This script is intentionally separate from `logistic_ca.py`.

Purpose:
- Train transition rules with 2020 -> 2025.
- Use the 2025 land-use map as the simulation base.
- Predict the 2030 land-use map.

Why this script exists:
- The general script uses `LogisticRegression`, which may be unstable on some
  Windows/BLAS environments when fitting a large multinomial model.
- This script uses `SGDClassifier(loss="log_loss")`, a more incremental
  logistic model that is usually more stable for large raster samples.

Outputs:
- GeoTIFF:
  output/logistic_ca/shenyang/shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5.tif
- CSV tables:
  tables/shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5_*.csv
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
from sklearn.linear_model import SGDClassifier
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
    model: SGDClassifier
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict 2030 with Logistic-CA.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/Logistic-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument("--fit-from", type=int, default=2020)
    parser.add_argument("--base-year", type=int, default=2025)
    parser.add_argument("--target-year", type=int, default=2030)
    parser.add_argument("--factor-year", type=int, default=2025)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=20000)
    parser.add_argument("--max-changed-samples", type=int, default=60000)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--sgd-max-iter", type=int, default=1000)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--logistic-weight", type=float, default=0.65)
    parser.add_argument("--neighbor-weight", type=float, default=0.30)
    parser.add_argument("--transition-weight", type=float, default=0.05)
    parser.add_argument("--random-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--landuse-dir", type=Path, default=None)
    parser.add_argument("--factor-dir", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.project_root
    args.landuse_dir = args.landuse_dir or root / "data" / "processed" / "landuse" / args.city
    args.factor_dir = args.factor_dir or root / "data" / "processed" / "suitability" / args.city
    args.tables_dir = args.tables_dir or root / "tables"
    args.output_dir = args.output_dir or root / "output" / "logistic_ca" / args.city
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


def log(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def output_stem(args: argparse.Namespace) -> str:
    return (
        f"{args.city}_logistic_ca_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
    )


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def factor_path(args: argparse.Namespace, name: str, year: int) -> Path:
    return args.factor_dir / f"{args.city}_{name}_{year}.tif"


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


def load_factors(args: argparse.Namespace, reference: RasterLayer) -> dict[str, np.ndarray]:
    return {name: load_factor(factor_path(args, name, args.factor_year), reference) for name in FACTOR_NAMES}


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


def class_counts(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.array([np.count_nonzero(valid & (array == code)) for code in CLASS_CODES], dtype=np.int64)


def neighborhood_denominator(valid: np.ndarray, size: int) -> np.ndarray:
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
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 0.0
    cells = ((array == class_code) & valid).astype(np.float32)
    counts = ndimage.convolve(cells, kernel, mode="constant", cval=0.0)
    return np.divide(
        counts,
        denominator,
        out=np.zeros_like(counts, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32)


def same_class_neighborhood_map(array: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    denominator = neighborhood_denominator(valid, size)
    out = np.zeros(array.shape, dtype=np.float32)
    for code in CLASS_CODES:
        frac = class_neighborhood_fraction(array, valid, denominator, int(code), size)
        mask = (array == code) & valid
        out[mask] = frac[mask]
    return out


def feature_names() -> list[str]:
    from_features = [f"from_{CLASS_NAMES[int(code)].lower().replace('/', '_')}" for code in CLASS_CODES]
    return from_features + FACTOR_NAMES + ["same_class_neighborhood"]


def build_feature_matrix(
    source_array: np.ndarray,
    factors: dict[str, np.ndarray],
    same_neigh: np.ndarray,
    flat_indices: np.ndarray,
) -> np.ndarray:
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
    source_array: np.ndarray,
    target_array: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    valid_flat = valid.ravel()
    source_flat = source_array.ravel()
    target_flat = target_array.ravel()
    samples = []

    for code in CLASS_CODES:
        idx = np.flatnonzero(valid_flat & (target_flat == code))
        if len(idx) == 0:
            continue
        if len(idx) > args.samples_per_class:
            idx = rng.choice(idx, size=args.samples_per_class, replace=False)
        samples.append(idx)

    changed = np.flatnonzero(valid_flat & (source_flat != target_flat))
    if len(changed) > args.max_changed_samples:
        changed = rng.choice(changed, size=args.max_changed_samples, replace=False)
    samples.append(changed)

    selected = np.unique(np.concatenate(samples))
    rng.shuffle(selected)
    return selected


def train_sgd_logistic(
    fit_layer: RasterLayer,
    base_layer: RasterLayer,
    factors: dict[str, np.ndarray],
    args: argparse.Namespace,
    log_path: Path,
) -> TrainedModel:
    valid = valid_mask(fit_layer, base_layer)
    log("Sampling training pixels...", log_path)
    sample_idx = sample_training_indices(fit_layer.array, base_layer.array, valid, args)
    log(f"Sampled pixels: {len(sample_idx):,}", log_path)

    log("Building neighborhood feature...", log_path)
    same_neigh = same_class_neighborhood_map(fit_layer.array, valid, args.neighborhood_size)

    log("Building training feature matrix...", log_path)
    x = build_feature_matrix(fit_layer.array, factors, same_neigh, sample_idx)
    y = base_layer.array.ravel()[sample_idx].astype(np.uint8)

    log("Scaling features...", log_path)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    log("Fitting SGD logistic model...", log_path)
    model = SGDClassifier(
        loss="log_loss",
        alpha=args.sgd_alpha,
        max_iter=args.sgd_max_iter,
        tol=1e-3,
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(x_scaled, y)
    log("Finished SGD logistic model.", log_path)
    log("Model target classes: " + ", ".join(str(int(code)) for code in model.classes_), log_path)
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
    scores = np.zeros(source_array.size, dtype=np.float32)
    model_classes = [int(code) for code in trained.model.classes_]
    if target_code not in model_classes:
        return scores.reshape(source_array.shape)

    target_col = model_classes.index(target_code)
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
    log_path: Path,
) -> tuple[np.ndarray, list[dict]]:
    valid = valid_landuse(base_layer.array, base_layer.nodata)
    current = base_layer.array.copy()
    current_flat = current.ravel()
    valid_flat = valid.ravel()
    rng = np.random.default_rng(args.seed)
    rows = []

    for iteration in range(1, args.iterations + 1):
        log(f"CA iteration {iteration}/{args.iterations}...", log_path)
        counts_before = class_counts(current, valid)
        same_neigh = same_class_neighborhood_map(current, valid, args.neighborhood_size)
        denominator = neighborhood_denominator(valid, args.neighborhood_size)
        changed = np.zeros(current_flat.shape, dtype=bool)
        remaining = args.iterations - iteration + 1

        source_idx = np.zeros(current_flat.shape, dtype=np.int16)
        for i, code in enumerate(CLASS_CODES):
            source_idx[current_flat == code] = i

        for target_i, target_code in enumerate(CLASS_CODES):
            counts_now = class_counts(current, valid)
            deficit = int(demand[target_i] - counts_now[target_i])
            if deficit <= 0:
                continue
            step_need = int(math.ceil(deficit / remaining))
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
        max_error = int(np.max(np.abs(counts_after - demand)))
        rows.append(
            {
                "iteration": iteration,
                "max_abs_demand_error_pixels": max_error,
                "changed_pixels": int(np.sum(np.abs(counts_after - counts_before))),
            }
        )
        log(f"  max demand error: {max_error} pixels", log_path)

    repair_remaining_demand(current, valid, factors, trained, demand, args, log_path)
    return current, rows


def repair_remaining_demand(
    current: np.ndarray,
    valid: np.ndarray,
    factors: dict[str, np.ndarray],
    trained: TrainedModel,
    demand: np.ndarray,
    args: argparse.Namespace,
    log_path: Path,
) -> None:
    log("Repairing remaining demand error...", log_path)
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


def pixel_area_km2(profile: dict) -> float:
    transform = profile["transform"]
    return abs(float(transform.a) * float(transform.e)) / 1_000_000.0


def write_prediction_raster(path: Path, reference: RasterLayer, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(dtype="uint8", count=1, compress="lzw", nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8), 1)


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
    pred_counts = class_counts(predicted, valid)
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
            pred_area = pred_counts[i] * area
            writer.writerow(
                {
                    "class_code": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "base_pixels": int(base_counts[i]),
                    "predicted_pixels": int(pred_counts[i]),
                    "base_area_km2": f"{base_area:.6f}",
                    "predicted_area_km2": f"{pred_area:.6f}",
                    "change_area_km2": f"{pred_area - base_area:.6f}",
                }
            )


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    max_error: int,
    output_raster: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "city",
            "mode",
            "fit_from",
            "base_year",
            "target_year",
            "factor_year",
            "neighborhood_size",
            "iterations",
            "samples_per_class",
            "max_changed_samples",
            "sgd_alpha",
            "sgd_max_iter",
            "max_abs_demand_error_pixels",
            "output_raster",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "city": args.city,
                "mode": "prediction",
                "fit_from": args.fit_from,
                "base_year": args.base_year,
                "target_year": args.target_year,
                "factor_year": args.factor_year,
                "neighborhood_size": args.neighborhood_size,
                "iterations": args.iterations,
                "samples_per_class": args.samples_per_class,
                "max_changed_samples": args.max_changed_samples,
                "sgd_alpha": args.sgd_alpha,
                "sgd_max_iter": args.sgd_max_iter,
                "max_abs_demand_error_pixels": int(max_error),
                "output_raster": str(output_raster),
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
    log_path = args.tables_dir / f"{stem}_sgd_run_log.txt"
    if log_path.exists():
        log_path.unlink()

    log(f"City: {args.city}", log_path)
    log(f"Train transition: {args.fit_from}->{args.base_year}", log_path)
    log(f"Predict target: {args.base_year}->{args.target_year}", log_path)
    log(f"Factor year: {args.factor_year}", log_path)

    log("Loading land-use rasters...", log_path)
    fit_layer = load_landuse(landuse_path(args, args.fit_from), args.fit_from)
    base_layer = load_landuse(landuse_path(args, args.base_year), args.base_year)
    validate_alignment(fit_layer, base_layer)

    log("Loading factors...", log_path)
    factors = load_factors(args, fit_layer)

    trained = train_sgd_logistic(fit_layer, base_layer, factors, args, log_path)

    log("Estimating Markov demand...", log_path)
    train_valid = valid_mask(fit_layer, base_layer)
    trans_counts = transition_counts(fit_layer.array, base_layer.array, train_valid)
    trans_probs = transition_probabilities(trans_counts)
    base_valid = valid_landuse(base_layer.array, base_layer.nodata)
    demand = markov_demand(base_layer.array, base_valid, trans_probs)

    log("Running Logistic-CA simulation...", log_path)
    predicted, rows = simulate_logistic_ca(
        base_layer=base_layer,
        factors=factors,
        trained=trained,
        demand=demand,
        transition_probs=trans_probs,
        args=args,
        log_path=log_path,
    )

    output_raster = args.output_dir / f"{stem}.tif"
    write_prediction_raster(output_raster, base_layer, predicted)
    max_error = int(np.max(np.abs(class_counts(predicted, base_valid) - demand)))

    write_area_projection_csv(
        args.tables_dir / f"{stem}_area_projection.csv",
        base_layer.array,
        predicted,
        base_valid,
        base_layer.profile,
    )
    write_summary_csv(args.tables_dir / f"{stem}_summary.csv", args, max_error, output_raster)
    write_simulation_log_csv(args.tables_dir / f"{stem}_simulation_log.csv", rows)
    write_coefficients_csv(args.tables_dir / f"{stem}_coefficients.csv", trained)

    log(f"Prediction raster: {output_raster}", log_path)
    log(f"Summary CSV: {args.tables_dir / f'{stem}_summary.csv'}", log_path)
    log(f"Run log: {log_path}", log_path)
    log(f"Max demand error: {max_error} pixels", log_path)
    log("Done.", log_path)


if __name__ == "__main__":
    main()
