"""Plot Logistic-CA validation or prediction results.

Run this after `logistic_ca.py`.

Default behavior:
- Read the 2015->2020 predict 2025 Logistic-CA output.
- If the observed 2025 CLCD raster exists, draw validation figures.
- Otherwise, draw future-prediction figures.

This script only reads existing GeoTIFF/CSV files. It does not rerun the model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling


CLASS_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
CLASS_COLORS = {
    1: "#f4e04d",
    2: "#26734d",
    3: "#8cc665",
    4: "#b8e186",
    5: "#4a90e2",
    6: "#ffffff",
    7: "#c2b280",
    8: "#d73027",
    9: "#80cdc1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Logistic-CA result figures.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/Logistic-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument("--fit-from", type=int, default=2015)
    parser.add_argument("--base-year", type=int, default=2020)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.project_root
    args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    args.output_dir = root / "output" / "logistic_ca" / args.city
    args.tables_dir = root / "tables"
    args.figures_dir = args.figures_dir or root / "figures" / "logistic_ca" / args.city
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    return args


def output_stem(args: argparse.Namespace) -> str:
    return (
        f"{args.city}_logistic_ca_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
    )


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def predicted_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"{output_stem(args)}.tif"


def confusion_path(args: argparse.Namespace) -> Path:
    return args.tables_dir / f"{output_stem(args)}_confusion_matrix.csv"


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")


def read_raster_for_plot(path: Path, max_size: int) -> np.ndarray:
    """Read a downsampled raster preview to keep plotting fast."""

    ensure_exists(path)
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_size
        if scale <= 1:
            return src.read(1)
        out_height = max(1, int(src.height / scale))
        out_width = max(1, int(src.width / scale))
        return src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )


def landuse_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = ["#000000"] + [CLASS_COLORS[code] for code in CLASS_CODES]
    cmap = ListedColormap(colors)
    bounds = [0, 0.5] + [code + 0.5 for code in CLASS_CODES]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def add_landuse_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=CLASS_COLORS[code], label=CLASS_NAMES[code])
        for code in CLASS_CODES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def plot_validation_comparison(args: argparse.Namespace) -> Path:
    base = read_raster_for_plot(landuse_path(args, args.base_year), args.max_size)
    observed = read_raster_for_plot(landuse_path(args, args.target_year), args.max_size)
    predicted = read_raster_for_plot(predicted_path(args), args.max_size)

    valid = np.isin(observed, CLASS_CODES) & np.isin(predicted, CLASS_CODES)
    correctness = np.zeros(observed.shape, dtype=np.uint8)
    correctness[valid & (observed == predicted)] = 1
    correctness[valid & (observed != predicted)] = 2

    cmap, norm = landuse_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        (axes[0, 0], base, f"Observed {args.base_year}"),
        (axes[0, 1], observed, f"Observed {args.target_year}"),
        (axes[1, 0], predicted, f"Logistic-CA predicted {args.target_year}"),
    ]
    for ax, array, title in panels:
        ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    correctness_cmap = ListedColormap(["#000000", "#2ca25f", "#de2d26"])
    correctness_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], correctness_cmap.N)
    axes[1, 1].imshow(correctness, cmap=correctness_cmap, norm=correctness_norm, interpolation="nearest")
    axes[1, 1].set_title("Correct / Incorrect")
    axes[1, 1].axis("off")

    add_landuse_legend(fig)
    fig.suptitle("Logistic-CA Validation", fontsize=14)
    path = args.figures_dir / f"{args.city}_logistic_ca_predict_{args.target_year}_comparison.png"
    save_figure(fig, path, args.dpi)
    return path


def read_confusion_matrix(path: Path) -> np.ndarray:
    ensure_exists(path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([int(value) for value in row[1:]])
    return np.array(rows, dtype=np.int64)


def plot_confusion_matrix(args: argparse.Namespace) -> Path:
    matrix = read_confusion_matrix(confusion_path(args))
    row_totals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals > 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(percent, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_CODES)), [CLASS_NAMES[code] for code in CLASS_CODES], rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_CODES)), [CLASS_NAMES[code] for code in CLASS_CODES])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title("Logistic-CA Confusion Matrix (%)")

    for i in range(percent.shape[0]):
        for j in range(percent.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{percent[i, j]:.1f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percent by observed class")
    path = args.figures_dir / f"{args.city}_logistic_ca_predict_{args.target_year}_confusion_matrix.png"
    save_figure(fig, path, args.dpi)
    return path


def plot_future_projection(args: argparse.Namespace) -> Path:
    base = read_raster_for_plot(landuse_path(args, args.base_year), args.max_size)
    predicted = read_raster_for_plot(predicted_path(args), args.max_size)
    changed = np.zeros(base.shape, dtype=np.uint8)
    valid = np.isin(base, CLASS_CODES) & np.isin(predicted, CLASS_CODES)
    changed[valid & (base == predicted)] = 1
    changed[valid & (base != predicted)] = 2

    cmap, norm = landuse_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(base, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title(f"Observed {args.base_year}")
    axes[1].imshow(predicted, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(f"Logistic-CA predicted {args.target_year}")

    change_cmap = ListedColormap(["#000000", "#d9d9d9", "#e34a33"])
    change_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], change_cmap.N)
    axes[2].imshow(changed, cmap=change_cmap, norm=change_norm, interpolation="nearest")
    axes[2].set_title("Changed / Unchanged")
    for ax in axes:
        ax.axis("off")

    add_landuse_legend(fig)
    fig.suptitle("Logistic-CA Future Projection", fontsize=14)
    path = args.figures_dir / f"{args.city}_logistic_ca_predict_{args.target_year}_future_projection.png"
    save_figure(fig, path, args.dpi)
    return path


def main() -> None:
    args = resolve_paths(parse_args())
    ensure_exists(predicted_path(args))

    if landuse_path(args, args.target_year).exists() and confusion_path(args).exists():
        plot_validation_comparison(args)
        plot_confusion_matrix(args)
    else:
        plot_future_projection(args)

    print("Done.")


if __name__ == "__main__":
    main()
