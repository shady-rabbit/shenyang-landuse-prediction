"""Dedicated plotting script for Logistic-CA 2030 prediction.

Run this after `predict_2030_logistic_ca.py` finishes.

This script is intentionally independent from `plot_logistic_ca_results.py`.
It only handles the 2025 -> 2030 future projection case:

- input base raster:
  data/processed/landuse/shenyang/shenyang_clcd_v01_2025_original.tif
- input prediction raster:
  output/logistic_ca/shenyang/shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5.tif
- optional input area table:
  tables/shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5_area_projection.csv

Outputs are written to:
figures/logistic_ca/shenyang/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
    parser = argparse.ArgumentParser(description="Plot Logistic-CA 2030 prediction.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/Logistic-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument("--max-size", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def output_stem(city: str) -> str:
    return f"{city}_logistic_ca_fit_2020_2025_predict_2030_n5_i5"


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def read_raster_preview(path: Path, max_size: int) -> np.ndarray:
    """Read a downsampled raster preview for plotting."""

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
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CLASS_COLORS[code],
            label=CLASS_NAMES[code],
        )
        for code in CLASS_CODES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}", flush=True)


def plot_future_projection(
    base_path: Path,
    predicted_path: Path,
    figures_dir: Path,
    city: str,
    max_size: int,
    dpi: int,
) -> Path:
    base = read_raster_preview(base_path, max_size)
    predicted = read_raster_preview(predicted_path, max_size)

    valid = np.isin(base, CLASS_CODES) & np.isin(predicted, CLASS_CODES)
    changed = np.zeros(base.shape, dtype=np.uint8)
    changed[valid & (base == predicted)] = 1
    changed[valid & (base != predicted)] = 2

    landuse_map, landuse_norm = landuse_cmap()
    change_map = ListedColormap(["#000000", "#d9d9d9", "#e34a33"])
    change_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], change_map.N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    axes[0].imshow(base, cmap=landuse_map, norm=landuse_norm, interpolation="nearest")
    axes[0].set_title("Observed 2025")
    axes[1].imshow(predicted, cmap=landuse_map, norm=landuse_norm, interpolation="nearest")
    axes[1].set_title("Logistic-CA Predicted 2030")
    axes[2].imshow(changed, cmap=change_map, norm=change_norm, interpolation="nearest")
    axes[2].set_title("Changed / Unchanged")

    for ax in axes:
        ax.axis("off")

    add_landuse_legend(fig)
    fig.suptitle("Logistic-CA 2030 Future Projection", fontsize=15)
    path = figures_dir / f"{city}_logistic_ca_predict_2030_future_projection.png"
    save_figure(fig, path, dpi)
    return path


def read_area_rows(area_csv: Path) -> list[dict]:
    ensure_exists(area_csv)
    with area_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_area_change(area_csv: Path, figures_dir: Path, city: str, dpi: int) -> Path:
    rows = read_area_rows(area_csv)
    labels = [row["class_name"] for row in rows]
    base = np.array([float(row["base_area_km2"]) for row in rows], dtype=float)
    predicted = np.array([float(row["predicted_area_km2"]) for row in rows], dtype=float)
    change = np.array([float(row["change_area_km2"]) for row in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].bar(x - width / 2, base, width, label="2025")
    axes[0].bar(x + width / 2, predicted, width, label="2030 predicted")
    axes[0].set_ylabel("Area (km2)")
    axes[0].set_title("Land-Use Area: 2025 vs Predicted 2030")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")

    colors = ["#de2d26" if value < 0 else "#2ca25f" for value in change]
    axes[1].bar(x, change, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Change (km2)")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_title("Predicted Area Change")

    fig.tight_layout()
    path = figures_dir / f"{city}_logistic_ca_predict_2030_area_change.png"
    save_figure(fig, path, dpi)
    return path


def main() -> None:
    args = parse_args()
    stem = output_stem(args.city)
    root = args.project_root

    base_path = (
        root
        / "data"
        / "processed"
        / "landuse"
        / args.city
        / f"{args.city}_clcd_v01_2025_original.tif"
    )
    predicted_path = root / "output" / "logistic_ca" / args.city / f"{stem}.tif"
    area_csv = root / "tables" / f"{stem}_area_projection.csv"
    figures_dir = root / "figures" / "logistic_ca" / args.city

    print("Logistic-CA 2030 plotting", flush=True)
    print(f"Base raster: {base_path}", flush=True)
    print(f"Prediction raster: {predicted_path}", flush=True)
    print(f"Area table: {area_csv}", flush=True)
    print(f"Figures dir: {figures_dir}", flush=True)

    plot_future_projection(
        base_path=base_path,
        predicted_path=predicted_path,
        figures_dir=figures_dir,
        city=args.city,
        max_size=args.max_size,
        dpi=args.dpi,
    )

    if area_csv.exists():
        plot_area_change(area_csv=area_csv, figures_dir=figures_dir, city=args.city, dpi=args.dpi)
    else:
        print(f"Skip area chart, missing table: {area_csv}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
