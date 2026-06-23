"""Plot CLCD and Markov baseline results.

这个脚本只负责把已经生成的实验结果画成论文/报告可用的 PNG 图。
它不会重新计算 Markov 转移矩阵，也不会修改任何 GeoTIFF 数据。

默认绘制沈阳市 CLCD 原始 9 类结果：
- 2000-2025 六期土地利用图
- 2025 真实图、Markov 预测图、预测正确/错误图
- 2000-2025 各类面积变化折线图
- 2025 验证混淆矩阵热力图
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。

    这个函数和 markov_baseline.py 里的逻辑保持一致，避免用解释器
    绝对路径运行时 rasterio 找不到 GDAL_DATA 或 PROJ_LIB。
    """

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"

    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_geospatial_data_paths()

# 使用 Agg 后端可以在没有弹窗图形界面的情况下直接保存 PNG。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling


DEFAULT_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]
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

# 为 CLCD 9 类设置固定颜色。所有图都用同一套颜色，便于对比。
CLASS_COLORS = {
    1: "#f4e04d",  # Cropland
    2: "#267300",  # Forest
    3: "#6dbb75",  # Shrub
    4: "#9cc36b",  # Grassland
    5: "#4b9cd3",  # Water
    6: "#f7fbff",  # Snow/Ice
    7: "#bdbdbd",  # Barren
    8: "#d7191c",  # Impervious
    9: "#41b6c4",  # Wetland
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    常用运行方式：
        python scripts/plot_markov_results.py --city shenyang
    """

    parser = argparse.ArgumentParser(
        description="Plot CLCD original-class Markov baseline results."
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
        help="City name used in processed land-use and Markov result filenames.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="CLCD years to draw in the multi-year map.",
    )
    parser.add_argument(
        "--fit-from",
        type=int,
        default=2015,
        help="Start year used to estimate the Markov transition.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="Base year used to generate the Markov prediction.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2025,
        help="Target year to compare prediction with observed CLCD.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1800,
        help="Maximum raster width/height used for plotting previews.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录补齐输入/输出目录。"""

    root = args.project_root
    args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    args.markov_dir = root / "output" / "markov_baseline" / args.city
    args.tables_dir = root / "tables"
    if args.figures_dir is None:
        args.figures_dir = root / "figures" / "markov_baseline" / args.city
    return args


def landuse_path(args: argparse.Namespace, year: int) -> Path:
    """拼出某年份 CLCD 裁剪结果路径。"""

    return args.landuse_dir / f"{args.city}_clcd_v01_{year}_original.tif"


def prediction_path(args: argparse.Namespace) -> Path:
    """拼出指定验证窗口的 Markov 预测图路径。"""

    name = (
        f"{args.city}_markov_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}.tif"
    )
    return args.markov_dir / name


def confusion_path(args: argparse.Namespace) -> Path:
    """拼出指定验证窗口的混淆矩阵 CSV 路径。"""

    name = (
        f"{args.city}_markov_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_confusion_matrix.csv"
    )
    return args.tables_dir / name


def area_summary_path(args: argparse.Namespace) -> Path:
    """拼出 CLCD 原始类别面积统计表路径。"""

    return args.tables_dir / f"{args.city}_clcd_original_class_summary.csv"


def ensure_input_exists(path: Path) -> None:
    """在绘图前检查输入文件是否存在，报错时给出清楚路径。"""

    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")


def read_raster_for_plot(path: Path, max_size: int) -> np.ndarray:
    """读取并按最近邻重采样为适合绘图的数组。

    原始 30m 栅格像元很多，直接画全分辨率 PNG 会比较慢。
    这里只为了制图预览，所以按 max_size 限制最长边，保持类别值不变。
    """

    ensure_input_exists(path)
    with rasterio.open(path) as src:
        scale = max(src.width / max_size, src.height / max_size, 1.0)
        out_width = max(1, int(round(src.width / scale)))
        out_height = max(1, int(round(src.height / scale)))
        return src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )


def landuse_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """创建 CLCD 0-9 值对应的颜色映射。

    0 是 NoData，画成白色；1-9 是 CLCD 原始土地利用类型。
    """

    colors = ["#ffffff"] + [CLASS_COLORS[code] for code in CLASS_CODES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), cmap.N)
    return cmap, norm


def add_landuse_legend(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """给土地利用图添加统一图例。"""

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=CLASS_COLORS[code],
            markeredgecolor="#333333",
            markersize=8,
            label=f"{code} {CLASS_NAMES[code]}",
        )
        for code in CLASS_CODES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """保存图片并释放内存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def plot_landuse_series(args: argparse.Namespace) -> Path:
    """绘制 2000-2025 六期土地利用图。"""

    cmap, norm = landuse_cmap()
    fig, axes_grid = plt.subplots(2, 3, figsize=(12, 10))
    axes = list(axes_grid.ravel())

    for ax, year in zip(axes, args.years, strict=True):
        array = read_raster_for_plot(landuse_path(args, year), args.max_size)
        ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(str(year), fontsize=12)

    add_landuse_legend(fig, axes)
    fig.suptitle(f"{args.city.capitalize()} CLCD Original 9-Class Land Use", fontsize=15)
    fig.subplots_adjust(bottom=0.12, wspace=0.04, hspace=0.12)

    output = args.figures_dir / f"{args.city}_clcd_2000_2025_maps.png"
    save_figure(fig, output, args.dpi)
    return output


def plot_prediction_comparison(args: argparse.Namespace) -> Path:
    """绘制真实图、预测图和预测正确/错误图。"""

    actual = read_raster_for_plot(landuse_path(args, args.target_year), args.max_size)
    predicted = read_raster_for_plot(prediction_path(args), args.max_size)

    # 正确/错误图只评价实际和预测都为 1-9 类的位置，NoData 保持为 0。
    valid = np.isin(actual, CLASS_CODES) & np.isin(predicted, CLASS_CODES)
    agreement = np.zeros(actual.shape, dtype=np.uint8)
    agreement[valid & (actual == predicted)] = 1
    agreement[valid & (actual != predicted)] = 2

    landuse_colors, landuse_norm = landuse_cmap()
    agreement_cmap = ListedColormap(["#ffffff", "#2ca25f", "#de2d26"])
    agreement_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], agreement_cmap.N)

    # 三联图标题较多，给顶部留出更多空间，避免总标题和子图标题重叠。
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2))
    axes[0].imshow(actual, cmap=landuse_colors, norm=landuse_norm, interpolation="nearest")
    axes[0].set_title(f"Observed {args.target_year}", fontsize=11)
    axes[1].imshow(
        predicted, cmap=landuse_colors, norm=landuse_norm, interpolation="nearest"
    )
    axes[1].set_title(
        f"Markov Prediction {args.target_year}\nfit {args.fit_from}->{args.base_year}",
        fontsize=11,
    )
    axes[2].imshow(
        agreement, cmap=agreement_cmap, norm=agreement_norm, interpolation="nearest"
    )
    axes[2].set_title("Correct / Error", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#2ca25f",
            markeredgecolor="#333333",
            markersize=8,
            label="Correct",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#de2d26",
            markeredgecolor="#333333",
            markersize=8,
            label="Error",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{args.city.capitalize()} Markov Baseline Validation",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.08)

    output = (
        args.figures_dir
        / f"{args.city}_markov_predict_{args.target_year}_comparison.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def read_area_summary(path: Path) -> dict[int, dict[int, float]]:
    """读取 crop_clcd.py 输出的面积统计表。

    返回结构：
        {year: {class_code: area_km2}}
    """

    ensure_input_exists(path)
    areas: dict[int, dict[int, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            class_code = int(row["class_code"])
            area = float(row["area_km2"])
            areas.setdefault(year, {})[class_code] = area
    return areas


def plot_area_trends(args: argparse.Namespace) -> Path:
    """绘制 2000-2025 各类面积变化折线图。

    沈阳市耕地面积远大于其他类别。为了让小类别变化也能看清，
    图中设置上下两个面板：上方面板展示所有非零类别，下方面板
    去掉耕地后放大显示其余类别。
    """

    areas = read_area_summary(area_summary_path(args))
    years = sorted(year for year in args.years if year in areas)

    nonzero_codes = []
    for class_code in CLASS_CODES:
        values = [areas.get(year, {}).get(class_code, 0.0) for year in years]
        if max(values, default=0.0) > 0:
            nonzero_codes.append(class_code)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, codes, title in [
        (axes[0], nonzero_codes, "All nonzero classes"),
        (
            axes[1],
            [code for code in nonzero_codes if code != 1],
            "Excluding Cropland",
        ),
    ]:
        for class_code in codes:
            values = [areas.get(year, {}).get(class_code, 0.0) for year in years]
            ax.plot(
                years,
                values,
                marker="o",
                linewidth=1.8,
                color=CLASS_COLORS[class_code],
                label=f"{class_code} {CLASS_NAMES[class_code]}",
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Area (km2)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[1].set_xlabel("Year")
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
    )
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(f"{args.city.capitalize()} CLCD Class Area Trends", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.82, 0.96))

    output = args.figures_dir / f"{args.city}_clcd_area_trends.png"
    save_figure(fig, output, args.dpi)
    return output


def read_confusion_matrix(path: Path) -> np.ndarray:
    """读取 markov_baseline.py 输出的混淆矩阵 CSV。"""

    ensure_input_exists(path)
    matrix = np.zeros((len(CLASS_CODES), len(CLASS_CODES)), dtype=np.int64)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            for col_index, class_code in enumerate(CLASS_CODES):
                matrix[row_index, col_index] = int(row[f"predicted_{class_code}"])
    return matrix


def plot_confusion_matrix(args: argparse.Namespace) -> Path:
    """绘制目标年份验证的混淆矩阵热力图。

    为了让小类别也能看见，这里显示的是按真实类别归一化后的百分比。
    """

    matrix = read_confusion_matrix(confusion_path(args))
    row_totals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=row_totals > 0,
    )
    percent_points = percent * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent_points, cmap="YlGnBu", vmin=0.0, vmax=100.0)
    ax.set_title(
        f"Confusion Matrix Percent\nfit {args.fit_from}->{args.base_year}, "
        f"predict {args.target_year}",
        fontsize=13,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_xticks(range(len(CLASS_CODES)))
    ax.set_yticks(range(len(CLASS_CODES)))
    ax.set_xticklabels([str(code) for code in CLASS_CODES])
    ax.set_yticklabels([str(code) for code in CLASS_CODES])

    # 只标注较大的比例，避免图面被数字挤满。
    for row in range(percent_points.shape[0]):
        for col in range(percent_points.shape[1]):
            if percent_points[row, col] >= 1.0:
                ax.text(
                    col,
                    row,
                    f"{percent_points[row, col]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percent by observed class (%)")
    fig.tight_layout()

    output = (
        args.figures_dir
        / f"{args.city}_markov_predict_{args.target_year}_confusion_matrix.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def main() -> None:
    """脚本入口：生成所有 Markov 基线结果图。"""

    args = resolve_paths(parse_args())
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"City: {args.city}")
    print(f"Land-use rasters: {args.landuse_dir}")
    print(f"Markov rasters: {args.markov_dir}")
    print(f"Tables: {args.tables_dir}")
    print(f"Figures: {args.figures_dir}")
    print("")

    outputs = [
        plot_landuse_series(args),
        plot_prediction_comparison(args),
        plot_area_trends(args),
        plot_confusion_matrix(args),
    ]

    print("")
    print("Done. Figure outputs:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
