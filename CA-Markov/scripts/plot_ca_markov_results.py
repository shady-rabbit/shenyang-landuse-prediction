"""Plot CA-Markov validation results.

这个脚本用于绘制 CA-Markov 实验结果。
它不会重新运行 CA-Markov，只读取已经生成的 GeoTIFF 和 CSV。

默认绘制适宜性因子 CA-Markov 的 2025 回测验证图：
- 2015->2020 估计转移概率，2020 作为基期，预测 2025
- 2025 真实图、Markov 基线预测图、CA-Markov 预测图、CA 正确/错误图
- Markov 与 CA-Markov 正确性对比图
- CA-Markov 混淆矩阵百分比热力图

如果 target_year 有真实 CLCD 栅格，则自动切换为验证模式，绘制：
- 目标年真实图、Markov 基线预测图、CA-Markov 预测图、CA 正确/错误图
- Markov 与 CA-Markov 正确性对比图
- CA-Markov 混淆矩阵百分比热力图

如果 target_year 是 2030 这类没有真实 CLCD 的未来年份，则自动切换为
未来预测模式，绘制：
- 基期图、目标年 CA-Markov 预测图、变化/未变化图
- 基期面积与目标年预测面积对比图
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。"""

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"

    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_geospatial_data_paths()

# 使用非交互后端，运行后直接保存 PNG。
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
    2: "#267300",
    3: "#6dbb75",
    4: "#9cc36b",
    5: "#4b9cd3",
    6: "#f7fbff",
    7: "#bdbdbd",
    8: "#d7191c",
    9: "#41b6c4",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    常用运行方式：
        python scripts/plot_ca_markov_results.py --city shenyang

    默认绘制适宜性因子版 2025 年回测验证结果。
    如果要绘制 2030 年未来预测结果，请显式传入：
        --fit-from 2020 --base-year 2025 --target-year 2030
    """

    parser = argparse.ArgumentParser(description="Plot CA-Markov validation figures.")
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
        "--model",
        choices=["ca_markov", "ca_markov_suitability"],
        default="ca_markov_suitability",
        help="Which CA-Markov output set to plot.",
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
        help="Target year. If its observed raster exists, validation mode is used.",
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=5,
        help="CA neighborhood size used by ca_markov.py.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="CA iteration count used by ca_markov.py.",
    )
    parser.add_argument(
        "--neighbor-weight",
        type=float,
        default=0.65,
        help="Neighbor weight used by ca_markov_suitability.py.",
    )
    parser.add_argument(
        "--suitability-weight",
        type=float,
        default=0.35,
        help="Suitability weight used by ca_markov_suitability.py.",
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
    """根据项目根目录补齐输入和输出路径。"""

    root = args.project_root
    args.landuse_dir = root / "data" / "processed" / "landuse" / args.city
    args.markov_dir = root / "output" / "markov_baseline" / args.city
    args.ca_dir = root / "output" / args.model / args.city
    args.tables_dir = root / "tables"
    if args.figures_dir is None:
        args.figures_dir = root / "figures" / args.model / args.city
    return args


def output_stem(args: argparse.Namespace) -> str:
    """生成和对应 CA-Markov 脚本一致的输出文件名前缀。"""

    stem = (
        f"{args.city}_{args.model}_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}_n{args.neighborhood_size}_i{args.iterations}"
    )
    if args.model == "ca_markov_suitability":
        stem += weight_suffix(args)
    return stem


def format_weight_for_filename(value: float) -> str:
    """把权重转换成和模型脚本一致的文件名片段。"""

    percent = value * 100
    rounded = round(percent)
    if abs(percent - rounded) < 1e-6:
        return f"{int(rounded):03d}"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def weight_suffix(args: argparse.Namespace) -> str:
    """生成适宜性因子权重后缀。"""

    return (
        f"_nw{format_weight_for_filename(args.neighbor_weight)}"
        f"_sw{format_weight_for_filename(args.suitability_weight)}"
    )


def figure_suffix(args: argparse.Namespace) -> str:
    """生成图件文件名后缀，避免不同权重图件互相覆盖。"""

    suffix = f"_n{args.neighborhood_size}_i{args.iterations}"
    if args.model == "ca_markov_suitability":
        suffix += weight_suffix(args)
    return suffix


def model_label(args: argparse.Namespace) -> str:
    """返回适合显示在图题中的模型名称。"""

    if args.model == "ca_markov_suitability":
        return "CA-Markov Suitability"
    return "CA-Markov"


def model_short_slug(args: argparse.Namespace) -> str:
    """返回适合输出文件名的短模型标识。"""

    if args.model == "ca_markov":
        return "ca"
    return args.model


def observed_path(args: argparse.Namespace) -> Path:
    """目标年份真实 CLCD 图路径。"""

    return args.landuse_dir / f"{args.city}_clcd_v01_{args.target_year}_original.tif"


def base_path(args: argparse.Namespace) -> Path:
    """基期年份真实 CLCD 图路径。"""

    return args.landuse_dir / f"{args.city}_clcd_v01_{args.base_year}_original.tif"


def markov_path(args: argparse.Namespace) -> Path:
    """Markov 基线预测图路径。"""

    name = (
        f"{args.city}_markov_fit_{args.fit_from}_{args.base_year}"
        f"_predict_{args.target_year}.tif"
    )
    return args.markov_dir / name


def ca_markov_path(args: argparse.Namespace) -> Path:
    """CA-Markov 预测图路径。"""

    return args.ca_dir / f"{output_stem(args)}.tif"


def ca_confusion_path(args: argparse.Namespace) -> Path:
    """CA-Markov 混淆矩阵 CSV 路径。"""

    return args.tables_dir / f"{output_stem(args)}_confusion_matrix.csv"


def ca_area_projection_path(args: argparse.Namespace) -> Path:
    """CA-Markov 面积预测表路径。"""

    return args.tables_dir / f"{output_stem(args)}_area_projection.csv"


def ensure_input_exists(path: Path) -> None:
    """检查输入文件是否存在。"""

    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")


def model_run_command(args: argparse.Namespace) -> str:
    """生成当前绘图参数对应的模型运行命令，便于缺文件时提示用户。"""

    script_name = f"{args.model}.py"
    script_path = args.project_root / "scripts" / script_name
    command = (
        f"{sys.executable} {script_path} "
        f"--fit-from {args.fit_from} "
        f"--base-year {args.base_year} "
        f"--target-year {args.target_year} "
        f"--neighborhood-size {args.neighborhood_size} "
        f"--iterations {args.iterations}"
    )
    if args.model == "ca_markov_suitability":
        command += (
            f" --neighbor-weight {args.neighbor_weight} "
            f"--suitability-weight {args.suitability_weight}"
        )
    return command


def check_required_inputs(args: argparse.Namespace, has_observed_target: bool) -> None:
    """在正式绘图前统一检查输入，给出更清楚的缺文件提示。"""

    required_paths = [base_path(args), ca_markov_path(args)]
    if has_observed_target:
        required_paths.extend(
            [
                observed_path(args),
                markov_path(args),
                ca_confusion_path(args),
            ]
        )
    else:
        required_paths.append(ca_area_projection_path(args))

    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "Some required inputs for plotting were not found:\n"
        f"{missing_text}\n\n"
        "Run the matching model script first, for example:\n"
        f"  {model_run_command(args)}"
    )


def read_raster_for_plot(path: Path, max_size: int) -> np.ndarray:
    """读取栅格并按最近邻重采样为适合绘图的尺寸。"""

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
    """创建 CLCD 0-9 类固定颜色映射。"""

    colors = ["#ffffff"] + [CLASS_COLORS[code] for code in CLASS_CODES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), cmap.N)
    return cmap, norm


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """保存图片并释放内存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def valid_class_mask(*arrays: np.ndarray) -> np.ndarray:
    """返回所有数组中都属于 CLCD 1-9 类的位置。"""

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isin(array, CLASS_CODES)
    return mask


def plot_ca_comparison(args: argparse.Namespace) -> Path:
    """绘制真实图、Markov 基线、CA-Markov 和 CA 正确/错误图。"""

    label = model_label(args)
    observed = read_raster_for_plot(observed_path(args), args.max_size)
    markov = read_raster_for_plot(markov_path(args), args.max_size)
    ca = read_raster_for_plot(ca_markov_path(args), args.max_size)

    valid = valid_class_mask(observed, ca)
    ca_agreement = np.zeros(observed.shape, dtype=np.uint8)
    ca_agreement[valid & (observed == ca)] = 1
    ca_agreement[valid & (observed != ca)] = 2

    landuse_colors, landuse_norm = landuse_cmap()
    agreement_cmap = ListedColormap(["#ffffff", "#2ca25f", "#de2d26"])
    agreement_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], agreement_cmap.N)

    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2))
    panels = [
        (observed, f"Observed {args.target_year}", landuse_colors, landuse_norm),
        (markov, "Markov Baseline", landuse_colors, landuse_norm),
        (
            ca,
            f"{label}\nn={args.neighborhood_size}, i={args.iterations}",
            landuse_colors,
            landuse_norm,
        ),
        (ca_agreement, f"{label} Correct / Error", agreement_cmap, agreement_norm),
    ]
    for ax, (array, title, cmap, norm) in zip(axes, panels, strict=True):
        ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=11)
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
        f"{args.city.capitalize()} {label} Validation",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.06)

    output = (
        args.figures_dir
        / f"{args.city}_{args.model}_predict_{args.target_year}_comparison"
        f"{figure_suffix(args)}.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def plot_markov_vs_ca_correctness(args: argparse.Namespace) -> Path:
    """绘制 Markov 和 CA-Markov 正确性对比图。

    图中分为四类：
    - Both correct：两个模型都预测正确；
    - Markov only：只有 Markov 基线正确；
    - CA only：只有 CA-Markov 正确；
    - Both error：两个模型都预测错误。
    """

    label = model_label(args)
    observed = read_raster_for_plot(observed_path(args), args.max_size)
    markov = read_raster_for_plot(markov_path(args), args.max_size)
    ca = read_raster_for_plot(ca_markov_path(args), args.max_size)

    valid = valid_class_mask(observed, markov, ca)
    markov_correct = valid & (observed == markov)
    ca_correct = valid & (observed == ca)

    comparison = np.zeros(observed.shape, dtype=np.uint8)
    comparison[markov_correct & ca_correct] = 1
    comparison[markov_correct & ~ca_correct] = 2
    comparison[~markov_correct & ca_correct] = 3
    comparison[valid & ~markov_correct & ~ca_correct] = 4

    cmap = ListedColormap(["#ffffff", "#2ca25f", "#fdae61", "#3182bd", "#de2d26"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(comparison, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"Markov vs {label} Correctness", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        ("Both correct", "#2ca25f"),
        ("Markov only", "#fdae61"),
        (f"{label} only", "#3182bd"),
        ("Both error", "#de2d26"),
    ]
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#333333",
            markersize=8,
            label=label,
        )
        for label, color in handles
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.12)

    output = (
        args.figures_dir
        / f"{args.city}_markov_vs_{model_short_slug(args)}_correctness_{args.target_year}"
        f"{figure_suffix(args)}.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def read_confusion_matrix(path: Path) -> np.ndarray:
    """读取 CA-Markov 混淆矩阵 CSV。"""

    ensure_input_exists(path)
    matrix = np.zeros((len(CLASS_CODES), len(CLASS_CODES)), dtype=np.int64)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            for col_index, class_code in enumerate(CLASS_CODES):
                matrix[row_index, col_index] = int(row[f"predicted_{class_code}"])
    return matrix


def plot_ca_confusion_matrix(args: argparse.Namespace) -> Path:
    """绘制 CA-Markov 混淆矩阵百分比热力图。"""

    label = model_label(args)
    matrix = read_confusion_matrix(ca_confusion_path(args))
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
        f"{label} Confusion Matrix Percent\nfit {args.fit_from}->{args.base_year}, "
        f"predict {args.target_year}",
        fontsize=13,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_xticks(range(len(CLASS_CODES)))
    ax.set_yticks(range(len(CLASS_CODES)))
    ax.set_xticklabels([str(code) for code in CLASS_CODES])
    ax.set_yticklabels([str(code) for code in CLASS_CODES])

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
        / f"{args.city}_{args.model}_predict_{args.target_year}_confusion_matrix"
        f"{figure_suffix(args)}.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def plot_future_projection(args: argparse.Namespace) -> Path:
    """绘制未来预测图。

    2030 没有真实图，不能做正确/错误评价，因此这里重点展示：
    1. 2025 基期土地利用；
    2. 2030 CA-Markov 预测土地利用；
    3. 2025->2030 的变化/未变化区域。
    """

    label = model_label(args)
    base = read_raster_for_plot(base_path(args), args.max_size)
    ca = read_raster_for_plot(ca_markov_path(args), args.max_size)

    valid = valid_class_mask(base, ca)
    change = np.zeros(base.shape, dtype=np.uint8)
    change[valid & (base == ca)] = 1
    change[valid & (base != ca)] = 2

    landuse_colors, landuse_norm = landuse_cmap()
    change_cmap = ListedColormap(["#ffffff", "#bdbdbd", "#de2d26"])
    change_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], change_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2))
    panels = [
        (base, f"Observed {args.base_year}", landuse_colors, landuse_norm),
        (
            ca,
            f"{label} Prediction {args.target_year}\n"
            f"fit {args.fit_from}->{args.base_year}",
            landuse_colors,
            landuse_norm,
        ),
        (change, f"Change {args.base_year}->{args.target_year}", change_cmap, change_norm),
    ]
    for ax, (array, title, cmap, norm) in zip(axes, panels, strict=True):
        ax.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#bdbdbd",
            markeredgecolor="#333333",
            markersize=8,
            label="No change",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#de2d26",
            markeredgecolor="#333333",
            markersize=8,
            label="Changed",
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
        f"{args.city.capitalize()} {label} Future Projection",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.06)

    output = (
        args.figures_dir
        / f"{args.city}_{args.model}_predict_{args.target_year}_future_projection"
        f"{figure_suffix(args)}.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def read_area_projection(path: Path) -> list[dict]:
    """读取 CA-Markov 面积预测表。"""

    ensure_input_exists(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_future_area_projection(args: argparse.Namespace) -> Path:
    """绘制基期面积与未来预测面积对比图。

    耕地面积远大于其他类型，单面板柱状图会把小类别压扁。
    因此这里使用上下两个面板：上方显示所有非零类别，下方排除耕地，
    便于观察建设用地、林地、水域等类别的变化。
    """

    label = model_label(args)
    rows = read_area_projection(ca_area_projection_path(args))
    records = []

    for row in rows:
        class_code = int(row["class_code"])
        base_area = int(row["base_pixels"]) * 30 * 30 / 1_000_000
        predicted_area = float(row["ca_predicted_area_km2"])
        if base_area <= 0 and predicted_area <= 0:
            continue
        records.append(
            {
                "class_code": class_code,
                "label": f"{class_code}\n{CLASS_NAMES[class_code]}",
                "base_area": base_area,
                "predicted_area": predicted_area,
                "color": CLASS_COLORS[class_code],
            }
        )

    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    for ax, panel_records, title in [
        (axes[0], records, "All nonzero classes"),
        (
            axes[1],
            [record for record in records if record["class_code"] != 1],
            "Excluding Cropland",
        ),
    ]:
        labels = [record["label"] for record in panel_records]
        base_values = [record["base_area"] for record in panel_records]
        predicted_values = [record["predicted_area"] for record in panel_records]
        colors = [record["color"] for record in panel_records]
        x = np.arange(len(labels))

        ax.bar(
            x - width / 2,
            base_values,
            width,
            color="#bdbdbd",
            label=str(args.base_year),
        )
        ax.bar(
            x + width / 2,
            predicted_values,
            width,
            color=colors,
            label=f"Predicted {args.target_year}",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Area (km2)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False)

    fig.suptitle(
        f"{args.city.capitalize()} {label} Area Projection "
        f"{args.base_year}->{args.target_year}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = (
        args.figures_dir
        / f"{args.city}_{args.model}_area_projection_{args.target_year}"
        f"{figure_suffix(args)}.png"
    )
    save_figure(fig, output, args.dpi)
    return output


def main() -> None:
    """脚本入口：绘制 CA-Markov 结果图。"""

    args = resolve_paths(parse_args())
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    has_observed_target = observed_path(args).exists()
    label = model_label(args)

    print(f"City: {args.city}")
    print(f"Model: {args.model}")
    print(f"Mode: {'validation' if has_observed_target else 'future projection'}")
    if has_observed_target:
        print(f"Observed target raster: {observed_path(args)}")
    else:
        print(f"Observed target raster: not found ({observed_path(args)})")
    print(f"Base raster: {base_path(args)}")
    if has_observed_target:
        print(f"Markov raster: {markov_path(args)}")
    print(f"{label} raster: {ca_markov_path(args)}")
    print(f"Tables: {args.tables_dir}")
    print(f"Figures: {args.figures_dir}")
    print("")

    check_required_inputs(args, has_observed_target)

    if has_observed_target:
        outputs = [
            plot_ca_comparison(args),
            plot_markov_vs_ca_correctness(args),
            plot_ca_confusion_matrix(args),
        ]
    else:
        outputs = [
            plot_future_projection(args),
            plot_future_area_projection(args),
        ]

    print("")
    print("Done. Figure outputs:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
