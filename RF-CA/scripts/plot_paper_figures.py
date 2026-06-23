"""生成论文可用的模型对比图。

当前 `ca` 环境中的 matplotlib 保存图片会静默失败，因此本脚本使用
Pillow 直接绘制 PNG/PDF 图件，不依赖 seaborn 或 matplotlib。

输出图件：
1. 2025 年验证 OA/Kappa 对比；
2. 2025 年各类别 F1 对比；
3. 2025 年真实图与各模型预测图对比；
4. 2030 年不同 CA 模型预测图对比；
5. RF-CA 2030 面积变化；
6. RF 随机森林特征重要性；
7. RF-CA 2025 验证混淆矩阵。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def configure_geospatial_data_paths() -> None:
    """补齐 Conda 环境中的 GDAL/PROJ 数据路径。"""

    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"
    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


def configure_console_encoding() -> None:
    """尽量让 Windows 终端正确显示中文提示。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


configure_console_encoding()
configure_geospatial_data_paths()

import rasterio
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
CLASS_SHORT = {
    1: "Crop.",
    2: "Forest",
    3: "Shrub",
    4: "Grass.",
    5: "Water",
    6: "Snow",
    7: "Barren",
    8: "Imperv.",
    9: "Wetland",
}
CLASS_COLORS = {
    0: (255, 255, 255),
    1: (244, 224, 77),
    2: (38, 115, 0),
    3: (109, 187, 117),
    4: (156, 195, 107),
    5: (75, 156, 211),
    6: (247, 251, 255),
    7: (189, 189, 189),
    8: (215, 25, 28),
    9: (65, 182, 196),
}
MODEL_ORDER = [
    "Markov baseline",
    "CA-Markov",
    "CA-Markov suitability",
    "Logistic-CA",
    "RF-CA",
]
MODEL_COLORS = {
    "Markov baseline": (140, 140, 140),
    "CA-Markov": (76, 120, 168),
    "CA-Markov suitability": (114, 183, 178),
    "Logistic-CA": (245, 133, 24),
    "RF-CA": (84, 162, 75),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Plot paper-ready RF-CA figures.")
    parser.add_argument("--project-root", type=Path, default=Path("E:/RF-CA"))
    parser.add_argument("--ca-markov-root", type=Path, default=Path("E:/CA-Markov"))
    parser.add_argument("--logistic-root", type=Path, default=Path("E:/Logistic-CA"))
    parser.add_argument("--city", default="shenyang")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="输出格式；PDF 为嵌入栅格图的 PDF。",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--max-map-width",
        type=int,
        default=900,
        help="单幅地图预览最大宽度，用于降采样控制图片体量。",
    )
    parser.add_argument(
        "--rf-ca-stem",
        default="shenyang_rf_ca_fit_2015_2020_predict_2025_driver2020_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42",
    )
    parser.add_argument(
        "--rf-ca-2030-stem",
        default="shenyang_rf_ca_fit_2020_2025_predict_2030_driver2025_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    """返回图件输出目录。"""

    return args.output_dir or args.project_root / "figures" / "paper" / args.city


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    """集中定义绘图输入文件。"""

    rf_tables = args.project_root / "tables"
    rf_output = args.project_root / "output" / "rf_ca" / args.city
    ca_ref = args.project_root / "references" / "ca_markov_tables"
    logistic_tables = args.logistic_root / "tables"
    logistic_output = args.logistic_root / "output" / "logistic_ca" / args.city
    return {
        "comparison": rf_tables
        / "model_comparison"
        / "shenyang_2025_validation_model_comparison.csv",
        "observed_2025": args.project_root
        / "data"
        / "processed"
        / "landuse"
        / args.city
        / "shenyang_clcd_v01_2025_original.tif",
        "markov_2025": args.ca_markov_root
        / "output"
        / "markov_baseline"
        / args.city
        / "shenyang_markov_fit_2015_2020_predict_2025.tif",
        "ca_markov_2025": args.ca_markov_root
        / "output"
        / "ca_markov"
        / args.city
        / "shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5.tif",
        "logistic_2025": logistic_output
        / "shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5.tif",
        "rf_ca_2025": rf_output / f"{args.rf_ca_stem}.tif",
        "ca_markov_2030": args.ca_markov_root
        / "output"
        / "ca_markov"
        / args.city
        / "shenyang_ca_markov_fit_2020_2025_predict_2030_n5_i5.tif",
        "logistic_2030": logistic_output
        / "shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5.tif",
        "rf_ca_2030": rf_output / f"{args.rf_ca_2030_stem}.tif",
        "markov_f1": ca_ref
        / "shenyang_markov_fit_2015_2020_predict_2025_per_class_accuracy.csv",
        "ca_markov_f1": ca_ref
        / "shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
        "logistic_f1": logistic_tables
        / "shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
        "rf_ca_f1": rf_tables
        / "rf_ca"
        / args.city
        / f"{args.rf_ca_stem}_per_class_accuracy.csv",
        "rf_feature_importance": rf_tables
        / "random_forest"
        / args.city
        / "shenyang_rf_fit_2015_2020_driver2020_n5_sampseed42_rf300_depthnone_leaf1_wtransition_trainseed42_feature_importance.csv",
        "rf_ca_confusion": rf_tables
        / "rf_ca"
        / args.city
        / f"{args.rf_ca_stem}_confusion_matrix.csv",
        "rf_ca_2030_area": rf_tables
        / "rf_ca"
        / args.city
        / f"{args.rf_ca_2030_stem}_area_projection.csv",
    }


def check_paths(paths: dict[str, Path]) -> list[str]:
    """返回缺失文件说明。"""

    return [f"{key}: {value}" for key, value in paths.items() if not value.exists()]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """读取常见 Windows 字体。"""

    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(34, bold=True)
FONT_SUBTITLE = font(25, bold=True)
FONT_AXIS = font(22)
FONT_SMALL = font(18)
FONT_TINY = font(15)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    """计算文字宽高。"""

    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill=(30, 30, 30)) -> None:
    """居中绘制文字。"""

    w, h = text_size(draw, text, fnt)
    draw.text((xy[0] - w // 2, xy[1] - h // 2), text, font=fnt, fill=fill)


def save_image(img: Image.Image, out_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    """保存图片。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "pdf":
            img.convert("RGB").save(path, resolution=dpi)
        else:
            img.save(path, dpi=(dpi, dpi))
        print(f"Wrote: {path}")


def hex_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_ticks: list[float], y_min: float, y_max: float) -> None:
    """绘制坐标轴和水平网格。"""

    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill=(60, 60, 60), width=2)
    draw.line((left, top, left, bottom), fill=(60, 60, 60), width=2)
    for tick in y_ticks:
        y = bottom - int((tick - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(220, 220, 220), width=1)
        draw.text((left - 72, y - 10), f"{tick:.2f}", font=FONT_TINY, fill=(70, 70, 70))


def plot_accuracy(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 OA/Kappa 对比。"""

    df = pd.read_csv(paths["comparison"])
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model")
    img = Image.new("RGB", (1900, 760), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (950, 40), "Validation Accuracy for 2025 Prediction", FONT_TITLE)

    panels = [
        ("Overall Accuracy", "overall_accuracy", (110, 115, 900, 610)),
        ("Kappa", "kappa", (1010, 115, 1800, 610)),
    ]
    for title, metric, box in panels:
        left, top, right, bottom = box
        values = df[metric].astype(float).to_numpy()
        y_min = max(values.min() - 0.012, 0.84)
        y_max = min(values.max() + 0.006, 1.0)
        ticks = np.linspace(round(y_min, 2), round(y_max, 2), 5).tolist()
        draw_centered(draw, ((left + right) // 2, top - 36), title, FONT_SUBTITLE)
        hex_grid(draw, box, ticks, y_min, y_max)

        n = len(df)
        slot = (right - left) / n
        bar_w = int(slot * 0.55)
        for i, row in enumerate(df.itertuples(index=False)):
            value = float(getattr(row, metric))
            model = str(row.model)
            x0 = int(left + i * slot + (slot - bar_w) / 2)
            x1 = x0 + bar_w
            y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
            draw.rectangle((x0, y, x1, bottom), fill=MODEL_COLORS[model], outline=(50, 50, 50))
            draw_centered(draw, ((x0 + x1) // 2, y - 20), f"{value:.4f}", FONT_TINY)
            label = model.replace(" ", "\n")
            for j, part in enumerate(label.split("\n")):
                draw_centered(draw, ((x0 + x1) // 2, bottom + 22 + j * 18), part, FONT_TINY)

    save_image(img, out_dir, "fig01_2025_model_accuracy_comparison", args.formats, args.dpi)


def read_f1(path: Path, model: str) -> pd.DataFrame:
    """读取 F1 表。"""

    df = pd.read_csv(path)
    out = df[["class_code", "f1_score"]].copy()
    out["class_code"] = out["class_code"].astype(int)
    out["f1_score"] = out["f1_score"].astype(float)
    out["model"] = model
    return out


def plot_f1(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制各类别 F1 对比。"""

    frames = [
        read_f1(paths["markov_f1"], "Markov baseline"),
        read_f1(paths["ca_markov_f1"], "CA-Markov"),
        read_f1(paths["logistic_f1"], "Logistic-CA"),
        read_f1(paths["rf_ca_f1"], "RF-CA"),
    ]
    df = pd.concat(frames, ignore_index=True)
    active = [1, 2, 3, 4, 5, 7, 8]
    models = ["Markov baseline", "CA-Markov", "Logistic-CA", "RF-CA"]

    img = Image.new("RGB", (1900, 820), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (950, 45), "Per-Class F1 Scores for 2025 Prediction", FONT_TITLE)
    box = (130, 120, 1780, 620)
    left, top, right, bottom = box
    hex_grid(draw, box, [0, 0.25, 0.5, 0.75, 1.0], 0, 1)
    draw.text((20, 320), "F1 Score", font=FONT_AXIS, fill=(30, 30, 30))
    group_w = (right - left) / len(active)
    bar_w = int(group_w * 0.18)
    for group_index, code in enumerate(active):
        center = left + group_index * group_w + group_w / 2
        draw_centered(draw, (int(center), bottom + 30), CLASS_SHORT[code], FONT_AXIS)
        for model_index, model in enumerate(models):
            value = float(
                df[(df["model"] == model) & (df["class_code"] == code)]["f1_score"].fillna(0).sum()
            )
            x0 = int(center + (model_index - 1.5) * bar_w * 1.2 - bar_w / 2)
            x1 = x0 + bar_w
            y = bottom - int(value * (bottom - top))
            draw.rectangle((x0, y, x1, bottom), fill=MODEL_COLORS[model], outline=(60, 60, 60))

    legend_x, legend_y = 430, 700
    for i, model in enumerate(models):
        x = legend_x + i * 260
        draw.rectangle((x, legend_y, x + 28, legend_y + 18), fill=MODEL_COLORS[model], outline=(60, 60, 60))
        draw.text((x + 38, legend_y - 2), model, font=FONT_SMALL, fill=(30, 30, 30))
    save_image(img, out_dir, "fig02_2025_per_class_f1_comparison", args.formats, args.dpi)


def read_raster_preview(path: Path, max_width: int) -> np.ndarray:
    """读取分类栅格预览数组。"""

    with rasterio.open(path) as src:
        scale = max(1, int(np.ceil(src.width / max_width)))
        out_height = int(np.ceil(src.height / scale))
        out_width = int(np.ceil(src.width / scale))
        return src.read(1, out_shape=(out_height, out_width), resampling=Resampling.nearest)


def class_image(array: np.ndarray) -> Image.Image:
    """把 0-9 分类数组转换为 RGB 图像。"""

    rgb = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        rgb[array == code] = color
    return Image.fromarray(rgb, mode="RGB")


def correctness_image(actual: np.ndarray, predicted: np.ndarray) -> Image.Image:
    """生成 RF-CA 正误图。"""

    rgb = np.full((actual.shape[0], actual.shape[1], 3), 255, dtype=np.uint8)
    valid = (actual > 0) & (predicted > 0)
    rgb[valid & (actual == predicted)] = (27, 158, 119)
    rgb[valid & (actual != predicted)] = (217, 95, 2)
    return Image.fromarray(rgb, mode="RGB")


def panel_image(title: str, img: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    """生成带标题的地图面板。"""

    pw, ph = panel_size
    canvas = Image.new("RGB", (pw, ph), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, (pw // 2, 24), title, FONT_SUBTITLE)
    max_w, max_h = pw - 20, ph - 58
    img = img.copy()
    img.thumbnail((max_w, max_h), Image.Resampling.NEAREST)
    x = (pw - img.width) // 2
    y = 48 + (max_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_landuse_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """绘制土地利用图例。"""

    for i, code in enumerate(CLASS_CODES):
        col = i % 5
        row = i // 5
        xx = x + col * 250
        yy = y + row * 32
        draw.rectangle((xx, yy, xx + 22, yy + 22), fill=CLASS_COLORS[code], outline=(60, 60, 60))
        draw.text((xx + 30, yy - 2), f"{code} {CLASS_NAMES[code]}", font=FONT_TINY, fill=(30, 30, 30))


def plot_2025_maps(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 2025 预测图对比。"""

    entries = [
        ("Observed 2025", paths["observed_2025"]),
        ("Markov baseline", paths["markov_2025"]),
        ("CA-Markov", paths["ca_markov_2025"]),
        ("Logistic-CA", paths["logistic_2025"]),
        ("RF-CA", paths["rf_ca_2025"]),
    ]
    arrays = [(title, read_raster_preview(path, args.max_map_width)) for title, path in entries]
    panels = [panel_image(title, class_image(array), (620, 850)) for title, array in arrays]
    panels.append(panel_image("RF-CA Correctness", correctness_image(arrays[0][1], arrays[-1][1]), (620, 850)))

    canvas = Image.new("RGB", (1960, 2020), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, (980, 42), "Observed and Predicted Land Use in 2025", FONT_TITLE)
    positions = [(20, 80), (670, 80), (1320, 80), (20, 930), (670, 930), (1320, 930)]
    for panel, pos in zip(panels, positions, strict=True):
        canvas.paste(panel, pos)
    draw_landuse_legend(draw, 260, 1840)
    draw.rectangle((1330, 1930, 1352, 1952), fill=(27, 158, 119), outline=(60, 60, 60))
    draw.text((1362, 1926), "Correct", font=FONT_TINY, fill=(30, 30, 30))
    draw.rectangle((1480, 1930, 1502, 1952), fill=(217, 95, 2), outline=(60, 60, 60))
    draw.text((1512, 1926), "Incorrect", font=FONT_TINY, fill=(30, 30, 30))
    save_image(canvas, out_dir, "fig03_2025_prediction_map_comparison", args.formats, args.dpi)


def plot_2030_maps(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 2030 预测图对比。"""

    entries = [
        ("Observed 2025", paths["observed_2025"]),
        ("CA-Markov 2030", paths["ca_markov_2030"]),
        ("Logistic-CA 2030", paths["logistic_2030"]),
        ("RF-CA 2030", paths["rf_ca_2030"]),
    ]
    panels = [
        panel_image(title, class_image(read_raster_preview(path, args.max_map_width)), (720, 920))
        for title, path in entries
    ]
    canvas = Image.new("RGB", (1500, 1980), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, (750, 42), "Land Use Projection for 2030", FONT_TITLE)
    positions = [(30, 80), (750, 80), (30, 1000), (750, 1000)]
    for panel, pos in zip(panels, positions, strict=True):
        canvas.paste(panel, pos)
    draw_landuse_legend(draw, 150, 1910)
    save_image(canvas, out_dir, "fig04_2030_projection_map_comparison", args.formats, args.dpi)


def plot_area_change(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 2030 面积变化图。"""

    df = pd.read_csv(paths["rf_ca_2030_area"])
    df = df[df["class_code"].isin([1, 2, 4, 5, 7, 8])].copy()
    df["change"] = df["rf_ca_predicted_area_km2"] - df["base_area_km2"]

    img = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (750, 42), "RF-CA Projected Area Change from 2025 to 2030", FONT_TITLE)
    box = (150, 100, 1390, 600)
    left, top, right, bottom = box
    max_abs = max(abs(df["change"]).max(), 1)
    y_min, y_max = -max_abs * 1.15, max_abs * 1.15
    hex_grid(draw, box, [round(v, 0) for v in np.linspace(y_min, y_max, 5)], y_min, y_max)
    zero_y = bottom - int((0 - y_min) / (y_max - y_min) * (bottom - top))
    draw.line((left, zero_y, right, zero_y), fill=(40, 40, 40), width=2)
    group_w = (right - left) / len(df)
    bar_w = int(group_w * 0.55)
    for i, row in enumerate(df.itertuples(index=False)):
        value = float(row.change)
        center = left + i * group_w + group_w / 2
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        x0, x1 = int(center - bar_w / 2), int(center + bar_w / 2)
        color = (26, 150, 65) if value >= 0 else (215, 25, 28)
        draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill=color, outline=(60, 60, 60))
        draw_centered(draw, (int(center), bottom + 30), CLASS_SHORT[int(row.class_code)], FONT_SMALL)
        label_y = y - 20 if value >= 0 else y + 18
        draw_centered(draw, (int(center), label_y), f"{value:.1f}", FONT_TINY)
    draw.text((20, 320), "Area Change (km²)", font=FONT_AXIS, fill=(30, 30, 30))
    save_image(img, out_dir, "fig05_rf_ca_2030_area_change", args.formats, args.dpi)


def plot_feature_importance(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 RF 特征重要性。"""

    df = pd.read_csv(paths["rf_feature_importance"]).head(12).iloc[::-1]
    img = Image.new("RGB", (1500, 920), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (750, 42), "Top Random Forest Feature Importances", FONT_TITLE)
    left, top, right, bottom = 520, 105, 1370, 830
    max_value = max(float(df["importance"].max()), 0.01)
    for i, row in enumerate(df.itertuples(index=False)):
        y = top + i * 55
        value = float(row.importance)
        bar_w = int(value / max_value * (right - left))
        draw.text((30, y - 4), str(row.feature), font=FONT_SMALL, fill=(30, 30, 30))
        draw.rectangle((left, y, left + bar_w, y + 28), fill=(84, 162, 75), outline=(60, 60, 60))
        draw.text((left + bar_w + 10, y - 2), f"{value:.3f}", font=FONT_TINY, fill=(30, 30, 30))
    save_image(img, out_dir, "fig06_rf_feature_importance", args.formats, args.dpi)


def blend_color(value: float) -> tuple[int, int, int]:
    """白色到蓝绿色的渐变。"""

    low = np.array([247, 252, 253])
    high = np.array([37, 115, 145])
    rgb = low * (1 - value) + high * value
    return tuple(int(v) for v in rgb)


def plot_confusion(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> None:
    """绘制 RF-CA 混淆矩阵。"""

    df = pd.read_csv(paths["rf_ca_confusion"])
    matrix = df[[f"predicted_{code}" for code in CLASS_CODES]].to_numpy(dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    img = Image.new("RGB", (1200, 1120), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (600, 42), "RF-CA Confusion Matrix for 2025 Prediction", FONT_TITLE)
    left, top, cell = 230, 140, 82
    for i, code in enumerate(CLASS_CODES):
        draw_centered(draw, (left - 60, top + i * cell + cell // 2), CLASS_SHORT[code], FONT_SMALL)
        draw_centered(draw, (left + i * cell + cell // 2, top - 35), CLASS_SHORT[code], FONT_SMALL)
    draw.text((470, 980), "Predicted Class", font=FONT_AXIS, fill=(30, 30, 30))
    draw.text((45, 95), "Observed Class", font=FONT_AXIS, fill=(30, 30, 30))
    for i in range(len(CLASS_CODES)):
        for j in range(len(CLASS_CODES)):
            value = float(norm[i, j])
            x0, y0 = left + j * cell, top + i * cell
            color = blend_color(value)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(230, 230, 230))
            if value >= 0.02:
                fill = (255, 255, 255) if value > 0.55 else (30, 30, 30)
                draw_centered(draw, (x0 + cell // 2, y0 + cell // 2), f"{value:.2f}", FONT_TINY, fill=fill)
    save_image(img, out_dir, "fig07_rf_ca_2025_confusion_matrix", args.formats, args.dpi)


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    out_dir = output_dir(args)
    paths = required_paths(args)
    missing = check_paths(paths)
    if missing:
        raise FileNotFoundError("以下绘图输入文件缺失：\n" + "\n".join(missing))

    if args.dry_run:
        print("Dry-run OK")
        print(f"Output dir: {out_dir}")
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    plot_accuracy(paths, out_dir, args)
    plot_f1(paths, out_dir, args)
    plot_2025_maps(paths, out_dir, args)
    plot_2030_maps(paths, out_dir, args)
    plot_area_change(paths, out_dir, args)
    plot_feature_importance(paths, out_dir, args)
    plot_confusion(paths, out_dir, args)
    print(f"All figures written to: {out_dir}")


if __name__ == "__main__":
    main()
