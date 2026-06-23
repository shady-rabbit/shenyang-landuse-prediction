from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image


def configure_geospatial_data_paths() -> None:
    prefix = Path(sys.prefix)
    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_lib = prefix / "Library" / "share" / "proj"
    if "GDAL_DATA" not in os.environ and gdal_data.exists():
        os.environ["GDAL_DATA"] = str(gdal_data)
    if "PROJ_LIB" not in os.environ and proj_lib.exists():
        os.environ["PROJ_LIB"] = str(proj_lib)


configure_geospatial_data_paths()

import rasterio
from rasterio.enums import Resampling


PRODUCT_DIR = Path(__file__).resolve().parent
OUT = PRODUCT_DIR / "static_webgis"
ASSETS = OUT / "assets"
MAPS = ASSETS / "maps"
DRIVERS = ASSETS / "drivers"
FIGURES = ASSETS / "figures"

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", r"E:\model"))


def project_root(dirname: str, required_relative: str) -> Path:
    candidate = MODEL_ROOT / dirname
    required_path = candidate / required_relative
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required project file not found: {required_path}. "
            f"Please keep the project under {MODEL_ROOT}\\{dirname}."
        )
    return candidate


RF = project_root(
    "RF-CA",
    r"tables\model_comparison\shenyang_2025_validation_model_comparison.csv",
)
LOGISTIC = project_root(
    "Logistic-CA",
    r"output\logistic_ca\shenyang\shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5.tif",
)
MARKOV = project_root(
    "CA-Markov",
    r"tables\shenyang_clcd_original_class_summary.csv",
)

MAX_PREVIEW_WIDTH = 1250

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
CLASS_CN = {
    1: "耕地",
    2: "林地",
    3: "灌木",
    4: "草地",
    5: "水体",
    6: "冰雪",
    7: "裸地",
    8: "不透水面",
    9: "湿地",
}
CLASS_COLORS = {
    0: (255, 255, 255, 0),
    1: (244, 224, 77, 255),
    2: (38, 115, 0, 255),
    3: (109, 187, 117, 255),
    4: (156, 195, 107, 255),
    5: (75, 156, 211, 255),
    6: (247, 251, 255, 255),
    7: (189, 189, 189, 255),
    8: (215, 25, 28, 255),
    9: (65, 182, 196, 255),
}


LANDUSE_LAYERS = [
    {
        "id": "actual_2015",
        "title": "CLCD 2015 真实土地利用",
        "group": "真实 CLCD",
        "year": 2015,
        "model": "Observed",
        "mode": "historical",
        "path": RF / "data" / "processed" / "landuse" / "shenyang" / "shenyang_clcd_v01_2015_original.tif",
        "out": "actual_2015.png",
        "description": "研究区 2015 年 CLCD 原始 9 类土地利用。",
    },
    {
        "id": "actual_2020",
        "title": "CLCD 2020 真实土地利用",
        "group": "真实 CLCD",
        "year": 2020,
        "model": "Observed",
        "mode": "historical",
        "path": RF / "data" / "processed" / "landuse" / "shenyang" / "shenyang_clcd_v01_2020_original.tif",
        "out": "actual_2020.png",
        "description": "作为 2025 回测和 2030 预测的重要基准年份。",
    },
    {
        "id": "actual_2025",
        "title": "CLCD 2025 真实土地利用",
        "group": "真实 CLCD",
        "year": 2025,
        "model": "Observed",
        "mode": "validation",
        "path": RF / "data" / "processed" / "landuse" / "shenyang" / "shenyang_clcd_v01_2025_original.tif",
        "out": "actual_2025.png",
        "description": "用于验证 2015-2020 训练、2020-2025 预测结果的真实参照。",
    },
    {
        "id": "markov_2025",
        "title": "Markov baseline 2025 预测",
        "group": "2025 回测预测",
        "year": 2025,
        "model": "Markov baseline",
        "mode": "validation",
        "path": MARKOV / "output" / "markov_baseline" / "shenyang" / "shenyang_markov_fit_2015_2020_predict_2025.tif",
        "out": "markov_2025.png",
        "description": "非空间 Markov 基准模型，作为传统基线。",
    },
    {
        "id": "ca_markov_2025",
        "title": "CA-Markov 2025 预测",
        "group": "2025 回测预测",
        "year": 2025,
        "model": "CA-Markov",
        "mode": "validation",
        "path": MARKOV / "output" / "ca_markov" / "shenyang" / "shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5.tif",
        "out": "ca_markov_2025.png",
        "description": "无外部驱动因子的 CA-Markov，突出邻域约束作用。",
    },
    {
        "id": "ca_markov_suit_2025",
        "title": "适宜性因子 CA-Markov 2025 预测",
        "group": "2025 回测预测",
        "year": 2025,
        "model": "CA-Markov suitability",
        "mode": "validation",
        "path": MARKOV / "output" / "ca_markov_suitability" / "shenyang" / "shenyang_ca_markov_suitability_fit_2015_2020_predict_2025_n5_i5_nw090_sw010.tif",
        "out": "ca_markov_suit_2025.png",
        "description": "最优权重组：neighbor_weight=0.90, suitability_weight=0.10。",
    },
    {
        "id": "logistic_ca_2025",
        "title": "Logistic-CA 2025 预测",
        "group": "2025 回测预测",
        "year": 2025,
        "model": "Logistic-CA",
        "mode": "validation",
        "path": LOGISTIC / "output" / "logistic_ca" / "shenyang" / "shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5.tif",
        "out": "logistic_ca_2025.png",
        "description": "Logistic Regression 学习转移概率，再结合 CA 空间分配。",
    },
    {
        "id": "rf_ca_2025",
        "title": "RF-CA 2025 预测",
        "group": "2025 回测预测",
        "year": 2025,
        "model": "RF-CA",
        "mode": "validation",
        "path": RF / "output" / "rf_ca" / "shenyang" / "shenyang_rf_ca_fit_2015_2020_predict_2025_driver2020_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42.tif",
        "out": "rf_ca_2025.png",
        "description": "随机森林转移概率 + CA 空间分配，本研究主模型之一。",
    },
    {
        "id": "ca_markov_2030",
        "title": "CA-Markov 2030 预测",
        "group": "2030 未来预测",
        "year": 2030,
        "model": "CA-Markov",
        "mode": "projection",
        "path": MARKOV / "output" / "ca_markov" / "shenyang" / "shenyang_ca_markov_fit_2020_2025_predict_2030_n5_i5.tif",
        "out": "ca_markov_2030.png",
        "description": "基于 2020-2025 训练、以 2025 为基准的 2030 预测。",
    },
    {
        "id": "ca_markov_suit_2030",
        "title": "适宜性因子 CA-Markov 2030 预测",
        "group": "2030 未来预测",
        "year": 2030,
        "model": "CA-Markov suitability",
        "mode": "projection",
        "path": MARKOV / "output" / "ca_markov_suitability" / "shenyang" / "shenyang_ca_markov_suitability_fit_2020_2025_predict_2030_n5_i5.tif",
        "out": "ca_markov_suit_2030.png",
        "description": "考虑外部适宜性因子的 CA-Markov 2030 情景预测。",
    },
    {
        "id": "logistic_ca_2030",
        "title": "Logistic-CA 2030 预测",
        "group": "2030 未来预测",
        "year": 2030,
        "model": "Logistic-CA",
        "mode": "projection",
        "path": LOGISTIC / "output" / "logistic_ca" / "shenyang" / "shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5.tif",
        "out": "logistic_ca_2030.png",
        "description": "Logistic-CA 未来预测结果。",
    },
    {
        "id": "rf_ca_2030",
        "title": "RF-CA 2030 预测",
        "group": "2030 未来预测",
        "year": 2030,
        "model": "RF-CA",
        "mode": "projection",
        "path": RF / "output" / "rf_ca" / "shenyang" / "shenyang_rf_ca_fit_2020_2025_predict_2030_driver2025_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42.tif",
        "out": "rf_ca_2030.png",
        "description": "RF-CA 未来预测结果，可作为报告主展示图层。",
    },
]


DRIVER_LAYERS = [
    {
        "id": "road_closeness_2025",
        "title": "道路邻近性 2025",
        "path": RF / "data" / "processed" / "drivers" / "shenyang" / "shenyang_road_closeness_2025.tif",
        "out": "driver_road_closeness_2025.png",
        "description": "数值越高表示越接近道路，对建设用地扩张具有解释意义。",
    },
    {
        "id": "water_closeness_2025",
        "title": "水系邻近性 2025",
        "path": RF / "data" / "processed" / "drivers" / "shenyang" / "shenyang_water_closeness_2025.tif",
        "out": "driver_water_closeness_2025.png",
        "description": "数值越高表示越接近水系。",
    },
    {
        "id": "nightlight_2025",
        "title": "夜光强度 2025",
        "path": RF / "data" / "processed" / "drivers" / "shenyang" / "shenyang_nightlight_2025.tif",
        "out": "driver_nightlight_2025.png",
        "description": "反映城市活动强度，是 RF 模型中排名靠前的驱动因子。",
    },
    {
        "id": "elevation_norm_2025",
        "title": "高程归一化 2025",
        "path": RF / "data" / "processed" / "drivers" / "shenyang" / "shenyang_elevation_norm_2025.tif",
        "out": "driver_elevation_norm_2025.png",
        "description": "归一化高程因子。",
    },
    {
        "id": "low_slope_2025",
        "title": "低坡度因子 2025",
        "path": RF / "data" / "processed" / "drivers" / "shenyang" / "shenyang_low_slope_2025.tif",
        "out": "driver_low_slope_2025.png",
        "description": "低坡度区域更适宜城镇建设和农业利用。",
    },
]


CSV_PATHS = {
    "model_comparison": RF / "tables" / "model_comparison" / "shenyang_2025_validation_model_comparison.csv",
    "historical_area": MARKOV / "tables" / "shenyang_clcd_original_class_summary.csv",
    "rf_feature_importance": RF / "tables" / "random_forest" / "shenyang" / "shenyang_rf_fit_2020_2025_driver2020_n5_sampseed42_rf300_depthnone_leaf1_wtransition_trainseed42_feature_importance.csv",
    "rf_ca_area_2030": RF / "tables" / "rf_ca" / "shenyang" / "shenyang_rf_ca_fit_2020_2025_predict_2030_driver2025_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42_area_projection.csv",
    "logistic_area_2030": LOGISTIC / "tables" / "shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5_area_projection.csv",
    "ca_markov_area_2030": MARKOV / "tables" / "shenyang_ca_markov_fit_2020_2025_predict_2030_n5_i5_area_projection.csv",
    "rf_ca_f1": RF / "tables" / "rf_ca" / "shenyang" / "shenyang_rf_ca_fit_2015_2020_predict_2025_driver2020_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42_per_class_accuracy.csv",
    "logistic_f1": LOGISTIC / "tables" / "shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
    "ca_markov_f1": MARKOV / "tables" / "shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
    "ca_markov_suit_f1": MARKOV / "tables" / "shenyang_ca_markov_suitability_fit_2015_2020_predict_2025_n5_i5_nw090_sw010_per_class_accuracy.csv",
    "markov_f1": MARKOV / "tables" / "shenyang_markov_fit_2015_2020_predict_2025_per_class_accuracy.csv",
}


PAPER_FIGURES = [
    ("fig01_2025_model_accuracy_comparison.png", "2025 模型精度对比"),
    ("fig02_2025_per_class_f1_comparison.png", "2025 各类别 F1 对比"),
    ("fig03_2025_prediction_map_comparison.png", "2025 真实-预测图对比"),
    ("fig04_2030_projection_map_comparison.png", "2030 预测图对比"),
    ("fig05_rf_ca_2030_area_change.png", "RF-CA 2030 面积变化"),
    ("fig06_rf_feature_importance.png", "随机森林特征重要性"),
    ("fig07_rf_ca_2025_confusion_matrix.png", "RF-CA 2025 混淆矩阵"),
]


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalise_row(row) for row in reader]


def normalise_row(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = None
            continue
        text = value.strip()
        try:
            if any(ch in text for ch in (".", "e", "E")):
                out[key] = float(text)
            else:
                out[key] = int(text)
            continue
        except ValueError:
            out[key] = text
    return out


def target_shape(width: int, height: int) -> tuple[int, int]:
    if width <= MAX_PREVIEW_WIDTH:
        return width, height
    ratio = MAX_PREVIEW_WIDTH / width
    return MAX_PREVIEW_WIDTH, max(1, int(round(height * ratio)))


def read_preview(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        out_width, out_height = target_shape(src.width, src.height)
        return src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )


def render_landuse(path: Path, out_path: Path) -> dict[str, object]:
    arr = read_preview(path)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        rgba[arr == code] = color
    image = Image.fromarray(rgba, mode="RGBA")
    image.save(out_path, optimize=True)
    values, counts = np.unique(arr, return_counts=True)
    class_counts = {
        str(int(v)): int(c)
        for v, c in zip(values, counts, strict=True)
        if int(v) in CLASS_CODES
    }
    return {"width": image.width, "height": image.height, "class_counts_preview": class_counts}


def gradient_color(value: np.ndarray) -> np.ndarray:
    low = np.array([247, 251, 255], dtype=np.float32)
    high = np.array([8, 81, 156], dtype=np.float32)
    rgb = low + (high - low) * value[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_driver(path: Path, out_path: Path) -> dict[str, object]:
    arr = read_preview(path).astype(np.float32)
    valid = np.isfinite(arr)
    if np.any(valid):
        p2, p98 = np.nanpercentile(arr[valid], [2, 98])
        if p98 <= p2:
            p2 = float(np.nanmin(arr[valid]))
            p98 = float(np.nanmax(arr[valid]))
        scaled = np.zeros_like(arr, dtype=np.float32)
        if p98 > p2:
            scaled = (arr - p2) / (p98 - p2)
        scaled = np.clip(scaled, 0, 1)
    else:
        p2, p98 = 0.0, 1.0
        scaled = np.zeros_like(arr, dtype=np.float32)
    rgb = gradient_color(scaled)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    image = Image.fromarray(rgba, mode="RGBA")
    image.save(out_path, optimize=True)
    return {"width": image.width, "height": image.height, "display_min": float(p2), "display_max": float(p98)}


def predicted_area(row: dict[str, object]) -> float | None:
    for key in (
        "rf_ca_predicted_area_km2",
        "predicted_area_km2",
        "ca_predicted_area_km2",
        "markov_demand_area_km2",
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def build_area_projection() -> list[dict[str, object]]:
    rows = read_csv(CSV_PATHS["rf_ca_area_2030"])
    out = []
    for row in rows:
        code = int(row["class_code"])
        base = float(row.get("base_area_km2") or 0)
        pred = predicted_area(row) or 0.0
        out.append(
            {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "class_cn": CLASS_CN[code],
                "base_2025_area_km2": base,
                "projected_2030_area_km2": pred,
                "change_area_km2": pred - base,
            }
        )
    return out


def build_historical_area() -> list[dict[str, object]]:
    rows = read_csv(CSV_PATHS["historical_area"])
    out = []
    for row in rows:
        code = int(row["class_code"])
        if code not in CLASS_CODES:
            continue
        out.append(
            {
                "year": int(row["year"]),
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "class_cn": CLASS_CN[code],
                "area_km2": float(row["area_km2"]),
            }
        )
    for row in build_area_projection():
        out.append(
            {
                "year": 2030,
                "class_code": row["class_code"],
                "class_name": row["class_name"],
                "class_cn": row["class_cn"],
                "area_km2": row["projected_2030_area_km2"],
                "source": "RF-CA projection",
            }
        )
    return out


def normalise_class_accuracy(rows: list[dict[str, object]], model: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        code = int(row["class_code"])
        precision = row.get("precision")
        recall = row.get("recall")
        if precision is None:
            precision = row.get("user_accuracy")
        if recall is None:
            recall = row.get("producer_accuracy")
        out.append(
            {
                "model": model,
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "class_cn": CLASS_CN[code],
                "precision": float(precision or 0),
                "recall": float(recall or 0),
                "f1_score": float(row.get("f1_score") or 0),
            }
        )
    return out


def build_class_accuracy() -> list[dict[str, object]]:
    mapping = [
        ("Markov baseline", "markov_f1"),
        ("CA-Markov", "ca_markov_f1"),
        ("CA-Markov suitability", "ca_markov_suit_f1"),
        ("Logistic-CA", "logistic_f1"),
        ("RF-CA", "rf_ca_f1"),
    ]
    rows: list[dict[str, object]] = []
    for model, key in mapping:
        rows.extend(normalise_class_accuracy(read_csv(CSV_PATHS[key]), model))
    return rows


def build_figures() -> list[dict[str, str]]:
    src_dir = RF / "figures" / "paper" / "shenyang"
    figures = []
    for filename, title in PAPER_FIGURES:
        src = src_dir / filename
        dst = FIGURES / filename
        shutil.copyfile(src, dst)
        figures.append({"title": title, "src": f"assets/figures/{filename}", "source": str(src)})
    return figures


def ensure_dirs() -> None:
    for path in (OUT, ASSETS, MAPS, DRIVERS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def remove_stale_pngs(path: Path) -> None:
    for item in path.glob("*.png"):
        item.unlink()


def main() -> None:
    ensure_dirs()
    remove_stale_pngs(MAPS)
    remove_stale_pngs(DRIVERS)
    remove_stale_pngs(FIGURES)

    rendered_layers = []
    for layer in LANDUSE_LAYERS:
        out_path = MAPS / str(layer["out"])
        meta = render_landuse(Path(layer["path"]), out_path)
        record = {
            key: value
            for key, value in layer.items()
            if key not in {"path", "out"}
        }
        record["src"] = f"assets/maps/{layer['out']}"
        record["source"] = str(layer["path"])
        record.update(meta)
        rendered_layers.append(record)

    rendered_drivers = []
    for layer in DRIVER_LAYERS:
        out_path = DRIVERS / str(layer["out"])
        meta = render_driver(Path(layer["path"]), out_path)
        record = {
            key: value
            for key, value in layer.items()
            if key not in {"path", "out"}
        }
        record["src"] = f"assets/drivers/{layer['out']}"
        record["source"] = str(layer["path"])
        record.update(meta)
        rendered_drivers.append(record)

    classes = [
        {
            "code": code,
            "name": CLASS_NAMES[code],
            "cn": CLASS_CN[code],
            "color": "#%02x%02x%02x" % CLASS_COLORS[code][:3],
        }
        for code in CLASS_CODES
    ]
    data = {
        "meta": {
            "title": "沈阳市土地利用变化预测展示系统",
            "subtitle": "CLCD 2015-2025 回测验证与 2030 情景预测",
            "generated_at": date.today().isoformat(),
            "study_area": "沈阳市",
            "data_source": "CLCD",
            "pixel_area_km2": 0.0009,
        },
        "classes": classes,
        "layers": rendered_layers,
        "drivers": rendered_drivers,
        "modelComparison": read_csv(CSV_PATHS["model_comparison"]),
        "areaProjection": build_area_projection(),
        "historicalArea": build_historical_area(),
        "classAccuracy": build_class_accuracy(),
        "featureImportance": read_csv(CSV_PATHS["rf_feature_importance"]),
        "figures": build_figures(),
        "paths": {key: str(path) for key, path in CSV_PATHS.items()},
    }
    data_js = "window.DASHBOARD_DATA = "
    data_js += json.dumps(data, ensure_ascii=False, indent=2)
    data_js += ";\n"
    (ASSETS / "data.js").write_text(data_js, encoding="utf-8")
    (ASSETS / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
