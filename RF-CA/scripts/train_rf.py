"""RF-CA 随机森林训练脚本。

本脚本读取 `scripts/build_rf_samples.py` 生成的 `.npz` 样本包，
训练随机森林模型，输出：

- `.joblib` 模型包；
- 训练/验证总体指标；
- 混淆矩阵；
- 各类别 Precision/Recall/F1；
- 特征重要性表。

说明：
样本构建阶段采用了按 `from_class -> to_class` 的分层抽样，稀有转移会被
有意放大。为了让模型和评价结果仍能反映真实像元数量，本脚本默认使用
`_transition_summary.csv` 计算样本权重：

    weight = available_pixels / sampled_pixels

如果想做“完全均衡样本”的对比实验，可使用：

    --sample-weight-mode none
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


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


@dataclass(frozen=True)
class TrainMetrics:
    """保存训练/验证阶段的总体评价指标。"""

    split: str
    sample_count: int
    weighted_pixel_count: float
    overall_accuracy: float
    weighted_overall_accuracy: float
    kappa: float
    weighted_kappa: float
    macro_f1: float
    weighted_f1: float


def configure_console_encoding() -> None:
    """尽量让 Windows 终端正确显示中文提示。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Train Random Forest for RF-CA.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("E:/RF-CA"),
        help="RF-CA 项目根目录。",
    )
    parser.add_argument(
        "--city",
        default="shenyang",
        help="研究区名称。",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2015,
        help="样本起始年份。仅在 --sample-file 为空时用于自动查找样本。",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=2020,
        help="样本目标年份。仅在 --sample-file 为空时用于自动查找样本。",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=None,
        help="样本 .npz 文件路径。为空时按 city/from-year/to-year 自动查找。",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="自动查找样本文件时使用的样本构建随机种子。",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["transition", "none"],
        default="transition",
        help="样本权重模式：transition 按真实转移像元数加权；none 不加权。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="验证集比例。",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="随机森林树数量。",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="单棵树最大深度，默认不限制。",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="叶节点最少样本数。",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="内部节点继续划分所需最少样本数。",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="每次分裂考虑的特征数，可用 sqrt、log2、None 或数字字符串。",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "balanced_subsample"],
        default="none",
        help="sklearn 类别权重策略。默认 none，避免与转移样本权重重复加权。",
    )
    parser.add_argument(
        "--oob-score",
        action="store_true",
        help="启用随机森林 OOB 分数。会稍微增加训练耗时。",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="并行线程数，-1 表示使用全部可用核心。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型输出目录，默认 models/random_forest/{city}。",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="表格输出目录，默认 tables/random_forest/{city}。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名输出文件。默认不覆盖。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查样本和输出路径，不训练模型。",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """根据项目根目录补齐输出路径。"""

    root = args.project_root
    if args.models_dir is None:
        args.models_dir = root / "models" / "random_forest" / args.city
    if args.tables_dir is None:
        args.tables_dir = root / "tables" / "random_forest" / args.city
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查训练参数是否合理。"""

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size 必须位于 0 和 1 之间。")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators 必须 >= 1。")
    if args.min_samples_leaf < 1:
        raise ValueError("--min-samples-leaf 必须 >= 1。")
    if args.min_samples_split < 2:
        raise ValueError("--min-samples-split 必须 >= 2。")


def parse_max_features(value: str) -> str | int | float | None:
    """把命令行 max_features 参数转换为 sklearn 可接受的类型。"""

    if value.lower() == "none":
        return None
    if value.lower() in {"sqrt", "log2"}:
        return value.lower()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise ValueError(
            "--max-features 只能是 sqrt、log2、None、整数或小数。"
        ) from exc


def find_sample_file(args: argparse.Namespace) -> Path:
    """根据年份自动查找样本 .npz 文件。"""

    if args.sample_file is not None:
        if not args.sample_file.exists():
            raise FileNotFoundError(f"样本文件不存在：{args.sample_file}")
        return args.sample_file

    sample_dir = args.project_root / "data" / "samples" / args.city
    pattern = (
        f"{args.city}_rf_samples_fit_{args.from_year}_{args.to_year}_"
        f"*_seed{args.sample_seed}.npz"
    )
    matches = sorted(sample_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"未找到样本文件：{sample_dir / pattern}\n"
            "请先运行 scripts/build_rf_samples.py，或用 --sample-file 明确指定。"
        )
    if len(matches) > 1:
        choices = "\n".join(str(path) for path in matches)
        raise ValueError(
            "找到多个候选样本文件，请用 --sample-file 明确指定：\n" + choices
        )
    return matches[0]


def metadata_path_for_sample(sample_file: Path) -> Path:
    """根据样本文件名推断 metadata JSON 路径。"""

    return sample_file.with_name(sample_file.stem + "_metadata.json")


def load_metadata(sample_file: Path) -> dict[str, Any]:
    """读取样本构建元数据。"""

    metadata_path = metadata_path_for_sample(sample_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"样本元数据不存在：{metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_samples(sample_file: Path) -> dict[str, np.ndarray]:
    """读取样本包。"""

    data = np.load(sample_file, allow_pickle=False)
    required = ["X", "y", "from_class", "to_class", "feature_names"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"样本文件缺少字段：{', '.join(missing)}")
    return {name: data[name] for name in data.files}


def read_transition_weights(summary_path: Path) -> dict[tuple[int, int], float]:
    """读取转移抽样摘要，计算每种转移对应的样本权重。"""

    if not summary_path.exists():
        raise FileNotFoundError(f"转移摘要不存在：{summary_path}")

    weights: dict[tuple[int, int], float] = {}
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_code = int(row["from_code"])
            to_code = int(row["to_code"])
            available = float(row["available_pixels"])
            sampled = float(row["sampled_pixels"])
            if sampled > 0:
                weights[(from_code, to_code)] = available / sampled
    return weights


def sample_weights(
    metadata: dict[str, Any],
    from_class: np.ndarray,
    to_class: np.ndarray,
    mode: str,
) -> np.ndarray:
    """生成每个样本的原始权重。"""

    if mode == "none":
        return np.ones(len(to_class), dtype=np.float64)

    summary_path = Path(metadata["transition_summary_csv"])
    transition_weights = read_transition_weights(summary_path)
    weights = np.ones(len(to_class), dtype=np.float64)
    for index, (from_code, to_code) in enumerate(zip(from_class, to_class, strict=True)):
        weights[index] = transition_weights.get((int(from_code), int(to_code)), 1.0)
    return weights


def normalized_train_weights(raw_weights: np.ndarray, mode: str) -> np.ndarray | None:
    """把训练权重归一化到均值约为 1，避免数值尺度过大。"""

    if mode == "none":
        return None
    total = float(raw_weights.sum())
    if total <= 0:
        return None
    return raw_weights * (len(raw_weights) / total)


def can_stratify(y: np.ndarray) -> bool:
    """判断验证集划分是否可以按类别分层。"""

    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and int(counts.min()) >= 2


def output_stem(metadata: dict[str, Any], args: argparse.Namespace) -> str:
    """生成 RF 训练输出文件名前缀。"""

    depth = "none" if args.max_depth is None else str(args.max_depth)
    driver_year = metadata.get("driver_year")
    sample_seed = metadata.get("seed", args.sample_seed)
    driver_part = f"driver{driver_year}" if driver_year is not None else "nodriver"
    return (
        f"{metadata.get('city', args.city)}_rf_fit_{metadata.get('from_year')}_{metadata.get('to_year')}"
        f"_{driver_part}_n{metadata.get('neighborhood_size')}"
        f"_sampseed{sample_seed}_rf{args.n_estimators}_depth{depth}"
        f"_leaf{args.min_samples_leaf}_w{args.sample_weight_mode}"
        f"_trainseed{args.seed}"
    )


def output_paths(stem: str, args: argparse.Namespace) -> dict[str, Path]:
    """生成所有输出文件路径。"""

    return {
        "model": args.models_dir / f"{stem}.joblib",
        "summary": args.tables_dir / f"{stem}_summary.csv",
        "confusion": args.tables_dir / f"{stem}_confusion_matrix.csv",
        "weighted_confusion": args.tables_dir / f"{stem}_weighted_confusion_matrix.csv",
        "per_class": args.tables_dir / f"{stem}_per_class_accuracy.csv",
        "feature_importance": args.tables_dir / f"{stem}_feature_importance.csv",
        "metadata": args.tables_dir / f"{stem}_train_metadata.json",
    }


def ensure_outputs_available(paths: dict[str, Path], overwrite: bool) -> None:
    """防止训练结果被意外覆盖。"""

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "输出文件已存在，为避免覆盖请调整参数或使用 --overwrite：\n" + joined
        )


def build_model(args: argparse.Namespace) -> RandomForestClassifier:
    """根据命令行参数创建随机森林分类器。"""

    class_weight = None if args.class_weight == "none" else args.class_weight
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=parse_max_features(args.max_features),
        class_weight=class_weight,
        oob_score=args.oob_score,
        bootstrap=True,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )


def compute_metrics(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_weights: np.ndarray,
) -> TrainMetrics:
    """计算总体精度指标。"""

    overall = accuracy_score(y_true, y_pred)
    weighted_overall = accuracy_score(y_true, y_pred, sample_weight=raw_weights)
    kappa = cohen_kappa_score(y_true, y_pred, labels=CLASS_CODES)
    weighted_kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        sample_weight=raw_weights,
    )
    macro = f1_score(y_true, y_pred, labels=CLASS_CODES, average="macro", zero_division=0)
    weighted = f1_score(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        average="weighted",
        sample_weight=raw_weights,
        zero_division=0,
    )
    return TrainMetrics(
        split=split,
        sample_count=int(len(y_true)),
        weighted_pixel_count=float(raw_weights.sum()),
        overall_accuracy=float(overall),
        weighted_overall_accuracy=float(weighted_overall),
        kappa=float(kappa),
        weighted_kappa=float(weighted_kappa),
        macro_f1=float(macro),
        weighted_f1=float(weighted),
    )


def write_summary_csv(
    path: Path,
    metrics: list[TrainMetrics],
    model: RandomForestClassifier,
    sample_file: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """写出训练总体摘要。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "city",
        "from_year",
        "to_year",
        "driver_year",
        "sample_weight_mode",
        "sample_count",
        "weighted_pixel_count",
        "overall_accuracy",
        "weighted_overall_accuracy",
        "kappa",
        "weighted_kappa",
        "macro_f1",
        "weighted_f1",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "class_weight",
        "oob_score",
        "sample_file",
    ]
    oob = getattr(model, "oob_score_", "")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            row.update(
                {
                    "city": metadata.get("city", args.city),
                    "from_year": metadata.get("from_year"),
                    "to_year": metadata.get("to_year"),
                    "driver_year": metadata.get("driver_year"),
                    "sample_weight_mode": args.sample_weight_mode,
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth if args.max_depth is not None else "",
                    "min_samples_leaf": args.min_samples_leaf,
                    "min_samples_split": args.min_samples_split,
                    "max_features": args.max_features,
                    "class_weight": args.class_weight,
                    "oob_score": f"{float(oob):.6f}" if oob != "" else "",
                    "sample_file": str(sample_file),
                }
            )
            for key in [
                "weighted_pixel_count",
                "overall_accuracy",
                "weighted_overall_accuracy",
                "kappa",
                "weighted_kappa",
                "macro_f1",
                "weighted_f1",
            ]:
                row[key] = f"{float(row[key]):.6f}"
            writer.writerow(row)


def write_confusion_csv(path: Path, matrix: np.ndarray, value_format: str) -> None:
    """写出混淆矩阵。行是真实类别，列是预测类别。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["actual_code", "actual_name"] + [
        f"predicted_{int(code)}" for code in CLASS_CODES
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
                row[f"predicted_{int(predicted_code)}"] = value_format.format(
                    matrix[row_index, col_index]
                )
            writer.writerow(row)


def per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_weights: np.ndarray,
) -> list[dict[str, Any]]:
    """整理每一类的未加权和加权精度。"""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        zero_division=0,
    )
    w_precision, w_recall, w_f1, w_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        sample_weight=raw_weights,
        zero_division=0,
    )
    rows = []
    for index, code in enumerate(CLASS_CODES):
        rows.append(
            {
                "class_code": int(code),
                "class_name": CLASS_NAMES[int(code)],
                "support_samples": int(support[index]),
                "estimated_support_pixels": float(w_support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "weighted_precision": float(w_precision[index]),
                "weighted_recall": float(w_recall[index]),
                "weighted_f1_score": float(w_f1[index]),
            }
        )
    return rows


def write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出各类别精度表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_code",
        "class_name",
        "support_samples",
        "estimated_support_pixels",
        "precision",
        "recall",
        "f1_score",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = row.copy()
            for key in [
                "estimated_support_pixels",
                "precision",
                "recall",
                "f1_score",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1_score",
            ]:
                clean[key] = f"{float(clean[key]):.6f}"
            writer.writerow(clean)


def write_feature_importance_csv(
    path: Path,
    feature_names: list[str],
    importances: np.ndarray,
) -> None:
    """写出随机森林特征重要性表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-importances, kind="stable")
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "feature", "importance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature": feature_names[index],
                    "importance": f"{float(importances[index]):.8f}",
                }
            )


def write_train_metadata_json(
    path: Path,
    args: argparse.Namespace,
    sample_file: Path,
    sample_metadata: dict[str, Any],
    output_files: dict[str, Path],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> None:
    """写出训练过程元数据。"""

    payload = {
        "script": "scripts/train_rf.py",
        "sample_file": str(sample_file),
        "sample_metadata": sample_metadata,
        "train_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_files": {key: str(path) for key, path in output_files.items()},
        "train_sample_count": int(len(train_indices)),
        "validation_sample_count": int(len(valid_indices)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_dry_run(
    args: argparse.Namespace,
    sample_file: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    paths: dict[str, Path],
) -> None:
    """输出 dry-run 检查信息。"""

    y = arrays["y"]
    values, counts = np.unique(y, return_counts=True)
    print("Dry-run OK")
    print(f"Sample file: {sample_file}")
    print(f"X shape: {arrays['X'].shape}")
    print(f"y shape: {arrays['y'].shape}")
    print(f"Feature count: {len(arrays['feature_names'])}")
    print(f"From year: {metadata.get('from_year')}")
    print(f"To year: {metadata.get('to_year')}")
    print(f"Driver year: {metadata.get('driver_year')}")
    print(f"Sample weight mode: {args.sample_weight_mode}")
    print("Class counts:")
    for value, count in zip(values, counts, strict=True):
        print(f"  {int(value)} {CLASS_NAMES.get(int(value), '')}: {int(count)}")
    print("Planned outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    """脚本入口：读取样本、训练 RF、输出模型和评价表。"""

    configure_console_encoding()
    args = resolve_paths(parse_args())
    validate_args(args)

    sample_file = find_sample_file(args)
    metadata = load_metadata(sample_file)
    arrays = load_samples(sample_file)
    stem = output_stem(metadata, args)
    paths = output_paths(stem, args)

    if args.dry_run:
        print_dry_run(args, sample_file, metadata, arrays, paths)
        return

    ensure_outputs_available(paths, overwrite=args.overwrite)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    x = arrays["X"].astype(np.float32, copy=False)
    y = arrays["y"].astype(np.uint8, copy=False)
    from_class = arrays["from_class"].astype(np.uint8, copy=False)
    to_class = arrays["to_class"].astype(np.uint8, copy=False)
    feature_names = [str(name) for name in arrays["feature_names"]]

    raw_weights = sample_weights(
        metadata,
        from_class=from_class,
        to_class=to_class,
        mode=args.sample_weight_mode,
    )

    indices = np.arange(len(y))
    stratify = y if can_stratify(y) else None
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    train_weights = normalized_train_weights(
        raw_weights[train_idx],
        mode=args.sample_weight_mode,
    )

    model = build_model(args)
    print(f"Training RF: samples={len(train_idx)}, features={x.shape[1]}")
    print(f"Validation samples: {len(valid_idx)}")
    print(f"Sample weight mode: {args.sample_weight_mode}")
    model.fit(x[train_idx], y[train_idx], sample_weight=train_weights)

    train_pred = model.predict(x[train_idx])
    valid_pred = model.predict(x[valid_idx])
    train_metrics = compute_metrics(
        "train",
        y_true=y[train_idx],
        y_pred=train_pred,
        raw_weights=raw_weights[train_idx],
    )
    valid_metrics = compute_metrics(
        "validation",
        y_true=y[valid_idx],
        y_pred=valid_pred,
        raw_weights=raw_weights[valid_idx],
    )

    valid_confusion = confusion_matrix(
        y[valid_idx],
        valid_pred,
        labels=CLASS_CODES,
    )
    valid_weighted_confusion = confusion_matrix(
        y[valid_idx],
        valid_pred,
        labels=CLASS_CODES,
        sample_weight=raw_weights[valid_idx],
    )
    per_class = per_class_rows(y[valid_idx], valid_pred, raw_weights[valid_idx])

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "class_codes": CLASS_CODES,
        "class_names": CLASS_NAMES,
        "sample_file": str(sample_file),
        "sample_metadata": metadata,
        "train_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    joblib.dump(artifact, paths["model"])

    write_summary_csv(
        paths["summary"],
        metrics=[train_metrics, valid_metrics],
        model=model,
        sample_file=sample_file,
        metadata=metadata,
        args=args,
    )
    write_confusion_csv(paths["confusion"], valid_confusion, "{:.0f}")
    write_confusion_csv(paths["weighted_confusion"], valid_weighted_confusion, "{:.6f}")
    write_per_class_csv(paths["per_class"], per_class)
    write_feature_importance_csv(
        paths["feature_importance"],
        feature_names=feature_names,
        importances=model.feature_importances_,
    )
    write_train_metadata_json(
        paths["metadata"],
        args=args,
        sample_file=sample_file,
        sample_metadata=metadata,
        output_files=paths,
        train_indices=train_idx,
        valid_indices=valid_idx,
    )

    print(f"Model: {paths['model']}")
    print(f"Summary: {paths['summary']}")
    print(
        "Validation: "
        f"OA={valid_metrics.overall_accuracy:.4f}, "
        f"weighted_OA={valid_metrics.weighted_overall_accuracy:.4f}, "
        f"Kappa={valid_metrics.kappa:.4f}, "
        f"weighted_Kappa={valid_metrics.weighted_kappa:.4f}"
    )
    if args.oob_score:
        print(f"OOB score: {model.oob_score_:.4f}")


if __name__ == "__main__":
    main()
