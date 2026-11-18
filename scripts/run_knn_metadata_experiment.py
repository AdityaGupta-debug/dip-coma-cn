"""Fit and evaluate the clean-training kNN DICOM metadata detector."""
from __future__ import annotations

import argparse
import json
import sys
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import find_dicom_files, relative_key
from src.knn_anomaly import fit_knn_metadata, score_knn_metadata


def _metrics(labels, predictions, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _study_metrics(frame, target):
    if frame.empty:
        return {}
    grouped = frame.groupby("study_group", as_index=False).agg(
        ground_truth=(target, "max"), prediction=("prediction", "max"),
        score=("score", "max"))
    return _metrics(grouped.ground_truth, grouped.prediction, grouped.score)


def _frame(paths, root, labels, scores, predictions, metadata_labels=None, pixel_labels=None):
    metadata_labels = labels if metadata_labels is None else metadata_labels
    pixel_labels = np.zeros(len(labels), dtype=np.uint8) if pixel_labels is None else pixel_labels
    rows = []
    for path, label, metadata_label, pixel_label, score, pred in zip(
            paths, labels, metadata_labels, pixel_labels, scores, predictions):
        key = relative_key(path, root)
        rows.append({"file": key, "study_group": key.split("/")[0],
                     "ground_truth": int(label), "metadata_label": int(metadata_label),
                     "pixel_label": int(pixel_label), "score": float(score),
                     "prediction": int(pred)})
    return pd.DataFrame(rows)


def _target_metric_rows(frame, attack):
    rows = []
    for target in ("ground_truth", "metadata_label", "pixel_label"):
        rows.append({"attack": attack, "target": target, "unit": "file",
                     **_metrics(frame[target], frame.prediction, frame.score)})
        rows.append({"attack": attack, "target": target, "unit": "study",
                     **_study_metrics(frame, target)})
    return rows


def _combined_labels(labels_csv):
    if not labels_csv:
        return {}
    table = pd.read_csv(labels_csv)
    required = {"condition", "file", "metadata_label", "pixel_label", "ground_truth"}
    if required - set(table.columns):
        raise ValueError(f"Combined label CSV must contain columns: {sorted(required)}")
    return {(str(row.condition), str(row.file)): row
            for row in table.itertuples(index=False)}


def run(args):
    start = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "metadata_knn.joblib"
    bundle = fit_knn_metadata(
        args.train, args.validation, model_path, n_neighbors=args.k,
        threshold_percentile=args.threshold_percentile,
        max_train_files=args.max_train_files, random_state=args.seed,
        metric=args.metric)
    val_paths = find_dicom_files(args.validation)
    val_scores, val_predictions = score_knn_metadata(val_paths, bundle)
    val_frame = _frame(val_paths, Path(args.validation), np.zeros(len(val_paths)),
                       val_scores, val_predictions)
    val_frame.to_csv(output / "validation_calibration_predictions.csv", index=False)

    test_paths = find_dicom_files(args.test)
    clean_scores, clean_predictions = score_knn_metadata(test_paths, bundle)
    clean_frame = _frame(test_paths, Path(args.test), np.zeros(len(test_paths)),
                         clean_scores, clean_predictions)
    clean_frame.to_csv(output / "clean_test_predictions.csv", index=False)
    rows = _target_metric_rows(clean_frame, "clean_test")

    attack_root = Path(args.attacks_root)
    combined = _combined_labels(args.labels_csv)
    prediction_frames = []
    for attack_dir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
        log_path = attack_dir / "attack_log.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        paths = find_dicom_files(attack_dir)
        scores, predictions = score_knn_metadata(paths, bundle)
        labels = []
        metadata_labels = []
        pixel_labels = []
        for path in paths:
            key = relative_key(path, attack_dir)
            combined_row = combined.get((attack_dir.name, key))
            if args.labels_csv and combined_row is None:
                raise ValueError(f"Combined labels missing {attack_dir.name}/{key}")
            if combined_row is not None:
                labels.append(int(combined_row.ground_truth))
                metadata_labels.append(int(combined_row.metadata_label))
                pixel_labels.append(int(combined_row.pixel_label))
            else:
                label = int(log.get("files", {}).get(key, {}).get("label", 0))
                labels.append(label)
                metadata_labels.append(label)
                pixel_labels.append(0)
        frame = _frame(paths, attack_dir, labels, scores, predictions,
                       metadata_labels, pixel_labels)
        frame.insert(0, "attack", attack_dir.name)
        frame.to_csv(output / f"{attack_dir.name}_predictions.csv", index=False)
        rows.extend(_target_metric_rows(frame, attack_dir.name))
        prediction_frames.append(frame)
    if not prediction_frames:
        raise ValueError(f"No attack_log.json directories found under {attack_root}")
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(output / "attack_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "configuration.json").write_text(json.dumps({
        "method": "kNN metadata anomaly detector",
        "training_files": bundle["training_files"],
        "validation_files": bundle["validation_files"],
        "test_files": len(test_paths), "k": bundle["n_neighbors"],
        "metric": bundle["metric"], "threshold": bundle["threshold"],
        "threshold_percentile": bundle["threshold_percentile"],
        "seed": args.seed,
        "score": bundle["score_definition"],
        "calibration": "clean validation only; higher mean kNN distance is anomalous",
        "study_grouping": "first path component (immutable directory-level study key)",
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }, indent=2), encoding="utf-8")
    report = ["# kNN metadata anomaly experiment", "",
              "The model is fitted on clean training metadata only. The score is mean distance to the k nearest training instances; the threshold is calibrated on clean validation data.",
              "", "## Results", metrics.to_markdown(index=False), "",
              "## Limitations", "",
              "- Results are specific to the selected CPAISD folds, engineered metadata representation, metric, k, and validation quantile.",
              "- Slice-level observations are clustered within studies; ten test studies are too few for precise population-level estimates.",
              "- A score is an anomaly signal, not proof of unauthorized activity.", ""]
    (output / "KNN_METADATA_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nSaved kNN metadata results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Clean official training fold")
    parser.add_argument("--validation", required=True, help="Clean official validation fold")
    parser.add_argument("--test", required=True, help="Clean official held-out test fold")
    parser.add_argument("--attacks-root", required=True, help="Attack folders with attack_log.json")
    parser.add_argument("--labels-csv", default=None,
                        help="Optional combined_predictions.csv with condition/file and separate metadata, pixel, union labels")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--max-train-files", type=int, default=0,
                        help="Optional exploratory cap; zero uses all training files")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
