"""Fit and evaluate a clean-only PatchCore-style CT anomaly baseline."""
from __future__ import annotations

import argparse
import json
import sys
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dicom_io import find_dicom_files, relative_key
from src.patchcore import fit_patchcore, load_patchcore, score_patchcore
from src.pixel_integrity import _load_ground_truth_mask
from src.no_reference_pixel import _save_anomaly_heatmap


def metrics(labels, predictions, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _prediction_rows(paths, root, labels, results, metadata_labels=None, pixel_labels=None):
    metadata_labels = np.zeros(len(labels), dtype=np.uint8) if metadata_labels is None else metadata_labels
    pixel_labels = labels if pixel_labels is None else pixel_labels
    rows = []
    for path, label, metadata_label, pixel_label, result in zip(
            paths, labels, metadata_labels, pixel_labels, results):
        key = relative_key(path, root)
        rows.append({"file": key, "study_group": key.split("/")[0],
                     "ground_truth": int(label), "metadata_label": int(metadata_label),
                     "pixel_label": int(pixel_label), "score": result["score"],
                     "prediction": result["prediction"]})
    return pd.DataFrame(rows)


def _study_metrics(frame, target):
    studies = frame.groupby("study_group", as_index=False).agg(
        ground_truth=(target, "max"), prediction=("prediction", "max"),
        score=("score", "max"))
    return metrics(studies.ground_truth, studies.prediction, studies.score)


def _target_metric_rows(frame, attack):
    rows = []
    for target in ("ground_truth", "metadata_label", "pixel_label"):
        rows.append({"attack": attack, "target": target, "unit": "file",
                     **metrics(frame[target], frame.prediction, frame.score)})
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
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model_path = output / "patchcore_model.pt"
    use_weights = args.weights == "imagenet"
    fit_patchcore(
        args.train, args.validation, model_path,
        max_train_files=args.max_train_files,
        max_memory_patches=args.max_memory_patches,
        patches_per_train_image=args.patches_per_train_image,
        image_size=args.image_size,
        threshold_percentile=args.threshold_percentile,
        seed=args.seed, use_imagenet_weights=use_weights,
        device_name=args.device)
    bundle = load_patchcore(model_path, args.device)

    validation_paths = find_dicom_files(args.validation)
    validation_results = score_patchcore(validation_paths, bundle)
    validation_frame = _prediction_rows(
        validation_paths, Path(args.validation), np.zeros(len(validation_paths)), validation_results)
    validation_frame.to_csv(output / "validation_calibration_predictions.csv", index=False)

    test_paths = find_dicom_files(args.test)
    clean_results = score_patchcore(test_paths, bundle)
    clean_frame = _prediction_rows(test_paths, Path(args.test),
                                   np.zeros(len(test_paths)), clean_results)
    clean_frame.to_csv(output / "clean_test_predictions.csv", index=False)
    metric_rows = _target_metric_rows(clean_frame, "clean_test")
    attack_rows = []
    attack_root = Path(args.attacks_root)
    combined = _combined_labels(args.labels_csv)
    for attack_dir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
        log_path = attack_dir / "attack_log.json"
        if not log_path.exists():
            continue
        attack_log = json.loads(log_path.read_text(encoding="utf-8"))
        paths = find_dicom_files(attack_dir)
        results = score_patchcore(paths, bundle)
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
                label = int(attack_log.get("files", {}).get(key, {}).get("label", 0))
                labels.append(label)
                metadata_labels.append(0)
                pixel_labels.append(label)
        frame = _prediction_rows(paths, attack_dir, labels, results,
                                 metadata_labels, pixel_labels)
        frame.insert(0, "attack", attack_dir.name)
        frame.to_csv(output / f"{attack_dir.name}_predictions.csv", index=False)
        metric_rows.extend(_target_metric_rows(frame, attack_dir.name))
        loc_tp = loc_fp = loc_fn = 0
        saved_heatmaps = 0
        for i, (path, label, result) in enumerate(zip(paths, labels, results)):
            key = relative_key(path, attack_dir)
            entry = attack_log.get("files", {}).get(key, {})
            mask_path = entry.get("mask_path")
            localization = {}
            if label and mask_path:
                truth = _load_ground_truth_mask(attack_dir, mask_path).astype(bool)
                pred = result["anomaly_map"] > float(bundle["pixel_threshold"])
                if pred.shape != truth.shape:
                    raise ValueError(f"PatchCore map/mask shape mismatch for {key}: {pred.shape} vs {truth.shape}")
                tp = int(np.count_nonzero(pred & truth))
                fp = int(np.count_nonzero(pred & ~truth))
                fn = int(np.count_nonzero(~pred & truth))
                loc_tp += tp
                loc_fp += fp
                loc_fn += fn
                localization = {"pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn}
                if saved_heatmaps < args.heatmaps_per_attack:
                    saved_heatmaps += 1
                    _save_anomaly_heatmap(result["image"], pred,
                        output / "heatmaps" / f"{attack_dir.name}_{i:04d}.png",
                        f"PatchCore anomaly map: {attack_dir.name}")
            attack_rows.append({"attack": attack_dir.name, "file": key,
                                "ground_truth": label, "score": result["score"],
                                "prediction": result["prediction"], **localization})
        if loc_tp + loc_fp + loc_fn:
            pixel_precision = loc_tp / (loc_tp + loc_fp) if loc_tp + loc_fp else 0.0
            pixel_recall = loc_tp / (loc_tp + loc_fn) if loc_tp + loc_fn else 0.0
            metric_rows.append({"attack": attack_dir.name, "unit": "pixel",
                                "pixel_precision": pixel_precision,
                                "pixel_recall": pixel_recall,
                                "pixel_f1": (2 * pixel_precision * pixel_recall /
                                             (pixel_precision + pixel_recall)
                                             if pixel_precision + pixel_recall else 0.0),
                                "pixel_iou": loc_tp / (loc_tp + loc_fp + loc_fn)})
    if not attack_rows:
        raise ValueError(f"No attack directories with attack_log.json found under {attack_root}")
    pd.DataFrame(attack_rows).to_csv(output / "attack_predictions.csv", index=False)
    result_frame = pd.DataFrame(metric_rows)
    result_frame.to_csv(output / "metrics.csv", index=False)
    config = {
        "model": "PatchCore-style nearest-neighbor patch detector",
        "backbone": "ResNet-18 layer2/layer3 concatenation, fixed weights",
        "pretraining": "ImageNet default weights" if use_weights else "random frozen initialization",
        "memory_subsampling": "deterministic random subsample (not k-center coreset)",
        "memory_patches": int(len(bundle["memory_bank"])),
        "train_files_used": bundle["training_files"],
        "image_size": args.image_size, "patch_grid": "16x16",
        "pixel_threshold": bundle["pixel_threshold"],
        "image_threshold": bundle["image_threshold"],
        "threshold_percentile": args.threshold_percentile,
        "seed": args.seed, "device": str(bundle["device"]),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "calibration": "clean validation only",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "pydicom_version": pydicom.__version__, "scikit_learn_version": sklearn.__version__,
        "pytorch_version": __import__("torch").__version__,
    }
    (output / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = ["# PatchCore-style pixel anomaly experiment", "",
              "The model uses fixed ResNet-18 patch features and a clean-training memory bank. Thresholds are calibrated on clean validation images only.",
              "", "## Results", result_frame.to_markdown(index=False), "",
              "## Interpretation limits", "",
              "- ImageNet pretraining is transferred to grayscale CT and may not represent CT-specific normality.",
              "- The memory bank uses deterministic random subsampling rather than the original paper's coreset procedure.",
              "- Attack masks are synthetic changed-pixel masks; performance is not clinical tamper validation.",
              "- Study-level estimates use only ten held-out studies.", ""]
    (output / "PATCHCORE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(result_frame.to_string(index=False))
    print(f"\nSaved PatchCore results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Clean official training fold")
    parser.add_argument("--validation", required=True, help="Clean official validation fold")
    parser.add_argument("--test", required=True, help="Clean held-out test fold")
    parser.add_argument("--attacks-root", required=True, help="Pixel attack folders with attack_log.json")
    parser.add_argument("--labels-csv", default=None,
                        help="Optional combined_predictions.csv with condition/file and separate metadata, pixel, union labels")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--max-train-files", type=int, default=0,
                        help="Optional cap for exploratory runs; zero uses the full clean training fold")
    parser.add_argument("--max-memory-patches", type=int, default=20000)
    parser.add_argument("--patches-per-train-image", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmaps-per-attack", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
