"""Clean-training, no-reference pixel anomaly model and mask evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.ndimage import median_filter

from .dicom_io import find_dicom_files, read_dataset, relative_key
from .pixel_integrity import _load_ground_truth_mask


PATCH_SIZE = 16


def _normalized_image(ds):
    image = np.asarray(ds.pixel_array)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected a single 2-D integer CT slice; got {image.shape}, {image.dtype}")
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _patch_feature_grid(image, patch_size=PATCH_SIZE):
    """Describe local intensity and texture without a clean-reference image."""
    height, width = image.shape
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect") if pad_h or pad_w else image
    gh, gw = padded.shape[0] // patch_size, padded.shape[1] // patch_size
    patches = padded.reshape(gh, patch_size, gw, patch_size).transpose(0, 2, 1, 3)
    mean = patches.mean(axis=(2, 3))
    std = patches.std(axis=(2, 3))
    grad_x = np.abs(patches[..., 1:] - patches[..., :-1]).mean(axis=(2, 3))
    grad_y = np.abs(patches[..., 1:, :] - patches[..., :-1, :]).mean(axis=(2, 3))
    centered = np.abs(patches - mean[..., None, None])
    high_deviation = (centered > (2.0 * std[..., None, None] + 1e-6)).mean(axis=(2, 3))
    features = np.stack([mean, std, grad_x, grad_y, high_deviation,
                         np.log1p(std * 100.0)], axis=-1).reshape(-1, 6)
    return features.astype(np.float32), (gh, gw), image.shape


def _mask_from_patch_scores(scores, grid_shape, image_shape, threshold):
    gh, gw = grid_shape
    patch_flags = (np.asarray(scores).reshape(gh, gw) < threshold)
    mask = np.repeat(np.repeat(patch_flags, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    return mask[:image_shape[0], :image_shape[1]]


def _binary_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(labels.size), "positive": int(labels.sum()), "accuracy": (tp + tn) / labels.size if labels.size else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _localization_counts(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=bool), np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    return tp, fp, fn, tn


def _save_anomaly_heatmap(image, anomaly_mask, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(image, cmap="gray")
    overlay = np.ma.masked_where(~anomaly_mask, anomaly_mask.astype(float))
    ax.imshow(overlay, cmap="autumn", alpha=0.65, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _residual_map(image):
    """A simple image-only high-frequency residual baseline."""
    return np.abs(image - median_filter(image, size=5, mode="reflect"))


def run_no_reference_pixel_experiment(train_dir, validation_dir, clean_test_dir,
                                      attacks_root, output_dir, *, seed=42,
                                      max_train_files=0, max_train_patches=50000,
                                      patches_per_slice=24, heatmaps_per_attack=2,
                                      pretrained_model_path=None):
    """Fit on clean training patches, calibrate on validation, score attacks.

    This detector consumes each evaluated image independently. It never reads
    a clean counterpart or a PixelData hash. Clean counterparts are used only
    afterward to score saved synthetic ground-truth masks.
    """
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_files = find_dicom_files(train_dir)
    val_files = find_dicom_files(validation_dir)
    test_files = find_dicom_files(clean_test_dir)
    if not train_files or not val_files or not test_files:
        raise ValueError("Training, validation, and clean test folders must each contain readable DICOM files.")
    if pretrained_model_path:
        bundle = joblib.load(pretrained_model_path)
        model = bundle["model"]
        training_files_used = int(bundle.get("training_files", 0))
        training_patches_used = int(bundle.get("training_patches", 0))
    else:
        if max_train_files and len(train_files) > max_train_files:
            train_files = random.Random(seed).sample(train_files, max_train_files)
        rng = random.Random(seed)
        rng.shuffle(train_files)
        train_feature_chunks = []
        train_patch_count = 0
        for path in train_files:
            ds = read_dataset(path, stop_before_pixels=False)
            grid, _, _ = _patch_feature_grid(_normalized_image(ds))
            take = min(patches_per_slice, len(grid))
            indices = rng.sample(range(len(grid)), take)
            selected = grid[indices]
            remaining = max_train_patches - train_patch_count
            if remaining <= 0:
                break
            selected = selected[:remaining]
            train_feature_chunks.append(selected)
            train_patch_count += len(selected)
            if train_patch_count >= max_train_patches:
                break
        if train_patch_count < 100:
            raise ValueError("Too few training patches to fit the no-reference detector.")
        train_features = np.concatenate(train_feature_chunks, axis=0)
        model = IsolationForest(n_estimators=100, max_samples="auto", contamination="auto",
                                random_state=seed, n_jobs=-1)
        model.fit(np.asarray(train_features, dtype=np.float32))
        training_files_used = len(train_feature_chunks)
        training_patches_used = len(train_features)

    val_patch_scores, val_file_scores = [], []
    val_residual_pixel_samples, val_residual_file_scores = [], []
    for path in val_files:
        ds = read_dataset(path, stop_before_pixels=False)
        image = _normalized_image(ds)
        features, _, _ = _patch_feature_grid(image)
        scores = model.score_samples(features)
        val_patch_scores.extend(scores.tolist())
        val_file_scores.append(float(np.min(scores)))
        residuals = _residual_map(image)
        # Bound calibration memory while sampling evenly from each image.
        flat = residuals.ravel()
        stride = max(1, len(flat) // 4096)
        val_residual_pixel_samples.extend(flat[::stride][:4096].tolist())
        val_residual_file_scores.append(float(np.percentile(residuals, 99.9)))
    # Calibrate both local map threshold and per-image alarm threshold only on
    # clean validation data. These are empirical tail cutoffs, not guarantees.
    patch_threshold = float(np.percentile(val_patch_scores, 1.0))
    file_threshold = float(np.percentile(val_file_scores, 1.0))
    residual_pixel_threshold = float(np.percentile(val_residual_pixel_samples, 99.9))
    residual_file_threshold = float(np.percentile(val_residual_file_scores, 99.0))
    joblib.dump({"model": model, "patch_size": PATCH_SIZE, "patch_threshold": patch_threshold,
                 "file_threshold": file_threshold, "seed": seed,
                 "training_files": training_files_used, "training_patches": training_patches_used},
                output / "no_reference_pixel_model.joblib")

    clean_scores = []
    clean_pixel_fp = 0
    clean_pixel_total = 0
    residual_clean_alarms = 0
    residual_clean_pixel_fp = 0
    residual_clean_pixel_total = 0
    for path in test_files:
        ds = read_dataset(path, stop_before_pixels=False)
        image = _normalized_image(ds)
        features, grid_shape, image_shape = _patch_feature_grid(image)
        scores = model.score_samples(features)
        clean_scores.append(float(np.min(scores)))
        clean_mask = _mask_from_patch_scores(scores, grid_shape, image_shape, patch_threshold)
        clean_pixel_fp += int(clean_mask.sum())
        clean_pixel_total += int(clean_mask.size)
        residual_map = _residual_map(image)
        residual_clean_alarms += int(float(np.percentile(residual_map, 99.9)) > residual_file_threshold)
        residual_clean_pixel_fp += int(np.count_nonzero(residual_map > residual_pixel_threshold))
        residual_clean_pixel_total += int(residual_map.size)
    clean_alarm = np.asarray(clean_scores) < file_threshold
    clean_summary = {"clean_test_files": len(test_files),
                     "clean_test_file_false_positive_rate": float(clean_alarm.mean()),
                     "clean_test_pixel_false_positive_rate": clean_pixel_fp / clean_pixel_total if clean_pixel_total else 0.0,
                     "median_residual_clean_test_file_false_positive_rate": residual_clean_alarms / len(test_files),
                     "median_residual_clean_test_pixel_false_positive_rate": residual_clean_pixel_fp / residual_clean_pixel_total if residual_clean_pixel_total else 0.0,
                     "patch_score_threshold": patch_threshold,
                     "file_score_threshold": file_threshold,
                     "median_residual_pixel_threshold": residual_pixel_threshold,
                     "median_residual_image_threshold": residual_file_threshold,
                     "validation_patch_count": len(val_patch_scores),
                     "validation_file_count": len(val_files)}
    (output / "calibration_and_clean_summary.json").write_text(json.dumps(clean_summary, indent=2), encoding="utf-8")

    attack_rows, metrics = [], []
    attacks_root = Path(attacks_root).resolve()
    for attack_dir in sorted(p for p in attacks_root.iterdir() if p.is_dir()):
        log_path = attack_dir / "attack_log.json"
        if not log_path.exists():
            continue
        attack_log = json.loads(log_path.read_text(encoding="utf-8"))
        rows, loc_tp, loc_fp, loc_fn, loc_tn = [], 0, 0, 0, 0
        image_scores, image_labels = [], []
        residual_image_scores, residual_image_labels = [], []
        residual_loc_tp = residual_loc_fp = residual_loc_fn = residual_loc_tn = 0
        heatmap_count = 0
        for path in find_dicom_files(attack_dir):
            key = relative_key(path, attack_dir)
            ds = read_dataset(path, stop_before_pixels=False)
            image = _normalized_image(ds)
            features, grid_shape, image_shape = _patch_feature_grid(image)
            scores = model.score_samples(features)
            anomaly_map = _mask_from_patch_scores(scores, grid_shape, image_shape, patch_threshold)
            image_score = float(np.min(scores))
            image_prediction = int(image_score < file_threshold)
            entry = attack_log.get("files", {}).get(key, {})
            label = int(entry.get("label", 0))
            image_scores.append(image_score)
            image_labels.append(label)
            mask_path = entry.get("mask_path")
            residual_map = _residual_map(image)
            residual_anomaly_map = residual_map > residual_pixel_threshold
            residual_image_score = float(np.percentile(residual_map, 99.9))
            residual_image_scores.append(residual_image_score)
            residual_image_labels.append(label)
            loc = None
            if label and mask_path:
                truth = _load_ground_truth_mask(attack_dir, mask_path)
                tp, fp, fn, tn = _localization_counts(truth, anomaly_map)
                loc_tp += tp; loc_fp += fp; loc_fn += fn; loc_tn += tn
                loc = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
                rtp, rfp, rfn, rtn = _localization_counts(truth, residual_anomaly_map)
                residual_loc_tp += rtp; residual_loc_fp += rfp
                residual_loc_fn += rfn; residual_loc_tn += rtn
                if heatmap_count < heatmaps_per_attack:
                    heatmap_count += 1
                    heatmap_path = output / "heatmaps" / f"{attack_dir.name}_{heatmap_count:02d}.png"
                    _save_anomaly_heatmap(np.asarray(ds.pixel_array), anomaly_map,
                                          heatmap_path, f"No-reference map: {attack_dir.name}")
            attack_rows.append({"attack": attack_dir.name, "file": key,
                                "ground_truth": label, "image_anomaly_score": image_score,
                                "image_prediction": image_prediction,
                                "predicted_anomalous_patch_count": int(np.sum(scores < patch_threshold)),
                                "mask_tp": loc["tp"] if loc else None,
                                "mask_fp": loc["fp"] if loc else None,
                                "mask_fn": loc["fn"] if loc else None})
        image_metrics = _binary_metrics(image_labels, np.asarray(image_scores) < file_threshold)
        residual_metrics = _binary_metrics(
            residual_image_labels,
            np.asarray(residual_image_scores) > residual_file_threshold)
        if loc_tp + loc_fp + loc_fn:
            precision = loc_tp / (loc_tp + loc_fp) if loc_tp + loc_fp else 0.0
            recall = loc_tp / (loc_tp + loc_fn) if loc_tp + loc_fn else 0.0
            localization = {"pixel_tp": loc_tp, "pixel_fp": loc_fp, "pixel_fn": loc_fn,
                            "pixel_tn": loc_tn, "pixel_precision": precision,
                            "pixel_recall": recall,
                            "pixel_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "pixel_iou": loc_tp / (loc_tp + loc_fp + loc_fn)}
        else:
            localization = {"pixel_tp": 0, "pixel_fp": 0, "pixel_fn": 0,
                            "pixel_tn": 0, "pixel_precision": 0.0,
                            "pixel_recall": 0.0, "pixel_f1": 0.0, "pixel_iou": 0.0}
        rprecision = residual_loc_tp / (residual_loc_tp + residual_loc_fp) if residual_loc_tp + residual_loc_fp else 0.0
        rrecall = residual_loc_tp / (residual_loc_tp + residual_loc_fn) if residual_loc_tp + residual_loc_fn else 0.0
        residual_localization = {
            "median_residual_pixel_tp": residual_loc_tp,
            "median_residual_pixel_fp": residual_loc_fp,
            "median_residual_pixel_fn": residual_loc_fn,
            "median_residual_pixel_precision": rprecision,
            "median_residual_pixel_recall": rrecall,
            "median_residual_pixel_f1": 2*rprecision*rrecall/(rprecision+rrecall) if rprecision+rrecall else 0.0,
            "median_residual_pixel_iou": residual_loc_tp/(residual_loc_tp+residual_loc_fp+residual_loc_fn) if residual_loc_tp+residual_loc_fp+residual_loc_fn else 0.0,
        }
        metrics.append({"attack": attack_dir.name, **image_metrics,
                        "median_residual_image_recall": residual_metrics["recall"],
                        "median_residual_image_fpr": residual_metrics["false_positive_rate"],
                        "median_residual_image_f1": residual_metrics["f1"],
                        **localization, **residual_localization})
    if not attack_rows:
        raise ValueError(f"No attack directories with attack_log.json found under {attacks_root}")
    pd.DataFrame(attack_rows).to_csv(output / "no_reference_pixel_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "no_reference_pixel_metrics.csv", index=False)
    report = ["# No-reference pixel anomaly experiment", "",
              f"- Training files used: {training_files_used}", f"- Training patches: {training_patches_used}",
              f"- Validation files: {len(val_files)}", f"- Clean test files: {len(test_files)}",
              f"- Patch size: {PATCH_SIZE}x{PATCH_SIZE}",
              "- Input to detector: each image independently; no clean test counterpart or hash reference is used for prediction.",
              "- Calibration: lower 1st percentile of clean validation patch scores and per-image minimum scores.", "",
              "## Clean test false alarms", pd.DataFrame([clean_summary]).to_markdown(index=False), "",
              "## Attack detection and mask localization", metric_frame.to_markdown(index=False), "",
              "## Interpretation and limitations",
              "- Two image-only no-reference baselines are included: patch-statistics Isolation Forest and a 5x5 median-filter residual threshold.",
              "- The residual thresholds are calibrated on clean validation images (99.9th percentile of sampled pixel residuals; 99th percentile of each image's 99.9th-percentile residual score).",
              "- The patch Isolation Forest's image threshold was calibrated from the validation minimum patch score; its patch map uses 16x16 blocks.",
              "- The predicted map is blockwise at 16x16 resolution; saved synthetic masks are used only for evaluation.",
              "- Synthetic masks represent edited pixels, not clinical lesion annotations.",
              "- Results remain exploratory and are limited to the tested CPAISD folds and synthetic edits.", ""]
    (output / "NO_REFERENCE_PIXEL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return metric_frame
