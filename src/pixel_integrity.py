"""Controlled pixel-data integrity experiments, separate from metadata attacks."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydicom.uid import UID

from .dicom_io import find_dicom_files, read_dataset, relative_key
from .fs_utils import link_or_copy
from .integrity import metadata_sha256, pixel_sha256

PIXEL_ATTACKS = ("localized_patch", "low_amplitude", "slice_inconsistency")


def _save_difference_heatmap(original: np.ndarray, modified: np.ndarray, output: Path, title: str):
    """Save a descriptive original/edited/absolute-difference figure."""
    original_float = np.asarray(original, dtype=np.float32)
    modified_float = np.asarray(modified, dtype=np.float32)
    if original_float.shape != modified_float.shape:
        raise ValueError(f"Cannot make a pixel difference map for unequal shapes: {original_float.shape} vs {modified_float.shape}")
    difference = np.abs(modified_float - original_float)
    low = float(min(original_float.min(), modified_float.min()))
    high = float(max(original_float.max(), modified_float.max()))
    if high <= low:
        high = low + 1.0
    vmax = max(1.0, float(difference.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(original_float, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("Clean reference")
    axes[1].imshow(modified_float, cmap="gray", vmin=low, vmax=high)
    axes[1].set_title("Modified copy")
    image = axes[2].imshow(difference, cmap="inferno", vmin=0, vmax=vmax)
    axes[2].set_title("Absolute pixel difference")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="Stored pixel-value difference")
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _study_series_groups(root: Path, files: list[Path]):
    groups: dict[str, dict[str, list[Path]]] = {}
    for path in files:
        ds = read_dataset(path, stop_before_pixels=True)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{path.parent}"
        series = str(getattr(ds, "SeriesInstanceUID", "")).strip() or f"MISSING:{path.parent}"
        groups.setdefault(study, {}).setdefault(series, []).append(path)
    return groups


def _pixel_limits(ds, array: np.ndarray) -> tuple[int, int]:
    bits = int(getattr(ds, "BitsStored", array.dtype.itemsize * 8))
    signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _edit_pixels(path: Path, attack: str, rng: random.Random, *,
                 patch_size: int = 5, patch_delta: int | None = None,
                 sparse_fraction: float = 0.01, sparse_delta: int = 1):
    ds = read_dataset(path, stop_before_pixels=False)
    transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax and UID(str(transfer_syntax)).is_compressed:
        raise ValueError(f"Compressed transfer syntax is not supported for pixel editing: {path}")
    array = np.asarray(ds.pixel_array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Expected a single 2-D integer CT slice, got shape={array.shape}, dtype={array.dtype}: {path}")
    low, high = _pixel_limits(ds, array)
    work = array.astype(np.int64, copy=True)
    rows, cols = work.shape
    mask = np.zeros(work.shape, dtype=bool)
    if attack == "localized_patch":
        height, width = min(patch_size, rows), min(patch_size, cols)
        top, left = max(0, (rows - height) // 2), max(0, (cols - width) // 2)
        region = work[top:top + height, left:left + width]
        changed = np.clip(region + (patch_delta if patch_delta is not None else 32), low, high)
        changed_mask = changed != region
        count = int(np.count_nonzero(changed_mask))
        mask[top:top + height, left:left + width] = changed_mask
        work[top:top + height, left:left + width] = changed
    elif attack == "low_amplitude":
        # Seeded sparse perturbation; clip to the DICOM stored-value range.
        if not 0 < sparse_fraction <= 1:
            raise ValueError("sparse_fraction must be in (0, 1].")
        count_to_change = max(1, round(work.size * sparse_fraction))
        indices = rng.sample(range(work.size), min(count_to_change, work.size))
        flat = work.reshape(-1)
        before = flat[indices].copy()
        flat[indices] = np.clip(before + sparse_delta, low, high)
        actual = flat[indices] != before
        count = int(np.count_nonzero(actual))
        mask.reshape(-1)[indices] = actual
    elif attack == "slice_inconsistency":
        # Introduce a localized alteration in a single slice; neighboring
        # slices in the series remain unchanged for contextual review.
        height, width = min(patch_size, rows), min(patch_size, cols)
        top, left = max(0, (rows - height) // 2), max(0, (cols - width) // 2)
        region = work[top:top + height, left:left + width]
        changed = np.clip(region + (patch_delta if patch_delta is not None else 8), low, high)
        changed_mask = changed != region
        count = int(np.count_nonzero(changed_mask))
        mask[top:top + height, left:left + width] = changed_mask
        work[top:top + height, left:left + width] = changed
    else:
        raise ValueError(f"Unknown pixel attack: {attack}")
    if count == 0:
        raise ValueError(f"Pixel edit made no representable change ({attack}): {path}")
    # Preserve the DICOM pixel encoding and only replace the pixel element.
    native_dtype = array.dtype.newbyteorder("<" if getattr(ds, "is_little_endian", True) else ">")
    ds.PixelData = work.astype(native_dtype, copy=False).tobytes(order="C")
    ds.save_as(str(path), enforce_file_format=True)
    return count, mask


def create_pixel_attack(input_dir, output_dir, attack: str, attack_fraction=0.20, seed=42,
                        *, patch_size=5, patch_delta=None, sparse_fraction=0.01,
                        sparse_delta=1):
    if attack not in PIXEL_ATTACKS:
        raise ValueError(f"attack must be one of {PIXEL_ATTACKS}")
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    files = find_dicom_files(source)
    if not files:
        raise ValueError(f"No DICOM files found in {source}")
    grouped = _study_series_groups(source, files)
    study_ids = sorted(grouped)
    rng = random.Random(seed)
    chosen_count = max(1, round(len(study_ids) * attack_fraction))
    chosen_studies = set(rng.sample(study_ids, min(chosen_count, len(study_ids))))
    if attack == "slice_inconsistency":
        selected = {sorted(series_files)[0]
                    for study in chosen_studies
                    for series_files in grouped[study].values() if series_files}
    else:
        selected = {path for study in chosen_studies
                    for series_files in grouped[study].values() for path in series_files}

    log = {"attack": attack, "attack_fraction_requested": attack_fraction,
           "sampling_unit": "study", "selected_study_count": len(chosen_studies),
           "total_study_count": len(study_ids), "seed": seed, "files": {},
           "modified_file_count": 0, "modified_pixel_count": 0,
           "note": "Synthetic pixel edits on copies only; no clinical interpretation is implied."}
    for path in files:
        key = relative_key(path, source)
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if path in selected:
            shutil.copy2(path, target)  # never mutate a hard-linked clean input
            changed_pixels, mask = _edit_pixels(
                target, attack, rng, patch_size=patch_size, patch_delta=patch_delta,
                sparse_fraction=sparse_fraction, sparse_delta=sparse_delta)
            mask_path = destination / "ground_truth_masks" / f"{key}.npz"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=np.packbits(mask.reshape(-1)), shape=np.asarray(mask.shape))
            log["files"][key] = {"label": 1, "changed_pixel_count": changed_pixels,
                                 "mask_path": mask_path.relative_to(destination).as_posix()}
            log["modified_file_count"] += 1
            log["modified_pixel_count"] += changed_pixels
        else:
            link_or_copy(path, target)
            log["files"][key] = {"label": 0, "changed_pixel_count": 0}
    (destination / "attack_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def _binary_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(labels.size), "positive": int(labels.sum()), "negative": int(labels.size-labels.sum()),
            "accuracy": (tp + tn) / labels.size if labels.size else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _load_ground_truth_mask(attack_dir: Path, mask_relative_path: str):
    with np.load(attack_dir / Path(mask_relative_path)) as saved:
        shape = tuple(int(x) for x in saved["shape"])
        packed = saved["mask"]
    return np.unpackbits(packed)[:int(np.prod(shape))].reshape(shape).astype(bool)


def _localization_metrics(truth: np.ndarray, predicted: np.ndarray):
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    union = tp + fp + fn
    return {"pixel_count": int(truth.size), "changed_pixels": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / union if union else 1.0}


def run_pixel_integrity_experiment(input_dir, output_dir, attack_fraction=0.20, seed=42,
                                   heatmaps_per_attack=3, patch_size=5, patch_delta=32,
                                   sparse_fraction=0.01, sparse_delta=1, slice_delta=8):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline = {}
    clean_files = find_dicom_files(source)
    for path in clean_files:
        ds = read_dataset(path, stop_before_pixels=False)
        baseline[relative_key(path, source)] = {
            "pixel_sha256": pixel_sha256(ds),
            "metadata_sha256": metadata_sha256(ds),
            "study_group": hashlib.sha256(str(getattr(ds, "StudyInstanceUID", "")).encode()).hexdigest()[:16],
        }
    if not baseline:
        raise ValueError(f"No readable DICOM files found under {source}")
    if heatmaps_per_attack < 0:
        raise ValueError("heatmaps_per_attack must be zero or greater")
    baseline_path = output / "clean_pixel_baseline.json"
    baseline_path.write_text(json.dumps({"description": "Trusted clean reference for this controlled experiment; protect separately.",
                                         "file_count": len(baseline), "files": baseline}, indent=2), encoding="utf-8")

    predictions, metric_rows, localization_rows = [], [], []
    for attack in PIXEL_ATTACKS:
        attack_dir = output / "attacks" / attack
        effective_delta = slice_delta if attack == "slice_inconsistency" else patch_delta
        log = create_pixel_attack(source, attack_dir, attack, attack_fraction, seed,
                                  patch_size=patch_size, patch_delta=effective_delta,
                                  sparse_fraction=sparse_fraction, sparse_delta=sparse_delta)
        attack_rows = []
        heatmap_count = 0
        attack_loc = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pixel_count": 0,
                      "changed_pixels": 0, "files_scored": 0}
        for path in find_dicom_files(attack_dir):
            key = relative_key(path, attack_dir)
            record = baseline[key]
            ds = read_dataset(path, stop_before_pixels=False)
            clean_ds = read_dataset(source / Path(key), stop_before_pixels=False)
            original_pixels = np.asarray(clean_ds.pixel_array)
            modified_pixels = np.asarray(ds.pixel_array)
            if original_pixels.shape != modified_pixels.shape:
                raise ValueError(f"Pixel array shape changed for {key}: {original_pixels.shape} -> {modified_pixels.shape}")
            delta = np.abs(modified_pixels.astype(np.float64) - original_pixels.astype(np.float64))
            label = int(log["files"].get(key, {}).get("label", 0))
            attack_entry = log["files"].get(key, {})
            if label and attack_entry.get("mask_path"):
                truth_mask = _load_ground_truth_mask(attack_dir, attack_entry["mask_path"])
                predicted_mask = delta > 0
                loc = _localization_metrics(truth_mask, predicted_mask)
                for metric in ("tp", "fp", "fn", "tn", "pixel_count", "changed_pixels"):
                    attack_loc[metric] += loc[metric]
                attack_loc["files_scored"] += 1
            else:
                loc = None
            heatmap = ""
            if label and heatmap_count < heatmaps_per_attack:
                heatmap_count += 1
                heatmap_path = output / "heatmaps" / f"{attack}_example_{heatmap_count:02d}.png"
                _save_difference_heatmap(original_pixels, modified_pixels, heatmap_path,
                                         f"{attack}: {Path(key).parent.name} / {Path(key).name}")
                heatmap = heatmap_path.relative_to(output).as_posix()
            row = {"attack": attack, "file": key, "study_group": record["study_group"], "ground_truth": label,
                   "pixel_hash_flag": int(pixel_sha256(ds) != record["pixel_sha256"]),
                   "metadata_hash_flag": int(metadata_sha256(ds) != record["metadata_sha256"]),
                   "changed_pixel_count": int(np.count_nonzero(delta)),
                   "changed_pixel_fraction": float(np.count_nonzero(delta) / delta.size) if delta.size else 0.0,
                   "mean_absolute_difference": float(delta.mean()) if delta.size else 0.0,
                   "root_mean_squared_difference": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
                   "maximum_absolute_difference": float(delta.max()) if delta.size else 0.0,
                   "localization_iou_reference_difference": loc["iou"] if loc else None,
                   "heatmap": heatmap}
            predictions.append(row)
            attack_rows.append(row)
        for method, column in (("Pixel SHA-256 reference", "pixel_hash_flag"),
                                ("Metadata SHA-256 negative control", "metadata_hash_flag")):
            vals = [r[column] for r in attack_rows]
            metric_rows.append({"attack": attack, "method": method, **_binary_metrics([r["ground_truth"] for r in attack_rows], vals)})
            frame = pd.DataFrame(attack_rows)
            study = frame.groupby("study_group", as_index=False).agg(ground_truth=("ground_truth", "max"), prediction=(column, "max"))
            metric_rows.append({"attack": attack, "method": method + " (study-level)", **_binary_metrics(study.ground_truth, study.prediction)})
        loc_precision = attack_loc["tp"] / (attack_loc["tp"] + attack_loc["fp"]) if attack_loc["tp"] + attack_loc["fp"] else 0.0
        loc_recall = attack_loc["tp"] / (attack_loc["tp"] + attack_loc["fn"]) if attack_loc["tp"] + attack_loc["fn"] else 0.0
        loc_union = attack_loc["tp"] + attack_loc["fp"] + attack_loc["fn"]
        localization_rows.append({"attack": attack, **attack_loc,
                                  "precision": loc_precision, "recall": loc_recall,
                                  "f1": 2 * loc_precision * loc_recall / (loc_precision + loc_recall) if loc_precision + loc_recall else 0.0,
                                  "iou": attack_loc["tp"] / loc_union if loc_union else 1.0,
                                  "source": "nonzero clean-reference pixel difference; descriptive reference-based localization check"})

    pd.DataFrame(predictions).to_csv(output / "pixel_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "pixel_metrics.csv", index=False)
    pd.DataFrame(localization_rows).to_csv(output / "pixel_localization_metrics.csv", index=False)
    prediction_frame = pd.DataFrame(predictions)
    attacked_frame = prediction_frame[prediction_frame["ground_truth"] == 1]
    difference_summary = attacked_frame.groupby("attack", as_index=False).agg(
        attacked_files=("file", "count"),
        mean_changed_pixel_fraction=("changed_pixel_fraction", "mean"),
        mean_absolute_difference=("mean_absolute_difference", "mean"),
        mean_rmse=("root_mean_squared_difference", "mean"),
        maximum_observed_difference=("maximum_absolute_difference", "max"))
    difference_summary.to_csv(output / "pixel_difference_summary.csv", index=False)
    config = {"input": str(source), "file_count": len(baseline), "attacks": list(PIXEL_ATTACKS),
              "attack_fraction": attack_fraction, "seed": seed,
              "heatmaps_per_attack": heatmaps_per_attack,
              "attack_parameters": {"localized_patch": f"central {patch_size}x{patch_size} stored-pixel patch, +{patch_delta}, on every slice in selected studies",
                                    "low_amplitude": f"seeded random {sparse_fraction:.4%} of stored pixels, +{sparse_delta}",
                                    "slice_inconsistency": f"central {patch_size}x{patch_size} stored-pixel patch, +{slice_delta}, on the first sorted slice of each selected series",
                                    "ground_truth_masks": "packed binary masks saved per modified DICOM instance"},
              "pixel_value_domain": "stored integer PixelData values; RescaleSlope/RescaleIntercept are not applied",
              "hash_scope": "raw DICOM PixelData value; metadata digest excludes PixelData",
              "limitations": ["Synthetic controlled edits only.", "Pixel hashing requires a trusted per-file original reference.",
                              "Only uncompressed single-frame 2-D integer pixel arrays are editable in this initial implementation.",
                              "No image anomaly model or lesion localization is evaluated."]}
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = ["# DICOM pixel-integrity experiment", "",
              "This is a separate pixel-data experiment; it does not modify or merge the metadata-only experiment results.", "",
              f"- Clean input files: {len(baseline)}", f"- Attacks: {', '.join(PIXEL_ATTACKS)}",
              f"- Selected study fraction: {attack_fraction:.0%}", f"- Seed: {seed}", "",
              "Edits operate on stored integer PixelData values without applying RescaleSlope/RescaleIntercept.", "",
              "## Hash-based detection results", pd.DataFrame(metric_rows).to_markdown(index=False), "",
              "## Pixel difference summary", difference_summary.to_markdown(index=False), "",
              "## Reference-based localization check", pd.DataFrame(localization_rows).to_markdown(index=False),
              "These localization scores threshold the clean-reference difference map at nonzero difference and compare it with the saved synthetic edit mask. They verify edit-mask and difference-map alignment; they are not no-reference detector performance.", "",
              "Difference heatmaps compare the clean reference, modified copy, and absolute stored-pixel difference. Per-image scaling is used for visualization; use the CSV values for numeric comparison.", "",
              "## Example heatmaps", *[f"![{Path(name).stem}](<{name}>)" for name in sorted({r["heatmap"] for r in predictions if r["heatmap"]})], "",
              "## Interpretation", "- Pixel SHA-256 is a known-original integrity comparator; mismatch detects changed PixelData bytes relative to the protected baseline.",
              "- Metadata SHA-256 is included as a negative control and should remain unchanged for pixel-only edits.",
              "- These synthetic edits do not establish clinical relevance, attack realism, image-anomaly detection, or localization.", "",
              "## Limitations", *[f"- {x}" for x in config["limitations"]]]
    (output / "PIXEL_EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return pd.DataFrame(metric_rows)


def run_combined_integrity_experiment(input_dir, output_dir, metadata_attacks, pixel_attacks,
                                     attack_fraction=0.20, seed=42):
    """Evaluate paired metadata+pixel edits against clean metadata and pixel references."""
    from .attacks import ATTACKS, create_attack

    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    invalid_meta = sorted(set(metadata_attacks) - set(ATTACKS))
    invalid_pixel = sorted(set(pixel_attacks) - set(PIXEL_ATTACKS))
    if invalid_meta or invalid_pixel:
        raise ValueError(f"Unsupported attacks: metadata={invalid_meta}, pixel={invalid_pixel}")
    if "study_uid_inconsistency" in metadata_attacks:
        raise ValueError("Combined experiment currently excludes study_uid_inconsistency because changing StudyInstanceUID corrupts grouping identity.")
    output.mkdir(parents=True, exist_ok=True)
    clean_files = find_dicom_files(source)
    baseline = {}
    for path in clean_files:
        ds = read_dataset(path, stop_before_pixels=False)
        baseline[relative_key(path, source)] = {
            "pixel_sha256": pixel_sha256(ds), "metadata_sha256": metadata_sha256(ds),
            "study_group": hashlib.sha256(str(getattr(ds, "StudyInstanceUID", "")).encode()).hexdigest()[:16]}
    if not baseline:
        raise ValueError(f"No readable DICOM files found under {source}")

    prediction_rows, metric_rows = [], []
    for meta_attack in metadata_attacks:
        metadata_source = output / "metadata_inputs" / meta_attack
        meta_log = create_attack(source, metadata_source, meta_attack, attack_fraction, seed)
        for pixel_attack in pixel_attacks:
            condition = f"{meta_attack}__{pixel_attack}"
            combined_dir = output / "combined_attacks" / condition
            # Use an independently seeded pixel-study sample so the two layers
            # can produce metadata-only, pixel-only, and overlapping cases.
            pixel_seed = seed + 2
            pixel_log = create_pixel_attack(metadata_source, combined_dir, pixel_attack,
                                            attack_fraction, pixel_seed)
            rows = []
            for path in find_dicom_files(combined_dir):
                key = relative_key(path, combined_dir)
                ds = read_dataset(path, stop_before_pixels=False)
                meta_label = int(meta_log["files"].get(key, {}).get("label", 0))
                pixel_label = int(pixel_log["files"].get(key, {}).get("label", 0))
                row = {"condition": condition, "metadata_attack": meta_attack,
                       "pixel_attack": pixel_attack, "file": key,
                       "study_group": baseline[key]["study_group"],
                       "metadata_label": meta_label, "pixel_label": pixel_label,
                       "ground_truth": int(meta_label or pixel_label),
                       "metadata_hash_flag": int(metadata_sha256(ds) != baseline[key]["metadata_sha256"]),
                       "pixel_hash_flag": int(pixel_sha256(ds) != baseline[key]["pixel_sha256"])}
                row["combined_hash_flag"] = int(row["metadata_hash_flag"] or row["pixel_hash_flag"])
                rows.append(row)
                prediction_rows.append(row)
            frame = pd.DataFrame(rows)
            methods = (("Metadata SHA-256 reference", "metadata_hash_flag"),
                       ("PixelData SHA-256 reference", "pixel_hash_flag"),
                       ("Metadata OR Pixel SHA-256", "combined_hash_flag"))
            for method, column in methods:
                metric_rows.append({"condition": condition, "level": "file", "method": method,
                                    **_binary_metrics(frame.ground_truth, frame[column])})
                study = frame.groupby("study_group", as_index=False).agg(
                    ground_truth=("ground_truth", "max"), prediction=(column, "max"))
                metric_rows.append({"condition": condition, "level": "study", "method": method,
                                    **_binary_metrics(study.ground_truth, study.prediction)})

    pd.DataFrame(prediction_rows).to_csv(output / "combined_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "combined_metrics.csv", index=False)
    report = ["# Combined DICOM metadata and pixel-integrity experiment", "",
              f"- Clean files: {len(baseline)}", f"- Metadata attacks: {', '.join(metadata_attacks)}",
              f"- Pixel attacks: {', '.join(pixel_attacks)}", f"- Attack fraction: {attack_fraction:.1%}",
              f"- Metadata study-selection seed: {seed}",
              f"- Pixel study-selection seed: {seed + 2}", "",
              "The metadata and pixel attacks use independently seeded study selections. Their overlap is recorded in the prediction table through the separate metadata_label and pixel_label columns. Metadata and pixel hash references were generated from the original clean files. The OR result is a reference-based integrity comparator, not evidence of a learned multimodal detector.", "",
              metrics.to_markdown(index=False), "",
              "Study identity is taken from the clean reference, not from the potentially edited DICOM header. These controlled hash results are expected when exact clean references are available and do not establish detection without a trusted reference."]
    (output / "COMBINED_EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return metrics
