"""Validation-calibrated late fusion of kNN metadata and PatchCore scores.

Both branch prediction folders must have been evaluated on the same clean test
files and the same paired attack folders. This script does not fit either
branch and does not use attack labels to select thresholds or weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _load(path):
    frame = pd.read_csv(path)
    required = {"file", "score", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _merge(metadata, image, key_columns):
    merged = metadata.merge(image, on=key_columns, how="inner", suffixes=("_metadata", "_image"),
                            validate="one_to_one")
    if len(merged) != len(metadata) or len(merged) != len(image):
        raise ValueError("Branch predictions do not contain the same unique file/attack keys")
    if "ground_truth_metadata" in merged and "ground_truth_image" in merged:
        if not np.array_equal(merged.ground_truth_metadata, merged.ground_truth_image):
            raise ValueError("Metadata and image branch attack labels do not match")
    for label in ("metadata_label", "pixel_label"):
        left, right = f"{label}_metadata", f"{label}_image"
        if left in merged and right in merged:
            if not np.array_equal(merged[left], merged[right]):
                raise ValueError(f"Branch files disagree on {label}")
            merged[label] = merged[left]
    if "ground_truth_metadata" in merged:
        merged["ground_truth"] = merged["ground_truth_metadata"]
    elif "ground_truth_image" in merged:
        merged["ground_truth"] = merged["ground_truth_image"]
    return merged


def _metrics(labels, predictions, scores):
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


def _add_fusion(frame, metadata_center, metadata_scale, image_center, image_scale,
                fusion_threshold):
    frame = frame.copy()
    frame["metadata_z"] = (frame.score_metadata - metadata_center) / metadata_scale
    frame["image_z"] = (frame.score_image - image_center) / image_scale
    # Equal weights are fixed in advance; the validation fold selects only the alarm cutoff.
    frame["fusion_score"] = 0.5 * frame.metadata_z + 0.5 * frame.image_z
    frame["weighted_score_flag"] = (frame.fusion_score > fusion_threshold).astype(int)
    frame["or_score"] = np.maximum(frame.metadata_z, frame.image_z)
    frame["and_score"] = np.minimum(frame.metadata_z, frame.image_z)
    frame["or_flag"] = ((frame.prediction_metadata.astype(int) == 1) |
                         (frame.prediction_image.astype(int) == 1)).astype(int)
    frame["and_flag"] = ((frame.prediction_metadata.astype(int) == 1) &
                          (frame.prediction_image.astype(int) == 1)).astype(int)
    return frame


def _evaluate(frame, group_columns, attack_name):
    rows = []
    methods = [
        ("kNN metadata", "prediction_metadata", "metadata_z"),
        ("PatchCore image", "prediction_image", "image_z"),
        ("Equal-weight score fusion", "weighted_score_flag", "fusion_score"),
        ("OR decision fusion", "or_flag", "or_score"),
        ("AND decision fusion", "and_flag", "and_score"),
    ]
    aggregations = {column: (column, "max") for _, column, _ in methods}
    aggregations.update({score: (score, "max") for _, _, score in methods})
    for target in ("ground_truth", "metadata_label", "pixel_label"):
        for unit, grouped in [("file", frame), ("study", frame.groupby(group_columns, as_index=False).agg(
                **{target: (target, "max")}, **aggregations))]:
            for method, pred_col, score_col in methods:
                rows.append({"attack": attack_name, "target": target, "unit": unit,
                             "method": method,
                             **_metrics(grouped[target], grouped[pred_col], grouped[score_col])})
    return rows


def run(args):
    if not 0.0 < args.threshold_percentile < 100.0:
        raise ValueError("threshold-percentile must be between 0 and 100")
    metadata_dir, image_dir = Path(args.metadata_results), Path(args.patchcore_results)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    m_val = _load(metadata_dir / "validation_calibration_predictions.csv")
    i_val = _load(image_dir / "validation_calibration_predictions.csv")
    val = _merge(m_val, i_val, ["file"])
    m_center, m_scale = float(m_val.score.median()), float(m_val.score.quantile(.75) - m_val.score.quantile(.25))
    i_center, i_scale = float(i_val.score.median()), float(i_val.score.quantile(.75) - i_val.score.quantile(.25))
    m_scale = m_scale if m_scale > 1e-12 else float(m_val.score.std()) or 1.0
    i_scale = i_scale if i_scale > 1e-12 else float(i_val.score.std()) or 1.0
    val = _add_fusion(val, m_center, m_scale, i_center, i_scale, float("inf"))
    threshold = float(np.quantile(val.fusion_score, args.threshold_percentile / 100.0))

    results, predictions = [], []
    clean = _merge(_load(metadata_dir / "clean_test_predictions.csv"),
                   _load(image_dir / "clean_test_predictions.csv"), ["file"])
    clean["study_group"] = clean.get("study_group_metadata", clean.get("study_group_image", "unknown"))
    clean = _add_fusion(clean, m_center, m_scale, i_center, i_scale, threshold)
    clean["ground_truth"] = 0
    clean.to_csv(output / "clean_test_fusion_predictions.csv", index=False)
    results.extend(_evaluate(clean, ["study_group"], "clean_test"))

    m_attack_files = {p.name for p in metadata_dir.glob("*_predictions.csv")
                      if p.name not in {"clean_test_predictions.csv", "validation_calibration_predictions.csv", "attack_predictions.csv"}}
    i_attack_files = {p.name for p in image_dir.glob("*_predictions.csv")
                      if p.name not in {"clean_test_predictions.csv", "validation_calibration_predictions.csv", "attack_predictions.csv"}}
    if m_attack_files != i_attack_files:
        raise ValueError("Metadata and PatchCore result directories have different per-attack prediction files")
    for filename in sorted(m_attack_files):
        m = _load(metadata_dir / filename)
        im = _load(image_dir / filename)
        merged = _merge(m, im, ["file"])
        if "attack_metadata" in merged and "attack_image" in merged:
            if not np.array_equal(merged.attack_metadata, merged.attack_image):
                raise ValueError(f"Attack family mismatch for {filename}")
            merged["attack"] = merged.attack_metadata
        else:
            merged["attack"] = Path(filename).stem.replace("_predictions", "")
        study_column = "study_group_metadata" if "study_group_metadata" in merged else "study_group_image"
        merged["study_group"] = merged[study_column]
        merged = _add_fusion(merged, m_center, m_scale, i_center, i_scale, threshold)
        merged.to_csv(output / f"fusion_{filename}", index=False)
        for attack, group in merged.groupby("attack"):
            results.extend(_evaluate(group, ["study_group"], str(attack)))
            predictions.append(group)
    if not predictions:
        raise ValueError("No matching per-attack prediction CSVs were found")
    pd.concat(predictions, ignore_index=True).to_csv(output / "all_attack_fusion_predictions.csv", index=False)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output / "fusion_metrics.csv", index=False)
    (output / "fusion_configuration.json").write_text(json.dumps({
        "branches": ["clean-training metadata kNN", "clean-training PatchCore"],
        "score_normalization": "branch score centered by clean validation median and scaled by clean validation IQR (std fallback)",
        "score_weights": {"metadata": 0.5, "image": 0.5},
        "score_threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "decision_fusions": ["OR", "AND"],
        "validation_attack_labels_used_for_selection": False,
        "paired_inputs_required": True,
    }, indent=2), encoding="utf-8")
    report = ["# Late-fusion experiment", "",
              "Metadata kNN and PatchCore are evaluated independently and then fused on matching file/attack inputs.",
              "Equal weights were fixed in advance. Score centering, scaling, and the score-fusion alarm cutoff use clean validation scores only.",
              "OR and AND use the branch flags calibrated by their respective model runs.", "",
              "## Results", result_frame.to_markdown(index=False), "",
              "## Interpretation limits", "",
              "- Fusion is meaningful only when both branch files refer to the same image copy and attack condition.",
              "- The validation fold contains ten studies, so study-level threshold calibration is coarse.",
              "- The combined output is anomaly detection, not an automated remediation or clinical defense.", ""]
    (output / "LATE_FUSION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(result_frame.to_string(index=False))
    print(f"\nSaved late-fusion results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-results", required=True,
                        help="Output folder from run_knn_metadata_experiment.py")
    parser.add_argument("--patchcore-results", required=True,
                        help="Output folder from run_patchcore_experiment.py")
    parser.add_argument("--output", required=True, help="Fresh output folder")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                        help="Clean validation percentile for equal-weight score fusion")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
