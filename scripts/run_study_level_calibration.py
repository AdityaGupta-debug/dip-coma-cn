"""Calibrate metadata anomaly scores at the study unit using clean validation studies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection import detect_directory
from src.integrity import create_baseline


def _study_scores(frame: pd.DataFrame, has_labels: bool):
    agg = {"isolation_severity": ("isolation_forest_score", lambda x: float((-x).max())),
           "autoencoder_severity": ("autoencoder_reconstruction_error", "max"),
           "file_count": ("file", "count")}
    if has_labels:
        agg["ground_truth"] = ("ground_truth", "max")
    return frame.groupby("study_group", as_index=False).agg(**agg)


def _conformal_p(calibration, score):
    calibration = np.asarray(calibration, dtype=float)
    return (1 + int(np.count_nonzero(calibration >= float(score)))) / (len(calibration) + 1)


def _binary_summary(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    positives, negatives = labels == 1, labels == 0
    tp = int(np.sum(positives & (predictions == 1)))
    fp = int(np.sum(negatives & (predictions == 1)))
    return {"n_studies": len(labels), "positive_studies": int(positives.sum()),
            "recall": tp / int(positives.sum()) if positives.any() else None,
            "false_positive_rate": fp / int(negatives.sum()) if negatives.any() else None,
            "tp": tp, "fp": fp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--test-baseline", required=True)
    parser.add_argument("--metadata-model", required=True)
    parser.add_argument("--autoencoder-model", required=True)
    parser.add_argument("--attack-predictions-root", required=True,
                        help="Root containing saved file_predictions_with_ground_truth.csv files")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    val_baseline = output / "validation_baseline.json"
    create_baseline(args.validation, val_baseline, include_pixel_hash=False)
    val_files = detect_directory(args.validation, val_baseline, args.metadata_model,
                                 output / "validation_file_predictions.csv", args.autoencoder_model)
    val_studies = _study_scores(val_files, has_labels=False)
    calibration = {"isolation_forest": val_studies.isolation_severity.to_list(),
                  "autoencoder": val_studies.autoencoder_severity.to_list()}
    test_files = detect_directory(args.test, args.test_baseline, args.metadata_model,
                                  output / "clean_test_file_predictions.csv", args.autoencoder_model)
    clean_studies = _study_scores(test_files.assign(ground_truth=0), has_labels=True)
    all_results = []

    def score_studies(name, studies):
        for alpha in args.alphas:
            for detector, column in [("Isolation Forest", "isolation_severity"),
                                     ("Autoencoder", "autoencoder_severity")]:
                pvalues = [_conformal_p(calibration["isolation_forest" if detector == "Isolation Forest" else "autoencoder"], x)
                           for x in studies[column]]
                pred = (np.asarray(pvalues) <= alpha).astype(int)
                row = {"condition": name, "alpha": alpha, "detector": detector,
                       **_binary_summary(studies.ground_truth, pred),
                       "validation_study_count": len(val_studies),
                       "minimum_attainable_conformal_p": 1 / (len(val_studies) + 1)}
                all_results.append(row)

    score_studies("clean_test", clean_studies)
    prediction_root = Path(args.attack_predictions_root).resolve()
    for pred_path in sorted(prediction_root.rglob("file_predictions_with_ground_truth.csv")):
        frame = pd.read_csv(pred_path)
        if not {"study_group", "ground_truth", "file", "isolation_forest_score",
                "autoencoder_reconstruction_error"}.issubset(frame.columns):
            continue
        studies = _study_scores(frame, has_labels=True)
        condition = pred_path.parent.relative_to(prediction_root).as_posix()
        score_studies(condition, studies)

    result = pd.DataFrame(all_results)
    result.to_csv(output / "study_level_calibrated_metrics.csv", index=False)
    val_studies.to_csv(output / "validation_study_calibration_scores.csv", index=False)
    config = {"alphas": args.alphas, "calibration_unit": "study",
              "validation_studies": len(val_studies),
              "minimum_conformal_p": 1 / (len(val_studies) + 1),
              "conformal_p_value": "(1 + number of clean validation study scores >= test score)/(n_validation_studies + 1)",
              "interpretation": "With ten validation studies, the smallest attainable p-value is 1/11; nominal alpha values below this cannot flag a study. Results are coarse and cohort-specific."}
    (output / "study_level_calibration_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = ["# Study-level metadata anomaly calibration", "",
              f"Clean validation units: {len(val_studies)} studies. Minimum attainable conformal p-value: {1/(len(val_studies)+1):.4f}.",
              "Scores aggregate slices using the maximum anomaly severity within each baseline-keyed study. Conformal p-values are computed from clean validation-study scores; they do not use test labels.",
              "", "## Results", result.to_markdown(index=False), "",
              "The validation set is small at the study unit. Discrete p-values limit threshold resolution, and the reported test metrics are descriptive for CPAISD, not population guarantees.", ""]
    (output / "STUDY_LEVEL_CALIBRATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"\nStudy-level calibration complete: {output}")


if __name__ == "__main__":
    main()
