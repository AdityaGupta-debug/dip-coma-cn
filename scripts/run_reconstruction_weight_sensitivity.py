"""Evaluate plain MSE, 2x PixelSpacing-weighted MSE, and a weighted mixed loss."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.attacks import _modify
from src.deep_anomaly import train_autoencoder, predict_autoencoder_with_explanations
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.mixed_anomaly import train_mixed_autoencoder, predict_mixed_autoencoder
from run_autoencoder_optimizer_comparison import (
    ATTACK_NAMES, load_cases, metrics,
)


def validation_cases(validation_dir: Path, fraction=0.20, seed=42):
    """Create attack copies in memory; original validation DICOMs stay untouched."""
    paths = find_dicom_files(validation_dir)
    cached = []
    by_study = {}
    for path in paths:
        ds = read_dataset(path, stop_before_pixels=True)
        key = relative_key(path, validation_dir)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{path.parent.name}"
        series = str(getattr(ds, "SeriesInstanceUID", "")).strip() or f"MISSING:{study}"
        cached.append((key, ds, study, series))
        by_study.setdefault(study, []).append((key, ds, series))
    studies = sorted(by_study)
    if not studies:
        raise FileNotFoundError(f"No readable DICOM files found in {validation_dir}")
    selected_count = max(1, round(len(studies) * fraction))
    selected_studies = set(random.Random(seed).sample(studies, min(selected_count, len(studies))))
    clean = [(key, ds) for key, ds, _, _ in cached]
    cases = {"clean": {"datasets": clean, "labels": [0] * len(clean)}}
    for attack in ATTACK_NAMES:
        changed_keys = set()
        if attack == "study_uid_inconsistency":
            seen_series = set()
            for study in sorted(selected_studies):
                for key, _ds, series in by_study[study]:
                    if series not in seen_series:
                        changed_keys.add(key)
                        seen_series.add(series)
        else:
            changed_keys = {key for study in selected_studies
                            for key, _ds, _series in by_study[study]}
        datasets, labels = [], []
        for key, ds, _study, _series in cached:
            attacked = copy.deepcopy(ds)
            changed = _modify(attacked, attack) if key in changed_keys else None
            label = int(changed is not None)
            datasets.append((key, attacked if label else ds))
            labels.append(label)
        cases[attack] = {"datasets": datasets, "labels": labels}
    return cases


def evaluate_model(predict_fn, model_path, cases):
    rows = {}
    recalls = []
    for condition, case in cases.items():
        prediction_records = predict_fn(case["datasets"], model_path, top_n=1)
        predicted = [int(prediction_records[key][0]) for key, _ in case["datasets"]]
        result = metrics(case["labels"], predicted)
        rows[condition] = result
        if condition != "clean":
            recalls.append(result["recall"])
    return {
        "edit_macro_recall": sum(recalls) / len(recalls),
        "clean_false_alarm_rate": rows["clean"]["fpr"],
        "clean_false_alarms": rows["clean"]["fp"],
        "clean_files": rows["clean"]["n"],
        "per_condition": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_cpaisd/reports/reconstruction_weight_sensitivity"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for label, directory in (("training", args.train), ("validation", args.validation),
                             ("test", args.test)):
        if not directory.is_dir() or not find_dicom_files(directory):
            raise FileNotFoundError(f"No readable DICOM files in {label} directory: {directory}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir()

    print("Loading validation edits in memory and existing test attack folders...", flush=True)
    val_cases = validation_cases(args.validation, args.attack_fraction, args.seed)
    test_cases = load_cases(args.test, args.attacks)
    specs = (
        ("plain_mse", "Plain MSE", train_autoencoder,
         predict_autoencoder_with_explanations, "mse", {}, {}),
        ("weighted_mse", "Weighted MSE (PixelSpacing x2)", train_autoencoder,
         predict_autoencoder_with_explanations, "mse",
         {"PixelSpacing_0": 2.0, "PixelSpacing_1": 2.0}, {}),
        ("weighted_mixed", "Mixed loss (0.20 / 0.20 / 0.60)",
         train_mixed_autoencoder, predict_mixed_autoencoder, None, {},
         {"numeric_weight": 0.20, "categorical_weight": 0.20,
          "binary_weight": 0.60}),
    )

    summaries, per_attack_rows = [], []
    for model_id, label, train_fn, predict_fn, loss_name, feature_weights, extra in specs:
        model_path = model_dir / f"{model_id}.joblib"
        print(f"\nTraining {label}...", flush=True)
        start = time.perf_counter()
        kwargs = {}
        if model_id != "weighted_mixed":
            kwargs.update({"optimizer_name": "rmsprop", "reconstruction_loss": loss_name,
                           "feature_loss_weights": feature_weights})
        kwargs.update(extra)
        bundle = train_fn(
            args.train, model_path, validation_dir=args.validation,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_percentile=args.threshold_percentile,
            random_state=args.seed, max_train_files=0, max_validation_files=0,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        validation_result = evaluate_model(predict_fn, model_path, val_cases)
        test_result = evaluate_model(predict_fn, model_path, test_cases)
        summaries.append({
            "setting": label, "threshold": bundle["threshold"],
            "validation_edit_macro_recall": validation_result["edit_macro_recall"],
            "validation_clean_false_alarm_rate": validation_result["clean_false_alarm_rate"],
            "validation_clean_false_alarms": validation_result["clean_false_alarms"],
            "validation_clean_files": validation_result["clean_files"],
            "test_edit_macro_recall": test_result["edit_macro_recall"],
            "test_clean_false_alarm_rate": test_result["clean_false_alarm_rate"],
            "test_clean_false_alarms": test_result["clean_false_alarms"],
            "test_clean_files": test_result["clean_files"],
            "epochs": args.epochs, "training_seconds": elapsed,
        })
        for split, result in (("validation", validation_result), ("test", test_result)):
            for condition, values in result["per_condition"].items():
                per_attack_rows.append({"setting": label, "split": split,
                                        "condition": condition, **values})
        print("  validation recall/FAR={:.3f}/{:.3f}; test recall/FAR={:.3f}/{:.3f}".format(
            validation_result["edit_macro_recall"],
            validation_result["clean_false_alarm_rate"],
            test_result["edit_macro_recall"],
            test_result["clean_false_alarm_rate"]), flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(per_attack_rows).to_csv(args.output / "per_attack_metrics.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "settings": [spec[1] for spec in specs],
        "weighted_mse_features": {"PixelSpacing_0": 2.0, "PixelSpacing_1": 2.0},
        "mixed_loss_weights": {"numeric": 0.20, "categorical": 0.20, "binary": 0.60},
        "thresholding": f"{args.threshold_percentile}th percentile of clean validation reconstruction scores for each model",
        "validation_attack_generation": "in-memory copies, 20% of studies, seed 42 by default; file-level macro recall over six scripted conditions",
        "test_attack_data": "existing six attack folders; test labels used only for evaluation",
        "notes": ["The validation clean false-alarm rate is measured on the same clean validation files used for threshold calibration, so it is a calibration-set statistic, not an independent estimate.",
                  "The screenshot values were not found in this repository; this run reports newly generated results under the documented protocol."]
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(
        "# Reconstruction-weight sensitivity comparison\n\n"
        "The displayed screenshot's exact result values were not found in the repository. "
        "This run compares its three stated settings using the current CPAISD pipeline: "
        "plain MSE; MSE with both PixelSpacing feature errors weighted 2x; and a mixed "
        "numeric/categorical/binary objective weighted 0.20/0.20/0.60. All models use "
        "RMSprop, the same folds and training budget, and independent clean-validation "
        "99th-percentile thresholds. Validation edits are generated in memory.\n\n"
        "Validation clean false-alarm rate is measured on the calibration files themselves; "
        "test clean false alarms are the held-out comparison. Edit recall is macro-averaged "
        "over the six scripted attack conditions. These results are exploratory and do not "
        "validate the screenshot's source values.\n\n## Summary\n\n"
        + summary.to_string(index=False) + "\n\nPer-attack metrics: `per_attack_metrics.csv`.\n",
        encoding="utf-8",
    )
    print("\nSummary saved:", args.output / "summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
