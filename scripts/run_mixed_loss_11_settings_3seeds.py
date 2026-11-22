"""Three-seed, study-disjoint 11-setting reconstruction-loss sweep."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
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
from run_autoencoder_optimizer_comparison import ATTACK_NAMES, load_cases, metrics


WEIGHT_GRID = (
    (0.20, 0.20, 0.60),
    (0.60, 0.20, 0.20),
    (0.20, 0.60, 0.20),
    (1 / 3, 1 / 3, 1 / 3),
    (0.50, 0.25, 0.25),
    (0.25, 0.50, 0.25),
    (0.25, 0.25, 0.50),
    (0.40, 0.20, 0.40),
    (0.20, 0.40, 0.40),
)
SEEDS = (42, 43, 44)


def settings():
    result = [
        {"name": "plain_mse", "kind": "mse", "weights": None,
         "feature_weights": {}},
        {"name": "weighted_mse_pixelspacing_x2", "kind": "mse", "weights": None,
         "feature_weights": {"PixelSpacing_0": 2.0, "PixelSpacing_1": 2.0}},
    ]
    for numeric, categorical, presence in WEIGHT_GRID:
        result.append({
            "name": f"mixed_n{numeric:.2f}_c{categorical:.2f}_p{presence:.2f}",
            "kind": "mixed",
            "weights": {"numeric_weight": numeric,
                        "categorical_weight": categorical,
                        "binary_weight": presence},
            "feature_weights": {},
        })
    return result


def split_validation(validation_dir: Path, seed: int):
    records = []
    by_study = {}
    for path in find_dicom_files(validation_dir):
        ds = read_dataset(path, stop_before_pixels=True)
        key = relative_key(path, validation_dir)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{path.parent.name}"
        series = str(getattr(ds, "SeriesInstanceUID", "")).strip() or f"MISSING:{study}"
        record = {"key": key, "path": path, "dataset": ds,
                  "study": study, "series": series}
        records.append(record)
        by_study.setdefault(study, []).append(record)
    study_ids = sorted(by_study)
    if len(study_ids) < 4:
        raise ValueError("Need at least four validation studies for disjoint calibration/evaluation")
    random.Random(seed).shuffle(study_ids)
    midpoint = len(study_ids) // 2
    calibration_ids = set(study_ids[:midpoint])
    evaluation_ids = set(study_ids[midpoint:])
    calibration_paths = [record["path"] for record in records
                         if record["study"] in calibration_ids]
    evaluation_records = [record for record in records
                          if record["study"] in evaluation_ids]
    return calibration_paths, evaluation_records, len(calibration_ids), len(evaluation_ids)


def validation_evaluation_cases(records):
    clean_datasets = [(r["key"], r["dataset"]) for r in records]
    cases = {"clean": {"datasets": clean_datasets,
                       "labels": [0] * len(clean_datasets)}}
    for attack in ATTACK_NAMES:
        if attack == "study_uid_inconsistency":
            changed_keys, seen_series = set(), set()
            for record in records:
                if record["series"] not in seen_series:
                    changed_keys.add(record["key"])
                    seen_series.add(record["series"])
        else:
            changed_keys = {record["key"] for record in records}
        datasets, labels = [], []
        for record in records:
            attacked = copy.deepcopy(record["dataset"])
            changed = _modify(attacked, attack) if record["key"] in changed_keys else None
            label = int(changed is not None)
            datasets.append((record["key"], attacked if label else record["dataset"]))
            labels.append(label)
        cases[attack] = {"datasets": datasets, "labels": labels}
    return cases


def evaluate_model(model_path, predict_fn, cases):
    per_condition = {}
    for condition, case in cases.items():
        predictions = predict_fn(case["datasets"], model_path, top_n=1)
        predicted = [int(predictions[key][0]) for key, _ in case["datasets"]]
        per_condition[condition] = metrics(case["labels"], predicted)
    edit_names = [name for name in ATTACK_NAMES if name != "study_uid_inconsistency"]
    return {
        "clean_false_alarm_rate": per_condition["clean"]["fpr"],
        "clean_false_alarms": per_condition["clean"]["fp"],
        "clean_files": per_condition["clean"]["n"],
        "edit_macro_recall_five": float(np.mean([per_condition[n]["recall"] for n in edit_names])),
        "edit_macro_recall_six": float(np.mean([per_condition[n]["recall"] for n in ATTACK_NAMES])),
        "uid_positive_control_recall": per_condition["study_uid_inconsistency"]["recall"],
        "per_condition": per_condition,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_cpaisd/reports/mixed_loss_11_settings_3seeds"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=98.0,
                        help="Calibration quantile corresponding to a 2% nominal clean-tail target.")
    parser.add_argument("--target-validation-fpr", type=float, default=0.02)
    args = parser.parse_args()

    for label, directory in (("training", args.train), ("validation", args.validation),
                             ("test", args.test)):
        if not directory.is_dir() or not find_dicom_files(directory):
            raise FileNotFoundError(f"No readable DICOM files in {label} directory: {directory}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    model_root = args.output / "models"
    model_root.mkdir()

    print("Loading the untouched test cases once...", flush=True)
    test_cases = load_cases(args.test, args.attacks)
    per_seed_rows, attack_rows = [], []
    grid = settings()
    total_runs = len(grid) * len(SEEDS)
    run_index = 0
    for seed in SEEDS:
        calibration_paths, validation_records, calibration_studies, evaluation_studies = split_validation(
            args.validation, seed
        )
        validation_cases = validation_evaluation_cases(validation_records)
        for setting in grid:
            run_index += 1
            model_id = f"{setting['name']}_seed{seed}"
            model_path = model_root / f"{model_id}.joblib"
            kind = setting["kind"]
            print(f"\n[{run_index}/{total_runs}] {setting['name']} seed={seed}", flush=True)
            train_fn = train_autoencoder if kind == "mse" else train_mixed_autoencoder
            predict_fn = (predict_autoencoder_with_explanations if kind == "mse"
                          else predict_mixed_autoencoder)
            kwargs = ({"optimizer_name": "rmsprop", "reconstruction_loss": "mse",
                       "feature_loss_weights": setting["feature_weights"]}
                      if kind == "mse" else setting["weights"])
            started = time.perf_counter()
            bundle = train_fn(
                args.train, model_path, validation_dir=args.validation,
                validation_paths=calibration_paths,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                threshold_percentile=args.threshold_percentile,
                random_state=seed, max_train_files=0, max_validation_files=0,
                **kwargs,
            )
            training_seconds = time.perf_counter() - started
            validation = evaluate_model(model_path, predict_fn, validation_cases)
            test = evaluate_model(model_path, predict_fn, test_cases)
            row = {
                "setting": setting["name"], "kind": kind, "seed": seed,
                "threshold": bundle["threshold"],
                "validation_calibration_studies": calibration_studies,
                "validation_evaluation_studies": evaluation_studies,
                "validation_edit_recall_five": validation["edit_macro_recall_five"],
                "validation_edit_recall_six": validation["edit_macro_recall_six"],
                "validation_clean_fpr": validation["clean_false_alarm_rate"],
                "validation_clean_false_alarms": validation["clean_false_alarms"],
                "validation_clean_files": validation["clean_files"],
                "test_edit_recall_five": test["edit_macro_recall_five"],
                "test_edit_recall_six": test["edit_macro_recall_six"],
                "test_uid_positive_control_recall": test["uid_positive_control_recall"],
                "test_clean_fpr": test["clean_false_alarm_rate"],
                "test_clean_false_alarms": test["clean_false_alarms"],
                "test_clean_files": test["clean_files"],
                "training_seconds": training_seconds,
                "numeric_weight": setting["weights"].get("numeric_weight") if setting["weights"] else None,
                "categorical_weight": setting["weights"].get("categorical_weight") if setting["weights"] else None,
                "presence_weight": setting["weights"].get("binary_weight") if setting["weights"] else None,
            }
            per_seed_rows.append(row)
            for split_name, evaluation in (("validation", validation), ("test", test)):
                for condition, values in evaluation["per_condition"].items():
                    attack_rows.append({"setting": setting["name"], "seed": seed,
                                        "split": split_name, "condition": condition,
                                        **values})
            print("  val recall/FPR={:.3f}/{:.3f}; test recall/FPR={:.3f}/{:.3f}".format(
                row["validation_edit_recall_five"], row["validation_clean_fpr"],
                row["test_edit_recall_five"], row["test_clean_fpr"]), flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(args.output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(args.output / "per_attack_metrics.csv", index=False)
    metric_cols = [
        "validation_edit_recall_five", "validation_clean_fpr",
        "test_edit_recall_five", "test_edit_recall_six",
        "test_uid_positive_control_recall", "test_clean_fpr",
    ]
    summary = per_seed.groupby(["setting", "kind"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
                       for col in summary.columns]
    summary = summary.reset_index(drop=True)
    pass_rates = per_seed.assign(
        validation_target_met=per_seed["validation_clean_fpr"] <= args.target_validation_fpr
    ).groupby("setting", as_index=False)["validation_target_met"].sum()
    pass_rates["validation_target_seed_count"] = pass_rates.pop("validation_target_met")
    summary = summary.merge(pass_rates, on="setting", how="left")
    summary.to_csv(args.output / "summary.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "seeds": list(SEEDS),
        "settings": grid,
        "setting_count": len(grid),
        "model_run_count": total_runs,
        "validation_protocol": "five validation studies calibrate each threshold at the 98th percentile; five disjoint studies evaluate clean FPR and in-memory synthetic edit recall",
        "attack_protocol": "five whole-study metadata edits are macro-averaged; UID inconsistency is reported separately and in six-condition recall",
        "setting_selection_rule": "settings meeting the held-out validation clean FPR target are eligible; choose by validation five-attack macro recall, never by test metrics",
        "interpretation": "This is a newly specified sweep because the screenshot did not provide all 11 settings or the original configuration. It cannot verify the screenshot's reported results.",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    eligible = summary.loc[summary["validation_target_seed_count"] == len(SEEDS)]
    (args.output / "REPORT.md").write_text(
        "# Three-seed reconstruction-weight sweep\n\n"
        "This run used 11 declared settings (plain MSE, 2x PixelSpacing-weighted MSE, "
        "and nine mixed-loss weight triples) and seeds 42, 43, and 44. For each seed, "
        "five validation studies calibrated the threshold at the 98th percentile; the "
        "other five measured validation clean false alarms and synthetic edit recall. "
        "The target was held-out validation clean FPR <= 2%. Test metrics are reported "
        "for comparison but were not used to select a setting. The screenshot's complete "
        "11-setting list was unavailable, so this is a transparent re-run protocol, not "
        "an exact reproduction of its numbers.\n\n"
        f"Settings meeting the <=2% validation FPR target for all three seeds: {len(eligible)}.\n\n"
        "## Mean and standard deviation across seeds\n\n"
        + summary.to_string(index=False) + "\n\n"
        "Per-seed results: `per_seed_metrics.csv`. Per-attack results: `per_attack_metrics.csv`.\n",
        encoding="utf-8",
    )
    print("\nSummary saved:", args.output / "summary.csv")
    print(summary.to_string(index=False))
    print(f"\nSettings meeting the <=2% validation FPR target in all seeds: {len(eligible)} / {len(grid)}")


if __name__ == "__main__":
    main()
