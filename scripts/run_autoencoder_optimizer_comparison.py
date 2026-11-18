"""Controlled optimizer comparison for the CPAISD metadata autoencoder."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.deep_anomaly import train_autoencoder, predict_autoencoder_with_explanations
from src.dicom_io import find_dicom_files, read_dataset, relative_key


OPTIMIZERS = ("adam", "rmsprop", "adamw", "sgd_momentum")
ATTACK_NAMES = (
    "modality", "patient_id", "pixel_spacing", "remove_protocol",
    "study_description", "study_uid_inconsistency",
)


def metrics(y_true, y_pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y_true), "n_positive": sum(y_true),
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def study_metrics(rows):
    groups = {}
    for row in rows:
        entry = groups.setdefault(row["study"], {"y": 0, "pred": 0})
        entry["y"] = max(entry["y"], row["label"])
        entry["pred"] = max(entry["pred"], row["prediction"])
    return metrics([x["y"] for x in groups.values()],
                   [x["pred"] for x in groups.values()])


def load_cases(test_root: Path, attacks_root: Path):
    cases = {}
    directories = [("clean", test_root)] + [
        (attack, attacks_root / attack) for attack in ATTACK_NAMES
    ]
    for name, directory in directories:
        files = find_dicom_files(directory)
        if not files:
            raise FileNotFoundError(f"No readable DICOM files found in {directory}")
        attack_log = {}
        if name != "clean":
            log_path = directory / "attack_log.json"
            if not log_path.exists():
                raise FileNotFoundError(f"Missing attack log: {log_path}")
            attack_log = json.loads(log_path.read_text(encoding="utf-8")).get("files", {})
        datasets, labels, studies = [], [], []
        for path in files:
            key = relative_key(path, directory).replace("\\", "/")
            datasets.append((key, read_dataset(path, stop_before_pixels=True)))
            labels.append(int(attack_log.get(key, {}).get("label", 0)))
            studies.append(Path(key).parts[0] if len(Path(key).parts) > 1 else key)
        cases[name] = {"datasets": datasets, "labels": labels, "studies": studies}
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path, default=Path("experiment_cpaisd/reports/autoencoder_optimizer_comparison"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir()

    print("Loading clean test and six attacked test conditions once...")
    cases = load_cases(args.test, args.attacks)
    attack_results, summary_results = [], []
    for optimizer in OPTIMIZERS:
        model_path = model_dir / f"metadata_autoencoder_{optimizer}.joblib"
        print(f"\nTraining {optimizer} for {args.epochs} epochs on clean training data...")
        started = time.perf_counter()
        bundle = train_autoencoder(
            args.train, model_path, validation_dir=args.validation,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_percentile=args.threshold_percentile,
            random_state=args.seed, max_train_files=0, max_validation_files=0,
            optimizer_name=optimizer,
        )
        elapsed = time.perf_counter() - started
        file_metrics, study_condition_metrics = [], []
        for condition, case in cases.items():
            predicted = predict_autoencoder_with_explanations(case["datasets"], model_path, top_n=1)
            preds = [int(predicted[key][0]) for key, _ in case["datasets"]]
            condition_rows = [{"study": study, "label": label, "prediction": prediction}
                              for study, label, prediction in zip(case["studies"], case["labels"], preds)]
            if condition == "clean":
                clean_file = metrics(case["labels"], preds)
                clean_study = study_metrics(condition_rows)
            else:
                fm, sm = metrics(case["labels"], preds), study_metrics(condition_rows)
                file_metrics.append(fm)
                study_condition_metrics.append(sm)
                attack_results.append({
                    "optimizer": optimizer, "condition": condition,
                    "threshold": bundle["threshold"], **fm,
                    "study_recall": sm["recall"], "study_fpr": sm["fpr"],
                    "study_f1": sm["f1"], "n_studies": sm["n"],
                })
        summary_results.append({
            "optimizer": optimizer, "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "momentum": 0.9 if optimizer == "sgd_momentum" else None,
            "weight_decay": 1e-4 if optimizer == "adamw" else 0.0,
            "threshold": bundle["threshold"],
            "training_files": bundle["training_files"],
            "validation_files": bundle["validation_files"],
            "training_and_preprocessing_seconds": elapsed,
            "macro_file_f1": sum(x["f1"] for x in file_metrics) / len(file_metrics),
            "macro_file_recall": sum(x["recall"] for x in file_metrics) / len(file_metrics),
            "macro_file_fpr": sum(x["fpr"] for x in file_metrics) / len(file_metrics),
            "macro_study_f1": sum(x["f1"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_recall": sum(x["recall"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_fpr": sum(x["fpr"] for x in study_condition_metrics) / len(study_condition_metrics),
            "clean_test_file_fpr": clean_file["fpr"],
            "clean_test_study_fpr": clean_study["fpr"],
        })
        result = summary_results[-1]
        print("  clean file FPR={:.4f}; attack macro file F1={:.4f}, recall={:.4f}, FPR={:.4f}; study macro F1={:.4f}".format(
            clean_file["fpr"], result["macro_file_f1"], result["macro_file_recall"],
            result["macro_file_fpr"], result["macro_study_f1"]))

    summary = pd.DataFrame(summary_results).sort_values(
        ["macro_study_f1", "macro_study_recall", "clean_test_study_fpr"],
        ascending=[False, False, True])
    summary.to_csv(args.output / "optimizer_summary.csv", index=False)
    pd.DataFrame(attack_results).to_csv(args.output / "per_attack_metrics.csv", index=False)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    config.update({
        "optimizers": list(OPTIMIZERS), "selection_metric": "macro_study_f1",
        "selection_note": "Exploratory single-seed comparison; the best row is a candidate, not a tuned optimum.",
        "data_protocol": "clean training fit; clean validation 99th-percentile threshold; untouched test and six existing attack conditions",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary saved:", args.output / "optimizer_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
