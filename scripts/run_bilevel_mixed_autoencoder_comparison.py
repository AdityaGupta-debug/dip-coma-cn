"""Compare MSE, fixed mixed, and Adam-bilevel mixed metadata autoencoders."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.deep_anomaly import train_autoencoder, predict_autoencoder_with_explanations
from src.dicom_io import find_dicom_files
from src.mixed_anomaly import (
    train_mixed_autoencoder, predict_mixed_autoencoder,
    train_bilevel_mixed_autoencoder,
)
from run_autoencoder_optimizer_comparison import load_cases, metrics, study_metrics


def _evaluate_model(display_name, bundle, model_path, predict_fn, cases,
                    training_seconds):
    per_file, per_study, rows = [], [], []
    clean_file = clean_study = None
    for condition, case in cases.items():
        predictions = predict_fn(case["datasets"], model_path, top_n=1)
        preds = [int(predictions[key][0]) for key, _ in case["datasets"]]
        study_rows = [{"study": study, "label": label, "prediction": pred}
                      for study, label, pred in zip(case["studies"], case["labels"], preds)]
        file_metric = metrics(case["labels"], preds)
        study_metric = study_metrics(study_rows)
        if condition == "clean":
            clean_file, clean_study = file_metric, study_metric
            continue
        per_file.append(file_metric)
        per_study.append(study_metric)
        rows.append({
            "model": display_name, "condition": condition,
            "threshold": bundle["threshold"], **file_metric,
            "study_recall": study_metric["recall"],
            "study_fpr": study_metric["fpr"],
            "study_f1": study_metric["f1"], "n_studies": study_metric["n"],
        })
    raw_weights = bundle.get("weights")
    normalized_weights = ([float(value) / sum(bundle["weights"])
                           for value in bundle["weights"]]
                          if raw_weights is not None else [None, None, None])
    summary = {
        "model": display_name,
        "optimizer": bundle.get("optimizer", "rmsprop"),
        "weight_optimizer": bundle.get("weight_optimizer", "none"),
        "epochs": bundle["epochs"], "threshold": bundle["threshold"],
        "numeric_weight": normalized_weights[0],
        "categorical_weight": normalized_weights[1],
        "binary_weight": normalized_weights[2],
        "training_files": bundle["training_files"],
        "validation_files": bundle["validation_files"],
        "training_seconds": training_seconds,
        "macro_file_f1": sum(row["f1"] for row in per_file) / len(per_file),
        "macro_file_recall": sum(row["recall"] for row in per_file) / len(per_file),
        "macro_file_fpr": sum(row["fpr"] for row in per_file) / len(per_file),
        "macro_study_f1": sum(row["f1"] for row in per_study) / len(per_study),
        "macro_study_recall": sum(row["recall"] for row in per_study) / len(per_study),
        "macro_study_fpr": sum(row["fpr"] for row in per_study) / len(per_study),
        "clean_test_file_fpr": clean_file["fpr"],
        "clean_test_study_fpr": clean_study["fpr"],
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_cpaisd/reports/bilevel_mixed_autoencoder_comparison"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--meta-learning-rate", type=float, default=1e-2,
                        help="Adam learning rate for the three loss-group weights.")
    parser.add_argument("--inner-step-size", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for label, directory in (("training", args.train), ("validation", args.validation),
                             ("test", args.test)):
        if not directory.is_dir() or not find_dicom_files(directory):
            raise FileNotFoundError(
                f"No readable DICOM files found in {label} directory: {directory}"
            )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a fresh path: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir()
    cases = load_cases(args.test, args.attacks)
    summaries, attack_rows = [], []
    histories = {}

    # Bilevel first so its immutable, study-disjoint threshold-calibration paths
    # can also calibrate both matched baselines.
    specifications = (
        ("bilevel_mixed", "Bilevel mixed", train_bilevel_mixed_autoencoder,
         predict_mixed_autoencoder),
        ("standard_mse", "Standard MSE", train_autoencoder,
         predict_autoencoder_with_explanations),
        ("fixed_mixed", "Fixed mixed (1/3 each)", train_mixed_autoencoder,
         predict_mixed_autoencoder),
    )
    bilevel_bundle = None
    calibration_paths = None
    for model_id, display_name, train_fn, predict_fn in specifications:
        model_path = model_dir / f"metadata_{model_id}.joblib"
        print(f"\nTraining {display_name} for {args.epochs} epochs...", flush=True)
        started = time.perf_counter()
        if model_id == "bilevel_mixed":
            bundle = train_fn(
                args.train, args.validation, model_path,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                meta_learning_rate=args.meta_learning_rate,
                inner_step_size=args.inner_step_size,
                threshold_percentile=args.threshold_percentile,
                random_state=args.seed, max_train_files=0,
            )
            bilevel_bundle = bundle
            calibration_paths = [Path(value) for value in bundle["threshold_validation_paths"]]
        elif model_id == "standard_mse":
            bundle = train_fn(
                args.train, model_path, validation_dir=args.validation,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                threshold_percentile=args.threshold_percentile,
                random_state=args.seed, max_train_files=0,
                max_validation_files=0, optimizer_name="rmsprop",
                reconstruction_loss="mse", validation_paths=calibration_paths,
            )
        else:
            bundle = train_fn(
                args.train, model_path, validation_dir=args.validation,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                threshold_percentile=args.threshold_percentile,
                random_state=args.seed, max_train_files=0,
                max_validation_files=0, validation_paths=calibration_paths,
            )
        training_seconds = time.perf_counter() - started
        histories[display_name] = bundle["loss_history"]
        summary, rows = _evaluate_model(
            display_name, bundle, model_path, predict_fn, cases, training_seconds
        )
        summaries.append(summary)
        attack_rows.extend(rows)
        print("  clean file/study FPR={:.4f}/{:.4f}; attack macro F1 file/study={:.4f}/{:.4f}".format(
            summary["clean_test_file_fpr"], summary["clean_test_study_fpr"],
            summary["macro_file_f1"], summary["macro_study_f1"]), flush=True)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["macro_study_f1", "macro_study_recall", "clean_test_study_fpr"],
        ascending=[False, False, True],
    )
    summary_df.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(args.output / "per_attack_metrics.csv", index=False)

    bilevel_history = histories["Bilevel mixed"]
    epochs = range(1, len(bilevel_history["numeric_weight"]) + 1)
    weight_frame = pd.DataFrame({
        "epoch": list(epochs),
        "numeric_weight": bilevel_history["numeric_weight"],
        "categorical_weight": bilevel_history["categorical_weight"],
        "binary_weight": bilevel_history["binary_weight"],
        "meta_objective": bilevel_history["meta_objective"],
        "hypergradient_norm": bilevel_history["hypergradient_norm"],
        "train_objective": bilevel_history["train_objective"],
        "validation_clean_score": bilevel_history["validation_clean_score"],
    })
    weight_frame.to_csv(args.output / "bilevel_weight_trajectory.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(weight_frame["epoch"], weight_frame["numeric_weight"], label="Numeric (Smooth L1)")
    axes[0].plot(weight_frame["epoch"], weight_frame["categorical_weight"], label="Categorical (CE)")
    axes[0].plot(weight_frame["epoch"], weight_frame["binary_weight"], label="Presence (BCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Learned loss weight")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Adam bilevel group-weight trajectory")
    axes[1].plot(weight_frame["epoch"], weight_frame["meta_objective"],
                 label="Validation ranking objective")
    axes[1].plot(weight_frame["epoch"], weight_frame["train_objective"],
                 label="Training reconstruction objective")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Objective value")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_title("Bilevel and reconstruction objectives")
    fig.savefig(args.output / "bilevel_weight_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "models": ["standard_mse", "fixed_mixed", "bilevel_mixed"],
        "model_optimizer": "RMSprop for matched comparison with current MSE baseline",
        "weight_optimizer": "Adam",
        "weight_method": "one-step differentiable unrolled hypergradient once per epoch",
        "outer_objective": "pairwise softplus ranking margin: synthetic attacked validation score should exceed its clean counterpart",
        "initial_group_weights": {"numeric": 1 / 3, "categorical": 1 / 3, "binary": 1 / 3},
        "validation_protocol": "validation studies split in half; synthetic attack copies on one half update weights, disjoint clean half calibrates all model thresholds",
        "validation_attack_families": bilevel_bundle["validation_attack_families"],
        "threshold_calibration_studies": bilevel_bundle["threshold_validation_studies"],
        "test_protocol": "same clean test and six existing attack conditions for all three models",
        "selection_note": "Exploratory single-seed comparison; test fold is not used for weight or threshold selection.",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(
        "# Adam bilevel mixed-loss comparison\n\n"
        "The three reconstruction groups are numeric Smooth L1, categorical "
        "cross-entropy, and binary presence BCE. The bilevel run initializes "
        "their weights equally and updates the group logits with Adam using a "
        "one-step unrolled hypergradient. The outer objective is a pairwise "
        "ranking loss on synthetic attacks generated from one half of the "
        "validation studies. The other half supplies clean-only threshold "
        "calibration for every model, including MSE and fixed mixed baselines. "
        "The autoencoder parameters use RMSprop to match the current MSE setup.\n\n"
        "Synthetic validation attack conditions are known to the optimization "
        "procedure; results therefore measure performance for these scripted "
        "families, not unseen real-world attacks. The test fold remains untouched "
        "during weight optimization and threshold calibration.\n\n"
        "## Summary\n\n" + summary_df.fillna("—").to_string(index=False) + "\n\n"
        "Per-attack results: `per_attack_metrics.csv`. Learned weights: "
        "`bilevel_weight_trajectory.csv` and `bilevel_weight_trajectory.png`.\n",
        encoding="utf-8",
    )
    print("\nSummary saved:", args.output / "summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
