"""Matched comparison of standard and denoising CPAISD metadata autoencoders.

This script is intentionally separate from the primary experiment runner. It
uses the existing six metadata attack folders and does not modify DICOM files.
"""
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
from src.denoising_anomaly import (
    train_denoising_autoencoder, predict_denoising_autoencoder,
)
from src.dicom_io import find_dicom_files
from run_autoencoder_optimizer_comparison import load_cases, metrics, study_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path,
                        default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_cpaisd/reports/denoising_autoencoder_comparison"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--noise-std", type=float, default=0.10,
                        help="Gaussian corruption standard deviation after feature standardization.")
    parser.add_argument("--mask-probability", type=float, default=0.05,
                        help="Per-feature corruption probability; masked standardized values are set to zero.")
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

    print("Loading clean test and six existing attacked conditions...", flush=True)
    cases = load_cases(args.test, args.attacks)
    model_specs = (
        ("standard_ae", "Standard AE", train_autoencoder,
         predict_autoencoder_with_explanations, {}),
        ("denoising_ae", "Denoising AE", train_denoising_autoencoder,
         predict_denoising_autoencoder,
         {"noise_std": args.noise_std,
          "mask_probability": args.mask_probability}),
    )
    summaries, attack_rows, curve_rows = [], [], []
    histories = {}
    for model_id, display_name, train_fn, predict_fn, extra in model_specs:
        model_path = model_dir / f"metadata_{model_id}.joblib"
        print(f"\nTraining {display_name} for {args.epochs} epochs...", flush=True)
        started = time.perf_counter()
        bundle = train_fn(
            args.train, model_path, validation_dir=args.validation,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_percentile=args.threshold_percentile,
            random_state=args.seed, max_train_files=0, max_validation_files=0,
            optimizer_name="rmsprop", **extra,
        )
        training_seconds = time.perf_counter() - started
        histories[display_name] = bundle["loss_history"]
        if model_id == "standard_ae":
            train_history = bundle["loss_history"]["train"]
            val_history = bundle["loss_history"]["validation"]
            history_pairs = zip(train_history, val_history)
            history_names = ("train_mse", "validation_mse")
        else:
            train_history = bundle["loss_history"]["train_denoising_mse"]
            val_history = bundle["loss_history"]["validation_clean_mse"]
            history_pairs = zip(train_history, val_history)
            history_names = ("train_denoising_mse", "validation_clean_mse")
        for epoch, (train_loss, validation_loss) in enumerate(history_pairs, start=1):
            curve_rows.append({
                "model": display_name, "epoch": epoch,
                history_names[0]: train_loss, history_names[1]: validation_loss,
            })

        file_metrics, study_condition_metrics = [], []
        clean_file = clean_study = None
        for condition, case in cases.items():
            predictions = predict_fn(case["datasets"], model_path, top_n=1)
            preds = [int(predictions[key][0]) for key, _ in case["datasets"]]
            rows = [{"study": study, "label": label, "prediction": pred}
                    for study, label, pred in zip(case["studies"], case["labels"], preds)]
            fm, sm = metrics(case["labels"], preds), study_metrics(rows)
            if condition == "clean":
                clean_file, clean_study = fm, sm
                continue
            file_metrics.append(fm)
            study_condition_metrics.append(sm)
            attack_rows.append({
                "model": display_name, "condition": condition,
                "threshold": bundle["threshold"], **fm,
                "study_recall": sm["recall"], "study_fpr": sm["fpr"],
                "study_f1": sm["f1"], "n_studies": sm["n"],
            })
        summaries.append({
            "model": display_name, "optimizer": "rmsprop",
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed,
            "threshold_percentile": args.threshold_percentile,
            "threshold": bundle["threshold"],
            "training_files": bundle["training_files"],
            "validation_files": bundle["validation_files"],
            "training_seconds": training_seconds,
            "noise_std": extra.get("noise_std", 0.0),
            "mask_probability": extra.get("mask_probability", 0.0),
            "macro_file_f1": sum(x["f1"] for x in file_metrics) / len(file_metrics),
            "macro_file_recall": sum(x["recall"] for x in file_metrics) / len(file_metrics),
            "macro_file_fpr": sum(x["fpr"] for x in file_metrics) / len(file_metrics),
            "macro_study_f1": sum(x["f1"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_recall": sum(x["recall"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_fpr": sum(x["fpr"] for x in study_condition_metrics) / len(study_condition_metrics),
            "clean_test_file_fpr": clean_file["fpr"],
            "clean_test_study_fpr": clean_study["fpr"],
        })
        row = summaries[-1]
        print("  clean FPR file/study={:.4f}/{:.4f}; attack macro F1 file/study={:.4f}/{:.4f}".format(
            clean_file["fpr"], clean_study["fpr"],
            row["macro_file_f1"], row["macro_study_f1"]), flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["macro_study_f1", "macro_study_recall", "clean_test_study_fpr"],
        ascending=[False, False, True],
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(args.output / "per_attack_metrics.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.output / "training_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (model_id, display_name, *_rest) in zip(axes, model_specs):
        history = histories[display_name]
        if model_id == "standard_ae":
            train_loss, val_loss = history["train"], history["validation"]
            train_label, val_label = "Training MSE", "Clean validation MSE"
        else:
            train_loss = history["train_denoising_mse"]
            val_loss = history["validation_clean_mse"]
            train_label, val_label = "Denoising training MSE", "Clean validation MSE"
        epochs = range(1, len(train_loss) + 1)
        axis.plot(epochs, train_loss, label=train_label)
        axis.plot(epochs, val_loss, label=val_label)
        axis.set_title(display_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean squared error")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Matched CPAISD metadata autoencoder comparison")
    fig.savefig(args.output / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "models": ["standard_ae", "denoising_ae"],
        "optimizer": "rmsprop",
        "denoising_corruption": {
            "gaussian_noise_std_on_standardized_features": args.noise_std,
            "per_feature_mask_probability": args.mask_probability,
            "masked_value": "zero in standardized space",
        },
        "data_protocol": "same clean training/validation/test folds and existing six attack folders; independent clean-validation threshold per model",
        "selection_note": "Exploratory single-seed matched baseline comparison; no test-driven tuning.",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(
        "# Denoising autoencoder comparison\n\n"
        "The standard and denoising autoencoders use the same metadata features, "
        "MLP architecture, RMSprop optimizer, folds, attack folders, and clean-validation "
        "threshold percentile. The denoising model receives Gaussian noise and random "
        "feature masking during training but reconstructs the original clean feature vector. "
        "Both models are scored using clean-input reconstruction MSE.\n\n"
        "Results are exploratory and use the supplied CPAISD fold and one random seed. "
        "The denoising setting is not assumed to be better; compare attack macro F1/recall "
        "and clean file- and study-level false-positive rates together.\n\n"
        "## Summary\n\n" + summary.to_string(index=False) + "\n\n"
        "Training curves: `training_curves.png`. Per-attack results: `per_attack_metrics.csv`.\n",
        encoding="utf-8",
    )
    print("\nSummary saved:", args.output / "summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
