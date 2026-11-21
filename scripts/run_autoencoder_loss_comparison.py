"""Compare reconstruction losses for the CPAISD metadata autoencoder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.deep_anomaly import train_autoencoder, predict_autoencoder_with_explanations
from src.dicom_io import find_dicom_files
from run_autoencoder_optimizer_comparison import load_cases, metrics, study_metrics

LOSSES = ("mse", "mae", "smooth_l1", "log_cosh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path, default=Path("experiment_cpaisd/reports/autoencoder_loss_comparison"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for label, directory in (("training", args.train), ("validation", args.validation),
                             ("test", args.test)):
        if not directory.is_dir() or not find_dicom_files(directory):
            raise FileNotFoundError(f"No readable DICOM files found in {label} directory: {directory}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir()

    print("Loading clean test and six attacked test conditions once...")
    cases = load_cases(args.test, args.attacks)
    summaries, attack_rows, curve_rows = [], [], []
    histories = {}
    for loss_name in LOSSES:
        model_path = model_dir / f"metadata_autoencoder_{loss_name}.joblib"
        print(f"\nTraining {loss_name} for {args.epochs} epochs (RMSprop)...", flush=True)
        started = time.perf_counter()
        bundle = train_autoencoder(
            args.train, model_path, validation_dir=args.validation,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_percentile=args.threshold_percentile,
            random_state=args.seed, max_train_files=0, max_validation_files=0,
            optimizer_name="rmsprop", reconstruction_loss=loss_name,
        )
        train_seconds = time.perf_counter() - started
        histories[loss_name] = bundle["loss_history"]
        for epoch, (train_loss, validation_loss) in enumerate(zip(
                bundle["loss_history"]["train"],
                bundle["loss_history"]["validation"]), start=1):
            curve_rows.append({"loss": loss_name, "epoch": epoch,
                               "train_loss": train_loss,
                               "validation_loss": validation_loss})
        file_condition_metrics, study_condition_metrics = [], []
        clean_file = clean_study = None
        for condition, case in cases.items():
            predictions = predict_autoencoder_with_explanations(
                case["datasets"], model_path, top_n=1
            )
            preds = [int(predictions[key][0]) for key, _ in case["datasets"]]
            rows = [{"study": study, "label": label, "prediction": pred}
                    for study, label, pred in zip(case["studies"], case["labels"], preds)]
            fm, sm = metrics(case["labels"], preds), study_metrics(rows)
            if condition == "clean":
                clean_file, clean_study = fm, sm
                continue
            file_condition_metrics.append(fm)
            study_condition_metrics.append(sm)
            attack_rows.append({
                "loss": loss_name, "condition": condition,
                "threshold": bundle["threshold"], **fm,
                "study_recall": sm["recall"], "study_fpr": sm["fpr"],
                "study_f1": sm["f1"], "n_studies": sm["n"],
            })
        summaries.append({
            "loss": loss_name, "optimizer": "rmsprop", "epochs": args.epochs,
            "learning_rate": args.learning_rate, "seed": args.seed,
            "threshold_percentile": args.threshold_percentile,
            "threshold": bundle["threshold"],
            "training_files": bundle["training_files"],
            "validation_files": bundle["validation_files"],
            "training_seconds": train_seconds,
            "macro_file_f1": sum(x["f1"] for x in file_condition_metrics) / len(file_condition_metrics),
            "macro_file_recall": sum(x["recall"] for x in file_condition_metrics) / len(file_condition_metrics),
            "macro_file_fpr": sum(x["fpr"] for x in file_condition_metrics) / len(file_condition_metrics),
            "macro_study_f1": sum(x["f1"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_recall": sum(x["recall"] for x in study_condition_metrics) / len(study_condition_metrics),
            "macro_study_fpr": sum(x["fpr"] for x in study_condition_metrics) / len(study_condition_metrics),
            "clean_test_file_fpr": clean_file["fpr"],
            "clean_test_study_fpr": clean_study["fpr"],
        })
        row = summaries[-1]
        print("  clean FPR file/study={:.4f}/{:.4f}; macro attack F1 file/study={:.4f}/{:.4f}; macro attack FPR file/study={:.4f}/{:.4f}".format(
            clean_file["fpr"], clean_study["fpr"], row["macro_file_f1"], row["macro_study_f1"],
            row["macro_file_fpr"], row["macro_study_fpr"]), flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["macro_study_f1", "macro_study_recall", "clean_test_study_fpr"],
        ascending=[False, False, True]
    )
    summary.to_csv(args.output / "loss_summary.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(args.output / "per_attack_metrics.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.output / "loss_curves.csv", index=False)
    final_losses = curves.groupby("loss", sort=False).tail(1)[
        ["loss", "epoch", "train_loss", "validation_loss"]
    ]
    final_losses.to_csv(args.output / "final_train_validation_loss.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, loss_name in zip(axes.flat, LOSSES):
        history = histories[loss_name]
        epochs = range(1, len(history["train"]) + 1)
        axis.plot(epochs, history["train"], label="Training", linewidth=1.8)
        if history["validation"] and history["validation"][0] is not None:
            axis.plot(epochs, history["validation"], label="Validation", linewidth=1.8)
        axis.set_title({"mse": "MSE", "mae": "MAE",
                        "smooth_l1": "Smooth L1", "log_cosh": "Log-cosh"}[loss_name])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean reconstruction loss")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("CPAISD metadata autoencoder reconstruction-loss curves")
    fig.savefig(args.output / "loss_curves.png", dpi=220, bbox_inches="tight")
    paper_figure = ROOT / "experiment_cpaisd" / "paper" / "figures" / "autoencoder_loss_curves.png"
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    config.update({
        "losses": list(LOSSES), "optimizer": "rmsprop",
        "selection_metric": "macro_study_f1",
        "selection_note": "Exploratory one-seed loss comparison; each loss uses its own clean-validation threshold.",
        "data_protocol": "all clean training instances; clean validation threshold; untouched test and six existing metadata attack conditions",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    lines = [
        "# Metadata autoencoder reconstruction-loss comparison", "",
        f"- Training files: {bundle['training_files']}",
        f"- Validation files: {bundle['validation_files']}",
        f"- Fixed optimizer: RMSprop; epochs: {args.epochs}; seed: {args.seed}",
        "- Each loss is calibrated independently at the same clean-validation percentile.",
        "- Metrics average the six existing attack conditions; they are descriptive for the CPAISD test fold.", "",
        f"- Loss curves: `{args.output / 'loss_curves.png'}`",
        summary.to_string(index=False), "",
        "Loss choice, thresholding, and optimizer are all part of the detector. A lower training loss is not evidence of better anomaly detection; compare clean false alarms and held-out attack detection together.",
    ]
    (args.output / "LOSS_COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSummary saved:", args.output / "loss_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
