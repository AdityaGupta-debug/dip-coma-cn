"""Compare standard MSE and mixed-type reconstruction for metadata AE."""
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
from src.mixed_anomaly import train_mixed_autoencoder, predict_mixed_autoencoder
from run_autoencoder_optimizer_comparison import load_cases, metrics, study_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, default=Path("experiment_cpaisd/attacks"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_cpaisd/reports/mixed_autoencoder_comparison"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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

    print("Loading clean test and six existing attacked conditions...", flush=True)
    cases = load_cases(args.test, args.attacks)
    model_specs = (
        ("standard_mse", "Standard MSE", train_autoencoder,
         predict_autoencoder_with_explanations, "mse"),
        ("mixed", "Mixed loss", train_mixed_autoencoder,
         predict_mixed_autoencoder, None),
    )
    summaries, attack_rows, curve_rows = [], [], []
    histories = {}
    for model_id, display_name, train_fn, predict_fn, loss_name in model_specs:
        model_path = model_dir / f"metadata_{model_id}.joblib"
        print(f"\nTraining {display_name} for {args.epochs} epochs...", flush=True)
        started = time.perf_counter()
        kwargs = {"optimizer_name": "rmsprop", "reconstruction_loss": loss_name} \
            if model_id == "standard_mse" else {}
        bundle = train_fn(
            args.train, model_path, validation_dir=args.validation,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold_percentile=args.threshold_percentile,
            random_state=args.seed, max_train_files=0, max_validation_files=0,
            **kwargs,
        )
        training_seconds = time.perf_counter() - started
        histories[display_name] = bundle["loss_history"]
        if model_id == "standard_mse":
            train_history = bundle["loss_history"]["train"]
            validation_history = bundle["loss_history"]["validation"]
        else:
            train_history = bundle["loss_history"]["train_total"]
            validation_history = bundle["loss_history"]["validation_clean_score"]
        for epoch, (train_loss, validation_loss) in enumerate(
                zip(train_history, validation_history), start=1):
            curve_rows.append({"model": display_name, "epoch": epoch,
                               "train_objective": train_loss,
                               "validation_clean_score": validation_loss})

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
            "model": display_name, "loss": bundle.get("loss", "mse"),
            "optimizer": "rmsprop", "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "seed": args.seed,
            "threshold_percentile": args.threshold_percentile,
            "threshold": bundle["threshold"],
            "numeric_weight": bundle.get("weights", (1.0, 1.0, 1.0))[0],
            "categorical_weight": bundle.get("weights", (1.0, 1.0, 1.0))[1],
            "binary_weight": bundle.get("weights", (1.0, 1.0, 1.0))[2],
            "training_files": bundle["training_files"],
            "validation_files": bundle["validation_files"],
            "training_seconds": training_seconds,
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
    pd.DataFrame(curve_rows).to_csv(args.output / "training_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (model_id, display_name, *_rest) in zip(axes, model_specs):
        history = histories[display_name]
        if model_id == "standard_mse":
            train_loss, val_loss = history["train"], history["validation"]
            train_label, val_label = "Training MSE", "Clean validation MSE"
        else:
            train_loss = history["train_total"]
            val_loss = history["validation_clean_score"]
            train_label, val_label = "Mixed training objective", "Clean validation mixed score"
        epochs = range(1, len(train_loss) + 1)
        axis.plot(epochs, train_loss, label=train_label)
        axis.plot(epochs, val_loss, label=val_label)
        axis.set_title(display_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss / score (different scales across models)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Standard MSE vs mixed-type metadata autoencoder")
    fig.savefig(args.output / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "models": ["standard_mse", "mixed"],
        "mixed_loss": {
            "numeric": "Smooth L1 / Huber beta=1 on standardized continuous targets",
            "categorical": "cross entropy over one-hot DictVectorizer groups",
            "binary": "binary cross entropy with logits on raw 0/1 presence targets",
            "group_weights": {"numeric": 1.0, "categorical": 1.0, "binary": 1.0},
            "score": "equal-weight mean of the three per-instance group losses",
        },
        "data_protocol": "same clean train/validation/test folds and existing six attack folders; independent clean-validation threshold per model",
        "selection_note": "Exploratory single-seed comparison; thresholds calibrated on validation only; no test-driven tuning.",
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(
        "# Mixed reconstruction-loss comparison\n\n"
        "The mixed autoencoder uses Smooth L1 for numeric features, categorical "
        "cross-entropy for one-hot feature groups, and binary cross-entropy for "
        "tag-presence indicators. Each type group has equal weight. The standard "
        "MSE model is trained as a matched baseline. Each model receives an "
        "independent threshold from clean validation scores. Unknown categories "
        "not present in the fitted DictVectorizer vocabulary are not represented, "
        "so this experiment does not address that feature-map limitation.\n\n"
        "Results are exploratory for one CPAISD test fold and one seed. Compare "
        "per-attack recall/F1 and clean file- and study-level false-positive rates.\n\n"
        "## Summary\n\n" + summary.to_string(index=False) + "\n\n"
        "Training curves: `training_curves.png`. Per-attack results: `per_attack_metrics.csv`.\n",
        encoding="utf-8",
    )
    print("\nSummary saved:", args.output / "summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
