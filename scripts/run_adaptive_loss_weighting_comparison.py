"""Matched study-disjoint comparison of fixed, learnable, and bilevel WBCE/Dice weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.supervised_localization import run_supervised_localization

MODES = ("fixed", "learnable", "bilevel")
FAMILIES = ("localized_patch", "sparse_low_amplitude", "copy_move_unseen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/cpaisd/train"))
    parser.add_argument("--validation", type=Path, default=Path("data/cpaisd/val"))
    parser.add_argument("--test", type=Path, default=Path("data/cpaisd/test"))
    parser.add_argument("--output", type=Path, default=Path("experiment_cpaisd/reports/adaptive_loss_weighting_comparison"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-cap", type=int, default=512)
    parser.add_argument("--validation-mask-cap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--alpha-learning-rate", type=float, default=0.001)
    parser.add_argument("--bilevel-meta-learning-rate", type=float, default=0.01)
    parser.add_argument("--inner-step-size", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    result_rows, study_rows, history_frames, configs = [], [], [], {}
    for mode in MODES:
        run_dir = args.output / mode
        print(f"\n=== Starting {mode} WBCE/Dice weighting ===", flush=True)
        summary = run_supervised_localization(
            args.train, args.validation, args.test, run_dir,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            image_size=args.image_size, max_train_files=args.train_cap,
            max_validation_files=0, max_validation_attack_files=args.validation_mask_cap,
            max_test_files=0, attack_fraction=0.20, pixel_quantile=99.5,
            image_quantile=99.0, base_channels=8, device_name=args.device,
            architecture="unet", loss_weighting=mode, initial_bce_weight=0.5,
            alpha_learning_rate=args.alpha_learning_rate,
            bilevel_meta_learning_rate=args.bilevel_meta_learning_rate,
            bilevel_inner_step_size=args.inner_step_size,
        )
        summary.insert(0, "weighting_mode", mode)
        result_rows.append(summary)
        run_studies = pd.read_csv(run_dir / "pixel_localization_study_metrics.csv")
        run_studies.insert(0, "weighting_mode", mode)
        study_rows.append(run_studies)
        history = pd.read_csv(run_dir / "training_history.csv")
        history.insert(0, "weighting_mode", mode)
        history_frames.append(history)
        configs[mode] = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
        print(f"=== Finished {mode}; selected WBCE weight={configs[mode]['selected_bce_weight']:.4f} ===", flush=True)

    metrics = pd.concat(result_rows, ignore_index=True)
    studies = pd.concat(study_rows, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    metrics.to_csv(args.output / "pixel_localization_comparison.csv", index=False)
    studies.to_csv(args.output / "study_metrics_comparison.csv", index=False)
    history.to_csv(args.output / "weighting_training_history.csv", index=False)
    (args.output / "comparison_config.json").write_text(json.dumps({
        "seed": args.seed, "epochs": args.epochs, "train_cap": args.train_cap,
        "validation_mask_cap": args.validation_mask_cap, "batch_size": args.batch_size,
        "image_size": args.image_size, "device": args.device,
        "modes": list(MODES), "runs": configs,
        "validation_protocol": "the same five validation studies for bilevel coefficient updates and the disjoint five studies for checkpoint selection and clean-threshold calibration; all modes use the latter for checkpoint selection/calibration",
        "test_protocol": "all official test images and the same seeded synthetic edit selection per mode",
        "bilevel_method": "one-step differentiable unrolled SGD proxy hypergradient once per epoch; outer objective is validation WBCE plus Dice",
    }, indent=2), encoding="utf-8")

    alpha_plot = history.pivot(index="epoch", columns="weighting_mode", values="bce_weight")
    ax = alpha_plot[[mode for mode in MODES if mode in alpha_plot]].plot(figsize=(8, 4.5), marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"WBCE coefficient $\alpha$")
    ax.set_ylim(0, 1)
    ax.set_title("Fixed, directly learned, and bilevel WBCE weights")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Weighting")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(args.output / "alpha_trajectory.png", dpi=220, bbox_inches="tight")
    paper_figure = ROOT / "experiment_cpaisd" / "paper" / "figures" / "adaptive_loss_weight_trajectory.png"
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    means = metrics.groupby("weighting_mode").agg(
        macro_image_f1=("f1", "mean"), macro_image_auroc=("image_auroc", "mean"),
        macro_pixel_precision=("pixel_precision", "mean"),
        macro_pixel_recall=("pixel_recall", "mean"), macro_pixel_iou=("pixel_iou", "mean"),
        clean_image_fpr=("clean_test_image_fpr", "first"),
        bce_weight=("bce_weight", "first"), dice_weight=("dice_weight", "first"),
    ).reset_index()
    means.to_csv(args.output / "macro_comparison.csv", index=False)
    report = [
        "# Adaptive WBCE/Dice loss-weight comparison", "",
        "The three runs share the same seed, 512-image training cap, U-Net, epochs, batch size, image resolution, synthetic edit generators, checkpoint-selection studies, threshold-calibration studies, and held-out test fold.",
        "The ten validation studies are split by study: five are reserved for bilevel coefficient updates, and five are used for checkpoint selection and clean threshold calibration. Test studies are never used for coefficient or checkpoint selection.",
        "The bilevel run uses a one-step differentiable unrolled SGD proxy to estimate the validation hypergradient once per epoch. This is a truncated hypergradient approximation, not exact convergence of the inner problem.",
        "The directly learnable mode updates alpha using training loss only; fixed weighting remains at alpha=0.5.", "",
        means.to_string(index=False), "",
        "Read pixel localization together with image false-positive rates and study-level results. The study counts are small, synthetic masks define labels, and all outcomes are descriptive.",
    ]
    (args.output / "ADAPTIVE_WEIGHTING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nAdaptive weighting comparison complete.\n", means.to_string(index=False), flush=True)
    print("Saved report:", args.output / "ADAPTIVE_WEIGHTING_REPORT.md", flush=True)


if __name__ == "__main__":
    main()

