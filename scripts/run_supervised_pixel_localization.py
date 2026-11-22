"""Train/evaluate a synthetic-supervision U-Net pixel-localization baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.supervised_localization import run_supervised_localization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/cpaisd/train", help="Official clean CPAISD training fold")
    parser.add_argument("--validation", default="data/cpaisd/val", help="Official clean CPAISD validation fold")
    parser.add_argument("--test", default="data/cpaisd/test", help="Official CPAISD test fold")
    parser.add_argument("--output", default="experiment_cpaisd_supervised_localization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-train-files", type=int, default=0,
                        help="0 uses all train files; set e.g. 512 for an exploratory run")
    parser.add_argument("--max-validation-files", type=int, default=0,
                        help="0 uses the full validation fold")
    parser.add_argument("--max-validation-attack-files", type=int, default=256,
                        help="Fixed synthetic-mask validation subset used for best-epoch selection")
    parser.add_argument("--max-test-files", type=int, default=0,
                        help="0 uses the full test fold")
    parser.add_argument("--attack-fraction", type=float, default=0.20,
                        help="Fraction of test studies attacked per condition")
    parser.add_argument("--pixel-quantile", type=float, default=99.5,
                        help="Clean-validation pixel-score quantile for mask threshold")
    parser.add_argument("--image-quantile", type=float, default=99.0,
                        help="Clean-validation per-image maximum-score quantile")
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--architecture", choices=["unet", "attention_unet"], default="unet",
                        help="Model architecture; use attention_unet for attention-gated skip connections")
    parser.add_argument("--loss-weighting", choices=["fixed", "learnable", "bilevel"],
                        default="fixed", help="WBCE/Dice coefficient strategy")
    parser.add_argument("--initial-bce-weight", type=float, default=0.5,
                        help="Initial/fixed WBCE coefficient; Dice weight is 1-alpha")
    parser.add_argument("--alpha-learning-rate", type=float, default=0.001,
                        help="Learning rate for directly learnable alpha updates")
    parser.add_argument("--bilevel-meta-learning-rate", type=float, default=0.01,
                        help="Adam learning rate for bilevel hypergradient updates")
    parser.add_argument("--bilevel-inner-step-size", type=float, default=1e-3,
                        help="One-step SGD proxy step used to compute the unrolled hypergradient")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    args = parser.parse_args()

    summary = run_supervised_localization(
        args.train, args.validation, args.test, args.output,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        image_size=args.image_size, max_train_files=args.max_train_files,
        max_validation_files=args.max_validation_files,
        max_validation_attack_files=args.max_validation_attack_files,
        max_test_files=args.max_test_files,
        attack_fraction=args.attack_fraction, pixel_quantile=args.pixel_quantile,
        image_quantile=args.image_quantile, base_channels=args.base_channels,
        device_name=args.device, architecture=args.architecture,
        loss_weighting=args.loss_weighting,
        initial_bce_weight=args.initial_bce_weight,
        alpha_learning_rate=args.alpha_learning_rate,
        bilevel_meta_learning_rate=args.bilevel_meta_learning_rate,
        bilevel_inner_step_size=args.bilevel_inner_step_size,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
