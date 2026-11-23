"""Train an online-PGD/bilevel U-Net and compare it with the 30-epoch baseline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.supervised_localization import run_supervised_localization


BASELINE_CHECKPOINT = ROOT / "experiment_cpaisd_supervised_localization_512_30ep_complete" / "unet_localizer.pt"
EVALUATOR = ROOT / "scripts" / "run_unet_fgsm_pgd_evaluation.py"


def _run_evaluation(checkpoint: Path, validation: Path, test: Path, output: Path,
                    *, seed: int, epsilon: float, pgd_steps: int,
                    pgd_step_size: float, max_images_per_family: int, device: str):
    command = [
        sys.executable, str(EVALUATOR),
        "--checkpoint", str(checkpoint), "--validation", str(validation),
        "--test", str(test), "--output", str(output),
        "--seed", str(seed), "--epsilon", str(epsilon),
        "--pgd-steps", str(pgd_steps), "--pgd-step-size", str(pgd_step_size),
        "--max-images-per-family", str(max_images_per_family),
        "--attack-methods", "baseline,fgsm,pgd", "--device", device,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/cpaisd/train")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/cpaisd/val")
    parser.add_argument("--test", type=Path, default=ROOT / "data/cpaisd/test")
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiment_cpaisd_bilevel_adversarial_unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Effective balanced batch size; split evenly across clean/standard/PGD")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-width", type=int, default=24)
    parser.add_argument("--max-train-files", type=int, default=0,
                        help="0 uses the full 8,376-image official training fold")
    parser.add_argument("--max-validation-files", type=int, default=0)
    parser.add_argument("--max-validation-attack-files", type=int, default=256)
    parser.add_argument("--max-test-files", type=int, default=0)
    parser.add_argument("--bilevel-update-interval", type=int, default=5)
    parser.add_argument("--bilevel-meta-learning-rate", type=float, default=0.01)
    parser.add_argument("--bilevel-inner-step-size", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--train-pgd-steps", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--step-size", type=float, default=0.004)
    parser.add_argument("--evaluation-pgd-steps", type=int, default=10)
    parser.add_argument("--max-eval-images-per-family", type=int, default=64,
                        help="Shared balanced test cap for baseline and robust model; 0 evaluates all positives")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; use a new path: {args.output}")
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    for path in (args.train, args.validation, args.test):
        if not path.is_dir():
            raise FileNotFoundError(f"CPAISD fold not found: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    robust_dir = args.output / "robust_training"
    baseline_eval = args.output / "baseline_fgsm_pgd"
    robust_eval = args.output / "robust_fgsm_pgd"

    robust_summary = run_supervised_localization(
        args.train, args.validation, args.test, robust_dir,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        image_size=args.image_size, max_train_files=args.max_train_files,
        max_validation_files=args.max_validation_files,
        max_validation_attack_files=args.max_validation_attack_files,
        max_test_files=args.max_test_files,
        attack_fraction=0.20, pixel_quantile=99.5, image_quantile=99.0,
        base_channels=args.base_width, device_name=args.device, architecture="unet",
        loss_weighting="bilevel", initial_bce_weight=0.5,
        bilevel_meta_learning_rate=args.bilevel_meta_learning_rate,
        bilevel_inner_step_size=args.bilevel_inner_step_size,
        adversarial_training=True, positive_weight=args.positive_weight,
        adversarial_epsilon=args.epsilon, adversarial_step_size=args.step_size,
        adversarial_steps=args.train_pgd_steps,
        bilevel_update_interval=args.bilevel_update_interval,
    )
    robust_checkpoint = robust_dir / "unet_localizer.pt"
    _run_evaluation(args.baseline_checkpoint, args.validation, args.test, baseline_eval,
                    seed=args.seed, epsilon=args.epsilon,
                    pgd_steps=args.evaluation_pgd_steps, pgd_step_size=args.step_size,
                    max_images_per_family=args.max_eval_images_per_family, device=args.device)
    _run_evaluation(robust_checkpoint, args.validation, args.test, robust_eval,
                    seed=args.seed, epsilon=args.epsilon,
                    pgd_steps=args.evaluation_pgd_steps, pgd_step_size=args.step_size,
                    max_images_per_family=args.max_eval_images_per_family, device=args.device)

    baseline_frame = pd.read_csv(baseline_eval / "adversarial_comparison_summary.csv")
    robust_frame = pd.read_csv(robust_eval / "adversarial_comparison_summary.csv")
    baseline_frame.insert(0, "model", "Baseline 30-epoch U-Net")
    robust_frame.insert(0, "model", "Bilevel + adversarial U-Net")
    comparison = pd.concat([baseline_frame, robust_frame], ignore_index=True)
    comparison_path = args.output / "baseline_vs_bilevel_adversarial.csv"
    comparison.to_csv(comparison_path, index=False)

    training_config = json.loads((robust_dir / "experiment_config.json").read_text(encoding="utf-8"))
    report = [
        "# Baseline vs. bilevel adversarial U-Net", "",
        "Both checkpoints are evaluated on the same CPAISD test studies, attack scripts, selected image cap, and clean-validation max-logit q99 thresholding protocol. Baseline and robust checkpoints receive independently calibrated thresholds from the same clean validation fold.",
        "The robust model is trained on balanced clean, standard synthetic, and live PGD-evaded samples. Copy-move is withheld from training. Its bilevel WBCE/Dice coefficient is updated with Adam at the configured interval; the hypergradient uses held-out validation synthetic and PGD examples plus a clean false-positive surrogate.",
        f"Training images used: {training_config['train_images_used']} of the available official training fold; epochs: {training_config['epochs_completed']}; width: {training_config['base_channels']}; device: {training_config['device']}.",
        f"Evaluation cap: {args.max_eval_images_per_family or 'all positives'} per edit family; PGD: {args.evaluation_pgd_steps} steps, epsilon {args.epsilon}, step size {args.step_size}.",
        "These are synthetic attack experiments. The results do not establish clinical plausibility or general robustness outside the tested white-box objective and perturbation budget.",
        "", "## Results", comparison.to_markdown(index=False), "",
        "## Validation-selected training summary", robust_summary.to_markdown(index=False), "",
        "Full per-model attack summaries and per-image scores are in `baseline_fgsm_pgd/` and `robust_fgsm_pgd/`.",
    ]
    report_path = args.output / "BASELINE_VS_BILEVEL_ADVERSARIAL_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nComparison saved: {comparison_path}")
    print(f"Report saved: {report_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
