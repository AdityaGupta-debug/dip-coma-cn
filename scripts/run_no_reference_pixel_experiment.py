"""Train and evaluate a no-reference pixel anomaly baseline."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.no_reference_pixel import run_no_reference_pixel_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Clean training DICOM fold")
    parser.add_argument("--validation", required=True, help="Clean validation DICOM fold")
    parser.add_argument("--test", required=True, help="Clean test fold")
    parser.add_argument("--attacks-root", required=True,
                        help="Pixel attack directory containing attack subfolders and attack_log.json files")
    parser.add_argument("--output", required=True, help="Fresh output folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-files", type=int, default=0,
                        help="Optional cap for quick runs; zero uses all training files")
    parser.add_argument("--max-train-patches", type=int, default=50000)
    parser.add_argument("--patches-per-slice", type=int, default=24)
    parser.add_argument("--heatmaps-per-attack", type=int, default=2)
    parser.add_argument("--pretrained-model", default=None,
                        help="Reuse a saved no-reference Isolation Forest bundle; validation calibration is recomputed")
    args = parser.parse_args()
    result = run_no_reference_pixel_experiment(
        args.train, args.validation, args.test, args.attacks_root, args.output,
        seed=args.seed, max_train_files=args.max_train_files,
        max_train_patches=args.max_train_patches,
        patches_per_slice=args.patches_per_slice,
        heatmaps_per_attack=args.heatmaps_per_attack,
        pretrained_model_path=args.pretrained_model)
    print(result.to_string(index=False))
    print(f"\nNo-reference pixel experiment complete: {args.output}")


if __name__ == "__main__":
    main()
