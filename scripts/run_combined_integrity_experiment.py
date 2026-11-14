"""Run paired metadata-plus-pixel edits as a separate integrity experiment."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import ATTACKS
from src.pixel_integrity import PIXEL_ATTACKS, run_combined_integrity_experiment


def main():
    parser = argparse.ArgumentParser(description="Run combined DICOM metadata and pixel-integrity attacks.")
    parser.add_argument("--input", required=True, help="Clean DICOM test-fold directory")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--metadata-attacks", nargs="+", choices=[a for a in ATTACKS if a != "study_uid_inconsistency"],
                        default=["patient_id", "modality"])
    parser.add_argument("--pixel-attacks", nargs="+", choices=PIXEL_ATTACKS, default=["localized_patch", "low_amplitude"])
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_combined_integrity_experiment(args.input, args.output, args.metadata_attacks,
                                               args.pixel_attacks, args.attack_fraction, args.seed)
    print(result.to_string(index=False))
    print(f"\nCombined integrity experiment complete: {args.output}")


if __name__ == "__main__":
    main()
