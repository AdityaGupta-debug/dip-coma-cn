"""Run separate controlled pixel-edit and trusted pixel-hash experiments."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# When invoked as ``python scripts/run_pixel_integrity_experiment.py``, some
# Windows environments put only the scripts directory on sys.path. Add the
# repository root so the sibling ``src`` package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pixel_integrity import run_pixel_integrity_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the separate DICOM pixel-integrity experiment.")
    parser.add_argument("--input", required=True, help="Clean DICOM test-fold directory")
    parser.add_argument("--output", required=True, help="Fresh output directory for pixel experiment artifacts")
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmaps-per-attack", type=int, default=3,
                        help="Number of representative modified slices to visualize per attack; use 0 to disable")
    parser.add_argument("--severity-sweep", action="store_true",
                        help="Run subtle, default, and stronger edit settings into separate subfolders")
    args = parser.parse_args()
    if args.severity_sweep:
        output = Path(args.output).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
        output.mkdir(parents=True, exist_ok=True)
        levels = {
            "subtle": {"patch_size": 3, "patch_delta": 1, "sparse_fraction": 0.001, "sparse_delta": 1, "slice_delta": 1},
            "default": {"patch_size": 5, "patch_delta": 32, "sparse_fraction": 0.01, "sparse_delta": 1, "slice_delta": 8},
            "stronger": {"patch_size": 9, "patch_delta": 32, "sparse_fraction": 0.05, "sparse_delta": 2, "slice_delta": 32},
        }
        summaries = []
        for name, settings in levels.items():
            print(f"\nRunning {name} severity: {json.dumps(settings)}", flush=True)
            frame = run_pixel_integrity_experiment(
                args.input, output / name, args.attack_fraction, args.seed,
                args.heatmaps_per_attack, **settings)
            frame.insert(0, "severity", name)
            summaries.append(frame)
        pd.concat(summaries, ignore_index=True).to_csv(output / "severity_sweep_metrics.csv", index=False)
        print(f"\nSeverity sweep complete: {output}")
    else:
        summary = run_pixel_integrity_experiment(args.input, args.output, args.attack_fraction,
                                                 args.seed, args.heatmaps_per_attack)
        print(summary.to_string(index=False))
        print(f"\nPixel experiment complete: {args.output}")


if __name__ == "__main__":
    main()
