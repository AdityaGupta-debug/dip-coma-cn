"""Recompute stratified study-cluster bootstrap summaries from saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Seed-sensitivity output directory")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    root = Path(args.input)
    rng = np.random.default_rng(args.seed)
    summaries = []
    for unit, filename in [("study", "study_level_predictions_all_runs.csv"),
                           ("series", "series_level_predictions_all_runs.csv")]:
        frame = pd.read_csv(root / filename)
        for (attack, method), subset in frame.groupby(["attack", "method"]):
            draws_by_seed, observed = [], []
            for _, run in subset.groupby("seed"):
                truth = run.ground_truth.to_numpy(dtype=int)
                prediction = run.prediction.to_numpy(dtype=int)
                pos, neg = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
                if not len(pos) or not len(neg):
                    continue
                observed.append((float(prediction[pos].mean()), float(prediction[neg].mean())))
                draws = np.empty((args.iterations, 2), dtype=float)
                for b in range(args.iterations):
                    draws[b, 0] = prediction[rng.choice(pos, len(pos), replace=True)].mean()
                    draws[b, 1] = prediction[rng.choice(neg, len(neg), replace=True)].mean()
                draws_by_seed.append(draws)
            if not draws_by_seed:
                continue
            combined = np.mean(np.stack(draws_by_seed), axis=0)
            summaries.append({
                "evaluation_unit": unit, "attack": attack, "method": method,
                "seed_count": len(observed),
                "mean_recall_across_seeds": float(np.mean([x[0] for x in observed])),
                "mean_fpr_across_seeds": float(np.mean([x[1] for x in observed])),
                "recall_ci_95_low": float(np.percentile(combined[:, 0], 2.5)),
                "recall_ci_95_high": float(np.percentile(combined[:, 0], 97.5)),
                "fpr_ci_95_low": float(np.percentile(combined[:, 1], 2.5)),
                "fpr_ci_95_high": float(np.percentile(combined[:, 1], 97.5)),
            })
    result = pd.DataFrame(summaries)
    result.to_csv(root / "study_bootstrap_summary.csv", index=False)
    per_run_path = root / "per_run_file_metrics.csv"
    per_run = pd.read_csv(per_run_path)
    configuration = {
        "seeds": sorted(int(x) for x in per_run.seed.unique()),
        "attacks": sorted(str(x) for x in per_run.attack.unique()),
        "attack_fraction": 0.20,
        "bootstrap_iterations": args.iterations,
        "bootstrap_unit": "baseline-keyed study or series, stratified by attack label",
        "bootstrap_seed": args.seed,
        "note": "Intervals are descriptive for this ten-study CPAISD test fold; they do not represent external-population uncertainty.",
    }
    (root / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Saved stratified study-cluster bootstrap summary to {root / 'study_bootstrap_summary.csv'}")


if __name__ == "__main__":
    main()
