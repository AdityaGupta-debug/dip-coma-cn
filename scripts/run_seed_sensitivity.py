"""Repeat representative metadata attacks with different study-selection seeds.

Reports per-run file/study/series metrics and study-cluster bootstrap intervals.
The bootstrap intervals describe this small CPAISD test cohort and attack
randomization; they are not external-population confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks import ATTACKS, create_attack
from src.evaluation import METHODS, evaluate_attack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean CPAISD test fold")
    parser.add_argument("--baseline", required=True, help="Clean test reference manifest")
    parser.add_argument("--metadata-model", required=True)
    parser.add_argument("--autoencoder-model", required=True)
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS,
                        default=["modality", "pixel_spacing", "manufacturer_same_length",
                                 "study_uid_inconsistency"])
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    study_rows, series_rows, run_rows = [], [], []

    for seed in args.seeds:
        for attack in args.attacks:
            tag = f"seed_{seed}/{attack}"
            attack_dir = output / "attacks" / f"seed_{seed}" / attack
            log = create_attack(args.input, attack_dir, attack, args.attack_fraction, seed)
            eval_dir = output / "evaluations" / f"seed_{seed}" / attack
            metrics, _ = evaluate_attack(
                attack_dir, args.baseline, args.metadata_model,
                attack_dir / "attack_log.json", eval_dir, args.autoencoder_model)
            for row in metrics.to_dict(orient="records"):
                run_rows.append({"seed": seed, "attack": attack,
                                 "modified_files": log["modified_file_count"], **row})
            for level, target in [("study", study_rows), ("series", series_rows)]:
                pred_path = eval_dir / f"{level}_level_predictions.csv"
                pred = pd.read_csv(pred_path)
                for method, column in METHODS.items():
                    group_column = "study_group" if level == "study" else "series_group"
                    for _, record in pred.iterrows():
                        target.append({"seed": seed, "attack": attack,
                                       "method": method,
                                       "group_id": record[group_column],
                                       "ground_truth": int(record["ground_truth"]),
                                       "prediction": int(record[column])})
            print(f"Finished {tag}: {log['modified_file_count']} modified files", flush=True)

    pd.DataFrame(run_rows).to_csv(output / "per_run_file_metrics.csv", index=False)
    study_frame = pd.DataFrame(study_rows)
    series_frame = pd.DataFrame(series_rows)
    study_frame.to_csv(output / "study_level_predictions_all_runs.csv", index=False)
    series_frame.to_csv(output / "series_level_predictions_all_runs.csv", index=False)
    rng = np.random.default_rng(20261114)
    summaries = []
    for unit, frame in [("study", study_frame), ("series", series_frame)]:
      for attack in args.attacks:
        for method in METHODS:
            subset = frame[(frame.attack == attack) & (frame.method == method)]
            per_seed = []
            for seed, run in subset.groupby("seed"):
                pos = run.loc[run.ground_truth == 1]
                neg = run.loc[run.ground_truth == 0]
                per_seed.append({"recall": float(pos.prediction.mean()) if len(pos) else np.nan,
                                 "fpr": float(neg.prediction.mean()) if len(neg) else np.nan})
            bootstrap_by_seed = []
            for _, run in subset.groupby("seed"):
                truth = run.ground_truth.to_numpy(dtype=int)
                prediction = run.prediction.to_numpy(dtype=int)
                positive_indices = np.flatnonzero(truth == 1)
                negative_indices = np.flatnonzero(truth == 0)
                if not len(positive_indices) or not len(negative_indices):
                    continue
                draws_for_run = np.empty((args.bootstrap_iterations, 2), dtype=float)
                for b in range(args.bootstrap_iterations):
                    pos_sample = rng.choice(positive_indices, size=len(positive_indices), replace=True)
                    neg_sample = rng.choice(negative_indices, size=len(negative_indices), replace=True)
                    draws_for_run[b, 0] = prediction[pos_sample].mean()
                    draws_for_run[b, 1] = prediction[neg_sample].mean()
                bootstrap_by_seed.append(draws_for_run)
            boot_draws = np.mean(np.stack(bootstrap_by_seed), axis=0) if bootstrap_by_seed else np.empty((0, 2))
            draws = np.asarray(boot_draws)
            summaries.append({"evaluation_unit": unit, "attack": attack, "method": method,
                              "seeds": len(per_seed),
                              "mean_study_recall_across_seeds": float(np.nanmean([x["recall"] for x in per_seed])),
                              "mean_study_fpr_across_seeds": float(np.nanmean([x["fpr"] for x in per_seed])),
                              "study_bootstrap_recall_ci_low": float(np.percentile(draws[:, 0], 2.5)),
                              "study_bootstrap_recall_ci_high": float(np.percentile(draws[:, 0], 97.5)),
                              "study_bootstrap_fpr_ci_low": float(np.percentile(draws[:, 1], 2.5)),
                              "study_bootstrap_fpr_ci_high": float(np.percentile(draws[:, 1], 97.5))})
    pd.DataFrame(summaries).to_csv(output / "study_bootstrap_summary.csv", index=False)
    config = {"seeds": args.seeds, "attacks": args.attacks,
              "attack_fraction": args.attack_fraction,
              "bootstrap_iterations": args.bootstrap_iterations,
              "bootstrap_unit": "baseline-keyed study, stratified by attack label",
              "note": "Intervals are descriptive for the ten-study CPAISD test fold and do not represent external-population uncertainty."}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSeed sensitivity experiment complete: {output}")


if __name__ == "__main__":
    main()
