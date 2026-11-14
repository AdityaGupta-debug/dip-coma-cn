"""Evaluate CPAISD-trained metadata models on an external DICOM cohort.

The external directory is treated as a clean reference cohort. Controlled
metadata edits are generated as copies for sensitivity measurements; clean
external files also provide a domain-shift alarm-rate estimate.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import ATTACKS, create_attack
from src.dicom_io import find_dicom_files
from src.evaluation import evaluate_attack
from src.integrity import create_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="External cohort root containing DICOM files")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--metadata-model", required=True, help="Isolation Forest trained on CPAISD")
    parser.add_argument("--autoencoder-model", required=True, help="Autoencoder trained on CPAISD")
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS,
                        default=["modality", "pixel_spacing", "manufacturer_same_length"])
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    files = find_dicom_files(source)
    if not files:
        raise ValueError(f"No readable DICOM files found under {source}")
    output.mkdir(parents=True, exist_ok=True)

    baseline_path = output / "external_clean_baseline.json"
    create_baseline(source, baseline_path, include_pixel_hash=False)
    all_metrics = []
    attack_root = output / "attacks"
    for attack in args.attacks:
        attack_dir = attack_root / attack
        attack_log = create_attack(source, attack_dir, attack,
                                   args.attack_fraction, args.seed)
        result_dir = output / "evaluations" / attack
        summary, _ = evaluate_attack(attack_dir, baseline_path,
                                     args.metadata_model, attack_dir / "attack_log.json",
                                     result_dir, args.autoencoder_model)
        summary.insert(0, "attack", attack)
        summary.insert(1, "modified_files", attack_log["modified_file_count"])
        all_metrics.append(summary)

    clean_result = output / "clean_external"
    from src.detection import detect_directory
    clean_predictions = detect_directory(source, baseline_path, args.metadata_model,
                                         clean_result / "predictions.csv", args.autoencoder_model)
    method_columns = {
        "Metadata SHA-256": "sha256_flag", "Structural rules": "rule_flag",
        "Isolation Forest": "isolation_forest_flag", "Autoencoder": "autoencoder_flag",
        "Hybrid": "hybrid_or_flag",
    }
    clean_alarm = pd.DataFrame([
        {"method": name, "external_files": len(clean_predictions),
         "alarm_count": int(clean_predictions[column].sum()),
         "alarm_rate": float(clean_predictions[column].mean())}
        for name, column in method_columns.items()
    ])
    clean_alarm.to_csv(clean_result / "alarm_rates.csv", index=False)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics.to_csv(output / "external_attack_metrics.csv", index=False)
    config = {"external_input": str(source), "external_file_count": len(files),
              "source_model_training_cohort": "CPAISD official training fold",
              "attacks": args.attacks, "attack_fraction": args.attack_fraction,
              "seed": args.seed,
              "interpretation": "Clean external alarm rates measure domain-shift sensitivity. Synthetic-attack metrics use an external clean reference generated from the same external files; hash performance is therefore a controlled integrity check, not cross-domain learned generalization."}
    (output / "external_evaluation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = ["# External DICOM cohort evaluation", "",
              f"- External files: {len(files)}", "- Source models: trained on CPAISD training data",
              f"- Attacks: {', '.join(args.attacks)}", "",
              "## Clean external alarm rates", clean_alarm.to_markdown(index=False), "",
              "## Controlled external attack metrics", metrics.to_markdown(index=False), "",
              "The clean alarm rates show how CPAISD-trained detectors respond to an untouched external cohort. Attack metrics use a clean per-file reference created from that external cohort; SHA-256 performance is expected under that protocol. Use the learned detector rows to assess transfer under the scripted attack conditions. This is not clinical validation.", ""]
    (output / "EXTERNAL_EVALUATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(clean_alarm.to_string(index=False))
    print(metrics.to_string(index=False))
    print(f"\nExternal evaluation complete: {output}")


if __name__ == "__main__":
    main()
