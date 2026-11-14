"""Test a feature-aware, same-length metadata edit against trained detectors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks import create_attack
from src.detection import detect_directory
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.features import metadata_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean test fold")
    parser.add_argument("--baseline", required=True, help="Clean per-file metadata manifest")
    parser.add_argument("--metadata-model", required=True)
    parser.add_argument("--autoencoder-model", required=True)
    parser.add_argument("--output", required=True, help="Fresh output folder")
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input).resolve()
    attack_dir = output / "attack"
    log = create_attack(source, attack_dir, "protocol_name_same_length",
                        args.attack_fraction, args.seed)
    clean = detect_directory(source, args.baseline, args.metadata_model,
                             output / "clean_predictions.csv", args.autoencoder_model)
    attacked = detect_directory(attack_dir, args.baseline, args.metadata_model,
                                output / "attacked_predictions.csv", args.autoencoder_model)
    clean = clean.set_index("file")
    attacked = attacked.set_index("file")
    changed_keys = [key for key, info in log["files"].items() if info.get("label") == 1]
    rows = []
    for key in changed_keys:
        clean_ds = read_dataset(source / key, stop_before_pixels=True)
        attacked_ds = read_dataset(attack_dir / key, stop_before_pixels=True)
        clean_features = metadata_features(clean_ds)
        attacked_features = metadata_features(attacked_ds)
        feature_collision = clean_features == attacked_features
        before, after = clean.loc[key], attacked.loc[key]
        rows.append({"file": key, "feature_vector_unchanged": int(feature_collision),
                     "metadata_sha256_detected": int(after.sha256_flag),
                     "structural_rules_detected": int(after.rule_flag),
                     "isolation_forest_detected": int(after.isolation_forest_flag),
                     "autoencoder_detected": int(after.autoencoder_flag),
                     "non_hash_detectors_all_evaded": int(not (after.rule_flag or after.isolation_forest_flag or after.autoencoder_flag)),
                     "if_score_change": float(after.isolation_forest_score - before.isolation_forest_score),
                     "ae_error_change": float(after.autoencoder_reconstruction_error - before.autoencoder_reconstruction_error)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "adaptive_evasion_per_file.csv", index=False)
    summary = {
        "attack": "ProtocolName same-length feature-collision edit",
        "seed": args.seed,
        "modified_files": len(frame),
        "feature_vector_collision_rate": float(frame.feature_vector_unchanged.mean()) if len(frame) else 0.0,
        "metadata_hash_detection_rate": float(frame.metadata_sha256_detected.mean()) if len(frame) else 0.0,
        "structural_rule_detection_rate": float(frame.structural_rules_detected.mean()) if len(frame) else 0.0,
        "isolation_forest_detection_rate": float(frame.isolation_forest_detected.mean()) if len(frame) else 0.0,
        "autoencoder_detection_rate": float(frame.autoencoder_detected.mean()) if len(frame) else 0.0,
        "all_non_hash_detectors_evaded_rate": float(frame.non_hash_detectors_all_evaded.mean()) if len(frame) else 0.0,
        "mean_absolute_if_score_change": float(frame.if_score_change.abs().mean()) if len(frame) else 0.0,
        "mean_absolute_ae_error_change": float(frame.ae_error_change.abs().mean()) if len(frame) else 0.0,
        "interpretation": "This attacker knows that ProtocolName is represented by presence and length, and chooses a same-length substitution. The experiment measures this explicit feature-map evasion; it is not an adaptive optimization against unknown detector internals.",
    }
    (output / "adaptive_evasion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output / "adaptive_evasion_summary.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"\nAdaptive metadata evasion experiment complete: {output}")


if __name__ == "__main__":
    main()
