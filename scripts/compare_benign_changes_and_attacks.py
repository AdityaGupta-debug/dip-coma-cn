"""Contrast detector alarms on a benign-like DICOM update and scripted attacks."""
import argparse
from pathlib import Path

import pandas as pd


METHODS = {
    "Metadata SHA-256": "sha256_flag",
    "Structural rules": "rule_flag",
    "Isolation Forest": "isolation_forest_flag",
    "Autoencoder": "autoencoder_flag",
    "Hybrid OR": "hybrid_or_flag",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign-results", required=True,
                        help="reserialization_results.csv from the synthetic benign-like control")
    parser.add_argument("--attack-evaluations-root", required=True,
                        help="Folder containing attack subfolders with file_predictions_with_ground_truth.csv")
    parser.add_argument("--output", required=True, help="Fresh output folder")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    benign = pd.read_csv(args.benign_results)
    rows = []
    for method, column in METHODS.items():
        if column not in benign:
            continue
        rows.append({"method": method, "condition": "synthetic benign-like file-meta rewrite",
                     "n_files": len(benign), "positive_files": 0,
                     "detection_rate_on_attack": None,
                     "alarm_rate_on_benign_change": float(benign[column].fillna(0).mean())})
    for pred_path in sorted(Path(args.attack_evaluations_root).resolve().rglob("file_predictions_with_ground_truth.csv")):
        frame = pd.read_csv(pred_path)
        if "ground_truth" not in frame:
            continue
        attack_name = pred_path.parent.relative_to(Path(args.attack_evaluations_root).resolve()).as_posix()
        positive = frame[frame.ground_truth == 1]
        for method, column in METHODS.items():
            if column not in positive:
                continue
            rows.append({"method": method, "condition": attack_name,
                         "n_files": len(frame), "positive_files": len(positive),
                         "detection_rate_on_attack": float(positive[column].astype(int).mean()) if len(positive) else None,
                         "alarm_rate_on_benign_change": None})
    result = pd.DataFrame(rows)
    result.to_csv(output / "benign_vs_attack_comparison.csv", index=False)
    (output / "BENIGN_VS_ATTACK_REPORT.md").write_text(
        "# Benign-change versus attack comparison\n\n" + result.to_markdown(index=False) +
        "\n\nThe benign condition is a synthetic ImplementationVersionName rewrite, not a vendor PACS export. Attack detection rates are computed only on labeled modified files; benign alarm rates are computed across the full benign-control cohort. They should not be interpreted as a universal malicious-versus-benign classifier without real workflow exports.\n",
        encoding="utf-8")
    print(result.to_string(index=False))
    print(f"\nBenign-versus-attack comparison complete: {output}")


if __name__ == "__main__":
    main()
