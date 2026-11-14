"""Compare detector alarms on controlled benign-like DICOM changes and attacks.

The benign labels mean only that a predeclared, non-pixel transformation was
applied. They are not clinical judgments and do not model a specific PACS.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import detect_directory
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.integrity import load_baseline, metadata_payload, metadata_sha256, pixel_sha256

METHODS = {
    "Metadata SHA-256": "sha256_flag",
    "Structural rules": "rule_flag",
    "Isolation Forest": "isolation_forest_flag",
    "Autoencoder": "autoencoder_flag",
    "Metadata OR hybrid": "hybrid_or_flag",
}

# This is a limited privacy-field cleanup proxy, not a complete DICOM
# de-identification implementation. CPAISD may already omit these attributes.
PRIVACY_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
)
BENIGN_ALLOWED_KEYWORDS = set(PRIVACY_TAGS) | {
    "PatientIdentityRemoved", "DeidentificationMethod"
}


def clinical_payload_without_allowed_changes(ds):
    payload = metadata_payload(ds)
    payload.pop("FILE_META", None)
    for keyword in BENIGN_ALLOWED_KEYWORDS:
        element = ds.data_element(keyword) if keyword in ds else None
        if element is not None:
            payload.pop(f"{element.tag.group:04X}{element.tag.element:04X}", None)
    return payload


def copy_clean_tree(source: Path, target: Path, files: list[Path]) -> None:
    for path in files:
        relative = relative_key(path, source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def make_benign_condition(source: Path, condition_root: Path, files: list[Path], condition: str):
    target_root = condition_root / condition
    copy_clean_tree(source, target_root, files)
    audit = []
    for path in files:
        relative = relative_key(path, source)
        target = target_root / relative
        original = read_dataset(path, stop_before_pixels=False)
        original_clinical = clinical_payload_without_allowed_changes(original)
        original_meta_hash = metadata_sha256(original)
        original_pixel_hash = pixel_sha256(original)

        edited = read_dataset(target, stop_before_pixels=False)
        changed_tags: list[str] = []
        if condition == "implementation_version_update":
            if edited.file_meta is None:
                raise ValueError(f"Missing DICOM file meta: {target}")
            edited.file_meta.ImplementationVersionName = "RCVR_UPDATE_V1"
            changed_tags.append("(0002,0013) ImplementationVersionName")
        elif condition == "privacy_field_cleanup_proxy":
            for keyword in PRIVACY_TAGS:
                if keyword in edited:
                    del edited[keyword]
                    changed_tags.append(keyword)
            edited.PatientIdentityRemoved = "YES"
            edited.DeidentificationMethod = "Synthetic limited privacy-field cleanup"
            changed_tags.extend(["PatientIdentityRemoved", "DeidentificationMethod"])
        elif condition == "lossless_reserialization":
            # Loading and saving through pydicom without changing attributes is
            # a controlled rewrite proxy. It is not a named PACS export.
            changed_tags.append("serialization only (no requested attribute edit)")
        else:
            raise ValueError(f"Unknown benign condition: {condition}")

        edited.save_as(str(target), enforce_file_format=True)
        result = read_dataset(target, stop_before_pixels=False)
        result_clinical = clinical_payload_without_allowed_changes(result)
        audit.append({
            "file": relative,
            "condition": condition,
            "ground_truth_benign_proxy": 1,
            "changed_tags": ";".join(changed_tags),
            "non_target_clinical_attributes_unchanged": int(original_clinical == result_clinical),
            "pixel_data_unchanged": int(original_pixel_hash == pixel_sha256(result)),
            "metadata_digest_changed": int(original_meta_hash != metadata_sha256(result)),
        })
    return target_root, audit


def summarize_predictions(frame: pd.DataFrame, labels: list[int], condition: str,
                          clean_frame: pd.DataFrame | None = None):
    rows = []
    labels_series = pd.Series(labels, index=frame.index, dtype=int)
    clean_by_file = (clean_frame.set_index("file") if clean_frame is not None else None)
    for method, column in METHODS.items():
        alarms = frame[column].fillna(0).astype(int)
        positive = labels_series == 1
        negative = labels_series == 0
        clean_flags = None
        added_alarms = None
        if clean_by_file is not None:
            clean_flags = frame["file"].map(clean_by_file[column]).fillna(0).astype(int)
            added_alarms = int(((alarms == 1) & (clean_flags == 0)).sum())
        rows.append({
            "condition": condition,
            "transformation_class": "benign-like proxy" if not positive.any() else "scripted attack",
            "method": method,
            "n_files": int(len(frame)),
            "positive_files": int(positive.sum()),
            "alarm_count": int(alarms.sum()),
            "alarm_rate": float(alarms.mean()) if len(alarms) else 0.0,
            "attack_recall": float(alarms[positive].mean()) if positive.any() else None,
            "attack_false_positive_rate": float(alarms[negative].mean()) if negative.any() else None,
            "clean_alarm_count": int(clean_flags.sum()) if clean_flags is not None else None,
            "new_alarm_count_vs_clean": added_alarms,
            "new_alarm_rate_vs_clean": (added_alarms / len(frame)
                                         if added_alarms is not None and len(frame) else None),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean DICOM test fold")
    parser.add_argument("--manifest", required=True, help="Clean test baseline JSON")
    parser.add_argument("--model", required=True, help="Trained Isolation Forest joblib")
    parser.add_argument("--ae-model", required=True, help="Trained autoencoder joblib")
    parser.add_argument("--attacks-root", required=True, help="Existing attack folders with attack_log.json")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    attacks_root = Path(args.attacks_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    files = find_dicom_files(source)
    if not files:
        raise ValueError(f"No DICOM files found under {source}")
    if not attacks_root.is_dir():
        raise FileNotFoundError(f"Attack directory not found: {attacks_root}")
    baseline = load_baseline(args.manifest)
    stale = []
    for path in files:
        key = relative_key(path, source)
        reference = baseline.get("files", {}).get(key)
        if reference is None or reference.get("metadata_sha256") != metadata_sha256(
            read_dataset(path, stop_before_pixels=True)
        ):
            stale.append(key)
    if stale:
        sample = ", ".join(stale[:3])
        raise ValueError(
            f"The baseline does not match the current clean input for {len(stale)} files "
            f"(examples: {sample}). Regenerate it from --input with the current code before running."
        )
    output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    audit_rows = []
    clean_predictions = detect_directory(
        source, args.manifest, args.model,
        output / "predictions" / "clean_test.csv", args.ae_model,
    )
    benign_conditions = (
        "implementation_version_update",
        "privacy_field_cleanup_proxy",
        "lossless_reserialization",
    )
    for condition in benign_conditions:
        condition_root, audit = make_benign_condition(source, output / "benign", files, condition)
        audit_rows.extend(audit)
        predictions = detect_directory(
            condition_root, args.manifest, args.model,
            output / "predictions" / f"{condition}.csv", args.ae_model,
        )
        summary_rows.extend(summarize_predictions(
            predictions, [0] * len(predictions), condition, clean_predictions
        ))

    for attack_dir in sorted(p for p in attacks_root.iterdir() if p.is_dir()):
        attack_log_path = attack_dir / "attack_log.json"
        if not attack_log_path.exists():
            continue
        predictions = detect_directory(
            attack_dir, args.manifest, args.model,
            output / "predictions" / f"attack_{attack_dir.name}.csv", args.ae_model,
        )
        attack_log = json.loads(attack_log_path.read_text(encoding="utf-8"))
        labels = [int(attack_log.get("files", {}).get(name, {}).get("label", 0))
                  for name in predictions["file"]]
        predictions = predictions.copy()
        predictions["ground_truth_attack"] = labels
        predictions.to_csv(output / "predictions" / f"attack_{attack_dir.name}_labeled.csv", index=False)
        if not any(labels):
            continue
        summary_rows.extend(summarize_predictions(
            predictions, labels, attack_dir.name, clean_predictions
        ))

    if not any(row["transformation_class"] == "scripted attack" for row in summary_rows):
        raise ValueError("No labeled attack files found; check --attacks-root contents.")
    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    summary.to_csv(output / "benign_vs_attack_summary.csv", index=False)
    audit.to_csv(output / "benign_transformation_audit.csv", index=False)

    benign_audit = audit.groupby("condition", as_index=False).agg(
        files=("file", "count"), non_target_clinical_attributes_unchanged_rate=("non_target_clinical_attributes_unchanged", "mean"),
        pixel_data_unchanged_rate=("pixel_data_unchanged", "mean"),
        metadata_digest_changed_rate=("metadata_digest_changed", "mean"),
    )
    benign_audit.to_csv(output / "benign_transformation_audit_summary.csv", index=False)
    report = [
        "# Controlled benign-like changes versus scripted attacks", "",
        "The labels identify the scripted transformation, not malicious intent or clinical harm.",
        "Benign-like conditions are proxies and do not represent a named PACS vendor or validated clinical workflow.",
        "The privacy-field cleanup is deliberately limited; it is not a complete standards-conformant de-identification process.",
        "All source files are copied before modification.", "",
        "## Benign transformation audit", benign_audit.to_markdown(index=False), "",
        "## Detector alarm rates", summary.to_markdown(index=False), "",
        "For benign-like conditions, alarm_rate is the fraction of transformed files flagged, while new_alarm_rate_vs_clean counts only alerts newly raised relative to each same untouched file. For attack conditions, attack_recall is computed on labeled edited files and attack_false_positive_rate on unedited files. These describe the specified transformations; they do not establish a general harmful-versus-harmless classifier.",
    ]
    (output / "BENIGN_VS_ATTACK_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nExperiment complete: {output}")


if __name__ == "__main__":
    main()
