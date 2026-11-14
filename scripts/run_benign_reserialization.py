"""Measure detector alarms after lossless DICOM parse/write reserialization."""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import detect_directory
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.integrity import metadata_payload, metadata_sha256, pixel_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean DICOM test fold")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--manifest", required=True, help="Clean metadata reference manifest")
    parser.add_argument("--model", required=True, help="Trained Isolation Forest model")
    parser.add_argument("--ae-model", required=True, help="Trained metadata autoencoder")
    args = parser.parse_args()

    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = find_dicom_files(source)
    if not files:
        raise ValueError(f"No readable DICOM files found under {source}")

    records = []
    for path in files:
        key = relative_key(path, source)
        target = output / "reserialized" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        original_bytes = path.read_bytes()
        original = read_dataset(path, stop_before_pixels=False)
        original_dataset_payload = metadata_payload(original)
        original_dataset_payload.pop("FILE_META", None)
        original_meta = metadata_sha256(original)
        original_pixel = pixel_sha256(original)
        # A receiving implementation may update its DICOM implementation
        # identifier while preserving the image and clinical dataset values.
        if getattr(original, "file_meta", None) is None:
            raise ValueError(f"DICOM file has no file-meta header: {path}")
        original.file_meta.ImplementationVersionName = "PACS_REEXPORT_V1"
        original.save_as(str(target), enforce_file_format=True)
        result = read_dataset(target, stop_before_pixels=False)
        result_dataset_payload = metadata_payload(result)
        result_dataset_payload.pop("FILE_META", None)
        records.append({
            "file": key,
            "raw_file_bytes_changed": int(original_bytes != target.read_bytes()),
            "clinical_dataset_attributes_unchanged": int(original_dataset_payload == result_dataset_payload),
            "full_header_digest_changed": int(original_meta != metadata_sha256(result)),
            "pixel_data_unchanged": int(original_pixel == pixel_sha256(result)),
        })

    predictions = detect_directory(output / "reserialized", args.manifest, args.model,
                                   output / "detector_predictions.csv", args.ae_model)
    detail = pd.DataFrame(records).merge(predictions, on="file", how="left")
    detail.to_csv(output / "reserialization_results.csv", index=False)
    flag_columns = {
        "Metadata SHA-256": "sha256_flag",
        "Structural rules": "rule_flag",
        "Isolation Forest": "isolation_forest_flag",
        "Autoencoder": "autoencoder_flag",
        "Metadata OR hybrid": "hybrid_or_flag",
    }
    summary = pd.DataFrame([
        {"method": name, "files": int(len(detail)),
         "false_positive_count": int(detail[column].fillna(0).sum()),
         "false_positive_rate": float(detail[column].fillna(0).mean())}
        for name, column in flag_columns.items()
    ])
    summary.to_csv(output / "reserialization_summary.csv", index=False)
    report = [
        "# Benign DICOM reserialization control", "",
        f"- Input files: {len(files)}",
        f"- Raw file bytes changed: {sum(x['raw_file_bytes_changed'] for x in records)}",
        f"- Clinical dataset attributes unchanged: {sum(x['clinical_dataset_attributes_unchanged'] for x in records)}/{len(files)}",
        f"- Full header digest changed: {sum(x['full_header_digest_changed'] for x in records)}/{len(files)}",
        f"- PixelData unchanged: {sum(x['pixel_data_unchanged'] for x in records)}/{len(files)}", "",
        "This control changes the DICOM ImplementationVersionName to a synthetic receiver identifier while preserving the clinical dataset attributes and PixelData. It is a controlled benign-like file-meta normalization, not a simulation of a specific PACS vendor.", "",
        summary.to_markdown(index=False), "",
        "See reserialization_results.csv for per-file byte, canonical-metadata, pixel, and detector outcomes.",
    ]
    (output / "RESERIALIZATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
