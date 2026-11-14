from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd
from .dicom_io import find_dicom_files, read_dataset, relative_key
from .integrity import load_baseline, metadata_sha256, pixel_sha256
from .validation import validate_datasets
from .anomaly import load_model, predict_anomalies
from .deep_anomaly import predict_autoencoder_with_explanations

def detect_directory(input_dir, baseline_path, model_path, output_csv=None, autoencoder_model_path=None):
    root = Path(input_dir).resolve()
    baseline = load_baseline(baseline_path)
    paths = find_dicom_files(root)
    datasets = [(relative_key(p, root), read_dataset(p, stop_before_pixels=True)) for p in paths]
    rules = validate_datasets(datasets)
    model_bundle = load_model(model_path)
    if_results = predict_anomalies(datasets, model_bundle)
    ae_results = (predict_autoencoder_with_explanations(datasets, autoencoder_model_path)
                  if autoencoder_model_path and Path(autoencoder_model_path).exists()
                  else {key: (0, 0.0, []) for key, _ in datasets})
    ae_threshold = (float(joblib.load(autoencoder_model_path).get("threshold", float("inf")))
                    if autoencoder_model_path and Path(autoencoder_model_path).exists()
                    else float("inf"))
    pixel_enabled = bool(baseline.get("pixel_hashing_enabled", False))
    rows = []
    for rel_path, ds in datasets:
        ref = baseline["files"].get(rel_path)
        # Keep study-level evaluation tied to the clean reference identity.
        # Attackers can alter StudyInstanceUID, so the evaluated header is not
        # a stable grouping key for the UID-consistency experiment.
        baseline_study_uid = str((ref or {}).get("study_instance_uid", "")).strip()
        baseline_series_uid = str((ref or {}).get("series_instance_uid", "")).strip()
        current_study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
        current_series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        study_uid = baseline_study_uid or current_study_uid
        series_uid = baseline_series_uid or current_series_uid
        if not study_uid:
            study_uid = "MISSING:" + Path(rel_path).parent.as_posix()
        if not series_uid:
            series_uid = "MISSING:" + Path(rel_path).parent.as_posix()
        meta_mismatch = int(ref is None or metadata_sha256(ds) != ref["metadata_sha256"])
        pixel_mismatch = None
        if pixel_enabled:
            full_ds = read_dataset(root / rel_path, stop_before_pixels=False)
            pixel_mismatch = int(ref is None or pixel_sha256(full_ds) != ref.get("pixel_sha256"))
            del full_ds
        rule_flag = int(bool(rules[rel_path]))
        if_flag, if_score = if_results[rel_path]
        ae_flag, ae_error, ae_top_features = ae_results[rel_path]
        alert_sources, explanations = [], []
        if meta_mismatch:
            alert_sources.append("metadata_sha256")
            explanations.append("metadata digest differs from the trusted clean reference; the hash does not identify which tag changed")
        if pixel_mismatch:
            alert_sources.append("pixel_sha256")
            explanations.append("PixelData digest differs from the trusted clean reference")
        if rule_flag:
            alert_sources.append("structural_rules")
            explanations.append("structural findings: " + ", ".join(rules[rel_path]))
        if if_flag:
            alert_sources.append("isolation_forest")
            explanations.append(f"Isolation Forest score {if_score:.6g} is below clean-validation threshold {model_bundle.get('threshold', 0.0):.6g}")
        if ae_flag:
            alert_sources.append("autoencoder")
            explanations.append(f"autoencoder reconstruction error {ae_error:.6g} exceeds clean-validation threshold {ae_threshold:.6g}; largest feature errors: " +
                                ", ".join(x["feature"] for x in ae_top_features[:3]))
        rows.append({
            # A pseudonymous group key allows study-level analysis without
            # exporting raw DICOM UIDs into routine prediction tables.
            "study_group": hashlib.sha256(study_uid.encode("utf-8")).hexdigest()[:16],
            "series_group": hashlib.sha256(series_uid.encode("utf-8")).hexdigest()[:16],
            "file": rel_path, "sha256_flag": meta_mismatch,
            "pixel_integrity_flag": pixel_mismatch, "rule_flag": rule_flag,
            "rule_findings": ";".join(rules[rel_path]),
            "isolation_forest_flag": if_flag, "isolation_forest_score": if_score,
            "isolation_forest_threshold": float(model_bundle.get("threshold", 0.0)),
            "autoencoder_flag": ae_flag, "autoencoder_reconstruction_error": ae_error,
            "autoencoder_threshold": ae_threshold,
            "autoencoder_top_features": json.dumps(ae_top_features, ensure_ascii=False),
            "alert_sources": ";".join(alert_sources),
            "alert_explanation": " | ".join(explanations),
            "if_ae_flag": int(bool(if_flag or ae_flag)),
            "hybrid_or_flag": int(bool(meta_mismatch or rule_flag or if_flag or ae_flag)),
            "rules_if_flag": int(bool(rule_flag or if_flag)),
        })
    frame = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    return frame
