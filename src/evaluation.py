from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .detection import detect_directory

METHODS = {
    "SHA-256 metadata integrity": "sha256_flag",
    "Structural rules": "rule_flag",
    "Isolation Forest": "isolation_forest_flag",
    "Autoencoder": "autoencoder_flag",
    "Isolation Forest + Autoencoder": "if_ae_flag",
    "Rules + Isolation Forest": "rules_if_flag",
    "Hybrid (SHA-256 + rules + IF + Autoencoder)": "hybrid_or_flag",
}

def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }

def evaluate_attack(input_dir, baseline_path, model_path, attack_log_path, output_dir, autoencoder_model_path=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    detection = detect_directory(input_dir, baseline_path, model_path, out / "file_predictions.csv", autoencoder_model_path)
    log = json.loads(Path(attack_log_path).read_text(encoding="utf-8"))
    y = [int(log["files"].get(f, {}).get("label", 0)) for f in detection["file"]]
    summary = []
    for method, column in METHODS.items():
        row = {"method": method, **_metrics(y, detection[column].astype(int).tolist())}
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "metrics.csv", index=False)
    detection["ground_truth"] = y
    detection.to_csv(out / "file_predictions_with_ground_truth.csv", index=False)
    # Slices within a study are correlated. Report one additional result per
    # study, where a study is positive if any instance was attacked/flagged.
    study = detection.groupby("study_group", as_index=False).agg(
        ground_truth=("ground_truth", "max"),
        **{column: (column, "max") for column in METHODS.values()},
        file_count=("file", "count"),
    )
    study_rows = [{"method": method, "n_studies": int(len(study)),
                   **_metrics(study["ground_truth"].astype(int).tolist(),
                              study[column].astype(int).tolist())}
                  for method, column in METHODS.items()]
    pd.DataFrame(study_rows).to_csv(out / "study_level_metrics.csv", index=False)
    study.to_csv(out / "study_level_predictions.csv", index=False)
    series = detection.groupby("series_group", as_index=False).agg(
        ground_truth=("ground_truth", "max"),
        **{column: (column, "max") for column in METHODS.values()},
        file_count=("file", "count"),
    )
    series_rows = [{"method": method, "n_series": int(len(series)),
                    **_metrics(series["ground_truth"].astype(int).tolist(),
                               series[column].astype(int).tolist())}
                   for method, column in METHODS.items()]
    pd.DataFrame(series_rows).to_csv(out / "series_level_metrics.csv", index=False)
    series.to_csv(out / "series_level_predictions.csv", index=False)
    return summary_df, detection

def run_ablation(attack_dirs, baseline_path, model_path, output_dir, autoencoder_model_path=None):
    """
    attack_dirs: dict attack_name -> directory containing attack_log.json and DICOMs.
    Aggregates file-level predictions across each attack dataset.
    """
    all_rows = []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for attack_name, directory in attack_dirs.items():
        directory = Path(directory)
        summary, frame = evaluate_attack(
            directory, baseline_path, model_path,
            directory / "attack_log.json", out / attack_name, autoencoder_model_path
        )
        frame["attack"] = attack_name
        all_rows.append(frame)
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if combined.empty:
        raise ValueError("No attack datasets supplied for ablation.")
    y = combined["ground_truth"].astype(int).tolist()
    rows = []
    for method, column in METHODS.items():
        rows.append({"method": method, **_metrics(y, combined[column].astype(int).tolist())})
    result = pd.DataFrame(rows)
    result.to_csv(out / "ablation_summary.csv", index=False)
    combined.to_csv(out / "all_file_predictions.csv", index=False)
    return result
