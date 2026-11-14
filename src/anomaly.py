from pathlib import Path
import joblib, random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import IsolationForest
from .dicom_io import find_dicom_files, read_dataset
from .features import metadata_features

def train_isolation_forest(clean_dir, model_path, contamination=0.05, random_state=42,
                           max_files=30000, validation_dir=None,
                           threshold_percentile=1.0, max_validation_files=10000):
    """Fit on clean data and calibrate a lower-tail anomaly cutoff on validation.

    ``contamination`` remains accepted for callers using the early prototype
    API; calibration, rather than that training-set estimate, sets the cutoff.
    """
    paths = find_dicom_files(clean_dir)
    if len(paths) < 5:
        raise ValueError("At least five clean DICOM files are recommended to train Isolation Forest.")
    if max_files and len(paths) > max_files:
        paths = random.Random(random_state).sample(paths, max_files)
    rows = [metadata_features(read_dataset(p, stop_before_pixels=True)) for p in paths]
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(rows)
    # Keep the forest's internal cutoff out of the decision rule. Calibrate the
    # score cutoff on independent clean validation studies instead.
    model = IsolationForest(
        n_estimators=300, contamination="auto",
        random_state=random_state, n_jobs=-1
    )
    model.fit(X)
    threshold_source = "clean_validation"
    if validation_dir:
        val_paths = find_dicom_files(validation_dir)
        if not val_paths:
            raise ValueError("Validation directory contains no readable DICOM files.")
        if max_validation_files and len(val_paths) > max_validation_files:
            val_paths = random.Random(random_state + 1).sample(val_paths, max_validation_files)
        val_rows = [metadata_features(read_dataset(p, stop_before_pixels=True)) for p in val_paths]
        val_scores = model.decision_function(vectorizer.transform(val_rows))
    else:
        # Backward-compatible fallback for standalone CLI use; research runs
        # should always provide the clean validation fold.
        val_scores = model.decision_function(X)
        threshold_source = "clean_training_fallback"
    if not 0.0 <= threshold_percentile <= 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100.")
    threshold = float(np.percentile(val_scores, threshold_percentile))
    bundle = {"vectorizer": vectorizer, "model": model, "training_files": len(paths),
              "threshold": threshold, "threshold_percentile": float(threshold_percentile),
              "threshold_source": threshold_source}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle

def load_model(model_path):
    return joblib.load(model_path)

def predict_anomalies(datasets, bundle):
    rows = [metadata_features(ds) for _, ds in datasets]
    X = bundle["vectorizer"].transform(rows)
    scores = bundle["model"].decision_function(X)  # lower = more anomalous
    threshold = bundle.get("threshold")
    if threshold is None:  # compatibility with models produced by v1
        threshold = 0.0
    return {datasets[i][0]: (int(scores[i] < threshold), float(scores[i])) for i in range(len(datasets))}
