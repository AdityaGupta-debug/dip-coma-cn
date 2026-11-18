"""Clean-training kNN anomaly scoring for DICOM metadata."""
from __future__ import annotations

from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .dicom_io import find_dicom_files, read_dataset
from .features import metadata_features


def _rows(paths):
    return [metadata_features(read_dataset(path, stop_before_pixels=True)) for path in paths]


def _transform(paths, vectorizer, scaler):
    matrix = vectorizer.transform(_rows(paths))
    return scaler.transform(matrix).astype(np.float32)


def fit_knn_metadata(clean_train_dir, clean_validation_dir, model_path, *,
                     n_neighbors=5, threshold_percentile=99.0,
                     max_train_files=0, random_state=42, metric="euclidean"):
    """Fit a clean-reference kNN distance model and calibrate on clean validation.

    The anomaly score is the mean distance to the k nearest clean training
    instances. Higher values mean more anomalous. Train and validation must be
    the official study-disjoint clean folds.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100")
    train_paths = find_dicom_files(clean_train_dir)
    val_paths = find_dicom_files(clean_validation_dir)
    if len(train_paths) <= n_neighbors or not val_paths:
        raise ValueError("Need more clean training files than k and a nonempty validation fold")
    if max_train_files and len(train_paths) > max_train_files:
        train_paths = random.Random(random_state).sample(train_paths, max_train_files)
        if len(train_paths) <= n_neighbors:
            raise ValueError("max_train_files must exceed n_neighbors")

    vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
    train_raw = vectorizer.fit_transform(_rows(train_paths))
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_raw).astype(np.float32)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto", n_jobs=-1)
    model.fit(train_x)
    val_x = _transform(val_paths, vectorizer, scaler)
    distances, _ = model.kneighbors(val_x, n_neighbors=n_neighbors)
    val_scores = distances.mean(axis=1)
    threshold = float(np.percentile(val_scores, threshold_percentile))

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "model": model,
        "threshold": threshold,
        "threshold_percentile": float(threshold_percentile),
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "training_files": len(train_paths),
        "validation_files": len(val_paths),
        "random_state": int(random_state),
        "score_definition": "mean_distance_to_k_nearest_clean_training_instances",
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def score_knn_metadata(paths, model_bundle):
    """Return high-is-anomalous kNN scores and calibrated binary flags."""
    paths = list(paths)
    if not paths:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint8)
    x = _transform(paths, model_bundle["vectorizer"], model_bundle["scaler"])
    k = int(model_bundle["n_neighbors"])
    distances, _ = model_bundle["model"].kneighbors(x, n_neighbors=k)
    scores = distances.mean(axis=1)
    return scores, (scores > float(model_bundle["threshold"])).astype(np.uint8)
