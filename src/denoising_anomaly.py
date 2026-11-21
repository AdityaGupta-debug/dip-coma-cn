"""Denoising metadata autoencoder for unsupervised DICOM anomaly detection.

The model is trained on clean feature vectors. During training, Gaussian noise
and random feature masking are applied to the input while the target remains
the original clean vector. Validation thresholds use clean-input reconstruction
MSE, so thresholds are comparable to the standard MSE autoencoder.
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .deep_anomaly import MetadataAutoencoder, _matrix
from .dicom_io import find_dicom_files
from .features import metadata_features


def _corrupt(batch: torch.Tensor, noise_std: float,
             mask_probability: float) -> torch.Tensor:
    """Apply reproducible-per-run Gaussian noise and feature masking."""
    noisy = batch + torch.randn_like(batch) * noise_std if noise_std else batch.clone()
    if mask_probability:
        keep = torch.rand_like(noisy).ge(mask_probability)
        noisy = noisy * keep
    return noisy


def _clean_errors(model, matrix: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(matrix), 1024):
            target = torch.from_numpy(matrix[start:start + 1024]).to(device)
            values.append(torch.mean((model(target) - target).square(), dim=1)
                          .cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def train_denoising_autoencoder(
    clean_dir,
    model_path,
    validation_dir=None,
    epochs=40,
    batch_size=32,
    learning_rate=1e-3,
    threshold_percentile=99.0,
    random_state=42,
    max_train_files=30000,
    max_validation_files=10000,
    optimizer_name="rmsprop",
    noise_std=0.10,
    mask_probability=0.05,
):
    """Fit a denoising AE using clean targets and calibrate on clean inputs."""
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    if not 0.0 <= mask_probability < 1.0:
        raise ValueError("mask_probability must be in [0, 1)")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100")

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    paths = find_dicom_files(clean_dir)
    if len(paths) < 10:
        raise ValueError("Denoising autoencoder needs at least 10 clean DICOM files")
    if max_train_files and len(paths) > max_train_files:
        paths = random.Random(random_state).sample(paths, max_train_files)
    X, vectorizer, scaler = _matrix(paths, fit=True)

    X_val = None
    validation_files = 0
    if validation_dir:
        val_paths = find_dicom_files(validation_dir)
        if not val_paths:
            raise ValueError("Validation directory contains no readable DICOM files")
        if max_validation_files and len(val_paths) > max_validation_files:
            val_paths = random.Random(random_state + 1).sample(
                val_paths, max_validation_files
            )
        X_val, _, _ = _matrix(val_paths, vectorizer, scaler)
        validation_files = len(val_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataAutoencoder(X.shape[1]).to(device)
    optimizer_name = str(optimizer_name).lower().replace(" ", "_")
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)
    elif optimizer_name == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=0.9)
    else:
        raise ValueError("optimizer_name must be adam, rmsprop, adamw, or sgd_momentum")

    tensor = torch.from_numpy(X)
    loader = DataLoader(
        TensorDataset(tensor), batch_size=max(1, min(batch_size, len(X))),
        shuffle=True,
    )
    history = {"train_denoising_mse": [], "validation_clean_mse": []}
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for (clean_batch,) in loader:
            clean_batch = clean_batch.to(device)
            corrupted_batch = _corrupt(clean_batch, noise_std, mask_probability)
            optimizer.zero_grad()
            loss = torch.mean((model(corrupted_batch) - clean_batch).square())
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        history["train_denoising_mse"].append(float(np.mean(batch_losses)))
        validation_source = X_val if X_val is not None else X
        history["validation_clean_mse"].append(
            float(_clean_errors(model, validation_source, device).mean())
        )

    calibration = _clean_errors(model, X_val if X_val is not None else X, device)
    threshold = float(np.percentile(calibration, threshold_percentile))
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(X.shape[1]), "vectorizer": vectorizer, "scaler": scaler,
        "threshold": threshold,
        "threshold_percentile": float(threshold_percentile),
        "training_files": len(paths), "validation_files": validation_files,
        "threshold_source": "clean_validation" if X_val is not None
        else "clean_training_fallback",
        "epochs": int(epochs), "optimizer": optimizer_name,
        "reconstruction_loss": "mse", "training_objective": "denoising_mse",
        "noise_std": float(noise_std),
        "mask_probability": float(mask_probability),
        "loss_history": history, "learning_rate": float(learning_rate),
        "random_state": int(random_state),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def predict_denoising_autoencoder(datasets, model_path, top_n=5):
    """Return flags, clean reconstruction MSE, and feature contributions."""
    bundle = joblib.load(model_path)
    model = MetadataAutoencoder(bundle["input_dim"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    rows = [metadata_features(ds) for _, ds in datasets]
    X = bundle["scaler"].transform(
        bundle["vectorizer"].transform(rows)
    ).astype(np.float32)
    with torch.no_grad():
        target = torch.from_numpy(X)
        contributions = (model(target) - target).square().numpy()
    errors = contributions.mean(axis=1)
    names = bundle["vectorizer"].get_feature_names_out()
    result = {}
    for i, (key, _) in enumerate(datasets):
        order = np.argsort(contributions[i])[::-1][:max(1, top_n)]
        top = [{"feature": str(names[j]),
                "squared_reconstruction_error": float(contributions[i, j])}
               for j in order]
        result[key] = (int(errors[i] > bundle["threshold"]),
                       float(errors[i]), top)
    return result
