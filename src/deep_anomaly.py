"""Metadata autoencoder for unsupervised DICOM metadata anomaly detection.

Fit only on clean training data. The validation directory, when supplied, is
used only to set the reconstruction-error threshold.
"""
from pathlib import Path
import random
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from .dicom_io import find_dicom_files, read_dataset
from .features import metadata_features

class MetadataAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def _matrix(paths, vectorizer=None, scaler=None, fit=False):
    rows = [metadata_features(read_dataset(path, stop_before_pixels=True)) for path in paths]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(rows) if fit else vectorizer.transform(rows)
    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    return X.astype(np.float32), vectorizer, scaler


def _reconstruction_loss_elements(target, reconstructed, loss_name="mse"):
    """Return one reconstruction-loss value per feature.

    Inputs have already been standardized. Robust alternatives reduce the
    influence of unusually large residuals while retaining a per-feature
    score for anomaly ranking and explanations.
    """
    residual = reconstructed - target
    loss_name = str(loss_name).lower().replace("-", "_")
    if loss_name == "mse":
        return residual.square()
    if loss_name == "mae":
        return residual.abs()
    if loss_name in {"smooth_l1", "huber"}:
        return torch.nn.functional.smooth_l1_loss(
            reconstructed, target, reduction="none", beta=1.0
        )
    if loss_name == "log_cosh":
        return torch.logaddexp(residual, -residual) - np.log(2.0)
    raise ValueError(
        f"Unsupported reconstruction loss {loss_name!r}; choose from "
        "mse, mae, smooth_l1, log_cosh"
    )

def train_autoencoder(clean_dir, model_path, validation_dir=None,
                      epochs=100, batch_size=32, learning_rate=1e-3,
                      threshold_percentile=99.0, random_state=42,
                      max_train_files=30000, max_validation_files=10000,
                      optimizer_name="adam", momentum=0.9, weight_decay=1e-4,
                      reconstruction_loss="mse", validation_paths=None,
                      feature_loss_weights=None):
    torch.manual_seed(random_state)
    paths = find_dicom_files(clean_dir)
    if len(paths) < 10:
        raise ValueError("Autoencoder training needs at least 10 clean DICOM files; substantially more are recommended for research.")
    if max_train_files and len(paths) > max_train_files:
        paths = random.Random(random_state).sample(paths, max_train_files)
    X, vectorizer, scaler = _matrix(paths, fit=True)
    if validation_dir:
        val_paths = (list(validation_paths) if validation_paths is not None
                     else find_dicom_files(validation_dir))
        if len(val_paths) < 1:
            raise ValueError("Validation directory contains no readable DICOM files.")
        if max_validation_files and len(val_paths) > max_validation_files:
            val_paths = random.Random(random_state + 1).sample(val_paths, max_validation_files)
        X_val, _, _ = _matrix(val_paths, vectorizer, scaler)
    else:
        X_val = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataAutoencoder(X.shape[1]).to(device)
    feature_names = vectorizer.get_feature_names_out()
    named_weights = dict(feature_loss_weights or {})
    unknown_weight_names = sorted(set(named_weights) - set(map(str, feature_names)))
    if unknown_weight_names:
        raise ValueError(f"Unknown feature loss weight names: {unknown_weight_names}")
    feature_weights = np.ones(X.shape[1], dtype=np.float32)
    feature_index = {str(name): index for index, name in enumerate(feature_names)}
    for name, weight in named_weights.items():
        if float(weight) <= 0:
            raise ValueError("Feature loss weights must be positive")
        feature_weights[feature_index[name]] = float(weight)
    feature_weights_tensor = torch.from_numpy(feature_weights).to(device)
    optimizer_name = str(optimizer_name).lower().replace(" ", "_")
    optimizer_options = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=learning_rate),
        "rmsprop": lambda: torch.optim.RMSprop(model.parameters(), lr=learning_rate),
        "adamw": lambda: torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        "sgd_momentum": lambda: torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        ),
    }
    if optimizer_name not in optimizer_options:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; choose from {', '.join(optimizer_options)}")
    optimizer = optimizer_options[optimizer_name]()
    reconstruction_loss = str(reconstruction_loss).lower().replace("-", "_")
    # Validate the requested name before spending time training.
    _reconstruction_loss_elements(torch.zeros(1), torch.zeros(1), reconstruction_loss)
    tensor = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(tensor), batch_size=max(1, min(batch_size, len(X))), shuffle=True)

    def matrix_loss(matrix):
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(matrix), 1024):
                batch = torch.from_numpy(matrix[start:start + 1024]).to(device)
                elements = _reconstruction_loss_elements(
                    batch, model(batch), reconstruction_loss
                )
                values.append(((elements * feature_weights_tensor).sum(dim=1) /
                               feature_weights_tensor.sum()).cpu().numpy())
        return float(np.concatenate(values).mean())

    loss_history = {"train": [], "validation": []}
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            elements = _reconstruction_loss_elements(
                batch, model(batch), reconstruction_loss
            )
            loss = (elements * feature_weights_tensor).sum(dim=1).mean() / feature_weights_tensor.sum()
            loss.backward()
            optimizer.step()
        loss_history["train"].append(matrix_loss(X))
        loss_history["validation"].append(
            matrix_loss(X_val) if X_val is not None else None
        )
        model.train()

    def errors(matrix):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(matrix).to(device)
            reconstructed = model(x)
            elements = _reconstruction_loss_elements(x, reconstructed, reconstruction_loss)
            return ((elements * feature_weights_tensor).sum(dim=1) /
                    feature_weights_tensor.sum()).cpu().numpy()

    calibration = errors(X_val if X_val is not None else X)
    threshold = float(np.percentile(calibration, threshold_percentile))
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(X.shape[1]), "vectorizer": vectorizer, "scaler": scaler,
        "threshold": threshold, "threshold_percentile": float(threshold_percentile),
        "training_files": len(paths), "validation_files": 0 if X_val is None else len(X_val),
        "threshold_source": "clean_validation" if X_val is not None else "clean_training_fallback",
        "epochs": int(epochs), "optimizer": optimizer_name,
        "reconstruction_loss": reconstruction_loss,
        "feature_loss_weights": named_weights,
        "loss_history": loss_history,
        "learning_rate": float(learning_rate), "momentum": float(momentum),
        "weight_decay": float(weight_decay), "random_state": int(random_state),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle

def predict_autoencoder_with_explanations(datasets, model_path, top_n=5):
    bundle = joblib.load(model_path)
    model = MetadataAutoencoder(bundle["input_dim"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    rows = [metadata_features(ds) for _, ds in datasets]
    X = bundle["vectorizer"].transform(rows)
    X = bundle["scaler"].transform(X).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(X)
        reconstructed = model(x)
        contributions = _reconstruction_loss_elements(
            x, reconstructed, bundle.get("reconstruction_loss", "mse")
        )
        raw_weights = bundle.get("feature_loss_weights", {})
        feature_names = bundle["vectorizer"].get_feature_names_out()
        feature_index = {str(name): index for index, name in enumerate(feature_names)}
        weights = torch.ones(contributions.shape[1], dtype=contributions.dtype)
        for name, weight in raw_weights.items():
            weights[feature_index[name]] = float(weight)
        contributions = contributions * weights
        errors = (contributions.sum(dim=1) / weights.sum()).numpy()
        contributions = contributions.numpy()
    threshold = bundle["threshold"]
    feature_names = bundle["vectorizer"].get_feature_names_out()
    result = {}
    for i, (key, _) in enumerate(datasets):
        order = np.argsort(contributions[i])[::-1][:max(1, top_n)]
        contribution_key = ("squared_reconstruction_error"
                            if bundle.get("reconstruction_loss", "mse") == "mse"
                            else "reconstruction_loss_contribution")
        top = [{"feature": str(feature_names[j]),
                contribution_key: float(contributions[i, j])}
               for j in order]
        result[key] = (int(errors[i] > threshold), float(errors[i]), top)
    return result


def predict_autoencoder(datasets, model_path):
    """Backward-compatible flag/error predictions without explanation details."""
    result = predict_autoencoder_with_explanations(datasets, model_path, top_n=1)
    return {key: (flag, error) for key, (flag, error, _) in result.items()}
