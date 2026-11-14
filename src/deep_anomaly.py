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

def train_autoencoder(clean_dir, model_path, validation_dir=None,
                      epochs=100, batch_size=32, learning_rate=1e-3,
                      threshold_percentile=99.0, random_state=42,
                      max_train_files=30000, max_validation_files=10000):
    torch.manual_seed(random_state)
    paths = find_dicom_files(clean_dir)
    if len(paths) < 10:
        raise ValueError("Autoencoder training needs at least 10 clean DICOM files; substantially more are recommended for research.")
    if max_train_files and len(paths) > max_train_files:
        paths = random.Random(random_state).sample(paths, max_train_files)
    X, vectorizer, scaler = _matrix(paths, fit=True)
    if validation_dir:
        val_paths = find_dicom_files(validation_dir)
        if len(val_paths) < 1:
            raise ValueError("Validation directory contains no readable DICOM files.")
        if max_validation_files and len(val_paths) > max_validation_files:
            val_paths = random.Random(random_state + 1).sample(val_paths, max_validation_files)
        X_val, _, _ = _matrix(val_paths, vectorizer, scaler)
    else:
        X_val = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataAutoencoder(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    tensor = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(tensor), batch_size=max(1, min(batch_size, len(X))), shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    def errors(matrix):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(matrix).to(device)
            reconstructed = model(x)
            return torch.mean((x - reconstructed) ** 2, dim=1).cpu().numpy()

    calibration = errors(X_val if X_val is not None else X)
    threshold = float(np.percentile(calibration, threshold_percentile))
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(X.shape[1]), "vectorizer": vectorizer, "scaler": scaler,
        "threshold": threshold, "threshold_percentile": float(threshold_percentile),
        "training_files": len(paths), "validation_files": 0 if X_val is None else len(X_val),
        "threshold_source": "clean_validation" if X_val is not None else "clean_training_fallback",
        "epochs": int(epochs)
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
        squared_errors = (x - reconstructed) ** 2
        errors = torch.mean(squared_errors, dim=1).numpy()
        contributions = squared_errors.numpy()
    threshold = bundle["threshold"]
    feature_names = bundle["vectorizer"].get_feature_names_out()
    result = {}
    for i, (key, _) in enumerate(datasets):
        order = np.argsort(contributions[i])[::-1][:max(1, top_n)]
        top = [{"feature": str(feature_names[j]),
                "squared_reconstruction_error": float(contributions[i, j])}
               for j in order]
        result[key] = (int(errors[i] > threshold), float(errors[i]), top)
    return result


def predict_autoencoder(datasets, model_path):
    """Backward-compatible flag/error predictions without explanation details."""
    result = predict_autoencoder_with_explanations(datasets, model_path, top_n=1)
    return {key: (flag, error) for key, (flag, error, _) in result.items()}
