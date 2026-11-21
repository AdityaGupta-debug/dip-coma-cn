"""Mixed-type reconstruction loss for DICOM metadata anomaly detection.

The MLP consumes standardized DictVectorizer features. Its reconstruction
objective is type-aware: Huber loss for continuous/numeric features,
binary-cross-entropy for field-presence indicators, and categorical
cross-entropy for one-hot DictVectorizer category groups. Categorical and
binary targets stay in their original 0/1 representation; they are not
standardized before computing their losses.
"""
from __future__ import annotations

import random
import copy
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from .deep_anomaly import MetadataAutoencoder
from .dicom_io import find_dicom_files, read_dataset
from .features import metadata_features
from .attacks import _modify
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


def _prepare(paths, vectorizer=None, scaler=None, fit=False):
    rows = [metadata_features(read_dataset(path, stop_before_pixels=True))
            for path in paths]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
    raw = vectorizer.fit_transform(rows) if fit else vectorizer.transform(rows)
    if scaler is None:
        scaler = StandardScaler()
    scaled = scaler.fit_transform(raw) if fit else scaler.transform(raw)
    return raw.astype(np.float32), scaled.astype(np.float32), vectorizer, scaler


def _feature_groups(feature_names):
    binary, categorical = [], {}
    for index, feature in enumerate(feature_names):
        name = str(feature)
        if name.endswith("_present"):
            binary.append(index)
        elif "_category=" in name:
            field = name.split("_category=", 1)[0]
            categorical.setdefault(field, []).append(index)
    categorical = {key: np.asarray(value, dtype=np.int64)
                   for key, value in categorical.items()}
    binary = np.asarray(binary, dtype=np.int64)
    excluded = set(binary.tolist())
    for indices in categorical.values():
        excluded.update(indices.tolist())
    numeric = np.asarray([i for i in range(len(feature_names)) if i not in excluded],
                         dtype=np.int64)
    if not len(numeric) or not len(binary) or not categorical:
        raise ValueError("Mixed loss requires numeric, presence, and categorical feature groups")
    return {"numeric": numeric, "binary": binary, "categorical": categorical}


def _per_sample_losses(logits, raw_target, scaled_target, groups):
    numeric = F.smooth_l1_loss(
        logits[:, groups["numeric"]], scaled_target[:, groups["numeric"]],
        reduction="none", beta=1.0,
    ).mean(dim=1)
    binary = F.binary_cross_entropy_with_logits(
        logits[:, groups["binary"]], raw_target[:, groups["binary"]],
        reduction="none",
    ).mean(dim=1)
    categorical = []
    for indices in groups["categorical"].values():
        # DictVectorizer category groups are one-hot; argmax gives each row's
        # category target. Unknown categories transform to all zeros and are
        # mapped to class zero for a defined, conservative score.
        target = torch.argmax(raw_target[:, indices], dim=1)
        group_loss = F.cross_entropy(logits[:, indices], target, reduction="none")
        # DictVectorizer encodes an unseen categorical value as all zeros. Treat
        # that as an out-of-vocabulary anomaly instead of silently mapping it
        # to class zero (e.g., CT -> unseen MR).
        unknown = raw_target[:, indices].sum(dim=1) == 0
        unknown_penalty = torch.full_like(group_loss, 10.0)
        categorical.append(torch.where(unknown, unknown_penalty, group_loss))
    categorical = torch.stack(categorical, dim=1).mean(dim=1)
    return numeric, categorical, binary


def _combined_score(logits, raw_target, scaled_target, groups, weights):
    numeric, categorical, binary = _per_sample_losses(
        logits, raw_target, scaled_target, groups
    )
    wn, wc, wb = weights
    return (wn * numeric + wc * categorical + wb * binary) / (wn + wc + wb)


def _score_matrix(model, raw, scaled, groups, weights, device, batch_size=1024):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            x = torch.from_numpy(scaled[start:start + batch_size]).to(device)
            target_raw = torch.from_numpy(raw[start:start + batch_size]).to(device)
            target_scaled = torch.from_numpy(scaled[start:start + batch_size]).to(device)
            scores.append(_combined_score(
                model(x), target_raw, target_scaled, groups, weights
            ).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def train_mixed_autoencoder(
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
    numeric_weight=1.0,
    categorical_weight=1.0,
    binary_weight=1.0,
    validation_paths=None,
):
    """Train a metadata AE with type-aware numeric/categorical/binary losses."""
    weights = (float(numeric_weight), float(categorical_weight),
               float(binary_weight))
    if any(weight <= 0 for weight in weights):
        raise ValueError("All mixed-loss group weights must be positive")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be between 0 and 100")

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    paths = find_dicom_files(clean_dir)
    if len(paths) < 10:
        raise ValueError("Mixed autoencoder needs at least 10 clean DICOM files")
    if max_train_files and len(paths) > max_train_files:
        paths = random.Random(random_state).sample(paths, max_train_files)
    raw, scaled, vectorizer, scaler = _prepare(paths, fit=True)
    groups = _feature_groups(vectorizer.get_feature_names_out())

    raw_val = scaled_val = None
    validation_files = 0
    if validation_dir:
        val_paths = (list(validation_paths) if validation_paths is not None
                     else find_dicom_files(validation_dir))
        if not val_paths:
            raise ValueError("Validation directory contains no readable DICOM files")
        if max_validation_files and len(val_paths) > max_validation_files:
            val_paths = random.Random(random_state + 1).sample(
                val_paths, max_validation_files
            )
        raw_val, scaled_val, _, _ = _prepare(
            val_paths, vectorizer=vectorizer, scaler=scaler
        )
        validation_files = len(val_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataAutoencoder(scaled.shape[1]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(raw), torch.from_numpy(scaled)),
        batch_size=min(batch_size, len(scaled)), shuffle=True,
    )
    torch_groups = {
        "numeric": torch.as_tensor(groups["numeric"], dtype=torch.long, device=device),
        "binary": torch.as_tensor(groups["binary"], dtype=torch.long, device=device),
        "categorical": {
            key: torch.as_tensor(value, dtype=torch.long, device=device)
            for key, value in groups["categorical"].items()
        },
    }
    history = {"train_total": [], "train_numeric": [],
               "train_categorical": [], "train_binary": [],
               "validation_clean_score": []}
    for _ in range(epochs):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        n_batches = 0
        for raw_batch, scaled_batch in loader:
            raw_batch = raw_batch.to(device)
            scaled_batch = scaled_batch.to(device)
            optimizer.zero_grad()
            numeric, categorical, binary = _per_sample_losses(
                model(scaled_batch), raw_batch, scaled_batch, torch_groups
            )
            components = (numeric.mean(), categorical.mean(), binary.mean())
            total = sum(weight * value for weight, value
                        in zip(weights, components)) / sum(weights)
            total.backward()
            optimizer.step()
            sums += np.asarray([float(total.detach().cpu())] +
                               [float(value.detach().cpu()) for value in components])
            n_batches += 1
        means = sums / max(1, n_batches)
        history["train_total"].append(float(means[0]))
        history["train_numeric"].append(float(means[1]))
        history["train_categorical"].append(float(means[2]))
        history["train_binary"].append(float(means[3]))
        score_raw = raw_val if raw_val is not None else raw
        score_scaled = scaled_val if scaled_val is not None else scaled
        history["validation_clean_score"].append(float(_score_matrix(
            model, score_raw, score_scaled, torch_groups, weights, device
        ).mean()))

    calibration = _score_matrix(
        model, raw_val if raw_val is not None else raw,
        scaled_val if scaled_val is not None else scaled,
        torch_groups, weights, device,
    )
    threshold = float(np.percentile(calibration, threshold_percentile))
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(scaled.shape[1]), "vectorizer": vectorizer,
        "scaler": scaler, "groups": groups, "weights": weights,
        "threshold": threshold,
        "threshold_percentile": float(threshold_percentile),
        "training_files": len(paths), "validation_files": validation_files,
        "threshold_source": "clean_validation" if raw_val is not None
        else "clean_training_fallback",
        "epochs": int(epochs), "optimizer": "rmsprop",
        "loss": "equal-weight mean of numeric Smooth L1, categorical cross-entropy, and binary BCE",
        "learning_rate": float(learning_rate), "random_state": int(random_state),
        "loss_history": history,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def _load_bilevel_validation(validation_dir, vectorizer, scaler, seed):
    """Split validation studies into disjoint hyperweight and threshold groups.

    Synthetic validation edits are applied only to in-memory DICOM objects.
    """
    from .dicom_io import find_dicom_files

    paths = find_dicom_files(validation_dir)
    grouped = {}
    for path in paths:
        ds = read_dataset(path, stop_before_pixels=True)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip()
        if not study:
            study = f"MISSING:{path.parent.name}"
        grouped.setdefault(study, []).append((path, ds))
    study_ids = sorted(grouped)
    if len(study_ids) < 4:
        raise ValueError("Bilevel weighting needs at least four validation studies")
    random.Random(seed).shuffle(study_ids)
    midpoint = len(study_ids) // 2
    meta_ids, calibration_ids = study_ids[:midpoint], study_ids[midpoint:]
    meta_records = [record for study in meta_ids for record in grouped[study]]
    calibration_records = [record for study in calibration_ids for record in grouped[study]]
    meta_datasets = [ds for _, ds in meta_records]
    calibration_datasets = [ds for _, ds in calibration_records]

    def matrices(datasets):
        raw = vectorizer.transform([metadata_features(ds) for ds in datasets]).astype(np.float32)
        scaled = scaler.transform(raw).astype(np.float32)
        return raw, scaled

    meta_clean_raw, meta_clean_scaled = matrices(meta_datasets)
    clean_pairs, attack_pairs = [], []
    validation_attacks = (
        "patient_id", "study_description", "pixel_spacing",
        "remove_protocol", "modality", "manufacturer_same_length",
    )
    for attack in validation_attacks:
        attack_datasets, clean_kept = [], []
        for dataset, raw_row in zip(meta_datasets, meta_clean_raw):
            edited = copy.deepcopy(dataset)
            try:
                changed = _modify(edited, attack)
            except (AttributeError, TypeError, ValueError):
                changed = None
            if changed is None:
                continue
            attack_datasets.append(edited)
            clean_kept.append(raw_row)
        if not attack_datasets:
            continue
        attack_raw = vectorizer.transform(
            [metadata_features(ds) for ds in attack_datasets]
        ).astype(np.float32)
        # Ignore edits that are invisible to this vectorizer; no objective can
        # learn a weight for information absent from model input.
        changed_rows = np.any(attack_raw != np.asarray(clean_kept), axis=1)
        if not np.any(changed_rows):
            continue
        clean_raw = np.asarray(clean_kept, dtype=np.float32)[changed_rows]
        attack_raw = attack_raw[changed_rows]
        clean_scaled = scaler.transform(clean_raw).astype(np.float32)
        attack_scaled = scaler.transform(attack_raw).astype(np.float32)
        clean_pairs.append((clean_raw, clean_scaled))
        attack_pairs.append((attack_raw, attack_scaled))
    if not attack_pairs:
        raise ValueError("No feature-visible synthetic attacks were generated for bilevel updates")
    return {
        "meta_clean_raw": meta_clean_raw,
        "meta_clean_scaled": meta_clean_scaled,
        "pair_clean_raw": np.concatenate([x[0] for x in clean_pairs]),
        "pair_clean_scaled": np.concatenate([x[1] for x in clean_pairs]),
        "pair_attack_raw": np.concatenate([x[0] for x in attack_pairs]),
        "pair_attack_scaled": np.concatenate([x[1] for x in attack_pairs]),
        "calibration_raw": matrices(calibration_datasets)[0],
        "calibration_scaled": matrices(calibration_datasets)[1],
        "meta_studies": len(meta_ids),
        "calibration_studies": len(calibration_ids),
        "calibration_paths": [path for path, _ in calibration_records],
        "validation_attack_families": list(validation_attacks),
    }


def _bilevel_group_weight_update(model, train_raw, train_scaled,
                                 validation_pairs, group_indices,
                                 weight_logits, meta_optimizer,
                                 inner_step_size, margin=0.1):
    """One-step unrolled hypergradient update using validation attack ranking."""
    model.eval()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    weights = torch.softmax(weight_logits, dim=0)
    train_logits = functional_call(model, (parameters, buffers), (train_scaled,))
    numeric, categorical, binary = _per_sample_losses(
        train_logits, train_raw, train_scaled, group_indices
    )
    train_components = torch.stack((numeric.mean(), categorical.mean(), binary.mean()))
    inner_objective = torch.sum(weights * train_components)
    gradients = torch.autograd.grad(
        inner_objective, tuple(parameters.values()), create_graph=True
    )
    updated_parameters = {
        name: parameter - inner_step_size * gradient
        for (name, parameter), gradient in zip(parameters.items(), gradients)
    }
    clean_raw, clean_scaled, attack_raw, attack_scaled = validation_pairs
    clean_logits = functional_call(
        model, (updated_parameters, buffers), (clean_scaled,)
    )
    attack_logits = functional_call(
        model, (updated_parameters, buffers), (attack_scaled,)
    )
    clean_scores = _combined_score(
        clean_logits, clean_raw, clean_scaled, group_indices, weights
    )
    attack_scores = _combined_score(
        attack_logits, attack_raw, attack_scaled, group_indices, weights
    )
    outer_objective = F.softplus(margin + clean_scores - attack_scores).mean()
    hypergradient = torch.autograd.grad(outer_objective, weight_logits)[0]
    meta_optimizer.zero_grad(set_to_none=True)
    weight_logits.grad = hypergradient.detach()
    meta_optimizer.step()
    with torch.no_grad():
        learned_weights = torch.softmax(weight_logits, dim=0)
    model.train()
    return (learned_weights.detach().cpu().numpy(),
            float(outer_objective.detach().cpu()),
            float(torch.linalg.vector_norm(hypergradient).detach().cpu()))


def train_bilevel_mixed_autoencoder(
    clean_dir,
    validation_dir,
    model_path,
    epochs=40,
    batch_size=32,
    learning_rate=1e-3,
    meta_learning_rate=1e-2,
    inner_step_size=1e-3,
    threshold_percentile=99.0,
    random_state=42,
    max_train_files=30000,
):
    """Train a mixed AE and learn type-group weights with Adam hypergradients.

    The autoencoder uses RMSprop to match the existing MSE baseline. Adam is
    used for the three bilevel group-weight logits. Synthetic validation edits
    guide the weight updates; disjoint clean validation studies calibrate the
    final anomaly threshold.
    """
    if epochs < 1 or batch_size < 1 or inner_step_size <= 0:
        raise ValueError("epochs/batch_size/inner_step_size must be positive")
    if meta_learning_rate <= 0:
        raise ValueError("meta_learning_rate must be positive")
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    paths = find_dicom_files(clean_dir)
    if len(paths) < 10:
        raise ValueError("Bilevel mixed autoencoder needs at least 10 training files")
    if max_train_files and len(paths) > max_train_files:
        paths = random.Random(random_state).sample(paths, max_train_files)
    raw, scaled, vectorizer, scaler = _prepare(paths, fit=True)
    groups = _feature_groups(vectorizer.get_feature_names_out())
    val = _load_bilevel_validation(validation_dir, vectorizer, scaler, random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataAutoencoder(scaled.shape[1]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    weight_logits = torch.nn.Parameter(torch.zeros(3, device=device))
    meta_optimizer = torch.optim.Adam([weight_logits], lr=meta_learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(raw), torch.from_numpy(scaled)),
        batch_size=min(batch_size, len(scaled)), shuffle=True,
    )
    group_indices = {
        "numeric": torch.as_tensor(groups["numeric"], dtype=torch.long, device=device),
        "binary": torch.as_tensor(groups["binary"], dtype=torch.long, device=device),
        "categorical": {
            key: torch.as_tensor(value, dtype=torch.long, device=device)
            for key, value in groups["categorical"].items()
        },
    }
    validation_pairs = tuple(torch.from_numpy(val[key]).to(device) for key in (
        "pair_clean_raw", "pair_clean_scaled", "pair_attack_raw", "pair_attack_scaled"
    ))
    history = {"numeric_weight": [], "categorical_weight": [], "binary_weight": [],
               "meta_objective": [], "hypergradient_norm": [],
               "train_objective": [], "validation_clean_score": []}
    for _epoch in range(epochs):
        epoch_iterator = iter(loader)
        first_batch = next(epoch_iterator)
        weights, meta_objective, gradient_norm = _bilevel_group_weight_update(
            model, first_batch[0].to(device), first_batch[1].to(device),
            validation_pairs, group_indices, weight_logits, meta_optimizer,
            inner_step_size,
        )
        epoch_losses = []
        for raw_batch, scaled_batch in chain((first_batch,), epoch_iterator):
            raw_batch = raw_batch.to(device)
            scaled_batch = scaled_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            numeric, categorical, binary = _per_sample_losses(
                model(scaled_batch), raw_batch, scaled_batch, group_indices
            )
            components = torch.stack((numeric.mean(), categorical.mean(), binary.mean()))
            weights_tensor = torch.as_tensor(weights, dtype=components.dtype, device=device)
            loss = torch.sum(weights_tensor * components)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        clean_score = _score_matrix(
            model, val["calibration_raw"], val["calibration_scaled"],
            group_indices, tuple(float(x) for x in weights), device,
        ).mean()
        history["numeric_weight"].append(float(weights[0]))
        history["categorical_weight"].append(float(weights[1]))
        history["binary_weight"].append(float(weights[2]))
        history["meta_objective"].append(meta_objective)
        history["hypergradient_norm"].append(gradient_norm)
        history["train_objective"].append(float(np.mean(epoch_losses)))
        history["validation_clean_score"].append(float(clean_score))

    final_weights = tuple(float(x) for x in torch.softmax(weight_logits, dim=0).detach().cpu())
    calibration_scores = _score_matrix(
        model, val["calibration_raw"], val["calibration_scaled"],
        group_indices, final_weights, device,
    )
    threshold = float(np.percentile(calibration_scores, threshold_percentile))
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(scaled.shape[1]), "vectorizer": vectorizer,
        "scaler": scaler, "groups": groups, "weights": final_weights,
        "threshold": threshold, "threshold_percentile": float(threshold_percentile),
        "training_files": len(paths), "validation_files": len(val["calibration_raw"]),
        "meta_validation_studies": val["meta_studies"],
        "threshold_validation_studies": val["calibration_studies"],
        "threshold_validation_paths": [str(path) for path in val["calibration_paths"]],
        "validation_attack_families": val["validation_attack_families"],
        "threshold_source": "disjoint_clean_validation_studies",
        "epochs": int(epochs), "optimizer": "rmsprop",
        "weight_optimizer": "adam", "weight_learning_rate": float(meta_learning_rate),
        "inner_step_size": float(inner_step_size),
        "outer_objective": "pairwise softplus ranking margin between clean and synthetic attacked validation metadata",
        "loss": "bilevel weighted mean of numeric Smooth L1, categorical cross-entropy, and binary BCE",
        "learning_rate": float(learning_rate), "random_state": int(random_state),
        "loss_history": history,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def predict_mixed_autoencoder(datasets, model_path, top_n=5):
    """Predict with mixed loss score and provide per-feature diagnostic ranks."""
    bundle = joblib.load(model_path)
    model = MetadataAutoencoder(bundle["input_dim"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    rows = [metadata_features(ds) for _, ds in datasets]
    raw = bundle["vectorizer"].transform(rows).astype(np.float32)
    scaled = bundle["scaler"].transform(raw).astype(np.float32)
    device = torch.device("cpu")
    groups = {
        "numeric": torch.as_tensor(bundle["groups"]["numeric"], dtype=torch.long),
        "binary": torch.as_tensor(bundle["groups"]["binary"], dtype=torch.long),
        "categorical": {
            key: torch.as_tensor(value, dtype=torch.long)
            for key, value in bundle["groups"]["categorical"].items()
        },
    }
    weights = bundle["weights"]
    result = {}
    names = bundle["vectorizer"].get_feature_names_out()
    with torch.no_grad():
        for start in range(0, len(scaled), 512):
            end = min(start + 512, len(scaled))
            x = torch.from_numpy(scaled[start:end])
            target_raw = torch.from_numpy(raw[start:end])
            target_scaled = torch.from_numpy(scaled[start:end])
            logits = model(x)
            scores = _combined_score(logits, target_raw, target_scaled,
                                     groups, weights).numpy()
            # Feature-level contributions are diagnostic rankings only. For a
            # categorical group, attach its CE to the observed category column.
            contribution = torch.zeros_like(logits)
            num_indices = groups["numeric"]
            contribution[:, num_indices] = F.smooth_l1_loss(
                logits[:, num_indices], target_scaled[:, num_indices],
                reduction="none", beta=1.0,
            ) * (weights[0] / sum(weights))
            bin_indices = groups["binary"]
            contribution[:, bin_indices] = F.binary_cross_entropy_with_logits(
                logits[:, bin_indices], target_raw[:, bin_indices], reduction="none"
            ) * (weights[2] / sum(weights))
            cat_scale = weights[1] / sum(weights) / len(groups["categorical"])
            for indices in groups["categorical"].values():
                targets = torch.argmax(target_raw[:, indices], dim=1)
                losses = F.cross_entropy(logits[:, indices], targets, reduction="none")
                unknown = target_raw[:, indices].sum(dim=1) == 0
                losses = torch.where(unknown, torch.full_like(losses, 10.0), losses)
                rows_index = torch.arange(len(targets), device=indices.device)
                # Add loss at the observed target category's feature index.
                contribution[rows_index, indices[targets]] += losses * cat_scale
            contribution_np = contribution.numpy()
            for local_index, score in enumerate(scores):
                absolute_index = start + local_index
                key = datasets[absolute_index][0]
                top_indices = np.argsort(contribution_np[local_index])[::-1][:max(1, top_n)]
                top = [{"feature": str(names[j]),
                        "mixed_loss_contribution": float(contribution_np[local_index, j])}
                       for j in top_indices if contribution_np[local_index, j] > 0]
                result[key] = (int(score > bundle["threshold"]), float(score), top)
    return result
