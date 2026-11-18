"""PatchCore-style clean-only CT anomaly detector.

Uses frozen ResNet-18 layer2/layer3 patch embeddings and a deterministic random
subsample of clean training patches as the nominal memory bank. torchvision is
an optional dependency; ImageNet weights are required by default and are
reported in the saved configuration.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .dicom_io import find_dicom_files, read_dataset
from .no_reference_pixel import _normalized_image


class ResNet18PatchFeatures(torch.nn.Module):
    def __init__(self, use_imagenet_weights=True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError(
                "PatchCore requires a compatible torchvision installation. "
                "Install the optional 'patchcore' dependencies for this project's PyTorch version."
            ) from exc
        weights = ResNet18_Weights.DEFAULT if use_imagenet_weights else None
        backbone = resnet18(weights=weights)
        self.conv1, self.bn1, self.relu = backbone.conv1, backbone.bn1, backbone.relu
        self.maxpool, self.layer1 = backbone.maxpool, backbone.layer1
        self.layer2, self.layer3 = backbone.layer2, backbone.layer3
        self.use_imagenet_weights = bool(use_imagenet_weights)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        low = self.layer2(x)
        high = self.layer3(low)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([low, high], dim=1)
        features = F.normalize(features, p=2, dim=1)
        return F.adaptive_avg_pool2d(features, (16, 16))


def _image_tensor(path, image_size=256):
    ds = read_dataset(path, stop_before_pixels=False)
    image = _normalized_image(ds)
    tensor = torch.from_numpy(image).float()[None, None]
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    # Replicate grayscale to RGB and use the documented ImageNet normalization.
    tensor = tensor.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ds, (tensor - mean) / std, image


def _feature_for_path(path, extractor, device, image_size=256):
    extractor.eval()
    with torch.inference_mode():
        ds, tensor, image = _image_tensor(path, image_size)
        fmap = extractor(tensor.to(device))[0].permute(1, 2, 0).reshape(-1, 768)
    return fmap.cpu().numpy(), image.shape, image


def fit_patchcore(clean_train_dir, clean_validation_dir, model_path, *,
                  max_train_files=0, max_memory_patches=20000,
                  patches_per_train_image=8, image_size=256,
                  threshold_percentile=99.0, seed=42,
                  use_imagenet_weights=True, device_name="auto"):
    if not 0 < threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100")
    if max_memory_patches < 10 or patches_per_train_image < 1:
        raise ValueError("Patch limits are too small")
    train_paths, val_paths = find_dicom_files(clean_train_dir), find_dicom_files(clean_validation_dir)
    if max_train_files:
        train_paths = random.Random(seed).sample(train_paths, min(max_train_files, len(train_paths)))
    if not train_paths or not val_paths:
        raise ValueError("Clean training and validation folds must be nonempty")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    extractor = ResNet18PatchFeatures(use_imagenet_weights=use_imagenet_weights).to(device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    train_chunks = []
    for path in train_paths:
        fmap, _, _ = _feature_for_path(path, extractor, device, image_size)
        take = min(patches_per_train_image, len(fmap))
        indices = rng.choice(len(fmap), size=take, replace=False)
        train_chunks.append(fmap[indices])
    candidates = np.concatenate(train_chunks, axis=0).astype(np.float32)
    # Deterministic random subsampling bounds the bank; this is not k-center coreset selection.
    if len(candidates) > max_memory_patches:
        take = rng.choice(len(candidates), size=max_memory_patches, replace=False)
        memory = candidates[take]
    else:
        memory = candidates
    val_patch_scores = []
    val_image_scores = []
    neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
    neighbor.fit(memory)
    for path in val_paths:
        chunk, _, _ = _feature_for_path(path, extractor, device, image_size)
        distances, _ = neighbor.kneighbors(chunk, return_distance=True)
        scores = distances[:, 0]
        val_patch_scores.extend(scores.tolist())
        val_image_scores.append(float(np.max(scores)))
    pixel_threshold = float(np.percentile(val_patch_scores, threshold_percentile))
    image_threshold = float(np.percentile(val_image_scores, threshold_percentile))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone_state_dict": extractor.state_dict(),
        "memory_bank": memory,
        "image_size": int(image_size),
        "pixel_threshold": pixel_threshold,
        "image_threshold": image_threshold,
        "threshold_percentile": float(threshold_percentile),
        "training_files": int(len(train_paths)),
        "candidate_patches": int(len(candidates)),
        "memory_patches": int(len(memory)),
        "patch_grid": [16, 16],
        "feature_dim": 768,
        "seed": int(seed),
        "device_used_for_fit": str(device),
        "backbone": "ResNet-18 layer2/layer3 concatenation",
        "pretraining": "ImageNet default weights" if use_imagenet_weights else "random frozen initialization",
        "memory_subsampling": "deterministic random patch subsample",
        "normalization": "per-image 1st/99th percentile; ImageNet channel normalization after grayscale replication",
    }, model_path)
    return torch.load(model_path, map_location="cpu", weights_only=False)


def load_patchcore(model_path, device_name="auto"):
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    use_weights = str(bundle["pretraining"]).startswith("ImageNet")
    extractor = ResNet18PatchFeatures(use_imagenet_weights=False)
    extractor.load_state_dict(bundle["backbone_state_dict"])
    extractor.to(device).eval()
    neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
    neighbor.fit(np.asarray(bundle["memory_bank"], dtype=np.float32))
    bundle["extractor"] = extractor
    bundle["neighbor"] = neighbor
    bundle["device"] = device
    bundle["use_imagenet_weights"] = use_weights
    return bundle


def score_patchcore(paths, bundle):
    paths = list(paths)
    if not paths:
        return []
    results = []
    for path in paths:
        chunk, shape, image = _feature_for_path(
            path, bundle["extractor"], bundle["device"], int(bundle["image_size"]))
        distances, _ = bundle["neighbor"].kneighbors(chunk, return_distance=True)
        score_grid = distances[:, 0].reshape(16, 16)
        anomaly_map = F.interpolate(
            torch.from_numpy(score_grid)[None, None].float(), size=shape,
            mode="bilinear", align_corners=False)[0, 0].numpy()
        results.append({
            "score": float(score_grid.max()),
            "prediction": int(score_grid.max() > float(bundle["image_threshold"])),
            "anomaly_map": anomaly_map,
            "image": image,
        })
    return results
