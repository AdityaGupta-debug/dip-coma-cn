"""Supervised pixel-tamper localization with synthetic training masks.

The model is trained on synthetic patch and sparse-value edits from CPAISD's
official training fold. A copy-move edit family is withheld from training and
used as an out-of-family test condition. No clean test counterpart is provided
to the network at inference time.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .dicom_io import find_dicom_files, read_dataset, relative_key
from .no_reference_pixel import _normalized_image


TRAIN_FAMILIES = ("localized_patch", "sparse_low_amplitude")
TEST_FAMILIES = (*TRAIN_FAMILIES, "copy_move_unseen")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNetLocalizer(nn.Module):
    """Compact U-Net that returns a per-pixel tamper-logit map."""
    def __init__(self, base_channels: int = 24):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(center), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


class AttentionGate(nn.Module):
    """Gate encoder skip features using the corresponding decoder context."""
    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int):
        super().__init__()
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, skip, gating):
        coefficients = self.attention(
            self.skip_projection(skip) + self.gating_projection(gating)
        )
        coefficients = F.interpolate(coefficients, size=skip.shape[-2:],
                                      mode="bilinear", align_corners=False)
        return skip * coefficients


class AttentionUNetLocalizer(nn.Module):
    """U-Net variant that gates skip features with decoder attention."""
    def __init__(self, base_channels: int = 24):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.gate4 = AttentionGate(b * 8, b * 8, b * 4)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.gate3 = AttentionGate(b * 4, b * 4, b * 2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.gate2 = AttentionGate(b * 2, b * 2, b)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.gate1 = AttentionGate(b, b, max(1, b // 2))
        self.dec1 = ConvBlock(b * 2, b)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))
        d4 = self.up4(center)
        d4 = self.dec4(torch.cat([d4, self.gate4(e4, d4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.gate3(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.gate2(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.gate1(e1, d1)], dim=1))
        return self.output(d1)


def build_localizer(architecture: str = "unet", base_channels: int = 24):
    if architecture == "unet":
        return UNetLocalizer(base_channels=base_channels)
    if architecture == "attention_unet":
        return AttentionUNetLocalizer(base_channels=base_channels)
    raise ValueError("architecture must be 'unet' or 'attention_unet'.")


def _study_group(ds, path: Path) -> str:
    identity = str(getattr(ds, "StudyInstanceUID", "")).strip() or str(path.parent.parent)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _image_tensor(path: Path, image_size: int) -> torch.Tensor:
    ds = read_dataset(path, stop_before_pixels=False)
    image = _normalized_image(ds)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor[0, 0].contiguous()


def _clip_edit(image: torch.Tensor, proposed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edited = proposed.clamp(0.0, 1.0)
    mask = edited.ne(image)
    return edited, mask


def _apply_train_edit(image: torch.Tensor, family: str, rng: random.Random):
    edited = image.clone()
    height, width = edited.shape
    if family == "localized_patch":
        size = rng.randint(8, min(28, height, width))
        top, left = rng.randint(0, height - size), rng.randint(0, width - size)
        delta = rng.uniform(0.15, 0.55) * rng.choice((-1.0, 1.0))
        edited[top:top + size, left:left + size] += delta
    elif family == "sparse_low_amplitude":
        count = max(1, round(edited.numel() * rng.uniform(0.002, 0.01)))
        indices = rng.sample(range(edited.numel()), count)
        flat = edited.view(-1)
        for index in indices:
            flat[index] += rng.uniform(0.06, 0.22) * rng.choice((-1.0, 1.0))
    else:
        raise ValueError(f"Unsupported training edit family: {family}")
    return _clip_edit(image, edited)


def _apply_test_edit(image: torch.Tensor, family: str, rng: random.Random):
    edited = image.clone()
    height, width = edited.shape
    if family == "localized_patch":
        size = max(8, round(min(height, width) * 0.07))
        top, left = rng.randint(0, height - size), rng.randint(0, width - size)
        edited[top:top + size, left:left + size] += 0.32
    elif family == "sparse_low_amplitude":
        count = max(1, round(edited.numel() * 0.01))
        indices = rng.sample(range(edited.numel()), count)
        flat = edited.view(-1)
        flat[indices] += 0.10
    elif family == "copy_move_unseen":
        size = max(16, round(min(height, width) * 0.10))
        source_top, source_left = rng.randint(0, height - size), rng.randint(0, width - size)
        destination_top, destination_left = rng.randint(0, height - size), rng.randint(0, width - size)
        attempts = 0
        while (abs(source_top - destination_top) < size and abs(source_left - destination_left) < size
               and attempts < 10):
            destination_top, destination_left = rng.randint(0, height - size), rng.randint(0, width - size)
            attempts += 1
        source = image[source_top:source_top + size, source_left:source_left + size].clone()
        edited[destination_top:destination_top + size, destination_left:destination_left + size] = source
    else:
        raise ValueError(f"Unsupported test edit family: {family}")
    return _clip_edit(image, edited)


class SyntheticTrainingDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, seed: int):
        self.paths, self.image_size, self.seed = paths, image_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = _image_tensor(self.paths[index], self.image_size)
        rng = random.Random(self.seed + self.epoch * len(self.paths) + index)
        family = TRAIN_FAMILIES[rng.randrange(len(TRAIN_FAMILIES))]
        edited, mask = _apply_train_edit(image, family, rng)
        return edited.unsqueeze(0), mask.float().unsqueeze(0)


class MixedAdversarialTrainingDataset(Dataset):
    """Return clean sources and fresh standard edits for online PGD training."""
    def __init__(self, paths: list[Path], image_size: int, seed: int):
        self.paths, self.image_size, self.seed = paths, image_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        clean = _image_tensor(self.paths[index], self.image_size)
        rng = random.Random(self.seed + self.epoch * len(self.paths) + index)
        family = TRAIN_FAMILIES[rng.randrange(len(TRAIN_FAMILIES))]
        edited, mask = _apply_train_edit(clean, family, rng)
        return clean.unsqueeze(0), edited.unsqueeze(0), mask.float().unsqueeze(0)


class CleanValidationDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths, self.image_size = paths, image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return _image_tensor(self.paths[index], self.image_size).unsqueeze(0)


class SyntheticValidationDataset(Dataset):
    """Fixed, deterministic synthetic edits on clean validation images."""
    def __init__(self, records: list[dict], seed: int, include_clean: bool = False):
        self.records, self.seed, self.include_clean = records, seed, include_clean

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = self.records[index]["image"]
        family = TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
        rng = random.Random(self.seed + 1_000_003 + index * 31)
        edited, mask = _apply_train_edit(image, family, rng)
        if self.include_clean:
            return image.unsqueeze(0), edited.unsqueeze(0), mask.float().unsqueeze(0), family
        return edited.unsqueeze(0), mask.float().unsqueeze(0), family


def _segmentation_components(logits, target, positive_weight):
    weight = torch.tensor([positive_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce, dice


def _segmentation_loss(logits, target, positive_weight, bce_weight=0.5):
    bce, dice = _segmentation_components(logits, target, positive_weight)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def _image_scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Use an unsaturated image score so clean quantiles remain discriminative."""
    return logits.flatten(start_dim=1).amax(dim=1)


def _generate_pgd_evasion(model, source, *, epsilon=0.02, step_size=0.004,
                          steps=6, seed=42, random_start=True):
    """Online targeted PGD toward an all-clean mask; preserve model BN buffers."""
    was_training = model.training
    model.eval()  # BatchNorm must not absorb adversarial-batch statistics.
    source = source.detach()
    generator = torch.Generator(device=source.device)
    generator.manual_seed(int(seed))
    if random_start:
        candidate = source + torch.empty_like(source).uniform_(-epsilon, epsilon, generator=generator)
        candidate = candidate.clamp(0.0, 1.0)
    else:
        candidate = source.clone()
    with torch.enable_grad():
        for _ in range(steps):
            candidate = candidate.detach().requires_grad_(True)
            logits = model(candidate)
            clean_target = torch.zeros_like(logits)
            evasion_loss = F.binary_cross_entropy_with_logits(logits, clean_target)
            gradient = torch.autograd.grad(evasion_loss, candidate, only_inputs=True)[0]
            candidate = candidate.detach() - step_size * gradient.sign()
            candidate = torch.maximum(torch.minimum(candidate, source + epsilon), source - epsilon)
            candidate = candidate.clamp(0.0, 1.0)
    if was_training:
        model.train()
    return candidate.detach()


def _bilevel_weight_update(model, train_inputs, train_masks, validation_inputs,
                           validation_masks, alpha_logit, meta_optimizer,
                           positive_weight, inner_step_size, clean_validation_inputs=None,
                           clean_fpr_weight=0.25, validation_families=None):
    """One-step unrolled hypergradient for robust IoU with a clean-FP penalty."""
    model.eval()  # Keep BatchNorm buffers fixed during the differentiable proxy step.
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    train_logits = functional_call(model, (parameters, buffers), (train_inputs,))
    train_bce, train_dice = _segmentation_components(
        train_logits, train_masks, positive_weight
    )
    alpha = torch.sigmoid(alpha_logit)
    inner_objective = alpha * train_bce + (1.0 - alpha) * train_dice
    gradients = torch.autograd.grad(
        inner_objective, tuple(parameters.values()), create_graph=True
    )
    updated_parameters = {
        name: parameter - inner_step_size * gradient
        for (name, parameter), gradient in zip(parameters.items(), gradients)
    }
    validation_logits = functional_call(
        model, (updated_parameters, buffers), (validation_inputs,)
    )
    validation_probability = torch.sigmoid(validation_logits)
    intersection = (validation_probability * validation_masks).sum(dim=(1, 2, 3))
    denominator = validation_probability.sum(dim=(1, 2, 3)) + validation_masks.sum(dim=(1, 2, 3))
    per_image_iou = (intersection + 1.0) / (denominator - intersection + 1.0)
    if validation_families is None:
        soft_iou = per_image_iou.mean()
    else:
        family_iou = [per_image_iou[
            torch.as_tensor([family == candidate for family in validation_families],
                            device=per_image_iou.device, dtype=torch.bool)
        ].mean() for candidate in sorted(set(validation_families))]
        soft_iou = torch.stack(family_iou).mean()
    outer_objective = 1.0 - soft_iou
    if clean_validation_inputs is not None:
        clean_logits = functional_call(
            model, (updated_parameters, buffers), (clean_validation_inputs,)
        )
        clean_fp_proxy = torch.sigmoid(clean_logits).mean()
        outer_objective = outer_objective + clean_fpr_weight * clean_fp_proxy
    hypergradient = torch.autograd.grad(outer_objective, alpha_logit)[0]
    meta_optimizer.zero_grad(set_to_none=True)
    alpha_logit.grad = hypergradient.detach()
    meta_optimizer.step()
    with torch.no_grad():
        alpha_logit.clamp_(-6.0, 6.0)
    model.train()
    return float(alpha.detach()), float(outer_objective.detach()), float(hypergradient.detach())


def _infer(model, loader, device):
    model.eval()
    maps = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            probability, _ = _predict_batch(model, batch.to(device, non_blocking=True))
            maps.extend(probability.squeeze(1).cpu().numpy())
    return maps


def _predict_batch(model, inputs):
    """Return per-pixel probabilities and pre-sigmoid max-logit image scores."""
    logits = model(inputs)
    return torch.sigmoid(logits), _image_scores_from_logits(logits)


def _load_paths(paths: list[Path], image_size: int):
    records = []
    for path in paths:
        try:
            image = _image_tensor(path, image_size)
            header = read_dataset(path, stop_before_pixels=True)
            records.append({"path": path, "relative": path.as_posix(),
                            "study_group": _study_group(header, path), "image": image})
        except Exception as exc:
            print(f"Skipping unreadable/decode-failed image {path}: {type(exc).__name__}: {exc}")
    return records


def calibrate_clean_image_threshold(model, validation_paths, image_size: int, *,
                                    quantile: float = 99.0, device="cpu", batch_size=8):
    """Calibrate legacy checkpoints with the current max-pre-sigmoid-logit score."""
    if not 0.0 < quantile < 100.0:
        raise ValueError("Image-score calibration quantile must be between 0 and 100.")
    records = _load_paths([Path(path) for path in validation_paths], image_size)
    if not records:
        raise ValueError("No clean validation images are available for threshold calibration.")
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = torch.stack([item["image"] for item in records[start:start + batch_size]])[:, None]
            logits = model(batch.to(target_device, non_blocking=True))
            scores.extend(_image_scores_from_logits(logits).cpu().tolist())
    return float(np.percentile(scores, quantile))


def _binary_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=np.uint8), np.asarray(predictions, dtype=np.uint8)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _localization_metrics(truths, predictions):
    tp = fp = fn = tn = 0
    for truth, prediction in zip(truths, predictions):
        truth = np.asarray(truth, dtype=bool)
        prediction = np.asarray(prediction, dtype=bool)
        tp += int(np.count_nonzero(truth & prediction))
        fp += int(np.count_nonzero(~truth & prediction))
        fn += int(np.count_nonzero(truth & ~prediction))
        tn += int(np.count_nonzero(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"pixel_precision": precision, "pixel_recall": recall,
            "pixel_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn, "pixel_tn": tn}


def _safe_auc(labels, scores, metric):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(metric(labels, scores))


def _validation_localization_metrics(model, loader, device):
    """Return macro-over-family hard Dice/IoU at a fixed 0.5 probability cutoff."""
    counts = {family: {"tp": 0, "fp": 0, "fn": 0, "n_images": 0}
              for family in TRAIN_FAMILIES}
    model.eval()
    with torch.no_grad():
        for inputs, masks, families in loader:
            probability = torch.sigmoid(model(inputs.to(device, non_blocking=True))).cpu()[:, 0]
            truth_batch = masks[:, 0].bool()
            prediction_batch = probability >= 0.5
            for truth, prediction, family in zip(truth_batch, prediction_batch, families):
                item = counts[family]
                item["tp"] += int(torch.count_nonzero(truth & prediction))
                item["fp"] += int(torch.count_nonzero(~truth & prediction))
                item["fn"] += int(torch.count_nonzero(truth & ~prediction))
                item["n_images"] += 1

    rows = []
    for family, item in counts.items():
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        rows.append({"attack_family": family, "n_images": item["n_images"],
                     "pixel_dice": dice, "pixel_iou": iou,
                     "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
                     "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
                     "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn})
    macro_iou = float(np.mean([row["pixel_iou"] for row in rows]))
    macro_dice = float(np.mean([row["pixel_dice"] for row in rows]))
    return {"macro_pixel_iou": macro_iou, "macro_pixel_dice": macro_dice,
            "by_family": rows}


def _robust_validation_metrics(model, attack_loader, clean_loader, device, *,
                               epsilon=0.02, step_size=0.004, pgd_steps=10, seed=42):
    """Evaluate synthetic masks before/after PGD and measure clean image scores."""
    counts = {(family, condition): {"tp": 0, "fp": 0, "fn": 0, "images": 0}
              for family in TRAIN_FAMILIES for condition in ("unattacked", "pgd10")}
    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, masks, families) in enumerate(attack_loader):
            inputs = inputs.to(device, non_blocking=True)
            truth_batch = masks[:, 0].bool().to(device)
            logits = model(inputs)
            probability = torch.sigmoid(logits)[:, 0]
            pgd_inputs = _generate_pgd_evasion(
                model, inputs, epsilon=epsilon, step_size=step_size, steps=pgd_steps,
                seed=seed + batch_index, random_start=True,
            )
            with torch.no_grad():
                pgd_probability = torch.sigmoid(model(pgd_inputs))[:, 0]
            for index, family in enumerate(families):
                for condition, prediction_map in (
                    ("unattacked", probability[index] >= 0.5),
                    ("pgd10", pgd_probability[index] >= 0.5),
                ):
                    item = counts[(family, condition)]
                    truth = truth_batch[index]
                    item["tp"] += int(torch.count_nonzero(truth & prediction_map))
                    item["fp"] += int(torch.count_nonzero(~truth & prediction_map))
                    item["fn"] += int(torch.count_nonzero(truth & ~prediction_map))
                    item["images"] += 1

    rows = []
    for (family, condition), item in counts.items():
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        rows.append({"attack_family": family, "condition": condition,
                     "n_images": item["images"],
                     "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                     "pixel_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                     "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
                     "pixel_recall": tp / (tp + fn) if tp + fn else 0.0})
    clean_scores = []
    with torch.no_grad():
        for batch in clean_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            logits = model(batch.to(device, non_blocking=True))
            clean_scores.extend(_image_scores_from_logits(logits).cpu().tolist())
    clean_threshold = float(np.percentile(clean_scores, 99.0))
    clean_fpr = float(np.mean(np.asarray(clean_scores) >= clean_threshold))
    return {"macro_pixel_iou": float(np.mean([row["pixel_iou"] for row in rows])),
            "macro_pixel_dice": float(np.mean([row["pixel_dice"] for row in rows])),
            "clean_image_fpr_at_q99": clean_fpr,
            "clean_image_threshold_at_q99": clean_threshold,
            "by_condition": rows}


def _study_rows(frame: pd.DataFrame, threshold: float):
    study = frame.groupby("study_group", as_index=False).agg(
        label=("label", "max"), score=("image_score", "max"))
    prediction = (study.score.to_numpy() >= threshold).astype(np.uint8)
    return {"n_studies": int(len(study)), **_binary_metrics(study.label, prediction)}


def _save_example(clean, edited, truth, prediction, output: Path, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Clean")
    axes[1].imshow(edited, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Synthetic edit")
    axes[2].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground-truth mask")
    axes[3].imshow(edited, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(~prediction, prediction), cmap="autumn", alpha=0.7, vmin=0, vmax=1)
    axes[3].set_title("Predicted localization")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_supervised_localization(train_dir, validation_dir, test_dir, output_dir, *,
                                seed=42, epochs=30, batch_size=8, image_size=256,
                                max_train_files=0, max_validation_files=0,
                                max_validation_attack_files=256, max_test_files=0,
                                attack_fraction=0.20,
                                pixel_quantile=99.5, image_quantile=99.0,
                                base_channels=24, device_name="auto", architecture="unet",
                                loss_weighting="fixed", initial_bce_weight=0.5,
                                alpha_learning_rate=1e-3,
                                bilevel_meta_learning_rate=0.01,
                                bilevel_inner_step_size=1e-3,
                                adversarial_training=False, positive_weight=3.0,
                                adversarial_epsilon=0.02, adversarial_step_size=0.004,
                                adversarial_steps=6, bilevel_update_interval=5):
    start_time = time.perf_counter()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    if image_size < 64 or image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16.")
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    if loss_weighting not in {"fixed", "learnable", "bilevel"}:
        raise ValueError("loss_weighting must be fixed, learnable, or bilevel.")
    if not 0.0 <= initial_bce_weight <= 1.0:
        raise ValueError("initial_bce_weight must be in [0, 1].")
    if positive_weight <= 0 or adversarial_epsilon < 0 or adversarial_step_size <= 0:
        raise ValueError("Positive weight and adversarial budgets must be valid positive values.")
    if adversarial_steps < 1 or bilevel_update_interval < 1:
        raise ValueError("Adversarial steps and bilevel update interval must be positive.")
    if adversarial_training and batch_size < 3:
        raise ValueError("Balanced clean/standard/PGD batches require --batch-size of at least 3.")
    output.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = (torch.device("cuda") if device_name == "auto" and torch.cuda.is_available()
              else torch.device("cpu") if device_name == "auto" else torch.device(device_name))

    train_paths = find_dicom_files(train_dir)
    validation_paths = find_dicom_files(validation_dir)
    test_paths = find_dicom_files(test_dir)
    if not train_paths or not validation_paths or not test_paths:
        raise ValueError("Train, validation, and test folders must each contain readable DICOM files.")
    selection_rng = random.Random(seed)
    if max_train_files and len(train_paths) > max_train_files:
        train_paths = sorted(selection_rng.sample(train_paths, max_train_files))
    if max_validation_files and len(validation_paths) > max_validation_files:
        validation_paths = sorted(selection_rng.sample(validation_paths, max_validation_files))
    if max_test_files and len(test_paths) > max_test_files:
        test_paths = sorted(selection_rng.sample(test_paths, max_test_files))

    print(f"Loading clean validation/test slices; device={device}", flush=True)
    validation_images = _load_paths(validation_paths, image_size)
    test_images = _load_paths(test_paths, image_size)
    if not validation_images or not test_images:
        raise ValueError("No decodable single-frame images found after pixel decoding.")

    train_dataset = (MixedAdversarialTrainingDataset(train_paths, image_size, seed)
                     if adversarial_training else SyntheticTrainingDataset(train_paths, image_size, seed))
    train_generator = torch.Generator().manual_seed(seed + 7_919)
    loader_batch_size = max(1, batch_size // 3) if adversarial_training else batch_size
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda",
                              generator=train_generator)
    model = build_localizer(architecture=architecture, base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    validation_groups = sorted({item["study_group"] for item in validation_images})
    if len(validation_groups) < 2:
        raise ValueError("Study-disjoint validation weighting needs at least two validation studies.")
    midpoint = max(1, len(validation_groups) // 2)
    alpha_validation_studies = set(validation_groups[:midpoint])
    selection_validation_studies = set(validation_groups[midpoint:])
    alpha_validation_records = [item for item in validation_images
                                if item["study_group"] in alpha_validation_studies]
    selection_validation_records = [item for item in validation_images
                                   if item["study_group"] in selection_validation_studies]
    validation_attack_records = selection_validation_records
    if max_validation_attack_files and len(validation_attack_records) > max_validation_attack_files:
        attack_selection_rng = random.Random(seed + 65_537)
        chosen = sorted(attack_selection_rng.sample(range(len(validation_attack_records)),
                                                    max_validation_attack_files))
        validation_attack_records = [validation_attack_records[index] for index in chosen]
    validation_attack_loader = DataLoader(
        SyntheticValidationDataset(validation_attack_records, seed),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda")
    selection_clean_tensor = (torch.stack([item["image"] for item in selection_validation_records])
                              .unsqueeze(1))
    selection_clean_loader = DataLoader(
        TensorDataset(selection_clean_tensor), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda")
    alpha_validation_records_for_meta = alpha_validation_records
    if max_validation_attack_files and len(alpha_validation_records_for_meta) > max_validation_attack_files:
        alpha_rng = random.Random(seed + 131_071)
        alpha_indices = sorted(alpha_rng.sample(range(len(alpha_validation_records_for_meta)),
                                                max_validation_attack_files))
        alpha_validation_records_for_meta = [alpha_validation_records_for_meta[index]
                                             for index in alpha_indices]
    validation_meta_loader = DataLoader(
        SyntheticValidationDataset(alpha_validation_records_for_meta, seed,
                                   include_clean=adversarial_training),
        batch_size=max(1, min(batch_size // 3, 4)) if adversarial_training else batch_size,
        shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed + 15_827))
    history = []
    best_state = None
    best_epoch = None
    best_validation_iou = -1.0
    best_validation_rows = None
    best_alpha = initial_bce_weight
    best_validation_clean_fpr = float("inf")
    alpha_initial_logit = float(np.log(max(initial_bce_weight, 1e-6) /
                                       max(1.0 - initial_bce_weight, 1e-6)))
    alpha_logit = nn.Parameter(torch.tensor(alpha_initial_logit, device=device,
                                            dtype=torch.float32))
    alpha_optimizer = (torch.optim.Adam([alpha_logit], lr=alpha_learning_rate)
                       if loss_weighting == "learnable" else None)
    meta_optimizer = (torch.optim.Adam([alpha_logit], lr=bilevel_meta_learning_rate)
                      if loss_weighting == "bilevel" else None)
    alpha_history = []
    meta_validation_history = []
    meta_gradient_history = []
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        validation_meta_iterator = iter(validation_meta_loader)
        losses = []
        epoch_meta_validation = float("nan")
        epoch_hypergradient = float("nan")
        for batch_index, batch_data in enumerate(train_loader):
            if adversarial_training:
                clean_inputs, standard_inputs, standard_masks = batch_data
                clean_inputs = clean_inputs.to(device, non_blocking=True)
                standard_inputs = standard_inputs.to(device, non_blocking=True)
                standard_masks = standard_masks.to(device, non_blocking=True)
                pgd_inputs = _generate_pgd_evasion(
                    model, standard_inputs, epsilon=adversarial_epsilon,
                    step_size=adversarial_step_size, steps=adversarial_steps,
                    seed=seed + epoch * len(train_loader) + batch_index,
                    random_start=True,
                )
                inputs = torch.cat([clean_inputs, standard_inputs, pgd_inputs], dim=0)
                masks = torch.cat([
                    torch.zeros_like(standard_masks), standard_masks, standard_masks,
                ], dim=0)
            else:
                inputs, masks = batch_data
                inputs = inputs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

            if (loss_weighting == "bilevel" and
                    batch_index % bilevel_update_interval == 0):
                try:
                    validation_meta_batch = next(validation_meta_iterator)
                except StopIteration:
                    validation_meta_iterator = iter(validation_meta_loader)
                    validation_meta_batch = next(validation_meta_iterator)
                if adversarial_training:
                    validation_clean = validation_meta_batch[0].to(device, non_blocking=True)
                    validation_standard = validation_meta_batch[1].to(device, non_blocking=True)
                    validation_masks = validation_meta_batch[2].to(device, non_blocking=True)
                    validation_pgd = _generate_pgd_evasion(
                        model, validation_standard, epsilon=adversarial_epsilon,
                        step_size=adversarial_step_size, steps=10,
                        seed=seed + 1_000_003 + epoch * len(train_loader) + batch_index,
                        random_start=True,
                    )
                    validation_meta_inputs = torch.cat([validation_standard, validation_pgd], dim=0)
                    validation_meta_masks = torch.cat([validation_masks, validation_masks], dim=0)
                    family_batch = tuple(validation_meta_batch[3])
                    validation_families = family_batch + family_batch
                else:
                    validation_meta_inputs = validation_meta_batch[0].to(device, non_blocking=True)
                    validation_meta_masks = validation_meta_batch[1].to(device, non_blocking=True)
                    validation_clean = None
                    validation_families = None
                _, epoch_meta_validation, epoch_hypergradient = _bilevel_weight_update(
                    model, inputs, masks, validation_meta_inputs, validation_meta_masks,
                    alpha_logit, meta_optimizer, positive_weight=positive_weight,
                    inner_step_size=bilevel_inner_step_size,
                    clean_validation_inputs=validation_clean,
                    validation_families=validation_families,
                )
            optimizer.zero_grad(set_to_none=True)
            if alpha_optimizer is not None:
                alpha_optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if loss_weighting == "fixed":
                alpha = initial_bce_weight
            else:
                alpha = torch.sigmoid(alpha_logit)
                if loss_weighting == "bilevel":
                    alpha = alpha.detach()
            bce, dice = _segmentation_components(logits, masks, positive_weight=positive_weight)
            loss = alpha * bce + (1.0 - alpha) * dice
            loss.backward()
            optimizer.step()
            if alpha_optimizer is not None:
                alpha_optimizer.step()
                with torch.no_grad():
                    alpha_logit.clamp_(-6.0, 6.0)
            losses.append(float(loss.detach().cpu()))
        learned_alpha = (float(torch.sigmoid(alpha_logit).detach().cpu())
                         if loss_weighting != "fixed" else float(initial_bce_weight))
        alpha_history.append(learned_alpha)
        meta_validation_history.append(epoch_meta_validation)
        meta_gradient_history.append(epoch_hypergradient)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if adversarial_training:
            validation_metrics = _robust_validation_metrics(
                model, validation_attack_loader, selection_clean_loader, device,
                epsilon=adversarial_epsilon, step_size=adversarial_step_size,
                pgd_steps=10, seed=seed + 2_000_003 + epoch,
            )
        else:
            validation_metrics = _validation_localization_metrics(model, validation_attack_loader, device)
        history_row = {"epoch": epoch + 1, "training_loss": mean_loss,
                       "bce_weight": learned_alpha,
                       "dice_weight": 1.0 - learned_alpha,
                       "bilevel_validation_objective": epoch_meta_validation,
                       "bilevel_hypergradient": epoch_hypergradient,
                       "validation_macro_pixel_dice": validation_metrics["macro_pixel_dice"],
                       "validation_macro_pixel_iou": validation_metrics["macro_pixel_iou"]}
        if adversarial_training:
            history_row["validation_clean_image_fpr_q99"] = validation_metrics["clean_image_fpr_at_q99"]
            history_row["validation_clean_image_threshold_q99"] = validation_metrics["clean_image_threshold_at_q99"]
            validation_rows = validation_metrics["by_condition"]
        else:
            validation_rows = validation_metrics["by_family"]
        for family_metrics in validation_rows:
            family = family_metrics["attack_family"]
            condition = family_metrics.get("condition", "unattacked")
            history_row[f"validation_{family}_{condition}_iou"] = family_metrics["pixel_iou"]
            history_row[f"validation_{family}_{condition}_dice"] = family_metrics["pixel_dice"]
        history.append(history_row)
        clean_fpr_for_selection = validation_metrics.get("clean_image_fpr_at_q99", 0.0)
        is_better = (validation_metrics["macro_pixel_iou"] > best_validation_iou or
                     (np.isclose(validation_metrics["macro_pixel_iou"], best_validation_iou) and
                      clean_fpr_for_selection < best_validation_clean_fpr))
        if is_better:
            best_validation_iou = validation_metrics["macro_pixel_iou"]
            best_validation_clean_fpr = clean_fpr_for_selection
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_validation_rows = validation_rows
            best_alpha = learned_alpha
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={mean_loss:.5f}; "
              f"val_Dice={validation_metrics['macro_pixel_dice']:.4f}; "
              f"val_IoU={validation_metrics['macro_pixel_iou']:.4f}; "
              f"best_epoch={best_epoch}", flush=True)
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; verify that the training fold is readable.")
    model.load_state_dict(best_state)
    model.to(device)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(best_validation_rows).assign(selected_epoch=best_epoch).to_csv(
        output / "validation_mask_metrics.csv", index=False)

    val_loader = DataLoader(CleanValidationDataset(
        [item["path"] for item in selection_validation_records], image_size),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda")
    rng = np.random.default_rng(seed)
    val_pixel_samples = []
    val_image_scores = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            probability_tensor, image_scores = _predict_batch(model, batch.to(device, non_blocking=True))
            probability_batch = probability_tensor.squeeze(1).cpu().numpy()
            for probability, image_score in zip(probability_batch, image_scores.cpu().numpy()):
                flat = np.asarray(probability, dtype=np.float32).ravel()
                sample_size = min(2048, len(flat))
                val_pixel_samples.extend(flat[rng.choice(len(flat), sample_size, replace=False)].tolist())
                val_image_scores.append(float(image_score))
    pixel_threshold = float(np.percentile(val_pixel_samples, pixel_quantile))
    image_threshold = float(np.percentile(val_image_scores, image_quantile))

    groups: dict[str, list[int]] = {}
    for index, item in enumerate(test_images):
        groups.setdefault(item["study_group"], []).append(index)
    study_ids = sorted(groups)
    attack_study_count = max(1, round(len(study_ids) * attack_fraction))
    selected_studies = {}
    for family_index, family in enumerate(TEST_FAMILIES):
        attack_rng = random.Random(seed + 1009 * (family_index + 1))
        selected_studies[family] = set(attack_rng.sample(study_ids, min(attack_study_count, len(study_ids))))

    model.eval()
    image_rows, summary_rows, study_metric_rows = [], [], []
    clean_image_rows = []
    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            batch_items = test_images[start:start + batch_size]
            batch = torch.stack([item["image"] for item in batch_items])[:, None]
            probability_tensor, image_scores = _predict_batch(model, batch.to(device, non_blocking=True))
            probability_batch = probability_tensor.squeeze(1).cpu().numpy()
            for item, probability, image_score_value in zip(batch_items, probability_batch, image_scores.cpu().numpy()):
                image_score = float(image_score_value)
                clean_image_rows.append({
                    "condition": "clean", "attack_family": "none", "file": relative_key(item["path"], test_dir),
                    "study_group": item["study_group"], "label": 0, "image_score": image_score,
                    "image_prediction": int(image_score >= image_threshold), "threshold": image_threshold,
                    "ground_truth_pixels": 0,
                    "predicted_pixels": int(np.count_nonzero(probability >= pixel_threshold)),
                    "pixel_threshold": pixel_threshold,
                })
    clean_frame = pd.DataFrame(clean_image_rows)
    clean_image_fpr = float(clean_frame["image_prediction"].mean())
    clean_pixel_alarm_rate = float(clean_frame["predicted_pixels"].sum() /
                                   (len(clean_frame) * image_size * image_size))
    clean_study = _study_rows(clean_frame, image_threshold)
    image_rows.extend(clean_image_rows)

    for family in TEST_FAMILIES:
        labels, scores, truths, predictions = [], [], [], []
        attack_example = None
        condition_rows = []
        family_index = TEST_FAMILIES.index(family)
        for start in range(0, len(test_images), batch_size):
            batch_items = test_images[start:start + batch_size]
            edited_batch, truth_batch, label_batch = [], [], []
            for offset, item in enumerate(batch_items):
                index = start + offset
                clean = item["image"]
                if item["study_group"] in selected_studies[family]:
                    edit_rng = random.Random(seed + index * 17 + 9001 * (family_index + 1))
                    edited, truth = _apply_test_edit(clean, family, edit_rng)
                    label = int(bool(torch.any(truth)))
                else:
                    edited, truth, label = clean, torch.zeros_like(clean, dtype=torch.bool), 0
                edited_batch.append(edited)
                truth_batch.append(truth)
                label_batch.append(label)
            with torch.no_grad():
                probability_tensor, image_scores = _predict_batch(
                    model, torch.stack(edited_batch)[:, None].to(device, non_blocking=True)
                )
                probability_batch = probability_tensor.squeeze(1).cpu().numpy()
                image_scores = image_scores.cpu().numpy()
            for item, edited, truth, label, probability, image_score_value in zip(
                    batch_items, edited_batch, truth_batch, label_batch, probability_batch, image_scores):
                image_score = float(image_score_value)
                image_prediction = int(image_score >= image_threshold)
                predicted_mask = probability >= pixel_threshold
                labels.append(label)
                scores.append(image_score)
                if label:
                    truths.append(truth.numpy())
                    predictions.append(predicted_mask)
                    if attack_example is None:
                        attack_example = (item["image"].numpy(), edited.numpy(), truth.numpy(), probability)
                condition_rows.append({
                    "condition": family, "attack_family": family,
                    "file": relative_key(item["path"], test_dir), "study_group": item["study_group"],
                    "label": label, "image_score": image_score, "image_prediction": image_prediction,
                    "threshold": image_threshold,
                    "ground_truth_pixels": int(torch.count_nonzero(truth)),
                    "predicted_pixels": int(np.count_nonzero(predicted_mask)),
                    "pixel_threshold": pixel_threshold,
                })
        frame = pd.DataFrame(condition_rows)
        image_rows.extend(condition_rows)
        binary = _binary_metrics(labels, frame["image_prediction"].to_numpy())
        binary.update({
            "condition": family,
            "n_images": int(len(frame)), "n_positive_images": int(sum(labels)),
            "loss_weighting": loss_weighting,
            "bce_weight": best_alpha,
            "dice_weight": 1.0 - best_alpha,
            "image_auroc": _safe_auc(np.asarray(labels), np.asarray(scores), roc_auc_score),
            "image_auprc": _safe_auc(np.asarray(labels), np.asarray(scores), average_precision_score),
            "clean_test_image_fpr": clean_image_fpr,
            "clean_test_pixel_alarm_fraction": clean_pixel_alarm_rate,
        })
        binary.update(_localization_metrics(truths, predictions))
        summary_rows.append(binary)
        study_metrics = {"condition": family, **_study_rows(frame, image_threshold)}
        study_metric_rows.append(study_metrics)
        if attack_example:
            clean_np, edited_np, truth_np, probability = attack_example
            prediction_np = probability >= pixel_threshold
            _save_example(clean_np, edited_np, truth_np, prediction_np,
                          output / "figures" / f"{family}.png",
                          f"Synthetic localization example: {family}")

    summary = pd.DataFrame(summary_rows)
    predictions_frame = pd.DataFrame(image_rows)
    study_metrics_frame = pd.DataFrame(study_metric_rows)
    summary.to_csv(output / "pixel_localization_summary.csv", index=False)
    predictions_frame.to_csv(output / "pixel_localization_per_image.csv", index=False)
    study_metrics_frame.to_csv(output / "pixel_localization_study_metrics.csv", index=False)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "config": {"image_size": image_size, "base_channels": base_channels,
                   "architecture": architecture,
                   "pixel_threshold": pixel_threshold, "image_threshold": image_threshold,
                   "image_score_type": "maximum_pre_sigmoid_logit",
                   "pixel_quantile": pixel_quantile, "image_quantile": image_quantile,
                   "seed": seed, "train_families": list(TRAIN_FAMILIES),
                   "withheld_test_family": "copy_move_unseen"},
        "loss_weighting": loss_weighting,
        "bce_weight": best_alpha,
        "dice_weight": 1.0 - best_alpha,
    }
    checkpoint_name = "attention_unet_localizer.pt" if architecture == "attention_unet" else "unet_localizer.pt"
    torch.save(checkpoint, output / checkpoint_name)
    config = {
        "dataset": "CPAISD", "train_images_used": len(train_paths),
        "validation_images_used": len(validation_images), "test_images_used": len(test_images),
        "test_studies": len(study_ids), "attack_fraction": attack_fraction,
        "attacked_studies_per_condition": attack_study_count,
        "train_families": list(TRAIN_FAMILIES), "test_families": list(TEST_FAMILIES),
        "withheld_attack_family": "copy_move_unseen", "image_size": image_size,
        "epochs_requested": epochs, "epochs_completed": len(history),
        "best_epoch_selected_by_validation_macro_pixel_iou": best_epoch,
        "best_validation_macro_pixel_iou": best_validation_iou,
        "best_validation_clean_image_fpr_at_q99": best_validation_clean_fpr,
        "validation_attack_images": len(validation_attack_records),
        "validation_metric_threshold": 0.5,
        "batch_size": batch_size, "base_channels": base_channels,
        "architecture": architecture,
        "loss_weighting": loss_weighting,
        "adversarial_training": bool(adversarial_training),
        "positive_pixel_bce_weight": positive_weight,
        "adversarial_epsilon_linf": adversarial_epsilon if adversarial_training else None,
        "adversarial_step_size": adversarial_step_size if adversarial_training else None,
        "adversarial_training_steps": adversarial_steps if adversarial_training else None,
        "bilevel_update_interval_batches": bilevel_update_interval if loss_weighting == "bilevel" else None,
        "image_score_type": "maximum_pre_sigmoid_logit",
        "initial_bce_weight": initial_bce_weight,
        "selected_bce_weight": best_alpha,
        "selected_dice_weight": 1.0 - best_alpha,
        "alpha_learning_rate": alpha_learning_rate,
        "bilevel_meta_learning_rate": bilevel_meta_learning_rate,
        "bilevel_inner_step_size": bilevel_inner_step_size,
        "alpha_validation_studies": len(alpha_validation_studies),
        "checkpoint_selection_validation_studies": len(selection_validation_studies),
        "alpha_validation_images_used": len(alpha_validation_records_for_meta),
        "device": str(device), "seed": seed, "pixel_threshold": pixel_threshold,
        "image_threshold": image_threshold, "pixel_threshold_clean_quantile": pixel_quantile,
        "image_threshold_clean_quantile": image_quantile,
        "clean_test_image_fpr": clean_image_fpr,
        "clean_test_pixel_alarm_fraction": clean_pixel_alarm_rate,
        "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        "limitations": [
            "Training and test edits are synthetic; they do not represent all real tampering.",
            "Copy-move is withheld as an edit family, but remains a scripted synthetic attack.",
            "Localization is evaluated at the resized model grid, not native DICOM pixel resolution.",
            "Only the official CPAISD test fold is used; it contains ten studies.",
            "The network is a supervised synthetic-tamper segmenter, not a clinical lesion model.",
        ],
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = [
        "# Supervised DICOM pixel-tamper localization", "",
        f"The {architecture} model is trained to segment synthetic pixel edits. The model receives each image independently; it does not receive a clean counterpart or a hash reference at inference time.",
        f"The compound segmentation loss is {best_alpha:.4f} WBCE + {1.0 - best_alpha:.4f} Dice at the selected checkpoint (positive-pixel BCE weight {positive_weight:g}). Weighting mode: {loss_weighting}.",
        "Training edit families are localized patches and sparse low-amplitude changes. The copy-move family is withheld from training and evaluated as a synthetic out-of-family condition.",
        "The image-level alarm score is the maximum pre-sigmoid logit, avoiding probability saturation. Thresholds are calibrated using clean validation images only. For study-disjoint adaptive weighting, validation studies are split into coefficient-update and checkpoint-selection/calibration groups; the test fold is untouched. Bilevel mode uses a one-inner-step differentiable hypergradient approximation. Adversarial mode balances clean, standard synthetic, and live PGD-evaded samples in each update; PGD inputs are generated with BatchNorm in evaluation mode and are trained against their original tamper masks. The validation-selected checkpoint maximizes mean pixel IoU across unattacked and 10-step PGD validation edits, with clean q99 false-alarm rate as the tie-break. Pixel masks are resized to the model grid before evaluation; they are not native-resolution clinical annotations.", "",
        f"- Train images: {len(train_paths)} (cap={max_train_files or 'all'})",
        f"- Validation images: {len(validation_images)}",
        f"- Test images: {len(test_images)} across {len(study_ids)} studies",
        f"- Device: {device}; epochs completed: {len(history)}; input: {image_size}x{image_size}",
        f"- Architecture: {architecture}; base channels: {base_channels}",
        f"- Loss-weighting mode: {loss_weighting}; selected WBCE/Dice coefficients: {best_alpha:.4f}/{1.0 - best_alpha:.4f}",
        f"- Validation studies: {len(validation_groups)} total; {len(alpha_validation_studies)} coefficient-update; {len(selection_validation_studies)} checkpoint-selection/calibration",
        f"- Best epoch selected by validation macro pixel IoU: {best_epoch} ({best_validation_iou:.4f})",
        f"- Synthetic-mask validation images: {len(validation_attack_records)}; selection threshold: 0.5",
        f"- Clean-test image FPR at calibrated threshold: {clean_image_fpr:.4f}",
        f"- Clean-test pixel alarm fraction: {clean_pixel_alarm_rate:.6f}", "",
        "## Image-level detection and pixel localization", summary.to_markdown(index=False), "",
        "## Best-checkpoint validation-mask metrics", pd.DataFrame(best_validation_rows).to_markdown(index=False), "",
        "## Study-level detection", study_metrics_frame.to_markdown(index=False), "",
        "## Interpretation limits",
        "- A withheld copy-move condition tests transfer to one unseen synthetic family, not generalization to natural attacks.",
        "- Scores and masks are evaluated on the held-out CPAISD test fold; the ten-study sample limits population-level inference.",
        "- Localization metrics use generated masks and the resized image grid.",
        "- This experiment does not establish clinical impact, real-world attack prevalence, or deployment readiness.",
    ]
    (output / "SUPERVISED_LOCALIZATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary
