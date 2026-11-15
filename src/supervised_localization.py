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
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, base_channels: int = 8):
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


class CleanValidationDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths, self.image_size = paths, image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return _image_tensor(self.paths[index], self.image_size).unsqueeze(0)


class SyntheticValidationDataset(Dataset):
    """Fixed, deterministic synthetic edits on clean validation images."""
    def __init__(self, records: list[dict], seed: int):
        self.records, self.seed = records, seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = self.records[index]["image"]
        family = TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
        rng = random.Random(self.seed + 1_000_003 + index * 31)
        edited, mask = _apply_train_edit(image, family, rng)
        return edited.unsqueeze(0), mask.float().unsqueeze(0), family


def _segmentation_loss(logits, target, positive_weight):
    weight = torch.tensor([positive_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def _infer(model, loader, device):
    model.eval()
    maps = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            logits = model(batch.to(device, non_blocking=True))
            maps.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    return maps


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
                                base_channels=8, device_name="auto"):
    start_time = time.perf_counter()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    if image_size < 64 or image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16.")
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
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

    train_dataset = SyntheticTrainingDataset(train_paths, image_size, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    model = UNetLocalizer(base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    validation_attack_records = validation_images
    if max_validation_attack_files and len(validation_attack_records) > max_validation_attack_files:
        attack_selection_rng = random.Random(seed + 65_537)
        chosen = sorted(attack_selection_rng.sample(range(len(validation_attack_records)),
                                                    max_validation_attack_files))
        validation_attack_records = [validation_attack_records[index] for index in chosen]
    validation_attack_loader = DataLoader(
        SyntheticValidationDataset(validation_attack_records, seed),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda")
    history = []
    best_state = None
    best_epoch = None
    best_validation_iou = -1.0
    best_validation_rows = None
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for inputs, masks in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = _segmentation_loss(logits, masks, positive_weight=12.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        validation_metrics = _validation_localization_metrics(model, validation_attack_loader, device)
        history_row = {"epoch": epoch + 1, "training_loss": mean_loss,
                       "validation_macro_pixel_dice": validation_metrics["macro_pixel_dice"],
                       "validation_macro_pixel_iou": validation_metrics["macro_pixel_iou"]}
        for family_metrics in validation_metrics["by_family"]:
            family = family_metrics["attack_family"]
            history_row[f"validation_{family}_dice"] = family_metrics["pixel_dice"]
            history_row[f"validation_{family}_iou"] = family_metrics["pixel_iou"]
        history.append(history_row)
        if validation_metrics["macro_pixel_iou"] > best_validation_iou:
            best_validation_iou = validation_metrics["macro_pixel_iou"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_validation_rows = validation_metrics["by_family"]
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
        [item["path"] for item in validation_images], image_size),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda")
    rng = np.random.default_rng(seed)
    val_pixel_samples = []
    val_image_scores = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            probability_batch = torch.sigmoid(model(batch.to(device, non_blocking=True))).squeeze(1).cpu().numpy()
            for probability in probability_batch:
                flat = np.asarray(probability, dtype=np.float32).ravel()
                sample_size = min(2048, len(flat))
                val_pixel_samples.extend(flat[rng.choice(len(flat), sample_size, replace=False)].tolist())
                val_image_scores.append(float(np.max(probability)))
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
            probability_batch = torch.sigmoid(model(batch.to(device, non_blocking=True))).squeeze(1).cpu().numpy()
            for item, probability in zip(batch_items, probability_batch):
                image_score = float(np.max(probability))
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
                probability_batch = torch.sigmoid(model(
                    torch.stack(edited_batch)[:, None].to(device, non_blocking=True)
                )).squeeze(1).cpu().numpy()
            for item, edited, truth, label, probability in zip(
                    batch_items, edited_batch, truth_batch, label_batch, probability_batch):
                image_score = float(np.max(probability))
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
                   "pixel_threshold": pixel_threshold, "image_threshold": image_threshold,
                   "pixel_quantile": pixel_quantile, "image_quantile": image_quantile,
                   "seed": seed, "train_families": list(TRAIN_FAMILIES),
                   "withheld_test_family": "copy_move_unseen"},
    }
    torch.save(checkpoint, output / "unet_localizer.pt")
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
        "validation_attack_images": len(validation_attack_records),
        "validation_metric_threshold": 0.5,
        "batch_size": batch_size, "base_channels": base_channels,
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
        "A compact U-Net is trained to segment synthetic pixel edits. The model receives each image independently; it does not receive a clean counterpart or a hash reference at inference time.",
        "Training edit families are localized patches and sparse low-amplitude changes. The copy-move family is withheld from training and evaluated as a synthetic out-of-family condition.",
        "Thresholds are calibrated using clean validation images only. Pixel masks are resized to the model grid before evaluation; they are not native-resolution clinical annotations.", "",
        f"- Train images: {len(train_paths)} (cap={max_train_files or 'all'})",
        f"- Validation images: {len(validation_images)}",
        f"- Test images: {len(test_images)} across {len(study_ids)} studies",
        f"- Device: {device}; epochs completed: {len(history)}; input: {image_size}x{image_size}",
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
