"""Evaluate bounded FGSM/PGD evasion against a trained synthetic-edit U-Net.

The attacks are applied only to synthetic edited test copies. The original
DICOM files remain unchanged, and the U-Net receives normalized image arrays,
not a clean reference or DICOM metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dicom_io import find_dicom_files, relative_key
from src.supervised_localization import (
    TEST_FAMILIES,
    build_localizer,
    _apply_test_edit,
    _load_paths,
)


def _attack_evasion(model, image, epsilon, method, steps, step_size, seed, device):
    """Reduce predicted tamper probability under an L-infinity budget."""
    source = image.detach().to(device).unsqueeze(0).unsqueeze(0)
    candidate = source.clone()
    if method == "pgd" and epsilon > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        candidate = candidate + torch.empty_like(candidate).uniform_(-epsilon, epsilon, generator=generator)
        candidate = candidate.clamp(0.0, 1.0)
    iterations = 1 if method == "fgsm" else steps
    alpha = epsilon if method == "fgsm" else step_size
    for _ in range(iterations):
        candidate = candidate.detach().requires_grad_(True)
        logits = model(candidate)
        # Target an all-clean prediction: suppress the detector's positive mask.
        target = torch.zeros_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        gradient = torch.autograd.grad(loss, candidate, only_inputs=True)[0]
        candidate = candidate.detach() - alpha * gradient.sign()
        candidate = torch.maximum(torch.minimum(candidate, source + epsilon), source - epsilon)
        candidate = candidate.clamp(0.0, 1.0)
    return candidate.detach()[0, 0]


def run_evaluation(checkpoint_path, test_dir, output_dir, seed=42, attack_fraction=0.20,
                   epsilon=0.02, pgd_steps=10, pgd_step_size=0.004, device_name="cpu"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    if epsilon < 0 or pgd_steps < 1 or pgd_step_size <= 0:
        raise ValueError("epsilon must be nonnegative, pgd_steps >= 1, and pgd_step_size > 0.")

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_localizer(config.get("architecture", "unet"),
                            base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    image_size = int(config["image_size"])
    pixel_threshold = float(config["pixel_threshold"])
    image_threshold = float(config["image_threshold"])

    records = _load_paths(find_dicom_files(Path(test_dir)), image_size)
    study_ids = sorted({record["study_group"] for record in records})
    positives_per_condition = max(1, round(len(study_ids) * attack_fraction))
    selected = {}
    for family_index, family in enumerate(TEST_FAMILIES):
        rng = random.Random(seed + 1009 * (family_index + 1))
        selected[family] = set(rng.sample(study_ids, min(positives_per_condition, len(study_ids))))

    rows = []
    summary_rows = []
    methods = ("baseline", "fgsm", "pgd")
    for family_index, family in enumerate(TEST_FAMILIES):
        examples = []
        for index, record in enumerate(records):
            if record["study_group"] not in selected[family]:
                continue
            rng = random.Random(seed + index * 17 + 9001 * (family_index + 1))
            edited, truth = _apply_test_edit(record["image"], family, rng)
            examples.append((record, edited, truth))

        outputs = {method: [] for method in methods}
        for example_index, (record, edited, truth) in enumerate(examples):
            outputs["baseline"].append(edited)
            outputs["fgsm"].append(_attack_evasion(
                model, edited, epsilon, "fgsm", 1, epsilon, seed + example_index, device))
            outputs["pgd"].append(_attack_evasion(
                model, edited, epsilon, "pgd", pgd_steps, pgd_step_size, seed + example_index, device))
        baseline_alarm = []
        family_rows = []
        for method in methods:
            method_predictions, truth_list, predicted_list = [], [], []
            for example_index, ((record, _, truth), image) in enumerate(zip(examples, outputs[method])):
                with torch.no_grad():
                    probability = torch.sigmoid(model(image.to(device).unsqueeze(0).unsqueeze(0)))[0, 0]
                score = float(probability.max().item())
                prediction = score >= image_threshold
                predicted_mask = probability >= pixel_threshold
                truth_mask = truth.to(device=device, dtype=torch.bool)
                method_predictions.append(int(prediction))
                truth_list.append(truth_mask)
                predicted_list.append(predicted_mask)
                if method == "baseline":
                    baseline_alarm.append(bool(prediction))
                union = torch.count_nonzero(truth_mask | predicted_mask).item()
                intersection = torch.count_nonzero(truth_mask & predicted_mask).item()
                rows.append({
                    "family": family, "method": method,
                    "file_id": hashlib.sha256(relative_key(record["path"], Path(test_dir)).encode()).hexdigest()[:16],
                    "study_group": record["study_group"],
                    "label": 1, "image_score": score, "image_alarm": int(prediction),
                    "image_threshold": image_threshold,
                    "pixel_threshold": pixel_threshold,
                    "pixel_precision": intersection / max(1, int(torch.count_nonzero(predicted_mask).item())),
                    "pixel_recall": intersection / max(1, int(torch.count_nonzero(truth_mask).item())),
                    "pixel_iou": intersection / max(1, union),
                    "linf_from_scripted_edit": float(torch.max(torch.abs(image - outputs["baseline"][example_index])).item()),
                })
            tp = fp = fn = tn = 0
            for truth_mask, prediction_mask in zip(truth_list, predicted_list):
                tp += int(torch.count_nonzero(truth_mask & prediction_mask).item())
                fp += int(torch.count_nonzero(~truth_mask & prediction_mask).item())
                fn += int(torch.count_nonzero(truth_mask & ~prediction_mask).item())
                tn += int(torch.count_nonzero(~truth_mask & ~prediction_mask).item())
            pixel_precision = tp / (tp + fp) if tp + fp else 0.0
            pixel_recall = tp / (tp + fn) if tp + fn else 0.0
            pixel_iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
            family_rows.append({
                "family": family, "method": method, "positive_images": len(examples),
                "image_recall": float(np.mean(method_predictions)) if method_predictions else 0.0,
                "baseline_detected_images": int(sum(baseline_alarm)) if method == "baseline" else int(sum(baseline_alarm)),
                "detected_images": int(sum(method_predictions)),
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall, "pixel_iou": pixel_iou,
                "attack_success_rate": (
                    sum(base and not pred for base, pred in zip(baseline_alarm, method_predictions)) /
                    sum(baseline_alarm) if method != "baseline" and sum(baseline_alarm) else 0.0
                ),
            })
        summary_rows.extend(family_rows)
        if family_rows and family_rows[0]["positive_images"] != len(examples):
            raise RuntimeError("Unexpected positive-image count during attack evaluation.")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fgsm_pgd_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "fgsm_pgd_per_image.csv", index=False)
    config_out = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "test_dir": str(Path(test_dir).resolve()),
        "seed": seed, "attack_fraction": attack_fraction, "epsilon_linf_normalized": epsilon,
        "fgsm_steps": 1, "pgd_steps": pgd_steps, "pgd_step_size": pgd_step_size,
        "device": str(device), "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "objective": "targeted evasion: minimize BCE-with-logits against an all-clean mask",
        "scope": "synthetic edited test images only; attacks constrained to normalized-image L-infinity epsilon",
    }
    (output_dir / "attack_config.json").write_text(__import__("json").dumps(config_out, indent=2), encoding="utf-8")
    report = [
        "# U-Net FGSM/PGD evasion evaluation", "",
        f"- Test studies: {len(study_ids)}", f"- Positive studies per edit family: {positives_per_condition}",
        f"- L-infinity budget in normalized image units: {epsilon}",
        f"- PGD steps / step size: {pgd_steps} / {pgd_step_size}",
        "- FGSM and PGD were targeted to suppress the U-Net tamper mask; attack success is reported only among synthetic edited images detected by the baseline model.",
        "- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.",
        "", "## Results", summary.to_markdown(index=False), "",
        "All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.", "",
    ]
    (output_dir / "FGSM_PGD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Saved unet_localizer.pt")
    parser.add_argument("--test", default="data/cpaisd/test")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="L-infinity bound in normalized [0,1] image units")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-step-size", type=float, default=0.004)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_evaluation(args.checkpoint, args.test, args.output, args.seed,
                            args.attack_fraction, args.epsilon, args.pgd_steps,
                            args.pgd_step_size, args.device)
    print(result.to_string(index=False))
    print(f"\nFGSM/PGD evaluation saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
