"""Compare targeted evasion methods against a trained synthetic-edit U-Net.

The attacks are applied only to synthetic edited test copies. The original
DICOM files remain unchanged, and the U-Net receives normalized image arrays,
not a clean reference or DICOM metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
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
    _image_scores_from_logits,
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


def _project_l2(candidate, source, radius):
    delta = candidate - source
    norm = torch.linalg.vector_norm(delta.reshape(-1), ord=2)
    if norm > radius:
        delta = delta * (radius / (norm + 1e-12))
    return (source + delta).clamp(0.0, 1.0)


def _cw_l2_evasion(model, image, *, steps=30, learning_rate=0.01,
                   constant=1.0, l2_radius=2.0, device="cpu"):
    """C&W-style targeted L2 optimization toward an all-clean segmentation.

    The objective is squared L2 distortion plus a hinge penalty on the largest
    tamper logit. A fixed L2 radius bounds the search; constants are prespecified
    and are not tuned against test labels.
    """
    source = image.detach().to(device).unsqueeze(0).unsqueeze(0)
    delta = torch.zeros_like(source, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    for _ in range(steps):
        candidate = (source + delta).clamp(0.0, 1.0)
        logits = model(candidate)
        distortion = torch.sum((candidate - source) ** 2)
        target_penalty = F.relu(logits.max())
        loss = distortion + constant * target_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = _project_l2(source + delta, source, l2_radius)
            delta.copy_(projected - source)
    return (source + delta).clamp(0.0, 1.0).detach()[0, 0]


def _deepfool_l2_evasion(model, image, *, max_steps=10, overshoot=0.02,
                         l2_radius=2.0, device="cpu"):
    """DeepFool-style iterative linearization of the highest tamper logit."""
    source = image.detach().to(device).unsqueeze(0).unsqueeze(0)
    candidate = source.clone()
    for _ in range(max_steps):
        candidate = candidate.detach().requires_grad_(True)
        logits = model(candidate)
        margin = logits.max()
        if float(margin.detach()) <= 0.0:
            break
        gradient = torch.autograd.grad(margin, candidate, only_inputs=True)[0]
        grad_sq = torch.sum(gradient ** 2).clamp_min(1e-12)
        step = (margin.detach() + overshoot) / grad_sq * gradient
        candidate = _project_l2(candidate.detach() - step, source, l2_radius).detach()
    return candidate[0, 0]


def _balanced_cap(examples, limit, seed):
    """Choose a reproducible, round-robin subset across the selected studies."""
    if not limit or len(examples) <= limit:
        return examples
    grouped = defaultdict(list)
    for item in examples:
        grouped[item[0]["study_group"]].append(item)
    rng = random.Random(seed)
    buckets = []
    for study in sorted(grouped):
        bucket = grouped[study]
        rng.shuffle(bucket)
        buckets.append(bucket)
    chosen = []
    while len(chosen) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop())
    return chosen


def run_evaluation(checkpoint_path, test_dir, output_dir, seed=42, attack_fraction=0.20,
                   epsilon=0.02, pgd_steps=10, pgd_step_size=0.004,
                   cw_steps=20, cw_learning_rate=0.01, cw_constant=1.0,
                   l2_radius=2.0, deepfool_steps=10,
                   max_images_per_family=64, device_name="cpu",
                   validation_dir="data/cpaisd/val", image_quantile=99.0,
                   attack_methods=("baseline", "fgsm", "pgd")):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    if (epsilon < 0 or pgd_steps < 1 or pgd_step_size <= 0 or cw_steps < 1
            or cw_learning_rate <= 0 or cw_constant <= 0 or l2_radius <= 0
            or deepfool_steps < 1 or max_images_per_family < 0):
        raise ValueError("Invalid attack budget or optimization parameters.")
    allowed_methods = {"baseline", "fgsm", "bim", "pgd", "cw_l2", "deepfool_l2"}
    attack_methods = tuple(dict.fromkeys(attack_methods))
    if "baseline" not in attack_methods or any(method not in allowed_methods for method in attack_methods):
        raise ValueError("Attack methods must include baseline and use supported method names.")

    device = (torch.device("cuda") if device_name == "auto" and torch.cuda.is_available()
              else torch.device("cpu") if device_name == "auto" else torch.device(device_name))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_localizer(config.get("architecture", "unet"),
                            base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    model.requires_grad_(False)
    image_size = int(config["image_size"])
    pixel_threshold = float(config["pixel_threshold"])
    validation_records = _load_paths(find_dicom_files(Path(validation_dir)), image_size)
    if not validation_records:
        raise ValueError("No clean validation images could be decoded for image-score calibration.")
    clean_validation_scores = []
    with torch.no_grad():
        for start in range(0, len(validation_records), 8):
            batch = torch.stack([item["image"] for item in validation_records[start:start + 8]])[:, None]
            logits = model(batch.to(device))
            clean_validation_scores.extend(_image_scores_from_logits(logits).cpu().tolist())
    image_threshold = float(np.percentile(clean_validation_scores, image_quantile))

    records = _load_paths(find_dicom_files(Path(test_dir)), image_size)
    study_ids = sorted({record["study_group"] for record in records})
    positives_per_condition = max(1, round(len(study_ids) * attack_fraction))
    selected = {}
    for family_index, family in enumerate(TEST_FAMILIES):
        rng = random.Random(seed + 1009 * (family_index + 1))
        selected[family] = set(rng.sample(study_ids, min(positives_per_condition, len(study_ids))))

    rows = []
    summary_rows = []
    methods = attack_methods
    clean_test_scores = []
    with torch.no_grad():
        for start in range(0, len(records), 8):
            batch = torch.stack([item["image"] for item in records[start:start + 8]])[:, None]
            logits = model(batch.to(device))
            clean_test_scores.extend(_image_scores_from_logits(logits).cpu().tolist())
    clean_frame = pd.DataFrame({
        "study_group": [item["study_group"] for item in records],
        "image_score": clean_test_scores,
    })
    clean_file_fpr = float(np.mean(np.asarray(clean_test_scores) >= image_threshold))
    clean_study_scores = clean_frame.groupby("study_group")["image_score"].max().to_numpy()
    clean_study_fpr = float(np.mean(clean_study_scores >= image_threshold))
    for family_index, family in enumerate(TEST_FAMILIES):
        examples = []
        for index, record in enumerate(records):
            if record["study_group"] not in selected[family]:
                continue
            rng = random.Random(seed + index * 17 + 9001 * (family_index + 1))
            edited, truth = _apply_test_edit(record["image"], family, rng)
            examples.append((record, edited, truth))
        candidate_count = len(examples)
        examples = _balanced_cap(examples, max_images_per_family,
                                 seed + family_index * 313)

        stats = {method: {"detected": 0, "baseline_detected": 0, "success": 0,
                          "tp": 0, "fp": 0, "fn": 0, "tn": 0}
                 for method in methods}
        for example_index, (record, edited, truth) in enumerate(examples):
            attack_images = {"baseline": edited}
            if "fgsm" in methods:
                attack_images["fgsm"] = _attack_evasion(
                    model, edited, epsilon, "fgsm", 1, epsilon, seed + example_index, device)
            if "bim" in methods:
                attack_images["bim"] = _attack_evasion(
                    model, edited, epsilon, "bim", pgd_steps, pgd_step_size,
                    seed + example_index, device)
            if "pgd" in methods:
                attack_images["pgd"] = _attack_evasion(
                    model, edited, epsilon, "pgd", pgd_steps, pgd_step_size,
                    seed + example_index, device)
            if "cw_l2" in methods:
                attack_images["cw_l2"] = _cw_l2_evasion(
                    model, edited, steps=cw_steps, learning_rate=cw_learning_rate,
                    constant=cw_constant, l2_radius=l2_radius, device=device)
            if "deepfool_l2" in methods:
                attack_images["deepfool_l2"] = _deepfool_l2_evasion(
                    model, edited, max_steps=deepfool_steps,
                    l2_radius=l2_radius, device=device)
            truth_mask = truth.to(device=device, dtype=torch.bool)
            baseline_alarm = None
            file_id = hashlib.sha256(relative_key(record["path"], Path(test_dir)).encode()).hexdigest()[:16]
            for method, image in attack_images.items():
                with torch.no_grad():
                    logits = model(image.to(device).unsqueeze(0).unsqueeze(0))
                    probability = torch.sigmoid(logits)[0, 0]
                    score = float(_image_scores_from_logits(logits)[0].item())
                prediction = int(score >= image_threshold)
                if method == "baseline":
                    baseline_alarm = bool(prediction)
                    for name in methods:
                        stats[name]["baseline_detected"] += int(baseline_alarm)
                else:
                    stats[method]["success"] += int(bool(baseline_alarm) and not bool(prediction))
                stats[method]["detected"] += prediction
                predicted_mask = probability >= pixel_threshold
                stats[method]["tp"] += int(torch.count_nonzero(truth_mask & predicted_mask).item())
                stats[method]["fp"] += int(torch.count_nonzero(~truth_mask & predicted_mask).item())
                stats[method]["fn"] += int(torch.count_nonzero(truth_mask & ~predicted_mask).item())
                stats[method]["tn"] += int(torch.count_nonzero(~truth_mask & ~predicted_mask).item())
                intersection = int(torch.count_nonzero(truth_mask & predicted_mask).item())
                union = int(torch.count_nonzero(truth_mask | predicted_mask).item())
                delta = image - edited
                rows.append({
                    "family": family, "method": method, "file_id": file_id,
                    "study_group": record["study_group"], "label": 1,
                    "image_score": score, "image_alarm": prediction,
                    "image_threshold": image_threshold, "pixel_threshold": pixel_threshold,
                    "pixel_precision": intersection / max(1, int(torch.count_nonzero(predicted_mask).item())),
                    "pixel_recall": intersection / max(1, int(torch.count_nonzero(truth_mask).item())),
                    "pixel_iou": intersection / max(1, union),
                    "linf_from_scripted_edit": float(torch.max(torch.abs(delta)).item()),
                    "l2_from_scripted_edit": float(torch.linalg.vector_norm(delta.reshape(-1), ord=2).item()),
                })

        for method in methods:
            item = stats[method]
            tp, fp, fn = item["tp"], item["fp"], item["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            summary_rows.append({
                "family": family, "method": method, "positive_images": len(examples),
                "image_recall": item["detected"] / max(1, len(examples)),
                "baseline_detected_images": item["baseline_detected"],
                "detected_images": item["detected"],
                "pixel_precision": precision, "pixel_recall": recall,
                "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                "attack_success_rate": (
                    item["success"] / item["baseline_detected"]
                    if method != "baseline" and item["baseline_detected"] else
                    float("nan") if method != "baseline" else 0.0
                ),
                "clean_test_file_fpr": clean_file_fpr,
                "clean_test_study_fpr": clean_study_fpr,
            })
        print(f"Completed {family}: evaluated {len(examples)} of {candidate_count} positive images.",
              flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "adversarial_comparison_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "adversarial_comparison_per_image.csv", index=False)
    config_out = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "test_dir": str(Path(test_dir).resolve()),
        "seed": seed, "attack_fraction": attack_fraction,
        "max_images_per_family": max_images_per_family,
        "epsilon_linf_normalized": epsilon,
        "attack_budgets": {
            "fgsm": {"norm": "L_inf", "epsilon": epsilon, "steps": 1},
            "bim": {"norm": "L_inf", "epsilon": epsilon, "steps": pgd_steps, "step_size": pgd_step_size, "random_start": False},
            "pgd": {"norm": "L_inf", "epsilon": epsilon, "steps": pgd_steps, "step_size": pgd_step_size, "random_start": True},
            "cw_l2": {"norm": "L2", "radius": l2_radius, "steps": cw_steps,
                      "learning_rate": cw_learning_rate, "constant": cw_constant},
            "deepfool_l2": {"norm": "L2", "radius": l2_radius, "max_steps": deepfool_steps},
        },
        "device": str(device), "image_threshold": image_threshold,
        "image_score_type": "maximum_pre_sigmoid_logit",
        "image_quantile": image_quantile, "validation_images_for_calibration": len(validation_records),
        "clean_test_file_fpr": clean_file_fpr, "clean_test_study_fpr": clean_study_fpr,
        "pixel_threshold": pixel_threshold, "attack_methods": list(methods),
        "objective": "targeted evasion: minimize BCE-with-logits against an all-clean mask",
        "scope": "balanced synthetic edited image subset only; L-infinity and L2 budgets are different and not directly equivalent",
    }
    (output_dir / "attack_config.json").write_text(__import__("json").dumps(config_out, indent=2), encoding="utf-8")
    report = [
        "# U-Net adversarial attack comparison", "",
        f"- Test studies: {len(study_ids)}", f"- Positive studies per edit family: {positives_per_condition}",
        f"- Maximum evaluated images per edit family: {max_images_per_family or 'all selected positives'}",
        f"- L-infinity budget in normalized image units: {epsilon}",
        f"- Image score: max pre-sigmoid logit; clean validation calibration quantile: {image_quantile}",
        f"- Clean test image FPR: {clean_file_fpr:.4f}; clean test study FPR: {clean_study_fpr:.4f}",
        f"- Methods evaluated: {', '.join(methods)}",
        f"- FGSM/BIM/PGD budget: L-infinity epsilon {epsilon}; iterative steps / step size {pgd_steps} / {pgd_step_size}",
        "- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.",
        "- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.",
        "- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.",
        "", "## Results", summary.to_markdown(index=False), "",
        "All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.", "",
    ]
    (output_dir / "ADVERSARIAL_COMPARISON_REPORT.md").write_text("\n".join(report), encoding="utf-8")
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
    parser.add_argument("--cw-steps", type=int, default=20)
    parser.add_argument("--cw-learning-rate", type=float, default=0.01)
    parser.add_argument("--cw-constant", type=float, default=1.0)
    parser.add_argument("--l2-radius", type=float, default=2.0,
                        help="L2 radius for C&W-style and DeepFool-style attacks")
    parser.add_argument("--deepfool-steps", type=int, default=10)
    parser.add_argument("--max-images-per-family", type=int, default=64,
                        help="Balanced, deterministic subset cap across selected studies; zero uses all positives")
    parser.add_argument("--validation", default="data/cpaisd/val",
                        help="Clean validation fold used to recalibrate max-logit image threshold")
    parser.add_argument("--image-quantile", type=float, default=99.0)
    parser.add_argument("--attack-methods", default="baseline,fgsm,pgd",
                        help="Comma-separated methods: baseline,fgsm,bim,pgd,cw_l2,deepfool_l2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_evaluation(args.checkpoint, args.test, args.output, args.seed,
                            args.attack_fraction, args.epsilon, args.pgd_steps,
                            args.pgd_step_size, args.cw_steps, args.cw_learning_rate,
                            args.cw_constant, args.l2_radius, args.deepfool_steps,
                            args.max_images_per_family,
                            args.device, args.validation, args.image_quantile,
                            tuple(method.strip() for method in args.attack_methods.split(",") if method.strip()))
    print(result.to_string(index=False))
    print(f"\nAdversarial attack comparison saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
