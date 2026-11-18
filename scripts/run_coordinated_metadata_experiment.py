"""Compare slice-local and coordinated study-wide metadata edits.

This synthetic experiment tests whether group-consistency checks detect a
plausible metadata value changed consistently across every instance in a
study. Clean DICOM files are read only; edits are applied to in-memory copies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks import _same_length_substitution
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.integrity import metadata_sha256
from src.semantic_consistency import validate_semantic_consistency
from src.validation import validate_datasets


def _study_key(ds, path):
    return str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{Path(path).parent}"


def _metrics(labels, predictions):
    tp = sum(bool(y and p) for y, p in zip(labels, predictions))
    tn = sum(bool(not y and not p) for y, p in zip(labels, predictions))
    fp = sum(bool(not y and p) for y, p in zip(labels, predictions))
    fn = sum(bool(y and not p) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_experiment(input_dir: Path, output_dir: Path, seed: int, attack_fraction: float):
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_dicom_files(input_dir)
    if not files:
        raise ValueError(f"No DICOM files found under {input_dir}")
    paths = [relative_key(path, input_dir) for path in files]
    clean = [read_dataset(path, stop_before_pixels=True) for path in files]
    groups = defaultdict(list)
    for index, ds in enumerate(clean):
        groups[_study_key(ds, paths[index])].append(index)
    study_ids = sorted(groups)
    count = min(len(study_ids), max(1, round(len(study_ids) * attack_fraction)))
    selected = sorted(random.Random(seed).sample(study_ids, count))
    selected_set = set(selected)
    labels_by_study = {study: int(study in selected_set) for study in study_ids}

    # Select one populated field that remains constant within CPAISD series.
    # ProtocolName is deliberately selected because a same-length edit tests
    # the current feature-map blind spot while preserving group consistency.
    populated = [str(getattr(ds, "ProtocolName", "")) for ds in clean]
    if not all(populated):
        raise ValueError("ProtocolName must be populated on every instance for this experiment.")

    summary_rows = []
    detail_rows = []
    attack_log = {
        "dataset": input_dir.name,
        "seed": seed,
        "attack_fraction": attack_fraction,
        "attack_unit": "baseline-keyed study",
        "edited_field": "ProtocolName",
        "edit": "same-length deterministic substitution applied to every instance in selected studies",
        "selected_studies_sha256_prefix": [hashlib.sha256(s.encode()).hexdigest()[:16] for s in selected],
        "original_files_modified": False,
        "conditions": {},
    }

    for condition in ("single_instance", "coordinated_study"):
        altered = [copy.deepcopy(ds) for ds in clean]
        changed_paths = []
        rng = random.Random(seed + (0 if condition == "single_instance" else 1))
        for study in selected:
            target_indices = [rng.choice(groups[study])] if condition == "single_instance" else groups[study]
            for index in target_indices:
                old = str(altered[index].ProtocolName)
                altered[index].ProtocolName = _same_length_substitution(old)
                changed_paths.append(paths[index])

        items = list(zip(paths, altered))
        semantic = validate_semantic_consistency(items)
        structural = validate_datasets(items)
        hash_flags = {
            path: int(metadata_sha256(ds) != metadata_sha256(original))
            for path, ds, original in zip(paths, altered, clean)
        }
        finding_maps = {
            "Semantic consistency": semantic,
            "Structural rules": structural,
            "Metadata SHA-256 reference": {path: ["metadata_digest_mismatch"] if hash_flags[path] else [] for path in paths},
        }
        attacked_instance_paths = set(changed_paths)
        attack_log["conditions"][condition] = {
            "modified_instance_count": len(changed_paths),
            "modified_studies": len(selected),
            "modified_file_ids": [hashlib.sha256(path.encode()).hexdigest()[:16] for path in changed_paths],
        }

        for detector, findings in finding_maps.items():
            predictions = {
                study: int(any(findings[paths[index]] for index in indices))
                for study, indices in groups.items()
            }
            labels = [labels_by_study[study] for study in study_ids]
            preds = [predictions[study] for study in study_ids]
            metrics = _metrics(labels, preds)
            summary_rows.append({
                "condition": condition,
                "detector": detector,
                "studies": len(study_ids),
                "positive_studies": len(selected),
                "modified_instances": len(changed_paths),
                **metrics,
            })
            for study in study_ids:
                group_paths = [paths[index] for index in groups[study]]
                detail_rows.append({
                    "condition": condition,
                    "detector": detector,
                    "study_group": hashlib.sha256(study.encode()).hexdigest()[:16],
                    "label": labels_by_study[study],
                    "prediction": predictions[study],
                    "modified_instances": sum(path in attacked_instance_paths for path in group_paths),
                    "finding_count": sum(bool(findings[path]) for path in group_paths),
                    "findings": ";".join(sorted({finding for path in group_paths for finding in findings[path]})),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "coordinated_metadata_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "coordinated_metadata_per_study.csv", index=False)
    (output_dir / "attack_config.json").write_text(json.dumps(attack_log, indent=2), encoding="utf-8")
    report = [
        "# Coordinated metadata consistency experiment", "",
        f"- DICOM instances: {len(paths)}", f"- Baseline-keyed studies: {len(study_ids)}",
        f"- Seed: {seed}", f"- Attack fraction: {attack_fraction}",
        "- The same-length ProtocolName substitution was applied either to one randomly selected instance per attacked study or to every instance in each attacked study.",
        "- Study-level labels and metrics are used; files were altered only in memory and original DICOMs were read only.",
        "- Metadata SHA-256 uses the clean input objects as an exact-reference control.",
        "", "## Results", summary.to_markdown(index=False), "",
        "A coordinated edit can remain consistent within a study and series while differing from the trusted original. The exact-reference hash is expected to detect that change; consistency checks are not designed to identify a coordinated but internally consistent value. This is a synthetic stress test, not evidence of a real clinical incident or a malicious actor.", "",
    ]
    (output_dir / "COORDINATED_METADATA_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean DICOM directory, such as data/cpaisd/test")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    args = parser.parse_args()
    summary = run_experiment(Path(args.input).resolve(), Path(args.output).resolve(), args.seed, args.attack_fraction)
    print(summary.to_string(index=False))
    print(f"\nCoordinated metadata experiment complete: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
