"""Evaluate study/series metadata consistency checks on controlled one-slice edits."""
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
from src.semantic_consistency import validate_semantic_consistency
from src.validation import validate_datasets


ATTACKS = (
    ("PatientID insertion/substitution", "PatientID"),
    ("ProtocolName same-length change", "ProtocolName"),
    ("Manufacturer same-length change", "Manufacturer"),
    ("HighBit dependency violation", "HighBit"),
    ("Study UID inconsistency (positive control)", "StudyInstanceUID"),
    ("Modality inconsistency (positive control)", "Modality"),
)


def _study_key(ds, path):
    return str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{Path(path).parent}"


def _metrics(labels, predictions):
    tp = sum(bool(y and p) for y, p in zip(labels, predictions))
    tn = sum(bool(not y and not p) for y, p in zip(labels, predictions))
    fp = sum(bool(not y and p) for y, p in zip(labels, predictions))
    fn = sum(bool(y and not p) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _edit(ds, field):
    if field in {"ProtocolName", "Manufacturer"}:
        old = str(getattr(ds, field, ""))
        if not old:
            raise ValueError(f"Required tag {field} is absent from selected study instance.")
        setattr(ds, field, _same_length_substitution(old))
        return
    if field == "PatientID":
        old = str(getattr(ds, field, ""))
        ds.PatientID = _same_length_substitution(old) if old else "SYNTHETIC-INSERTED-ID"
        return
    if field == "HighBit":
        bits_stored = int(ds.BitsStored)
        old = int(ds.HighBit)
        ds.HighBit = bits_stored - 2 if bits_stored > 1 else old + 1
        if int(ds.HighBit) == old:
            ds.HighBit = old - 1
        return
    if field == "StudyInstanceUID":
        ds.StudyInstanceUID = "2.25.999999999999999999999999999999999999"
        return
    if field == "Modality":
        ds.Modality = "MR" if str(getattr(ds, "Modality", "")).upper() == "CT" else "CT"
        return
    raise ValueError(f"Unknown semantic attack field: {field}")


def _study_predictions(paths, datasets, finding_map, study_groups):
    predictions = {study: 0 for study in set(study_groups.values())}
    for path, ds in zip(paths, datasets):
        findings = finding_map.get(path, [])
        if findings:
            predictions[study_groups[path]] = 1
    return predictions


def run_experiment(input_dir: Path, output_dir: Path, seed: int, attack_fraction: float):
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = find_dicom_files(input_dir)
    paths = [relative_key(path, input_dir) for path in files]
    clean = [read_dataset(path, stop_before_pixels=True) for path in files]
    items = list(zip(paths, clean))
    studies = defaultdict(list)
    for index, ds in enumerate(clean):
        studies[_study_key(ds, paths[index])].append(index)
    study_groups = {paths[index]: study for study, indices in studies.items() for index in indices}
    study_ids = sorted(studies)
    positive_count = min(len(study_ids), max(1, round(len(study_ids) * attack_fraction)))
    clean_semantic = validate_semantic_consistency(items)
    clean_structural = validate_datasets(items)
    summary_rows = []
    detail_rows = []
    attack_log = {"seed": seed, "attack_fraction": attack_fraction,
                  "unit": "baseline-keyed study", "attacks": {}}

    clean_sem_flags = _study_predictions(paths, clean, clean_semantic, study_groups)
    clean_struct_flags = _study_predictions(paths, clean, clean_structural, study_groups)
    clean_labels = [0] * len(study_ids)
    for detector, flags in (("semantic", clean_sem_flags), ("structural", clean_struct_flags)):
        metrics = _metrics(clean_labels, [flags[study] for study in study_ids])
        summary_rows.append({"condition": "clean_control", "detector": detector,
                             "studies": len(study_ids), "positive_studies": 0,
                             "findings": metrics["tp"] + metrics["fp"], **metrics})

    for attack_index, (name, field) in enumerate(ATTACKS):
        attack_rng = random.Random(seed + attack_index * 101)
        selected = sorted(attack_rng.sample(study_ids, positive_count))
        altered = [copy.deepcopy(ds) for ds in clean]
        selected_paths = []
        for study in selected:
            index = attack_rng.choice(studies[study])
            _edit(altered[index], field)
            selected_paths.append(paths[index])
        altered_items = list(zip(paths, altered))
        semantic_findings = validate_semantic_consistency(altered_items)
        structural_findings = validate_datasets(altered_items)
        labels = {study: int(study in selected) for study in study_ids}
        attack_log["attacks"][name] = {
            "field": field, "selected_studies": [hashlib.sha256(s.encode()).hexdigest()[:16] for s in selected],
            "modified_files": selected_paths,
        }
        for detector, finding_map in (("semantic", semantic_findings),
                                      ("structural", structural_findings)):
            predictions = _study_predictions(paths, altered, finding_map, study_groups)
            metrics = _metrics([labels[s] for s in study_ids], [predictions[s] for s in study_ids])
            flagged_studies = [s for s in study_ids if predictions[s]]
            summary_rows.append({"condition": name, "detector": detector,
                                 "studies": len(study_ids), "positive_studies": len(selected),
                                 "findings": sum(bool(finding_map[p]) for p in paths), **metrics})
            for study in study_ids:
                detail_rows.append({
                    "condition": name, "detector": detector,
                    "study_group": hashlib.sha256(study.encode()).hexdigest()[:16],
                    "label": labels[study], "prediction": predictions[study],
                    "findings": ";".join(sorted({finding for p in studies[study]
                                                   for finding in finding_map[paths[p]]})),
                })
    (output_dir / "attack_config.json").write_text(json.dumps(attack_log, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "semantic_consistency_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "semantic_consistency_per_study.csv", index=False)
    report = [
        "# Semantic metadata consistency experiment", "",
        f"- Input DICOM instances: {len(paths)}", f"- Baseline-keyed studies: {len(study_ids)}",
        f"- Seed: {seed}", f"- Attack fraction: {attack_fraction}",
        "- Attacks changed one instance in each selected study; original DICOM files were read only.",
        "- Metrics use the study as the unit because group consistency rules may flag peer slices.",
        "", "## Study-level results", summary.to_markdown(index=False), "",
        "The semantic checker reports conflicts among fields expected to be stable within a study or series, plus selected tag dependencies. It cannot detect a coordinated change that remains internally consistent, establish malicious intent, or replace a trusted reference. Results are descriptive for this dataset and these synthetic edits.", "",
    ]
    (output_dir / "SEMANTIC_CONSISTENCY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean DICOM directory (for example data/cpaisd/test)")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    args = parser.parse_args()
    summary = run_experiment(Path(args.input).resolve(), Path(args.output).resolve(),
                             args.seed, args.attack_fraction)
    print(summary.to_string(index=False))
    print(f"\nSemantic consistency experiment complete: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
