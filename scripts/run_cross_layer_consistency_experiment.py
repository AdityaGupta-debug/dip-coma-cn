"""Evaluate header-versus-pixel-layout consistency rules on clean and attacked DICOMs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.fs_utils import link_or_copy
from src.pixel_metadata_consistency import check_pixel_metadata_consistency


def make_shape_attack(source: Path, destination: Path, field: str, fraction: float, seed: int):
    files = find_dicom_files(source)
    by_study = defaultdict(list)
    for path in files:
        ds = read_dataset(path, stop_before_pixels=True)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{path.parent}"
        by_study[study].append(path)
    studies = sorted(by_study)
    count = max(1, round(len(studies) * fraction))
    selected = set(random.Random(seed).sample(studies, min(count, len(studies))))
    destination.mkdir(parents=True, exist_ok=True)
    log = {"attack": f"{field.lower()}_declared_plus_one", "seed": seed,
           "selected_study_count": len(selected), "total_study_count": len(studies),
           "modified_file_count": 0, "files": {}}
    for path in files:
        key = relative_key(path, source)
        target = destination / key
        ds = read_dataset(path, stop_before_pixels=True)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{path.parent}"
        if study in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            full = read_dataset(target, stop_before_pixels=False)
            setattr(full, field, int(getattr(full, field)) + 1)
            full.save_as(str(target), enforce_file_format=True)
            log["files"][key] = {"label": 1, "changed_fields": [field]}
            log["modified_file_count"] += 1
        else:
            link_or_copy(path, target)
            log["files"][key] = {"label": 0, "changed_fields": []}
    (destination / "attack_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def evaluate_condition(name: str, root: Path, output: Path, positive_metadata_attack=False):
    log_path = root / "attack_log.json"
    attack_log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    rows = []
    for path in find_dicom_files(root):
        key = relative_key(path, root)
        ds = read_dataset(path, stop_before_pixels=False)
        findings = check_pixel_metadata_consistency(ds)
        label = int(attack_log.get("files", {}).get(key, {}).get("label", 0)) if positive_metadata_attack else 0
        rows.append({"condition": name, "file": key, "ground_truth_layout_mismatch": label,
                     "flag": int(bool(findings)), "findings": ";".join(findings)})
    frame = pd.DataFrame(rows)
    positives = frame[frame.ground_truth_layout_mismatch == 1]
    negatives = frame[frame.ground_truth_layout_mismatch == 0]
    tp = int(((frame.ground_truth_layout_mismatch == 1) & (frame.flag == 1)).sum())
    fn = int(len(positives) - tp)
    fp = int(((frame.ground_truth_layout_mismatch == 0) & (frame.flag == 1)).sum())
    tn = int(len(negatives) - fp)
    summary = {"condition": name, "files": len(frame), "positive_layout_attack_files": len(positives),
               "flagged_files": int(frame.flag.sum()), "recall": tp / len(positives) if len(positives) else None,
               "false_positive_rate": fp / len(negatives) if len(negatives) else None,
               "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{name}_per_file.csv", index=False)
    finding_counts = (frame.loc[frame.findings != "", "findings"].str.split(";").explode()
                      .value_counts().rename_axis("finding").reset_index(name="file_count"))
    finding_counts.to_csv(output / f"{name}_finding_counts.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Clean CPAISD test fold")
    parser.add_argument("--output", required=True, help="Fresh output folder")
    parser.add_argument("--pixel-negative-control", required=True,
                        help="Pixel-only low-amplitude attack folder; expected to preserve header/payload layout")
    parser.add_argument("--attack-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input).resolve()
    attacks_root = output / "attacks"
    configs = [("rows_mismatch", "Rows"), ("columns_mismatch", "Columns")]
    summaries = [evaluate_condition("clean_control", source, output / "evaluation")]
    for name, field in configs:
        attack_dir = attacks_root / name
        make_shape_attack(source, attack_dir, field, args.attack_fraction, args.seed)
        summaries.append(evaluate_condition(name, attack_dir, output / "evaluation", True))
    summaries.append(evaluate_condition("pixel_only_negative_control",
                                        Path(args.pixel_negative_control).resolve(),
                                        output / "evaluation", False))
    table = pd.DataFrame(summaries)
    table.to_csv(output / "cross_layer_summary.csv", index=False)
    report = ["# Header-to-pixel layout consistency experiment", "",
              "The detector checks DICOM-declared dimensions, frame/sample layout, stored bit range, decoded array shape, and (for uncompressed data) PixelData byte length. It does not compare against a clean image or hash.",
              "", "## Results", table.to_markdown(index=False), "",
              "The pixel-only condition is a negative control: these edits change pixel values but preserve the declared layout. Rows/Columns attacks are deliberately detectable header-to-payload contradictions, not subtle or clinically realistic tampering. Clean-control findings must be zero before interpreting attack recall.", ""]
    (output / "CROSS_LAYER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nCross-layer consistency experiment complete: {output}")


if __name__ == "__main__":
    main()
