from pathlib import Path
from collections import defaultdict
import argparse, json, random, shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dicom_io import find_dicom_files, read_dataset
from src.fs_utils import link_or_copy

def main():
    ap = argparse.ArgumentParser(description="Select a lightweight complete-study subset from CPAISD.")
    ap.add_argument("--input", required=True, help="Extracted CPAISD root containing train, val, test")
    ap.add_argument("--output", default="data/real_ct")
    ap.add_argument("--train-studies", type=int, default=92)
    ap.add_argument("--validation-studies", type=int, default=10)
    ap.add_argument("--test-studies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src, out = Path(a.input).resolve(), Path(a.output).resolve()
    limits = {"train": a.train_studies, "val": a.validation_studies, "test": a.test_studies}
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {out}")
    out.mkdir(parents=True)
    summary = {"dataset": "CPAISD", "seed": a.seed, "official_folds_preserved": True, "folds": {}}
    patient_folds = defaultdict(set)
    for fold in limits:
        fold_root = src / fold
        if not fold_root.exists():
            raise FileNotFoundError(f"Missing fold: {fold_root}")
        for f in find_dicom_files(fold_root):
            ds = read_dataset(f, stop_before_pixels=True)
            patient = str(getattr(ds, "PatientID", "")).strip()
            if patient:
                patient_folds[patient].add(fold)
    overlaps = {patient: sorted(folds) for patient, folds in patient_folds.items() if len(folds) > 1}
    if overlaps:
        raise ValueError(f"PatientID leakage across official folds ({len(overlaps)} IDs); inspect fold assignments before proceeding.")
    summary["patient_id_cross_fold_leakage_count"] = 0
    for fold, limit in limits.items():
        root = src / fold
        if not root.exists(): raise FileNotFoundError(f"Missing fold: {root}")
        groups = defaultdict(list)
        for f in find_dicom_files(root):
            ds = read_dataset(f, stop_before_pixels=True)
            uid = str(getattr(ds, "StudyInstanceUID", "")).strip() or str(f.parent)
            groups[uid].append(f)
        if len(groups) < limit: raise ValueError(f"{fold}: asked for {limit} studies, found {len(groups)}")
        keys = sorted(groups)
        random.Random(a.seed + {"train":0,"val":1,"test":2}[fold]).shuffle(keys)
        chosen = keys[:limit]
        count = 0
        for key in chosen:
            for f in groups[key]:
                link_or_copy(f, out / fold / f.relative_to(root))
                count += 1
        summary["folds"][fold] = {"selected_studies":len(chosen), "selected_dicom_files":count, "available_studies":len(groups)}
    (out / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Subset ready: {out}")
if __name__ == "__main__": main()
