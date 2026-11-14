"""Create patient-grouped train/validation/test folders from a DICOM directory."""
from pathlib import Path
from collections import defaultdict
import argparse, random, shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.fs_utils import link_or_copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/real_ct")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train <= 0 or args.validation < 0 or args.train + args.validation >= 1:
        raise ValueError("Require train > 0, validation >= 0, and train + validation < 1.")
    root, out = Path(args.input).resolve(), Path(args.output).resolve()
    files = find_dicom_files(root)
    if not files:
        raise ValueError(f"No DICOM files found under {root}")
    groups = defaultdict(list)
    for p in files:
        ds = read_dataset(p, stop_before_pixels=True)
        # PatientID is preferred. Study UID is a fallback when PatientID is absent.
        patient = str(getattr(ds, "PatientID", "")).strip()
        group = "PATIENT:" + patient if patient else "STUDY:" + str(getattr(ds, "StudyInstanceUID", ""))
        if group in ("PATIENT:", "STUDY:"):
            group = "FILE:" + relative_key(p, root)
        groups[group].append(p)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    if n < 3:
        raise ValueError(f"At least 3 independent patient/study groups are required; found {n}. More groups are strongly recommended.")
    n_train = max(1, int(n * args.train))
    n_val = max(1, int(n * args.validation)) if args.validation > 0 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    partitions = {
        "train": keys[:n_train],
        "validation": keys[n_train:n_train+n_val],
        "test": keys[n_train+n_val:],
    }
    if out.exists():
        shutil.rmtree(out)
    summary = {}
    for split, group_keys in partitions.items():
        count = 0
        for group in group_keys:
            for source in groups[group]:
                rel = source.resolve().relative_to(root)
                target = out / split / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(source, target)
                count += 1
        summary[split] = {"patients_or_groups": len(group_keys), "dicom_files": count}
    import json
    (out / "split_summary.json").write_text(json.dumps({
        "seed": args.seed, "grouping": "PatientID; StudyInstanceUID fallback",
        "groups": summary
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Keep this split fixed. Train only on train; tune on validation; evaluate on test.")

if __name__ == "__main__":
    main()
