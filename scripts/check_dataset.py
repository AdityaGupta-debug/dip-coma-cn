from pathlib import Path
from collections import Counter
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dicom_io import find_dicom_files, read_dataset

def main():
    parser = argparse.ArgumentParser(description="Inspect a DICOM dataset directory")
    parser.add_argument("--input", default="data/real_ct")
    args = parser.parse_args()
    root = Path(args.input)
    files = find_dicom_files(root)
    modalities = Counter()
    studies, series = set(), set()
    for path in files:
        ds = read_dataset(path, stop_before_pixels=True)
        modalities[str(getattr(ds, "Modality", "MISSING"))] += 1
        studies.add(str(getattr(ds, "StudyInstanceUID", "")))
        series.add(str(getattr(ds, "SeriesInstanceUID", "")))
    print(f"DICOM files: {len(files)}")
    print(f"Modalities: {dict(modalities)}")
    print(f"Unique studies: {len(studies)}")
    print(f"Unique series: {len(series)}")
    print("This checks DICOM readability; it does not certify dataset quality or de-identification.")
    if not files:
        raise SystemExit("No readable DICOM files found. Check the input directory.")

if __name__ == "__main__":
    main()
