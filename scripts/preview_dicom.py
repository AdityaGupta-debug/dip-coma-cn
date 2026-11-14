"""Render a representative DICOM slice for visual dataset sanity checking."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from src.dicom_io import find_dicom_files, read_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/real_ct")
    p.add_argument("--output", default="reports/sample_ct_slice.png")
    args = p.parse_args()
    files = find_dicom_files(args.input)
    if not files:
        raise SystemExit("No readable DICOM files found.")
    chosen = files[len(files)//2]
    ds = read_dataset(chosen)
    if not hasattr(ds, "PixelData"):
        raise SystemExit(f"No PixelData in selected file: {chosen}")
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(hu, cmap="gray", vmin=-100, vmax=200)
    plt.axis("off")
    plt.title("Representative CT slice (soft-tissue window)")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out}")
    print(f"Source relative path: {chosen}")
    print(f"Shape: {hu.shape}; HU range: {hu.min():.1f} to {hu.max():.1f}")

if __name__ == "__main__":
    main()
