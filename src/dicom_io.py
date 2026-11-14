from pathlib import Path
import pydicom

def find_dicom_files(root):
    """Recursively find readable DICOM instances, including extensionless files."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    result = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
            if getattr(ds, "SOPClassUID", None) is not None and getattr(ds, "SOPInstanceUID", None) is not None:
                result.append(path)
        except Exception:
            continue
    return sorted(result)

def read_dataset(path, stop_before_pixels=False):
    return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=False)

def relative_key(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
