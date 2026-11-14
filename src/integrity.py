import hashlib
import json
from pathlib import Path
from .dicom_io import find_dicom_files, read_dataset, relative_key

EXCLUDED_TAGS = {"7FE00010"}  # PixelData: verified separately

def _clean_value(value):
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return str(value.value)
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.hex()
    return str(value)

def _dataset_elements_payload(ds):
    payload = {}
    for elem in ds:
        tag = f"{elem.tag.group:04X}{elem.tag.element:04X}"
        if tag in EXCLUDED_TAGS or elem.tag.element == 0x0000:
            continue
        if elem.VR == "SQ":
            seq = []
            for item in elem.value:
                seq.append(metadata_payload(item))
            payload[tag] = seq
        else:
            payload[tag] = _clean_value(elem.value)
    return payload

def metadata_payload(ds):
    """Canonical DICOM header payload, including group 0002 file meta.

    PixelData and group-length elements are excluded by design. File-meta
    elements are part of the DICOM header and must be covered by integrity
    checks just like the main dataset elements.
    """
    payload = _dataset_elements_payload(ds)
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None:
        payload["FILE_META"] = _dataset_elements_payload(file_meta)
    return payload

def metadata_sha256(ds):
    raw = json.dumps(metadata_payload(ds), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def pixel_sha256(ds):
    value = getattr(ds, "PixelData", b"")
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(bytes(value)).hexdigest()

def create_baseline(input_dir, output_file, include_pixel_hash=False):
    root = Path(input_dir).resolve()
    entries = {}
    paths = find_dicom_files(root)
    for path in paths:
        ds = read_dataset(path, stop_before_pixels=True)
        key = relative_key(path, root)
        record = {
            "metadata_sha256": metadata_sha256(ds),
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
            "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        }
        if include_pixel_hash:
            full_ds = read_dataset(path, stop_before_pixels=False)
            record["pixel_sha256"] = pixel_sha256(full_ds)
            del full_ds
        else:
            record["pixel_sha256"] = None
        entries[key] = record
    result = {
        "schema_version": 1,
        "description": "Trusted clean-data baseline. Protect this file separately from the DICOM dataset.",
        "pixel_hashing_enabled": bool(include_pixel_hash),
        "source_directory_name": root.name,
        "file_count": len(entries),
        "files": entries,
    }
    if not entries:
        raise ValueError(f"No readable DICOM files found under {root}")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def load_baseline(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
