"""Versioned canonical DICOM content hashing profile."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydicom.dataset import Dataset


PROFILE = "dicom-json-v1"


def _value(value: Any) -> Any:
    if isinstance(value, Dataset):
        return _dataset(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"encoding": "base64", "value": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [_value(item) for item in value]
    if value is None:
        return None
    # DICOM value classes stringify consistently after parsing and preserve
    # decimal/integer lexical values without lossy float conversion.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _dataset(ds: Dataset, *, include_pixel_data: bool = False) -> list[dict[str, Any]]:
    elements = []
    for elem in sorted(ds, key=lambda item: int(item.tag)):
        if elem.tag.element == 0x0000:
            continue
        if not include_pixel_data and elem.tag == 0x7FE00010:
            continue
        if elem.VR == "SQ":
            value = [_dataset(item, include_pixel_data=include_pixel_data) for item in elem.value]
        else:
            value = _value(elem.value)
        elements.append({"tag": f"{int(elem.tag):08X}", "vr": elem.VR, "value": value})
    return elements


def canonical_metadata_bytes(ds: Dataset) -> bytes:
    """Serialize dataset and file meta; exclude PixelData/group-length tags."""
    payload = {
        "profile": PROFILE,
        "dataset": _dataset(ds),
        "file_meta": _dataset(ds.file_meta) if getattr(ds, "file_meta", None) is not None else [],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def metadata_sha256(ds: Dataset) -> str:
    return hashlib.sha256(canonical_metadata_bytes(ds)).hexdigest()


def pixel_data_sha256(ds: Dataset) -> str:
    elem = ds.get(0x7FE00010)
    if elem is None:
        raise ValueError("DICOM object has no PixelData element")
    value = elem.value
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("PixelData is not available as raw encoded bytes")
    return hashlib.sha256(bytes(value)).hexdigest()


def raw_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
