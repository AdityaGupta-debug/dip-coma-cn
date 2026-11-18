"""Interpretable checks for metadata that should agree within DICOM groups.

These checks detect internal inconsistency only. A consistent but incorrect
value cannot be identified without a trusted reference or a stronger semantic
source of truth.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict


STUDY_CONSTANT_FIELDS = (
    "PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyDate",
)

SERIES_CONSTANT_FIELDS = (
    "StudyInstanceUID", "Modality", "Manufacturer", "ProtocolName",
    "Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "PixelSpacing", "ImageOrientationPatient", "SliceThickness",
)

_NUMERIC_FIELDS = {
    "Rows", "Columns", "SamplesPerPixel", "BitsAllocated", "BitsStored",
    "HighBit", "PixelRepresentation", "PixelSpacing", "ImageOrientationPatient",
    "SliceThickness",
}
_CASE_INSENSITIVE_FIELDS = {
    "PatientSex", "Modality", "PhotometricInterpretation", "Manufacturer",
    "ProtocolName",
}


def _value(ds, field: str):
    value = getattr(ds, field, None)
    if value is None:
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value.casefold() if field in _CASE_INSENSITIVE_FIELDS else value
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray))):
        values = list(value)
        if not values:
            return None
        if field in _NUMERIC_FIELDS:
            try:
                return tuple(_normalize_number(item) for item in values)
            except (TypeError, ValueError, OverflowError):
                pass
        return tuple(str(item).strip() for item in values)
    if field in _NUMERIC_FIELDS:
        try:
            return _normalize_number(value)
        except (TypeError, ValueError, OverflowError):
            pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.casefold() if field in _CASE_INSENSITIVE_FIELDS else text or None


def _normalize_number(value):
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return round(number, 8)


def _mark_group_conflicts(items, fields, group_name, findings):
    if len(items) < 2:
        return
    for field in fields:
        values = [_value(ds, field) for _, ds in items]
        present = [value for value in values if value is not None and value != ""]
        if not present:
            continue
        conflict = len(present) != len(values) or len(set(present)) > 1
        if conflict:
            finding = f"{group_name}_metadata_inconsistent:{field}"
            for path, _ in items:
                findings[path].add(finding)


def _instance_dependencies(path, ds, findings):
    try:
        bits_stored = int(ds.BitsStored)
        high_bit = int(ds.HighBit)
        bits_allocated = int(ds.BitsAllocated)
        if bits_stored > 0 and high_bit != bits_stored - 1:
            findings[path].add("tag_dependency_inconsistent:HighBit_BitsStored")
        if bits_stored > bits_allocated:
            findings[path].add("tag_dependency_inconsistent:BitsStored_BitsAllocated")
    except (AttributeError, TypeError, ValueError):
        pass

    modality = str(getattr(ds, "Modality", "")).strip().upper()
    photo = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    try:
        samples = int(getattr(ds, "SamplesPerPixel", 1))
    except (TypeError, ValueError):
        return
    if modality == "CT" and samples == 1 and photo not in {"MONOCHROME1", "MONOCHROME2"}:
        findings[path].add("tag_dependency_inconsistent:CT_PhotometricInterpretation_SamplesPerPixel")


def validate_semantic_consistency(items):
    """Return ``dict[path, sorted findings]`` for study/series relationships.

    ``items`` is an iterable of ``(relative_path, pydicom.Dataset)``. Study-level
    patient descriptors are compared within each StudyInstanceUID; acquisition
    and stored-pixel layout descriptors are compared within each
    SeriesInstanceUID. Missing values are ignored when all instances omit a
    field, but mixed presence or conflicting present values are reported for
    every member of the affected group so callers can aggregate at study or
    series level.
    """
    items = list(items)
    findings = {path: set() for path, _ in items}
    studies = defaultdict(list)
    series = defaultdict(list)
    for path, ds in items:
        _instance_dependencies(path, ds, findings)
        study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
        series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        if study_uid:
            studies[study_uid].append((path, ds))
        if series_uid:
            series[series_uid].append((path, ds))
    for group in studies.values():
        _mark_group_conflicts(group, STUDY_CONSTANT_FIELDS, "study", findings)
    for group in series.values():
        _mark_group_conflicts(group, SERIES_CONSTANT_FIELDS, "series", findings)
    return {path: sorted(values) for path, values in findings.items()}
