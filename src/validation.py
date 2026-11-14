import math
from collections import defaultdict

REQUIRED = ("SOPClassUID", "SOPInstanceUID", "StudyInstanceUID",
            "SeriesInstanceUID", "Modality", "Rows", "Columns")

def validate_datasets(items):
    """
    items: iterable of (relative_path, pydicom Dataset)
    Returns dict[path] -> list[str] of structural findings.
    """
    findings = {path: [] for path, _ in items}
    series_studies = defaultdict(set)
    for path, ds in items:
        for field in REQUIRED:
            if not getattr(ds, field, None):
                findings[path].append(f"missing_{field}")
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            findings[path].append("modality_not_CT")
        try:
            if int(ds.Rows) <= 0 or int(ds.Columns) <= 0:
                findings[path].append("invalid_image_dimensions")
        except Exception:
            findings[path].append("invalid_image_dimensions")
        try:
            spacing = [float(x) for x in ds.PixelSpacing]
            if len(spacing) != 2 or any(not math.isfinite(x) or x <= 0 for x in spacing):
                findings[path].append("invalid_pixel_spacing")
        except Exception:
            findings[path].append("missing_or_invalid_pixel_spacing")
        if getattr(ds, "SliceThickness", None) is not None:
            try:
                if float(ds.SliceThickness) <= 0:
                    findings[path].append("invalid_slice_thickness")
            except Exception:
                findings[path].append("invalid_slice_thickness")
        series = str(getattr(ds, "SeriesInstanceUID", ""))
        study = str(getattr(ds, "StudyInstanceUID", ""))
        if series:
            series_studies[series].add(study)
    inconsistent_series = {s for s, studies in series_studies.items() if len(studies) > 1}
    for path, ds in items:
        if str(getattr(ds, "SeriesInstanceUID", "")) in inconsistent_series:
            findings[path].append("study_uid_inconsistent_within_series")
    return findings
