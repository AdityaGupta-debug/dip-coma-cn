import math

def _number(ds, name, default=0.0):
    try:
        value = getattr(ds, name)
        if isinstance(value, (list, tuple)):
            vals = [float(x) for x in value]
            return sum(vals) / len(vals) if vals else default
        return float(value)
    except Exception:
        return default

def metadata_features(ds):
    fields = [
        "PatientID", "PatientName", "StudyDescription", "SeriesDescription",
        "ProtocolName", "Modality", "ImageType", "Manufacturer",
        "PhotometricInterpretation", "BodyPartExamined"
    ]
    out = {}
    for field in fields:
        value = getattr(ds, field, "")
        if isinstance(value, (list, tuple)):
            value = "\\".join(map(str, value))
        value = str(value)
        out[f"{field}_length"] = float(len(value))
        out[f"{field}_present"] = 1.0 if value.strip() else 0.0
        # A stable categorical feature; no patient text is stored in the model.
        out[f"{field}_category"] = value[:80] if field in ("Modality", "PhotometricInterpretation", "BodyPartExamined", "Manufacturer") else str(len(value))
    for field in ("Rows", "Columns", "SamplesPerPixel", "BitsAllocated", "BitsStored",
                  "HighBit", "PixelRepresentation", "SliceThickness", "SpacingBetweenSlices",
                  "RescaleIntercept", "RescaleSlope", "KVP", "XRayTubeCurrent"):
        out[field] = _number(ds, field)
    try:
        spacing = getattr(ds, "PixelSpacing", [0, 0])
        out["PixelSpacing_0"] = float(spacing[0])
        out["PixelSpacing_1"] = float(spacing[1])
    except Exception:
        out["PixelSpacing_0"] = 0.0
        out["PixelSpacing_1"] = 0.0
    try:
        out["ImageType_count"] = float(len(ds.ImageType))
    except Exception:
        out["ImageType_count"] = 0.0
    return out
