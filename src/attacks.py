from pathlib import Path
import json, random, shutil, hashlib
from .dicom_io import find_dicom_files, read_dataset, relative_key
from .fs_utils import link_or_copy

ATTACKS = ("patient_id", "study_description", "protocol_name_same_length",
           "manufacturer_same_length", "pixel_spacing",
           "study_uid_inconsistency", "remove_protocol", "modality")


def _same_length_substitution(value):
    """Deterministically change alphanumerics while preserving string length."""
    original = str(value)
    if not original:
        return original
    result = []
    for char in original:
        if "0" <= char <= "9":
            result.append(str((int(char) + 1) % 10))
        elif "A" <= char <= "Z":
            result.append(chr((ord(char) - ord("A") + 1) % 26 + ord("A")))
        elif "a" <= char <= "z":
            result.append(chr((ord(char) - ord("a") + 1) % 26 + ord("a")))
        else:
            result.append(char)
    substituted = "".join(result)
    if substituted == original:
        substituted = ("X" + original[1:]) if len(original) else original
    return substituted

def _modify(ds, attack):
    if attack == "patient_id":
        ds.PatientID = "ALTERED-" + str(getattr(ds, "PatientID", "UNKNOWN"))
        return "PatientID"
    if attack == "study_description":
        ds.StudyDescription = "UNAUTHORIZED MODIFICATION"
        return "StudyDescription"
    if attack in {"protocol_name_same_length", "manufacturer_same_length"}:
        field = "ProtocolName" if attack == "protocol_name_same_length" else "Manufacturer"
        current = str(getattr(ds, field, ""))
        if not current:
            return None
        setattr(ds, field, _same_length_substitution(current))
        return field
    if attack == "pixel_spacing":
        spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        try:
            ds.PixelSpacing = [str(float(spacing[0]) * 1.7), str(float(spacing[1]) * 1.7)]
        except Exception:
            ds.PixelSpacing = ["1.7", "1.7"]
        return "PixelSpacing"
    if attack == "study_uid_inconsistency":
        ds.StudyInstanceUID = "2.25.999999999999999999999999999999999999"
        return "StudyInstanceUID"
    if attack == "remove_protocol":
        if "ProtocolName" not in ds:
            return None
        del ds.ProtocolName
        return "ProtocolName"
    if attack == "modality":
        ds.Modality = "MR"
        return "Modality"
    raise ValueError(f"Unknown attack: {attack}")

def create_attack(input_dir, output_dir, attack, attack_fraction=0.20, seed=42):
    if attack not in ATTACKS:
        raise ValueError(f"attack must be one of {ATTACKS}")
    if not 0 < attack_fraction <= 1:
        raise ValueError("attack_fraction must be in (0, 1].")
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh path: {destination}")
    destination.mkdir(parents=True)
    files = find_dicom_files(source)
    if not files:
        raise ValueError(f"No DICOM files found in {source}")
    rng = random.Random(seed)
    by_study = {}
    for p in files:
        ds = read_dataset(p, stop_before_pixels=True)
        study = str(getattr(ds, "StudyInstanceUID", "")).strip() or f"MISSING:{p.parent}"
        by_study.setdefault(study, []).append(p)
    study_ids = sorted(by_study)
    count = max(1, round(len(study_ids) * attack_fraction))
    selected_studies = set(rng.sample(study_ids, min(count, len(study_ids))))
    selected = {p for study in selected_studies for p in by_study[study]}
    if attack == "study_uid_inconsistency":
        # Change one instance per series in selected studies to create a genuine
        # within-series UID disagreement while keeping the attack unit at study level.
        by_series = {}
        for study in selected_studies:
            for p in by_study[study]:
                ds = read_dataset(p, stop_before_pixels=True)
                series = str(getattr(ds, "SeriesInstanceUID", "MISSING_SERIES"))
                by_series.setdefault((study, series), []).append(p)
        selected = {sorted(group)[0] for group in by_series.values() if group}

    log = {"attack": attack, "attack_fraction_requested": attack_fraction,
           "sampling_unit": "StudyInstanceUID", "selected_study_count": len(selected_studies),
           "total_study_count": len(study_ids), "seed": seed, "files": {}, "modified_file_count": 0,
           "selected_study_ids_sha256": sorted(hashlib.sha256(x.encode()).hexdigest()[:16] for x in selected_studies),
           "note": "Only files actually modified are labelled positive; attacks are selected by study, not slice."}
    for path in files:
        key = relative_key(path, source)
        target = destination / key
        if path in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)  # never edit a hard link to the clean source
            ds = read_dataset(target)
            before_value = None
            if attack in {"protocol_name_same_length", "manufacturer_same_length"}:
                field = "ProtocolName" if attack == "protocol_name_same_length" else "Manufacturer"
                before_value = str(getattr(ds, field, ""))
            changed_field = _modify(ds, attack)
            if changed_field:
                ds.save_as(str(target), enforce_file_format=True)
                record = {"label": 1, "changed_fields": [changed_field]}
                if before_value is not None:
                    after_value = str(getattr(ds, changed_field, ""))
                    record["length_preserved"] = len(before_value) == len(after_value)
                    record["original_length"] = len(before_value)
                    record["modified_length"] = len(after_value)
                log["files"][key] = record
                log["modified_file_count"] += 1
            else:
                target.unlink(missing_ok=True)
                link_or_copy(path, target)
                log["files"][key] = {"label": 0, "changed_fields": []}
        else:
            link_or_copy(path, target)
            log["files"][key] = {"label": 0, "changed_fields": []}
    (destination / "attack_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log
