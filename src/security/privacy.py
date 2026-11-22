"""Conservative research-copy de-identification and AES-GCM file storage.

This is a limited transformation helper, not a claim of full DICOM PS3.15
confidentiality-profile conformance or burned-in pixel de-identification.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

import pydicom
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydicom.uid import generate_uid
from pydicom.datadict import keyword_for_tag

from .canonical import raw_file_sha256


_REMOVE_KEYWORDS = (
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "RequestingPhysician", "AccessionNumber",
    "PatientBirthTime", "StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
    "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime",
    "InstanceCreationDate", "InstanceCreationTime", "InstitutionName", "StationName",
    "InstitutionalDepartmentName", "IssuerOfPatientID", "StudyID",
    "PerformedStationAETitle", "ScheduledStationAETitle", "PerformedProcedureStepID",
    "OtherPatientIDsSequence", "ReferringPhysicianIdentificationSequence",
    "RequestingPhysicianIdentificationSequence",
)
_UID_CLASS_KEYWORDS = {"SOPClassUID", "ReferencedSOPClassUID", "MediaStorageSOPClassUID",
                       "TransferSyntaxUID", "ImplementationClassUID"}


def _walk_datasets(ds):
    yield ds
    for element in ds:
        if element.VR == "SQ":
            for item in element.value:
                yield from _walk_datasets(item)


def _pseudonym(key: bytes, value: str, purpose: str) -> str:
    digest = hmac.new(key, (purpose + "\0" + value).encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return "R" + digest


def deidentify_research_copy(source: str | Path, destination: str | Path, pseudonym_key: bytes,
                             *, overwrite: bool = False) -> dict:
    """Write a de-identified copy; never modifies source. Pseudonyms are stable per key."""
    if len(pseudonym_key) < 32:
        raise ValueError("Pseudonymization key must contain at least 32 bytes")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    ds = pydicom.dcmread(str(source), force=False)
    patient_id = str(getattr(ds, "PatientID", ""))
    if patient_id:
        ds.PatientID = _pseudonym(pseudonym_key, patient_id, "patient")
    uid_map = {}
    for item in list(_walk_datasets(ds)):
        item.remove_private_tags()
        for keyword in _REMOVE_KEYWORDS:
            if keyword in item:
                del item[keyword]
        for elem in item:
            if elem.VR != "UI" or keyword_for_tag(elem.tag) in _UID_CLASS_KEYWORDS:
                continue
            old_values = list(elem.value) if isinstance(elem.value, (list, tuple)) else [elem.value]
            new_values = []
            for old in old_values:
                text = str(old)
                if text not in uid_map:
                    entropy = hmac.new(pseudonym_key, ("uid\0" + text).encode("ascii", "ignore"), hashlib.sha256).hexdigest()
                    uid_map[text] = generate_uid(prefix="2.25.", entropy_srcs=[entropy])
                new_values.append(uid_map[text])
            elem.value = new_values if len(new_values) > 1 else new_values[0]
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None:
        file_meta.remove_private_tags()
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None:
        for attr, ref in (("MediaStorageSOPInstanceUID", "SOPInstanceUID"),):
            if getattr(ds, ref, None):
                setattr(file_meta, attr, getattr(ds, ref))
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Limited research pseudonymization; private tags removed"
    ds.LongitudinalTemporalInformationModified = "REMOVED"
    burned_in = str(getattr(ds, "BurnedInAnnotation", "")).upper()
    ds.save_as(str(destination), enforce_file_format=True)
    return {
        "source_sha256": raw_file_sha256(source),
        "output_sha256": raw_file_sha256(destination),
        "removed_private_tags": True,
        "uid_relationships_preserved": True,
        "burned_in_annotation": burned_in or "UNKNOWN",
        "pixel_review_required": burned_in not in {"NO"},
        "warning": "Limited metadata transformation; review DICOM confidentiality requirements and burned-in pixels.",
    }


def generate_aes256_key(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(AESGCM.generate_key(bit_length=256))


def encrypt_file(source: str | Path, destination: str | Path, key: bytes) -> dict:
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")
    if destination.exists():
        raise FileExistsError(destination)
    nonce = secrets.token_bytes(12)
    plaintext = source.read_bytes()
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, b"DICOM-SECURE-FILE-v1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"DSF1" + nonce + ciphertext)
    return {"plaintext_bytes": len(plaintext), "ciphertext_bytes": destination.stat().st_size,
            "source_sha256": hashlib.sha256(plaintext).hexdigest(),
            "encrypted_sha256": raw_file_sha256(destination)}


def decrypt_file(source: str | Path, destination: str | Path, key: bytes, *, overwrite: bool = False) -> None:
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    source, destination = Path(source), Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    blob = source.read_bytes()
    if len(blob) < 32 or blob[:4] != b"DSF1":
        raise ValueError("Unsupported or malformed encrypted-file envelope")
    plaintext = AESGCM(key).decrypt(blob[4:16], blob[16:], b"DICOM-SECURE-FILE-v1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plaintext)
