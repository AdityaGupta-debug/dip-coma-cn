"""External, signed DICOM integrity manifests and verification outcomes."""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pydicom

from .canonical import PROFILE, canonical_json_bytes, metadata_sha256, pixel_data_sha256, raw_file_sha256
from .signing import (key_id_for_public_key, load_private_key, sign_bytes,
                      trusted_public_key, verify_bytes)

OUTCOMES = {
    "VERIFIED", "INTEGRITY_MISMATCH", "INVALID_SIGNATURE", "UNTRUSTED_SIGNER",
    "MISSING_REFERENCE", "MALFORMED_DICOM", "POLICY_REVIEW_REQUIRED",
}


def create_signed_manifest(dicom_path: str | Path, private_key_path: str | Path,
                           *, include_raw_file_hash: bool = True) -> dict:
    dicom_path = Path(dicom_path)
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False, force=False)
    private_key = load_private_key(private_key_path)
    signer_id = key_id_for_public_key(private_key.public_key())
    content = {
        "schema_version": 1,
        "manifest_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "canonicalization_profile": PROFILE,
        "signature_algorithm": "Ed25519",
        "signer_id": signer_id,
        "dicom": {
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
            "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        },
        "digests": {
            "metadata_sha256": metadata_sha256(ds),
            "pixel_data_sha256": pixel_data_sha256(ds),
            "raw_file_sha256": raw_file_sha256(dicom_path) if include_raw_file_hash else None,
        },
    }
    signature = sign_bytes(private_key, canonical_json_bytes(content))
    return {"manifest": content, "signature": {"algorithm": "Ed25519", "value_base64": base64.b64encode(signature).decode("ascii")}}


def save_signed_manifest(envelope: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(envelope) + b"\n")


def _result(status: str, *, details: dict | None = None) -> dict:
    if status not in OUTCOMES:
        raise ValueError(f"Unknown security outcome: {status}")
    return {"status": status, "details": details or {}}


def verify_signed_manifest(dicom_path: str | Path, envelope_or_path: dict | str | Path,
                           trust_registry: Mapping, *, expected_signer: str | None = None,
                           replay_cache_path: str | Path | None = None) -> dict:
    """Verify trust/signature first, then independently compare each digest scope."""
    try:
        envelope = (json.loads(Path(envelope_or_path).read_text(encoding="utf-8"))
                    if isinstance(envelope_or_path, (str, Path)) else envelope_or_path)
        manifest = envelope["manifest"]
        signature_info = envelope["signature"]
        if manifest.get("schema_version") != 1 or manifest.get("canonicalization_profile") != PROFILE:
            return _result("POLICY_REVIEW_REQUIRED", details={"reason": "unsupported manifest schema or canonicalization profile"})
        if signature_info.get("algorithm") != "Ed25519" or manifest.get("signature_algorithm") != "Ed25519":
            return _result("POLICY_REVIEW_REQUIRED", details={"reason": "unsupported signature algorithm"})
        signature = base64.b64decode(signature_info["value_base64"], validate=True)
        signer_id = manifest["signer_id"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _result("MISSING_REFERENCE", details={"reason": "manifest or signature is missing or malformed"})

    if expected_signer is not None and signer_id != expected_signer:
        return _result("UNTRUSTED_SIGNER", details={"reason": "manifest signer differs from expected sender", "signer_id": signer_id})
    public_key = trusted_public_key(trust_registry, signer_id)
    if public_key is None:
        return _result("UNTRUSTED_SIGNER", details={"reason": "signer is not in the trusted-key registry", "signer_id": signer_id})
    if not verify_bytes(public_key, signature, canonical_json_bytes(manifest)):
        return _result("INVALID_SIGNATURE", details={"reason": "manifest signature does not verify"})

    manifest_id = str(manifest.get("manifest_id", ""))
    if replay_cache_path is not None:
        cache_path = Path(replay_cache_path)
        if cache_path.exists():
            try:
                seen = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return _result("POLICY_REVIEW_REQUIRED", details={"reason": "replay cache is unreadable"})
            if manifest_id and manifest_id in seen.get("accepted_manifest_ids", []):
                return _result("POLICY_REVIEW_REQUIRED", details={"reason": "replayed manifest identifier", "manifest_id": manifest_id})

    try:
        dicom_path = Path(dicom_path)
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False, force=False)
        if not manifest_id or not all(manifest.get("dicom", {}).get(key) for key in
                                      ("sop_instance_uid", "study_instance_uid", "series_instance_uid")):
            return _result("MISSING_REFERENCE", details={"reason": "manifest lacks required identifiers"})
        if not all(isinstance(manifest.get("digests", {}).get(key), str) and len(manifest["digests"][key]) == 64
                   for key in ("metadata_sha256", "pixel_data_sha256")):
            return _result("MISSING_REFERENCE", details={"reason": "manifest lacks required content digests"})
    except Exception as exc:
        return _result("MALFORMED_DICOM", details={"reason": type(exc).__name__})
    if not isinstance(manifest.get("dicom"), dict) or not isinstance(manifest.get("digests"), dict):
        return _result("POLICY_REVIEW_REQUIRED", details={"reason": "required manifest sections are absent"})
    actual_ids = {
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
    }
    try:
        actual = {"metadata_sha256": metadata_sha256(ds), "pixel_data_sha256": pixel_data_sha256(ds)}
    except (ValueError, TypeError) as exc:
        return _result("MALFORMED_DICOM", details={"reason": str(exc)})
    expected_digests = manifest["digests"]
    if expected_digests.get("raw_file_sha256") is not None:
        actual["raw_file_sha256"] = raw_file_sha256(dicom_path)
    mismatches = {scope: {"expected": expected_digests.get(scope), "actual": value}
                  for scope, value in actual.items() if expected_digests.get(scope) != value}
    id_mismatches = {key: {"expected": manifest["dicom"].get(key), "actual": value}
                     for key, value in actual_ids.items() if manifest["dicom"].get(key) != value}
    if mismatches or id_mismatches:
        return _result("INTEGRITY_MISMATCH", details={"digest_mismatches": mismatches, "identifier_mismatches": id_mismatches})
    if replay_cache_path is not None:
        cache_path = Path(replay_cache_path)
        seen = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"schema_version": 1, "accepted_manifest_ids": []}
        seen.setdefault("accepted_manifest_ids", []).append(manifest_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(seen, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _result("VERIFIED", details={"signer_id": signer_id, "verified_scopes": list(actual), "signature_valid": True})
