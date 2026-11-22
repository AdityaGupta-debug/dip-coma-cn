"""Command line interface: ``python -m src.security.cli --help``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit import append_audit_event
from .canonical import canonical_json_bytes, raw_file_sha256
from .manifest import create_signed_manifest, save_signed_manifest, verify_signed_manifest
from .privacy import (decrypt_file, deidentify_research_copy, encrypt_file,
                      generate_aes256_key)
from .signing import (add_trusted_signer, generate_ed25519_keypair,
                      generate_hmac_key, hmac_sha256, revoke_trusted_signer,
                      verify_hmac_sha256)
from .workflow import verify_directory_then_analyze, verify_then_analyze


def _key_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DICOM signed-manifest and AI-assisted integrity prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("keygen", help="create a local Ed25519 signer keypair")
    p.add_argument("--private-key", required=True); p.add_argument("--public-key", required=True)
    p = sub.add_parser("trust-add", help="explicitly add a public key to a trusted registry")
    p.add_argument("--registry", required=True); p.add_argument("--signer-id", required=True); p.add_argument("--public-key", required=True)
    p = sub.add_parser("trust-revoke", help="deactivate a signer in the local trusted registry")
    p.add_argument("--registry", required=True); p.add_argument("--signer-id", required=True)
    p = sub.add_parser("hmac-keygen", help="generate a 256-bit shared HMAC key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("pseudonym-keygen", help="generate a separate 256-bit pseudonymization key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("aes-keygen", help="generate an AES-256-GCM file encryption key")
    p.add_argument("--key", required=True)
    p = sub.add_parser("sign", help="sign hashes for one DICOM file")
    p.add_argument("--dicom", required=True); p.add_argument("--private-key", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--hmac-key", help="optional shared key for an independent HMAC experiment")
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-signer")
    p = sub.add_parser("verify", help="verify trust, manifest signature, and DICOM hashes")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--expected-signer"); p.add_argument("--hmac-key"); p.add_argument("--audit-log", default="reports/security_audit.jsonl")
    p.add_argument("--replay-cache", help="optional stateful manifest-ID cache; protect it against tampering")
    p.add_argument("--actor-id", default="local-verifier")
    p = sub.add_parser("analyze", help="verify first, then run existing metadata/rule detectors")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--if-model"); p.add_argument("--ae-model"); p.add_argument("--pixel-model"); p.add_argument("--unet-checkpoint")
    p.add_argument("--allow-unverified-ml", action="store_true")
    p.add_argument("--output", required=True)
    p = sub.add_parser("analyze-directory", help="verify per-file manifests then run study/series-aware analysis")
    p.add_argument("--dicom-dir", required=True); p.add_argument("--manifest-dir", required=True)
    p.add_argument("--trust-registry", required=True); p.add_argument("--if-model"); p.add_argument("--ae-model")
    p.add_argument("--pixel-model"); p.add_argument("--unet-checkpoint"); p.add_argument("--output", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-verifier")
    p = sub.add_parser("deidentify", help="write a pseudonymized research copy (limited transformation)")
    p.add_argument("--dicom", required=True); p.add_argument("--output", required=True); p.add_argument("--pseudonym-key", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="local-transform")
    p = sub.add_parser("encrypt", help="AES-GCM encrypt a file at rest")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--key", required=True)
    p = sub.add_parser("decrypt", help="AES-GCM decrypt a protected file")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--key", required=True)
    p = sub.add_parser("transform", help="verify, de-identify a copy, and sign the transformed object")
    p.add_argument("--dicom", required=True); p.add_argument("--manifest", required=True); p.add_argument("--trust-registry", required=True)
    p.add_argument("--output", required=True); p.add_argument("--output-manifest", required=True)
    p.add_argument("--pseudonym-key", required=True); p.add_argument("--private-key", required=True)
    p.add_argument("--audit-log", default="reports/security_audit.jsonl"); p.add_argument("--actor-id", default="authorized-transformer")
    args = parser.parse_args(argv)

    if args.command == "keygen":
        print(json.dumps({"signer_id": generate_ed25519_keypair(args.private_key, args.public_key),
                          "private_key_path": str(Path(args.private_key).resolve()), "public_key_path": str(Path(args.public_key).resolve())}, indent=2))
    elif args.command == "trust-add":
        add_trusted_signer(args.registry, args.signer_id, args.public_key); print(f"Trusted signer registered: {args.signer_id}")
    elif args.command == "trust-revoke":
        revoke_trusted_signer(args.registry, args.signer_id); print(f"Signer deactivated: {args.signer_id}")
    elif args.command == "hmac-keygen":
        generate_hmac_key(args.key); print(f"Generated HMAC key: {Path(args.key).resolve()}")
    elif args.command == "pseudonym-keygen":
        generate_hmac_key(args.key); print(f"Generated pseudonymization key: {Path(args.key).resolve()}")
    elif args.command == "aes-keygen":
        generate_aes256_key(args.key); print(f"Generated AES-256 key: {Path(args.key).resolve()}")
    elif args.command == "sign":
        envelope = create_signed_manifest(args.dicom, args.private_key)
        save_signed_manifest(envelope, args.manifest)
        if args.hmac_key:
            tag = hmac_sha256(_key_bytes(args.hmac_key), canonical_json_bytes(envelope["manifest"]))
            Path(str(args.manifest) + ".hmac").write_text(json.dumps({"algorithm": "HMAC-SHA256", "tag": tag}) + "\n", encoding="utf-8")
        append_audit_event(args.audit_log, "sign_manifest", "SIGNED", actor_id=args.actor_id,
                           source_sha256=raw_file_sha256(args.dicom), output_sha256=raw_file_sha256(args.manifest),
                           details={"signer_id": envelope["manifest"]["signer_id"], "hmac_created": bool(args.hmac_key)})
        print(json.dumps({"manifest": args.manifest, "signer_id": envelope["manifest"]["signer_id"],
                          "hmac_sidecar": str(args.manifest) + ".hmac" if args.hmac_key else None}, indent=2))
    elif args.command == "verify":
        envelope = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        registry = json.loads(Path(args.trust_registry).read_text(encoding="utf-8"))
        result = verify_signed_manifest(args.dicom, envelope, registry, expected_signer=args.expected_signer,
                                        replay_cache_path=args.replay_cache)
        hmac_sidecar = Path(str(args.manifest) + ".hmac")
        if args.hmac_key:
            try:
                sidecar = json.loads(hmac_sidecar.read_text(encoding="utf-8"))
                valid = verify_hmac_sha256(_key_bytes(args.hmac_key), canonical_json_bytes(envelope["manifest"]), sidecar["tag"])
                result["hmac"] = {"status": "VALID" if valid else "INVALID"}
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                result["hmac"] = {"status": "MISSING_OR_INVALID"}
        append_audit_event(args.audit_log, "verify_manifest", result["status"], actor_id=args.actor_id,
                           source_sha256=raw_file_sha256(args.dicom), details={"signer_id": envelope.get("manifest", {}).get("signer_id")})
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "VERIFIED" and result.get("hmac", {}).get("status", "VALID") == "VALID" else 2
    elif args.command == "analyze":
        result = verify_then_analyze(args.dicom, args.manifest, args.trust_registry,
                                     if_model_path=args.if_model, autoencoder_model_path=args.ae_model,
                                     pixel_model_path=args.pixel_model, unet_checkpoint_path=args.unet_checkpoint,
                                     run_ml_on_unverified=args.allow_unverified_ml)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "analyze-directory":
        result = verify_directory_then_analyze(args.dicom_dir, args.manifest_dir, args.trust_registry,
                                               if_model_path=args.if_model, autoencoder_model_path=args.ae_model,
                                               pixel_model_path=args.pixel_model,
                                               unet_checkpoint_path=args.unet_checkpoint)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(result, indent=2) + "\n"
        Path(args.output).write_text(serialized, encoding="utf-8")
        status_counts = {}
        for item in result["results"].values():
            status = item["cryptographic"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        append_audit_event(args.audit_log, "verify_directory_and_analyze", "COMPLETED", actor_id=args.actor_id,
                           details={"files_discovered": result["files_discovered"],
                                    "files_verified": result["files_verified"],
                                    "crypto_status_counts": status_counts,
                                    "report_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()})
        print(json.dumps({"files_discovered": result["files_discovered"],
                          "files_verified": result["files_verified"],
                          "files_not_verified": result.get("files_not_verified", 0),
                          "report": str(Path(args.output).resolve()), "crypto_status_counts": status_counts}, indent=2))
    elif args.command == "deidentify":
        report = deidentify_research_copy(args.dicom, args.output, _key_bytes(args.pseudonym_key))
        append_audit_event(args.audit_log, "deidentify_research_copy", "TRANSFORMED", actor_id=args.actor_id,
                           source_sha256=report["source_sha256"], output_sha256=report["output_sha256"],
                           details={"pixel_review_required": report["pixel_review_required"]})
        print(json.dumps(report, indent=2))
    elif args.command == "encrypt":
        print(json.dumps(encrypt_file(args.input, args.output, _key_bytes(args.key)), indent=2))
    elif args.command == "decrypt":
        decrypt_file(args.input, args.output, _key_bytes(args.key)); print(f"Decrypted file: {args.output}")
    elif args.command == "transform":
        registry = json.loads(Path(args.trust_registry).read_text(encoding="utf-8"))
        verification = verify_signed_manifest(args.dicom, args.manifest, registry)
        if verification["status"] != "VERIFIED":
            print(json.dumps({"original_verification": verification, "transformation": "NOT_PERFORMED"}, indent=2))
            return 2
        report = deidentify_research_copy(args.dicom, args.output, _key_bytes(args.pseudonym_key))
        envelope = create_signed_manifest(args.output, args.private_key)
        save_signed_manifest(envelope, args.output_manifest)
        append_audit_event(args.audit_log, "verify_deidentify_resign", "TRANSFORMED_AND_RESIGNED", actor_id=args.actor_id,
                           source_sha256=report["source_sha256"], output_sha256=report["output_sha256"],
                           details={"original_manifest_id": json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("manifest", {}).get("manifest_id"),
                                    "new_manifest_id": envelope["manifest"]["manifest_id"],
                                    "new_signer_id": envelope["manifest"]["signer_id"],
                                    "pixel_review_required": report["pixel_review_required"]})
        print(json.dumps({"original_verification": verification, "transformation": report,
                          "new_manifest": args.output_manifest, "new_signer_id": envelope["manifest"]["signer_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
