"""Ed25519 and HMAC helpers backed by the ``cryptography`` package."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def key_id_for_public_key(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "ed25519-" + hashlib.sha256(raw).hexdigest()[:24]


def generate_ed25519_keypair(private_path: str | Path, public_path: str | Path) -> str:
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path.exists() or public_path.exists():
        raise FileExistsError("Refusing to overwrite an existing key file")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    private_bytes = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption())
    public_bytes = public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    _write_secret(private_path, private_bytes)
    public_path.write_bytes(public_bytes)
    return key_id_for_public_key(public)


def _write_secret(path: Path, value: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Expected an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Expected an Ed25519 public key")
    return key


def sign_bytes(private_key: Ed25519PrivateKey, message: bytes) -> bytes:
    return private_key.sign(message)


def verify_bytes(public_key: Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


def add_trusted_signer(registry_path: str | Path, signer_id: str, public_key_path: str | Path) -> None:
    """Explicitly trust a public key via a local, administrator-managed registry."""
    public = load_public_key(public_key_path)
    if key_id_for_public_key(public) != signer_id:
        raise ValueError("Signer identifier does not match this public key")
    path = Path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "signers": {}}
    registry.setdefault("signers", {})[signer_id] = {
        "algorithm": "Ed25519",
        "public_key_pem": public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii"),
        "active": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def trusted_public_key(registry: Mapping, signer_id: str) -> Ed25519PublicKey | None:
    entry = registry.get("signers", {}).get(signer_id)
    if not entry or not entry.get("active", True) or entry.get("algorithm") != "Ed25519":
        return None
    try:
        key = serialization.load_pem_public_key(entry["public_key_pem"].encode("ascii"))
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(key, Ed25519PublicKey) or key_id_for_public_key(key) != signer_id:
        return None
    return key


def revoke_trusted_signer(registry_path: str | Path, signer_id: str) -> None:
    path = Path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if signer_id not in registry.get("signers", {}):
        raise KeyError(f"Unknown signer id: {signer_id}")
    registry["signers"][signer_id]["active"] = False
    path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def generate_hmac_key(path: str | Path, length: int = 32) -> None:
    if length < 32:
        raise ValueError("HMAC-SHA256 keys must contain at least 32 random bytes")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_secret(target, secrets.token_bytes(length))


def hmac_sha256(key: bytes, message: bytes) -> str:
    if len(key) < 32:
        raise ValueError("HMAC-SHA256 keys must contain at least 32 bytes")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_hmac_sha256(key: bytes, message: bytes, expected_hex: str) -> bool:
    try:
        actual = hmac_sha256(key, message)
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected_hex.lower())
