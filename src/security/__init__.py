"""Cryptographic and privacy-aware DICOM integrity utilities.

This package is additive: the original research baseline format and inference
modules remain unchanged for backward compatibility.
"""

from .manifest import create_signed_manifest, verify_signed_manifest

__all__ = ["create_signed_manifest", "verify_signed_manifest"]
