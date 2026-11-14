"""Cross-check DICOM pixel payloads against metadata describing their layout."""
from __future__ import annotations

import math

import numpy as np


def check_pixel_metadata_consistency(ds):
    """Return deterministic header/payload consistency findings for one object.

    The pixel array is decoded locally; no clean counterpart or reference hash
    is required. Compressed transfer syntaxes skip the raw-length calculation,
    but the decoded shape and value representation are still checked.
    """
    findings = []
    try:
        rows, columns = int(ds.Rows), int(ds.Columns)
        samples = int(getattr(ds, "SamplesPerPixel", 1))
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        bits_allocated = int(ds.BitsAllocated)
        bits_stored = int(getattr(ds, "BitsStored", bits_allocated))
        pixel_representation = int(getattr(ds, "PixelRepresentation", 0))
    except (AttributeError, TypeError, ValueError):
        return ["missing_or_invalid_pixel_layout_metadata"]

    if min(rows, columns, samples, frames, bits_allocated, bits_stored) <= 0:
        findings.append("nonpositive_pixel_layout_value")
    if bits_stored > bits_allocated:
        findings.append("bits_stored_exceeds_bits_allocated")
    if pixel_representation not in (0, 1):
        findings.append("invalid_pixel_representation")
    if samples > 1:
        try:
            planar = int(ds.PlanarConfiguration)
            if planar not in (0, 1):
                findings.append("invalid_planar_configuration")
        except (AttributeError, TypeError, ValueError):
            findings.append("missing_planar_configuration_for_multisample_pixels")

    pixel_data = getattr(ds, "PixelData", None)
    if pixel_data is None:
        findings.append("missing_pixel_data")
    else:
        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        compressed = False
        if transfer_syntax is not None:
            try:
                compressed = bool(transfer_syntax.is_compressed)
            except Exception:
                compressed = False
        if not compressed and rows > 0 and columns > 0 and samples > 0 and frames > 0 and bits_allocated > 0:
            total_bits = rows * columns * samples * frames * bits_allocated
            expected_bytes = math.ceil(total_bits / 8)
            allowed_lengths = {expected_bytes, expected_bytes + (expected_bytes % 2)}
            if bits_allocated != 1 and bits_allocated % 8:
                findings.append("bits_allocated_not_byte_aligned")
            elif len(pixel_data) not in allowed_lengths:
                findings.append("pixel_data_length_inconsistent_with_declared_layout")

    try:
        array = np.asarray(ds.pixel_array)
    except Exception as exc:
        findings.append(f"pixel_decode_failed:{type(exc).__name__}")
        return findings

    if array.ndim == 0:
        findings.append("decoded_pixel_array_has_invalid_rank")
        return findings
    if samples == 1:
        expected_shapes = {(rows, columns)} if frames == 1 else {(frames, rows, columns)}
    else:
        expected_shapes = {(rows, columns, samples)} if frames == 1 else {(frames, rows, columns, samples)}
    if tuple(array.shape) not in expected_shapes:
        findings.append("decoded_pixel_shape_inconsistent_with_header")

    if bits_stored > 0 and bits_stored <= 63 and array.size:
        if pixel_representation == 0:
            low, high = 0, (1 << bits_stored) - 1
        else:
            low, high = -(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1
        try:
            observed_low, observed_high = int(np.min(array)), int(np.max(array))
            if observed_low < low or observed_high > high:
                findings.append("decoded_values_outside_declared_stored_bit_range")
        except (TypeError, ValueError, OverflowError):
            findings.append("decoded_pixel_values_not_integer_like")

    return sorted(set(findings))
