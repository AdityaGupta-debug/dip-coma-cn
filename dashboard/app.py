"""Preloaded interactive research demo for CPAISD DICOM metadata and pixels."""
from __future__ import annotations

import json
import random
import subprocess
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import _same_length_substitution
from src.detection import detect_directory
from src.dicom_io import find_dicom_files, read_dataset, relative_key
from src.integrity import create_baseline, metadata_sha256, pixel_sha256
from src.no_reference_pixel import (
    _mask_from_patch_scores,
    _normalized_image,
    _patch_feature_grid,
    _residual_map,
)
from src.pixel_integrity import _edit_pixels
from src.pixel_metadata_consistency import check_pixel_metadata_consistency
from src.semantic_consistency import validate_semantic_consistency
from src.security.audit import append_audit_event
from src.security.canonical import canonical_json_bytes, raw_file_sha256
from src.security.manifest import create_signed_manifest, save_signed_manifest, verify_signed_manifest
from src.security.privacy import (decrypt_file, deidentify_research_copy, encrypt_file,
                                  generate_aes256_key)
from src.security.signing import (add_trusted_signer, generate_ed25519_keypair,
                                  generate_hmac_key, hmac_sha256, key_id_for_public_key,
                                  load_public_key, revoke_trusted_signer, verify_hmac_sha256)
from src.security.workflow import verify_directory_then_analyze, verify_then_analyze


DATASET_DIR = PROJECT_ROOT / "data" / "cpaisd" / "test"
IF_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "isolation_forest.joblib"
AE_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "metadata_autoencoder.joblib"
PIXEL_MODEL = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "no_reference_pixel_model.joblib"
PIXEL_CALIBRATION = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "calibration_and_clean_summary.json"
SECURITY_HOME = Path.home() / ".dicom_security_demo"
SECURITY_KEYS = SECURITY_HOME / "keys"
SECURITY_TRUST_REGISTRY = SECURITY_HOME / "trusted-senders.json"
SECURITY_REPLAY_CACHE = SECURITY_HOME / "accepted-manifests.json"
SECURITY_UI_OUTPUT = PROJECT_ROOT / "reports" / "security_ui"
SECURITY_AUDIT_LOG = SECURITY_UI_OUTPUT / "security_audit.jsonl"
SECURITY_EXPERIMENT_OUTPUT = PROJECT_ROOT / "reports" / "crypto_security_full"
_completed_unet = (PROJECT_ROOT / "experiment_cpaisd_supervised_localization_512_30ep_complete"
                  / "unet_localizer.pt")
_unet_candidates = sorted(
    PROJECT_ROOT.glob("experiment_cpaisd_supervised_localization*/unet_localizer.pt"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
_preferred_unet = PROJECT_ROOT / "experiment_cpaisd_supervised_localization" / "unet_localizer.pt"
UNET_MODEL = (_completed_unet if _completed_unet.is_file()
              else (_unet_candidates[0] if _unet_candidates else _preferred_unet))
UNET_RESULTS_DIR = UNET_MODEL.parent

st.set_page_config(page_title="DICOM Security Demo Lab", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner="Loading the preloaded CPAISD test fold...")
def get_dicom_paths(root_string: str) -> tuple[str, ...]:
    return tuple(str(path) for path in find_dicom_files(root_string))


@st.cache_resource(show_spinner="Indexing study/series metadata for consistency checks...")
def get_semantic_study_index(root_string: str):
    grouped = {}
    root = Path(root_string)
    for path_string in get_dicom_paths(root_string):
        path = Path(path_string)
        dataset = read_dataset(path, stop_before_pixels=True)
        study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
        if study_uid:
            grouped.setdefault(study_uid, []).append((relative_key(path, root), dataset))
    return grouped


@st.cache_resource(show_spinner=False)
def load_pixel_bundle(path_string: str):
    return joblib.load(path_string)


@st.cache_resource(show_spinner="Loading the supervised pixel-localization model...")
def load_unet_bundle(path_string: str):
    import torch
    from src.supervised_localization import build_localizer, calibrate_clean_image_threshold

    checkpoint = torch.load(path_string, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_localizer(config.get("architecture", "unet"),
                            base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if config.get("image_score_type") != "maximum_pre_sigmoid_logit":
        validation_paths = find_dicom_files(PROJECT_ROOT / "data" / "cpaisd" / "val")
        if validation_paths:
            config["image_threshold"] = calibrate_clean_image_threshold(
                model, validation_paths, int(config["image_size"]), quantile=99.0,
            )
            config["image_score_type"] = "maximum_pre_sigmoid_logit"
            config["image_quantile"] = 99.0
    return model, config


def unet_probability_map(dataset, model, config):
    import torch
    import torch.nn.functional as functional

    normalized = _normalized_image(dataset)
    image_size = int(config["image_size"])
    tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32))[None, None]
    tensor = functional.interpolate(tensor, size=(image_size, image_size),
                                    mode="bilinear", align_corners=False)
    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        image_score = float(logits.flatten(start_dim=1).amax(dim=1)[0].item())
    if config.get("image_score_type") != "maximum_pre_sigmoid_logit":
        image_score = float(probability.max())
    pixel_mask = probability >= float(config["pixel_threshold"])
    return normalized, probability, pixel_mask, image_score


def apply_copy_move_edit(path: Path, seed: int = 42) -> tuple[int, np.ndarray]:
    """Copy a stored-pixel patch to a separate location in a temporary DICOM copy."""
    dataset = read_dataset(path, stop_before_pixels=False)
    array = np.asarray(dataset.pixel_array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Copy-move demo needs a single-frame 2-D integer CT, got {array.shape} {array.dtype}.")
    rows, columns = array.shape
    size = max(8, round(min(rows, columns) * 0.10))
    if rows < size * 2 or columns < size * 2:
        raise ValueError("This slice is too small for two non-overlapping copy-move regions.")
    rng = random.Random(seed)
    destination = None
    patch = None
    changed_mask = None
    for _ in range(200):
        source_top = rng.randint(0, rows - size)
        source_left = rng.randint(0, columns - size)
        dest_top = rng.randint(0, rows - size)
        dest_left = rng.randint(0, columns - size)
        overlap = not (dest_top + size <= source_top or source_top + size <= dest_top
                       or dest_left + size <= source_left or source_left + size <= dest_left)
        if not overlap:
            candidate_patch = array[source_top:source_top + size, source_left:source_left + size].copy()
            candidate_target = array[dest_top:dest_top + size, dest_left:dest_left + size]
            candidate_mask = candidate_target != candidate_patch
            if np.any(candidate_mask):
                destination = (dest_top, dest_left)
                patch = candidate_patch
                changed_mask = candidate_mask
                break
    if destination is None:
        raise ValueError("Could not find a non-identical, non-overlapping copy-move patch in this slice.")
    edited = array.copy()
    dest_top, dest_left = destination
    target = edited[dest_top:dest_top + size, dest_left:dest_left + size]
    changed = int(np.count_nonzero(changed_mask))
    target[:] = patch
    byte_order = "<" if getattr(dataset, "is_little_endian", True) else ">"
    dataset.PixelData = edited.astype(array.dtype.newbyteorder(byte_order), copy=False).tobytes(order="C")
    dataset.save_as(str(path), enforce_file_format=True)
    mask = np.zeros(array.shape, dtype=bool)
    mask[dest_top:dest_top + size, dest_left:dest_left + size] = changed_mask
    return changed, mask


def apply_metadata_edit(dataset, edit_name: str) -> tuple[str, str, str]:
    if edit_name == "Modality to MR":
        old = str(getattr(dataset, "Modality", ""))
        dataset.Modality = "MR"
        return "Modality", old, "MR"
    if edit_name == "PixelSpacing x 1.7":
        old_values = list(getattr(dataset, "PixelSpacing", []))
        if len(old_values) != 2:
            raise ValueError("This file does not have a two-value PixelSpacing field.")
        new_values = [str(float(value) * 1.7) for value in old_values]
        dataset.PixelSpacing = new_values
        return "PixelSpacing", str(old_values), str(new_values)
    if edit_name == "ProtocolName same-length substitution":
        old = str(getattr(dataset, "ProtocolName", ""))
        if not old:
            raise ValueError("This file has no ProtocolName value to edit.")
        new = _same_length_substitution(old)
        dataset.ProtocolName = new
        return "ProtocolName", old, new
    if edit_name == "Manufacturer same-length substitution":
        old = str(getattr(dataset, "Manufacturer", ""))
        if not old:
            raise ValueError("This file has no Manufacturer value to edit.")
        new = _same_length_substitution(old)
        dataset.Manufacturer = new
        return "Manufacturer", old, new
    if edit_name == "Remove ProtocolName":
        if "ProtocolName" not in dataset:
            raise ValueError("This file has no ProtocolName field to remove.")
        old = str(dataset.ProtocolName)
        del dataset.ProtocolName
        return "ProtocolName", old, "<removed>"
    raise ValueError(f"Unsupported metadata edit: {edit_name}")


def detector_rows(clean_row, edited_row, pixel_hash_changed: bool,
                  clean_pixel_flag: int, edited_pixel_flag: int,
                  pixel_model_rows: list[dict], clean_layout: list[str], edited_layout: list[str],
                  clean_semantic: list[str], edited_semantic: list[str]):
    rows = []

    def add(name, clean_alert, edited_alert, detail):
        rows.append({
            "Check": name,
            "Clean scan": "ALERT" if clean_alert else "Clear",
            "Edited scan": "ALERT" if edited_alert else "Clear",
            "New alert": "Yes" if edited_alert and not clean_alert else "No",
            "What it checks": detail,
        })

    add("Metadata SHA-256", int(clean_row.sha256_flag), int(edited_row.sha256_flag),
        "Did a covered DICOM header value change from the clean reference?")
    add("PixelData SHA-256", clean_pixel_flag, int(pixel_hash_changed),
        "Did the stored pixel payload change from the clean reference?")
    add("Structural rules", int(clean_row.rule_flag), int(edited_row.rule_flag),
        str(edited_row.rule_findings) or "Required fields and basic CT constraints")
    add("Semantic study/series consistency", int(bool(clean_semantic)), int(bool(edited_semantic)),
        "; ".join(edited_semantic) if edited_semantic else
        "Do patient/study fields and stable acquisition fields agree across peer slices?")
    add("Isolation Forest", int(clean_row.isolation_forest_flag), int(edited_row.isolation_forest_flag),
        f"Score {edited_row.isolation_forest_score:.5f}; threshold {edited_row.isolation_forest_threshold:.5f}")
    add("Metadata autoencoder", int(clean_row.autoencoder_flag), int(edited_row.autoencoder_flag),
        f"Reconstruction error {edited_row.autoencoder_reconstruction_error:.6g}; threshold {edited_row.autoencoder_threshold:.6g}")
    for item in pixel_model_rows:
        add(item["name"], item["clean_flag"], item["edited_flag"], item["detail"])
    add("Pixel/header consistency", int(bool(clean_layout)), int(bool(edited_layout)),
        "; ".join(edited_layout) if edited_layout else "Do pixel layout and header dimensions agree?")
    clean_or = int(bool(clean_row.sha256_flag or clean_pixel_flag or clean_row.rule_flag
                         or clean_semantic
                         or clean_row.isolation_forest_flag or clean_row.autoencoder_flag
                         or any(x["clean_flag"] for x in pixel_model_rows) or clean_layout))
    edited_or = int(bool(edited_row.sha256_flag or pixel_hash_changed or edited_row.rule_flag
                         or edited_semantic
                         or edited_row.isolation_forest_flag or edited_row.autoencoder_flag
                         or any(x["edited_flag"] for x in pixel_model_rows) or edited_layout))
    add("All-checks OR summary", clean_or, edited_or,
        "Any metadata, pixel, anomaly, structural, or layout check raised an alert")
    return pd.DataFrame(rows)


def run_live_demo(source_file: Path, edit_mode: str, metadata_edit: str | None,
                  pixel_edit: str | None):
    relative = relative_key(source_file, DATASET_DIR)
    with tempfile.TemporaryDirectory(prefix="dicom_security_live_demo_") as temp_name:
        temp_root = Path(temp_name)
        clean_root, edited_root = temp_root / "clean", temp_root / "edited"
        clean_file, edited_file = clean_root / relative, edited_root / relative
        clean_file.parent.mkdir(parents=True, exist_ok=True)
        edited_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, clean_file)
        shutil.copy2(source_file, edited_file)

        edit_description = "No edit selected"
        if edit_mode in {"Metadata only", "Metadata + pixels"}:
            dataset = read_dataset(edited_file, stop_before_pixels=False)
            tag, old_value, new_value = apply_metadata_edit(dataset, metadata_edit or "")
            dataset.save_as(str(edited_file), enforce_file_format=True)
            edit_description = f"{tag}: {old_value} -> {new_value}"

        changed_pixel_count = 0
        synthetic_pixel_mask = None
        if edit_mode in {"Pixels only", "Metadata + pixels"}:
            if pixel_edit == "Copy-move patch (withheld U-Net family)":
                changed_pixel_count, synthetic_pixel_mask = apply_copy_move_edit(edited_file, seed=42)
                pixel_description = "10% patch copied to a non-overlapping location (withheld U-Net family)"
            else:
                attack_name = "localized_patch" if pixel_edit == "Localized patch (5x5, +32)" else "low_amplitude"
                changed_pixel_count, synthetic_pixel_mask = _edit_pixels(
                    edited_file, attack_name, random.Random(42),
                    patch_size=5, patch_delta=32, sparse_fraction=0.01, sparse_delta=1,
                )
                pixel_description = ("5x5 central patch, +32 stored values" if attack_name == "localized_patch"
                                     else "1% seeded sparse edit, +1 stored value")
            edit_description = (edit_description + " | " if edit_description != "No edit selected" else "") + pixel_description

        baseline_path = temp_root / "demo_clean_reference.json"
        create_baseline(clean_root, baseline_path, include_pixel_hash=True)
        clean_detection = detect_directory(clean_root, baseline_path, IF_MODEL, autoencoder_model_path=AE_MODEL).iloc[0]
        edited_detection = detect_directory(edited_root, baseline_path, IF_MODEL, autoencoder_model_path=AE_MODEL).iloc[0]
        clean_ds = read_dataset(clean_file, stop_before_pixels=False)
        edited_ds = read_dataset(edited_file, stop_before_pixels=False)
        clean_study_uid = str(getattr(clean_ds, "StudyInstanceUID", "")).strip()
        study_peers = get_semantic_study_index(str(DATASET_DIR)).get(clean_study_uid, [])
        clean_semantic_items = [(key, peer) for key, peer in study_peers if key != relative]
        clean_semantic_items.append((relative, clean_ds))
        edited_semantic_items = [(key, peer) for key, peer in study_peers if key != relative]
        edited_semantic_items.append((relative, edited_ds))
        clean_semantic = validate_semantic_consistency(clean_semantic_items).get(relative, [])
        edited_semantic = validate_semantic_consistency(edited_semantic_items).get(relative, [])
        clean_pixels = np.asarray(clean_ds.pixel_array)
        edited_pixels = np.asarray(edited_ds.pixel_array)
        difference = np.abs(edited_pixels.astype(np.float32) - clean_pixels.astype(np.float32))
        pixel_hash_changed = pixel_sha256(clean_ds) != pixel_sha256(edited_ds)
        clean_layout = check_pixel_metadata_consistency(clean_ds)
        edited_layout = check_pixel_metadata_consistency(edited_ds)

        bundle = load_pixel_bundle(str(PIXEL_MODEL))
        calibration = json.loads(PIXEL_CALIBRATION.read_text(encoding="utf-8"))
        image_results = []
        image_masks = []
        for label, ds in (("Clean", clean_ds), ("Edited", edited_ds)):
            normalized = _normalized_image(ds)
            features, grid_shape, image_shape = _patch_feature_grid(normalized, bundle.get("patch_size", 16))
            patch_scores = bundle["model"].score_samples(features)
            patch_flag = int(float(np.min(patch_scores)) < float(bundle["file_threshold"]))
            patch_mask = _mask_from_patch_scores(
                patch_scores, grid_shape, image_shape, float(bundle["patch_threshold"])
            )
            residual = _residual_map(normalized)
            residual_score = float(np.percentile(residual, 99.9))
            residual_flag = int(residual_score > float(calibration["median_residual_image_threshold"]))
            image_results.append({
                "label": label, "patch_flag": patch_flag,
                "patch_score": float(np.min(patch_scores)),
                "residual_flag": residual_flag, "residual_score": residual_score,
            })
            image_masks.append((normalized, patch_mask))

        no_reference_rows = [
            {"name": "No-reference patch Isolation Forest",
             "clean_flag": image_results[0]["patch_flag"],
             "edited_flag": image_results[1]["patch_flag"],
             "detail": f"Edited minimum patch score {image_results[1]['patch_score']:.5f}; image threshold {bundle['file_threshold']:.5f}. Heatmap uses the 16x16-patch model."},
            {"name": "Median-residual baseline",
             "clean_flag": image_results[0]["residual_flag"],
             "edited_flag": image_results[1]["residual_flag"],
             "detail": f"Edited 99.9th-percentile residual {image_results[1]['residual_score']:.5f}; threshold {calibration['median_residual_image_threshold']:.5f}."},
        ]
        unet_outputs = None
        if UNET_MODEL.is_file():
            unet_model, unet_config = load_unet_bundle(str(UNET_MODEL))
            clean_norm_unet, clean_probability, clean_unet_mask, clean_score = unet_probability_map(
                clean_ds, unet_model, unet_config
            )
            edited_norm_unet, edited_probability, edited_unet_mask, edited_score = unet_probability_map(
                edited_ds, unet_model, unet_config
            )
            image_threshold = float(unet_config["image_threshold"])
            no_reference_rows.append({
                "name": "Supervised U-Net pixel localizer",
                "clean_flag": int(clean_score >= image_threshold),
                "edited_flag": int(edited_score >= image_threshold),
                "detail": (f"Edited image score {edited_score:.4f} ({unet_config.get('image_score_type', 'legacy max probability')}); threshold "
                           f"{image_threshold:.4f}; predicted mask pixels "
                           f"{int(np.count_nonzero(edited_unet_mask)):,} at "
                           f"{int(unet_config['image_size'])}x{int(unet_config['image_size'])}."),
            })
            unet_outputs = {
                "clean_image": clean_norm_unet,
                "edited_image": edited_norm_unet,
                "clean_probability": clean_probability,
                "edited_probability": edited_probability,
                "clean_mask": clean_unet_mask,
                "edited_mask": edited_unet_mask,
                "synthetic_mask": synthetic_pixel_mask,
                "pixel_threshold": float(unet_config["pixel_threshold"]),
                "image_threshold": image_threshold,
                "clean_score": clean_score,
                "edited_score": edited_score,
            }
        rows = detector_rows(
            clean_detection, edited_detection, pixel_hash_changed,
            int(clean_detection.pixel_integrity_flag or 0), int(edited_detection.pixel_integrity_flag or 0),
            no_reference_rows, clean_layout, edited_layout, clean_semantic, edited_semantic,
        )
        clean_norm, clean_mask = image_masks[0]
        edited_norm, edited_mask = image_masks[1]
        return {
            "edit_description": edit_description,
            "changed_pixel_count": changed_pixel_count,
            "difference": difference,
            "clean_pixels": clean_pixels,
            "edited_pixels": edited_pixels,
            "synthetic_mask": synthetic_pixel_mask,
            "clean_norm": clean_norm,
            "edited_norm": edited_norm,
            "clean_mask": clean_mask,
            "edited_mask": edited_mask,
            "detectors": rows,
            "clean_pixel_hash": pixel_sha256(clean_ds),
            "edited_pixel_hash": pixel_sha256(edited_ds),
            "clean_metadata_hash": metadata_sha256(clean_ds),
            "edited_metadata_hash": metadata_sha256(edited_ds),
            "clean_layout": clean_layout,
            "edited_layout": edited_layout,
            "clean_semantic": clean_semantic,
            "edited_semantic": edited_semantic,
            "unet": unet_outputs,
        }


def show_saved_results():
    results = [
        ("Metadata experiment", PROJECT_ROOT / "experiment_cpaisd" / "reports" / "macro_average_metrics.csv"),
        ("Metadata clean-test alarms", PROJECT_ROOT / "experiment_cpaisd" / "reports" / "clean_test_false_positive_rates.csv"),
        ("Pixel integrity and masks", PROJECT_ROOT / "experiment_cpaisd_pixel_masks" / "pixel_metrics.csv"),
        ("No-reference pixel detector", PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "no_reference_pixel_metrics.csv"),
        ("Header-to-pixel consistency", PROJECT_ROOT / "experiment_cpaisd_cross_layer" / "cross_layer_summary.csv"),
        ("Semantic consistency study metrics", PROJECT_ROOT / "experiment_cpaisd_semantic_consistency" / "semantic_consistency_summary.csv"),
        ("Semantic consistency per-study findings", PROJECT_ROOT / "experiment_cpaisd_semantic_consistency" / "semantic_consistency_per_study.csv"),
        ("Adaptive feature evasion", PROJECT_ROOT / "experiment_cpaisd_adaptive_evasion" / "adaptive_evasion_summary.csv"),
        ("Benign changes versus attacks", PROJECT_ROOT / "experiment_cpaisd_benign_attack_v3" / "benign_vs_attack_summary.csv"),
        ("Combined metadata + pixel checks", PROJECT_ROOT / "experiment_cpaisd_combined_independent" / "combined_metrics.csv"),
        ("Supervised U-Net localization metrics", UNET_RESULTS_DIR / "pixel_localization_summary.csv"),
        ("Supervised U-Net study-level metrics", UNET_RESULTS_DIR / "pixel_localization_study_metrics.csv"),
        ("Supervised U-Net training history", UNET_RESULTS_DIR / "training_history.csv"),
        ("Signed-manifest security scenarios", SECURITY_EXPERIMENT_OUTPUT / "security_scenarios.csv"),
        ("Signed-manifest operation timings", SECURITY_EXPERIMENT_OUTPUT / "security_performance.csv"),
    ]
    for title, path in results:
        with st.expander(title, expanded=False):
            if path.is_file():
                try:
                    st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
                    st.caption(str(path.relative_to(PROJECT_ROOT)))
                except Exception as exc:
                    st.warning(f"Could not read this saved result: {exc}")
            else:
                st.info(f"No saved result file was found at {path.relative_to(PROJECT_ROOT)}")
    figure_dir = UNET_RESULTS_DIR / "figures"
    if figure_dir.is_dir():
        with st.expander("Supervised U-Net localization examples", expanded=False):
            figure_columns = st.columns(3)
            figure_specs = [
                ("localized_patch.png", "Localized patch"),
                ("sparse_low_amplitude.png", "Sparse low-amplitude edit"),
                ("copy_move_unseen.png", "Withheld copy-move edit"),
            ]
            for column, (filename, caption) in zip(figure_columns, figure_specs):
                figure_path = figure_dir / filename
                if figure_path.is_file():
                    column.image(str(figure_path), caption=caption, use_container_width=True)
            st.caption("Each figure shows a clean slice, synthetic edit, generated mask, and U-Net prediction at the resized model grid.")


def _security_registry() -> dict:
    if not SECURITY_TRUST_REGISTRY.is_file():
        return {"schema_version": 1, "signers": {}}
    return json.loads(SECURITY_TRUST_REGISTRY.read_text(encoding="utf-8"))


def _security_key_files(pattern: str) -> list[Path]:
    return sorted(SECURITY_KEYS.glob(pattern)) if SECURITY_KEYS.is_dir() else []


def _security_result_summary(result: dict) -> dict:
    """Show verification status without exposing UID or digest values in the UI."""
    details = result.get("details", {})
    safe_details = {}
    for key in ("reason", "signer_id", "verified_scopes", "signature_valid"):
        if key in details:
            safe_details[key] = details[key]
    if "digest_mismatches" in details:
        safe_details["digest_mismatch_scopes"] = sorted(details["digest_mismatches"])
    if "identifier_mismatches" in details:
        safe_details["identifier_mismatch_fields"] = sorted(details["identifier_mismatches"])
    summary = {"status": result.get("status"), "details": safe_details}
    if "hmac" in result:
        summary["hmac"] = {"status": result["hmac"].get("status")}
    return summary


def _show_secure_exchange(paths: list[Path]) -> None:
    st.subheader("Authenticated DICOM exchange")
    st.write("Create a signed manifest, check sender trust and content integrity, and run the existing analysis only after verification.")
    st.warning("This is a local file-level research demo. It does not send data over TCP/TLS or open a DICOM network association. Keys are stored on the Streamlit app host.")

    selected_index = st.selectbox(
        "CPAISD test slice for this security workflow", range(len(paths)), index=0,
        format_func=lambda index: f"Test slice {index + 1:03d} of {len(paths)}",
        key="security_exchange_slice",
    )
    dicom_path = paths[selected_index]
    st.caption("The selected original is read-only in these controls. Signed manifests contain linkable DICOM UIDs; handle them as sensitive. Demo outputs are written under `reports/security_ui/`.")

    key_tab, manifest_tab, analysis_tab, batch_tab, privacy_tab, storage_tab, results_tab = st.tabs([
        "Keys and trust", "Sign and verify", "Verify + AI", "Batch workflow",
        "Research-copy transform", "At-rest encryption", "Security experiment",
    ])

    with key_tab:
        st.markdown("#### Ed25519 signer keys")
        st.caption(f"Keys are kept outside the repository in `{SECURITY_HOME}`. Share public keys through a trusted out-of-band process; do not upload a received key and trust it automatically.")
        key_label = st.text_input("Key name", value="research-sender", key="security_key_label")
        cleaned_label = "".join(char for char in key_label.strip() if char.isalnum() or char in "-_")
        if st.button("Generate Ed25519 keypair", key="security_generate_signer"):
            if not cleaned_label:
                st.error("Choose a key name containing letters, numbers, hyphens, or underscores.")
            else:
                private_path = SECURITY_KEYS / f"{cleaned_label}-private.pem"
                public_path = SECURITY_KEYS / f"{cleaned_label}-public.pem"
                try:
                    signer_id = generate_ed25519_keypair(private_path, public_path)
                    st.success(f"Created signer `{signer_id}`. Private key: `{private_path}`")
                except FileExistsError:
                    st.error("That key name already exists. Choose a new name; existing keys are never overwritten.")
                except Exception as exc:
                    st.error(f"Key generation failed: {type(exc).__name__}: {exc}")

        public_files = _security_key_files("*-public.pem")
        st.markdown("#### Local trusted-signer registry")
        if public_files:
            public_choice = st.selectbox("Public key to register", public_files, format_func=lambda path: path.name,
                                         key="security_public_key_choice")
            try:
                signer_id = key_id_for_public_key(load_public_key(public_choice))
                st.code(f"Signer ID: {signer_id}")
                st.download_button(
                    "Download selected public key",
                    public_choice.read_bytes(), file_name=public_choice.name,
                    mime="application/x-pem-file", key="security_public_key_download",
                )
                if st.button("Add public key to trusted registry", key="security_trust_add"):
                    add_trusted_signer(SECURITY_TRUST_REGISTRY, signer_id, public_choice)
                    st.success("Public key added to the local registry.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read this public key: {type(exc).__name__}: {exc}")
        else:
            st.info("Generate a signer keypair to enable trust registration.")

        st.markdown("#### Import and explicitly trust a sender key")
        st.caption("Only trust a public key after confirming its fingerprint with the sender through a separate trusted channel.")
        imported_public = st.file_uploader("Sender Ed25519 public key (.pem/.pub)", type=["pem", "pub", "txt"],
                                           key="security_import_public_key")
        if imported_public is not None:
            try:
                with tempfile.TemporaryDirectory(prefix="dicom_trust_key_") as temp_name:
                    candidate = Path(temp_name) / "sender-public.pem"
                    candidate.write_bytes(imported_public.getvalue())
                    imported_id = key_id_for_public_key(load_public_key(candidate))
                st.code(f"Fingerprint / signer ID: {imported_id}")
                if st.button("Trust this sender key", key="security_trust_imported_key"):
                    SECURITY_KEYS.mkdir(parents=True, exist_ok=True)
                    persisted_key = SECURITY_KEYS / f"{imported_id}-public.pem"
                    if not persisted_key.exists():
                        persisted_key.write_bytes(imported_public.getvalue())
                    add_trusted_signer(SECURITY_TRUST_REGISTRY, imported_id, persisted_key)
                    st.success("Sender key explicitly added to this app's trusted registry.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Uploaded key is not a valid Ed25519 public key: {type(exc).__name__}: {exc}")

        registry = _security_registry()
        signer_entries = registry.get("signers", {})
        if signer_entries:
            signer_ids = sorted(signer_entries)
            revoke_choice = st.selectbox("Signer to deactivate", signer_ids, key="security_revoke_choice")
            if st.button("Deactivate signer", key="security_revoke"):
                try:
                    revoke_trusted_signer(SECURITY_TRUST_REGISTRY, revoke_choice)
                    st.success("Signer deactivated. Future verification will reject it.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not deactivate signer: {type(exc).__name__}: {exc}")
            st.dataframe(pd.DataFrame([
                {"Signer ID": signer_id, "Algorithm": entry.get("algorithm"),
                 "Active": bool(entry.get("active", True))}
                for signer_id, entry in signer_entries.items()
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No trusted signers registered yet.")

        st.markdown("#### Optional local secrets")
        secret_columns = st.columns(3)
        secret_specs = [
            ("HMAC-SHA256", "demo-hmac.secret", generate_hmac_key),
            ("Pseudonymization", "demo-pseudonym.secret", generate_hmac_key),
            ("AES-256-GCM", "demo-aes.secret", generate_aes256_key),
        ]
        for column, (title, filename, generator) in zip(secret_columns, secret_specs):
            secret_path = SECURITY_KEYS / filename
            column.caption(title)
            if column.button(f"Generate {title} key", key=f"generate_{filename}"):
                try:
                    generator(secret_path)
                    column.success(f"Saved `{secret_path.name}`")
                except FileExistsError:
                    column.warning("Key already exists; it was not overwritten.")
                except Exception as exc:
                    column.error(f"Generation failed: {type(exc).__name__}: {exc}")

    local_manifests_dir = SECURITY_UI_OUTPUT / "manifests"
    local_manifests = sorted(local_manifests_dir.glob("*.manifest.json")) if local_manifests_dir.is_dir() else []
    private_keys = _security_key_files("*-private.pem")
    hmac_keys = sorted(set(_security_key_files("*-hmac.secret") + _security_key_files("demo-hmac.secret")))
    pseudonym_keys = sorted(set(_security_key_files("*-pseudonym.secret") + _security_key_files("demo-pseudonym.secret")))
    aes_keys = sorted(set(_security_key_files("*-aes.secret") + _security_key_files("demo-aes.secret")))

    with manifest_tab:
        st.markdown("#### Sign selected DICOM object")
        st.caption("The manifest covers metadata and PixelData. The optional raw-file digest also detects byte-level reserialization differences.")
        if private_keys:
            signing_key = st.selectbox("Ed25519 private key", private_keys,
                                       format_func=lambda path: path.name, key="security_signing_key")
            include_raw = st.checkbox("Include raw-file SHA-256", value=True, key="security_raw_hash")
            sign_hmac = st.checkbox("Also create an HMAC-SHA256 sidecar", value=False, key="security_sign_hmac")
            signing_hmac_key = None
            if sign_hmac and hmac_keys:
                signing_hmac_key = st.selectbox("HMAC key", hmac_keys, format_func=lambda path: path.name,
                                                key="security_hmac_signing_key")
            elif sign_hmac:
                st.warning("Generate an HMAC key in the Keys and trust tab first.")
            if st.button("Sign selected DICOM", type="primary", key="security_sign_object",
                          disabled=sign_hmac and signing_hmac_key is None):
                try:
                    SECURITY_UI_OUTPUT.mkdir(parents=True, exist_ok=True)
                    local_manifests_dir.mkdir(parents=True, exist_ok=True)
                    manifest_path = local_manifests_dir / f"test-slice-{selected_index + 1:04d}-{uuid.uuid4().hex[:8]}.manifest.json"
                    envelope = create_signed_manifest(dicom_path, signing_key, include_raw_file_hash=include_raw)
                    save_signed_manifest(envelope, manifest_path)
                    if signing_hmac_key:
                        tag = hmac_sha256(signing_hmac_key.read_bytes(), canonical_json_bytes(envelope["manifest"]))
                        Path(str(manifest_path) + ".hmac").write_text(
                            json.dumps({"algorithm": "HMAC-SHA256", "tag": tag}, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    append_audit_event(
                        SECURITY_AUDIT_LOG, "sign_manifest", "SIGNED", actor_id="streamlit-local-signer",
                        source_sha256=raw_file_sha256(dicom_path), output_sha256=raw_file_sha256(manifest_path),
                        details={"signer_id": envelope["manifest"]["signer_id"], "hmac_created": bool(signing_hmac_key)},
                    )
                    st.session_state["security_last_manifest"] = str(manifest_path)
                    st.success(f"Signed manifest created: `{manifest_path.relative_to(PROJECT_ROOT)}`")
                    st.caption(f"Signer ID: `{envelope['manifest']['signer_id']}` · Manifest ID: `{envelope['manifest']['manifest_id']}`")
                    st.download_button("Download signed manifest", manifest_path.read_bytes(),
                                       file_name=manifest_path.name, mime="application/json", key="download_signed_manifest")
                except Exception as exc:
                    st.error(f"Signing failed: {type(exc).__name__}: {exc}")
        else:
            st.info("Generate an Ed25519 keypair in the Keys and trust tab before signing.")

        st.markdown("#### Verify a signed manifest")
        if local_manifests:
            remembered = st.session_state.get("security_last_manifest")
            default_manifest = next((idx for idx, path in enumerate(local_manifests) if str(path) == remembered), 0)
            manifest_path = st.selectbox("Manifest sidecar", local_manifests, index=default_manifest,
                                         format_func=lambda path: path.name, key="security_verify_manifest")
            current_registry = _security_registry()
            trusted = sorted(current_registry.get("signers", {}))
            expected_options = ["Any active trusted signer"] + trusted
            expected_signer = st.selectbox("Expected signer", expected_options, key="security_expected_signer")
            use_hmac = st.checkbox("Verify optional HMAC sidecar", value=False, key="security_verify_hmac")
            verify_hmac_key = st.selectbox("HMAC key", hmac_keys, format_func=lambda path: path.name,
                                           key="security_hmac_verify_key") if use_hmac and hmac_keys else None
            use_replay = st.checkbox("Use local replay cache (second verification of the same manifest is flagged)",
                                     value=False, key="security_use_replay")
            if use_hmac and not verify_hmac_key:
                st.warning("No HMAC key is available. Generate one in the Keys and trust tab.")
            if st.button("Verify selected file and manifest", key="security_verify_object",
                         disabled=use_hmac and verify_hmac_key is None):
                try:
                    registry = _security_registry()
                    replay_path = SECURITY_REPLAY_CACHE if use_replay else None
                    result = verify_signed_manifest(
                        dicom_path, manifest_path, registry,
                        expected_signer=None if expected_signer == expected_options[0] else expected_signer,
                        replay_cache_path=replay_path,
                    )
                    if verify_hmac_key:
                        sidecar_path = Path(str(manifest_path) + ".hmac")
                        try:
                            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                            envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
                            valid = verify_hmac_sha256(verify_hmac_key.read_bytes(),
                                                       canonical_json_bytes(envelope["manifest"]), sidecar["tag"])
                            result["hmac"] = {"status": "VALID" if valid else "INVALID"}
                        except (OSError, ValueError, KeyError, json.JSONDecodeError):
                            result["hmac"] = {"status": "MISSING_OR_INVALID"}
                    append_audit_event(SECURITY_AUDIT_LOG, "verify_manifest", result["status"],
                                       actor_id="streamlit-local-verifier",
                                       source_sha256=raw_file_sha256(dicom_path),
                                       details={"manifest_file": manifest_path.name})
                    hmac_status = result.get("hmac", {}).get("status")
                    if result["status"] == "VERIFIED" and (not use_hmac or hmac_status == "VALID"):
                        st.success("Cryptographic verification passed.")
                    elif result["status"] == "VERIFIED":
                        st.error(f"Signed manifest passed, but HMAC verification failed: {hmac_status}.")
                    else:
                        st.error(f"Verification outcome: {result['status']}")
                    st.json(_security_result_summary(result))
                except Exception as exc:
                    st.error(f"Verification failed: {type(exc).__name__}: {exc}")
        else:
            st.info("No local signed manifests yet. Sign a preloaded slice above to create one.")

        st.divider()
        st.markdown("#### Sign a DICOM file for exchange")
        st.caption("Upload is processed in temporary storage for this session; the source DICOM is not added to the project dataset.")
        exchange_file = st.file_uploader("DICOM file to sign", key="security_exchange_sign_upload")
        if private_keys:
            exchange_signer = st.selectbox(
                "Signing key for uploaded file", private_keys,
                format_func=lambda path: path.name, key="security_exchange_signer",
            )
            exchange_include_raw = st.checkbox(
                "Include raw-file SHA-256 in exchange manifest", value=True,
                key="security_exchange_raw_hash",
            )
            exchange_hmac = st.checkbox("Create an additional HMAC sidecar", value=False,
                                         key="security_exchange_hmac")
            exchange_hmac_key = st.selectbox(
                "Shared HMAC key", hmac_keys, format_func=lambda path: path.name,
                key="security_exchange_hmac_key",
            ) if exchange_hmac and hmac_keys else None
            if exchange_hmac and exchange_hmac_key is None:
                st.warning("Generate an HMAC key in the Keys and trust tab first.")
            if st.button("Create exchange manifest", key="security_exchange_sign_button",
                         disabled=exchange_file is None or (exchange_hmac and exchange_hmac_key is None)):
                try:
                    with tempfile.TemporaryDirectory(prefix="dicom_exchange_sign_") as temp_name:
                        temp_dir = Path(temp_name)
                        source = temp_dir / Path(exchange_file.name).name
                        source.write_bytes(exchange_file.getvalue())
                        envelope = create_signed_manifest(
                            source, exchange_signer,
                            include_raw_file_hash=exchange_include_raw,
                        )
                        manifest_bytes = canonical_json_bytes(envelope) + b"\n"
                        manifest_name = f"{source.stem}.manifest.json"
                        event_path = temp_dir / manifest_name
                        event_path.write_bytes(manifest_bytes)
                        hmac_bytes = None
                        if exchange_hmac_key is not None:
                            hmac_tag = hmac_sha256(
                                exchange_hmac_key.read_bytes(),
                                canonical_json_bytes(envelope["manifest"]),
                            )
                            hmac_bytes = (json.dumps({"algorithm": "HMAC-SHA256", "tag": hmac_tag},
                                                     sort_keys=True) + "\n").encode("utf-8")
                        append_audit_event(
                            SECURITY_AUDIT_LOG, "sign_uploaded_manifest", "SIGNED",
                            actor_id="streamlit-local-signer",
                            source_sha256=raw_file_sha256(source),
                            output_sha256=raw_file_sha256(event_path),
                            details={"signer_id": envelope["manifest"]["signer_id"]},
                        )
                        st.session_state["security_exchange_manifest_bytes"] = manifest_bytes
                        st.session_state["security_exchange_manifest_name"] = manifest_name
                        st.session_state["security_exchange_hmac_bytes"] = hmac_bytes
                        st.session_state["security_exchange_hmac_name"] = f"{manifest_name}.hmac"
                        st.session_state["security_exchange_signer_id"] = envelope["manifest"]["signer_id"]
                    st.success("Exchange manifest signed. Send it alongside the original DICOM file.")
                    st.caption(f"Signer ID: `{st.session_state['security_exchange_signer_id']}`")
                    st.download_button(
                        "Download exchange manifest",
                        st.session_state["security_exchange_manifest_bytes"],
                        file_name=st.session_state["security_exchange_manifest_name"],
                        mime="application/json", key="security_exchange_manifest_download",
                    )
                    if st.session_state.get("security_exchange_hmac_bytes") is not None:
                        st.download_button(
                            "Download HMAC sidecar",
                            st.session_state["security_exchange_hmac_bytes"],
                            file_name=st.session_state["security_exchange_hmac_name"],
                            mime="application/json", key="security_exchange_hmac_download",
                        )
                except Exception as exc:
                    st.error(f"Could not sign uploaded DICOM: {type(exc).__name__}: {exc}")
        else:
            st.info("Generate an Ed25519 keypair first to sign an uploaded DICOM file.")

        st.markdown("#### Verify received DICOM and manifest")
        received_file = st.file_uploader("Received DICOM file", key="security_exchange_verify_dicom")
        received_manifest = st.file_uploader("Received signed manifest (.json)", type=["json"],
                                             key="security_exchange_verify_manifest")
        verify_exchange_hmac = st.checkbox("Also verify a shared-secret HMAC sidecar", value=False,
                                            key="security_exchange_verify_hmac")
        received_hmac = st.file_uploader("Received HMAC sidecar", type=["hmac", "json"],
                                         key="security_exchange_verify_hmac_upload") if verify_exchange_hmac else None
        received_hmac_key = st.selectbox(
            "Local HMAC key", hmac_keys, format_func=lambda path: path.name,
            key="security_exchange_verify_hmac_key",
        ) if verify_exchange_hmac and hmac_keys else None
        use_exchange_replay = st.checkbox(
            "Apply local replay check to this manifest", value=False,
            key="security_exchange_replay_check",
        )
        if st.button("Verify received pair", key="security_exchange_verify_button",
                     disabled=(received_file is None or received_manifest is None or
                               (verify_exchange_hmac and (received_hmac is None or received_hmac_key is None)))):
            try:
                with tempfile.TemporaryDirectory(prefix="dicom_exchange_verify_") as temp_name:
                    temp_dir = Path(temp_name)
                    received_path = temp_dir / Path(received_file.name).name
                    manifest_path = temp_dir / "received.manifest.json"
                    received_path.write_bytes(received_file.getvalue())
                    manifest_path.write_bytes(received_manifest.getvalue())
                    result = verify_signed_manifest(
                        received_path, manifest_path, _security_registry(),
                        replay_cache_path=SECURITY_REPLAY_CACHE if use_exchange_replay else None,
                    )
                    if verify_exchange_hmac and received_hmac is not None and received_hmac_key is not None:
                        try:
                            hmac_sidecar = json.loads(received_hmac.getvalue().decode("utf-8"))
                            manifest_envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
                            hmac_valid = (
                                hmac_sidecar.get("algorithm") == "HMAC-SHA256" and
                                verify_hmac_sha256(
                                    received_hmac_key.read_bytes(),
                                    canonical_json_bytes(manifest_envelope["manifest"]),
                                    hmac_sidecar["tag"],
                                )
                            )
                            result["hmac"] = {"status": "VALID" if hmac_valid else "INVALID"}
                        except (UnicodeDecodeError, OSError, ValueError, KeyError, TypeError):
                            result["hmac"] = {"status": "MISSING_OR_INVALID"}
                    append_audit_event(
                        SECURITY_AUDIT_LOG, "verify_received_manifest", result["status"],
                        actor_id="streamlit-local-verifier",
                        source_sha256=raw_file_sha256(received_path),
                        details={"manifest_file": Path(received_manifest.name).name},
                    )
                hmac_status = result.get("hmac", {}).get("status")
                if result["status"] == "VERIFIED" and (not verify_exchange_hmac or hmac_status == "VALID"):
                    st.success("Received DICOM passed signature and integrity verification.")
                elif result["status"] == "VERIFIED":
                    st.error(f"Signed manifest is valid, but the requested HMAC check failed: {hmac_status}.")
                else:
                    st.error(f"Received pair verification outcome: {result['status']}")
                st.json(_security_result_summary(result))
                st.caption("The sender’s public key must already be trusted in this app’s local signer registry.")
            except Exception as exc:
                st.error(f"Could not verify received pair: {type(exc).__name__}: {exc}")

    with analysis_tab:
        st.markdown("#### Verification-first analysis")
        st.write("The analysis workflow will not run AI models when signature or content verification fails.")
        analysis_manifests = sorted(local_manifests_dir.glob("*.manifest.json")) if local_manifests_dir.is_dir() else []
        if analysis_manifests:
            selected_manifest = st.selectbox("Signed reference for this slice", analysis_manifests,
                                             format_func=lambda path: path.name, key="security_analysis_manifest")
            analysis_registry_path = st.text_input("Trusted-signer registry", value=str(SECURITY_TRUST_REGISTRY),
                                                   key="security_analysis_registry")
            model_cols = st.columns(4)
            run_if = model_cols[0].checkbox("Isolation Forest", value=True, key="security_ai_if")
            run_ae = model_cols[1].checkbox("Metadata autoencoder", value=True, key="security_ai_ae")
            run_pixel = model_cols[2].checkbox("Patch pixel IF", value=True, key="security_ai_pixel")
            run_unet = model_cols[3].checkbox("U-Net (slower)", value=False, key="security_ai_unet")
            use_analysis_replay = st.checkbox("Apply local replay-cache check", value=False, key="security_analysis_replay")
            if st.button("Verify, then analyze", type="primary", key="security_verify_analyze"):
                try:
                    registry = json.loads(Path(analysis_registry_path).read_text(encoding="utf-8"))
                    crypto_result = verify_signed_manifest(
                        dicom_path, selected_manifest, registry,
                        replay_cache_path=SECURITY_REPLAY_CACHE if use_analysis_replay else None,
                    )
                    if crypto_result["status"] == "VERIFIED":
                        result = verify_then_analyze(
                            dicom_path, selected_manifest, analysis_registry_path,
                            if_model_path=IF_MODEL if run_if else None,
                            autoencoder_model_path=AE_MODEL if run_ae else None,
                            pixel_model_path=PIXEL_MODEL if run_pixel else None,
                            unet_checkpoint_path=UNET_MODEL if run_unet and UNET_MODEL.is_file() else None,
                        )
                        result["cryptographic"] = _security_result_summary(crypto_result)
                    else:
                        result = {"cryptographic": _security_result_summary(crypto_result),
                                  "ml_status": "SKIPPED_CRYPTOGRAPHIC_POLICY",
                                  "reason": "AI analysis is blocked when the signed manifest is not verified."}
                    SECURITY_UI_OUTPUT.mkdir(parents=True, exist_ok=True)
                    analysis_path = SECURITY_UI_OUTPUT / f"single_analysis_{uuid.uuid4().hex[:8]}.json"
                    analysis_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
                    st.session_state["security_last_analysis"] = str(analysis_path)
                    if crypto_result["status"] == "VERIFIED":
                        st.success("Verified object analyzed. Cryptographic and AI findings are shown separately.")
                    else:
                        st.error(f"Cryptographic status: {crypto_result['status']} · AI analysis skipped.")
                    st.json(result)
                    st.download_button("Download structured analysis", analysis_path.read_bytes(),
                                       file_name=analysis_path.name, mime="application/json", key="download_security_analysis")
                except Exception as exc:
                    st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        else:
            st.info("Sign this slice first, then select its manifest here.")

    with batch_tab:
        st.markdown("#### Sign the test fold")
        st.write("Creates one signed sidecar per preloaded test DICOM, preserving the relative directory layout. Original DICOM files remain untouched.")
        if private_keys:
            batch_signer = st.selectbox("Batch signing key", private_keys, format_func=lambda path: path.name,
                                        key="security_batch_signer")
            batch_raw = st.checkbox("Include raw-file digest in every manifest", value=True, key="security_batch_raw")
            if st.button(f"Sign all {len(paths)} test files", key="security_batch_sign"):
                batch_root = (SECURITY_UI_OUTPUT / "batch_manifests" /
                              f"{batch_signer.stem.replace('-private', '')}_{uuid.uuid4().hex[:8]}")
                batch_root.mkdir(parents=True, exist_ok=True)
                progress = st.progress(0.0, text="Signing test-fold files…")
                signed_count = 0
                try:
                    for idx, source in enumerate(paths):
                        relative = Path(relative_key(source, DATASET_DIR))
                        sidecar = batch_root / f"{relative}.manifest.json"
                        envelope = create_signed_manifest(source, batch_signer, include_raw_file_hash=batch_raw)
                        save_signed_manifest(envelope, sidecar)
                        signed_count += 1
                        if (idx + 1) % 10 == 0 or idx + 1 == len(paths):
                            progress.progress((idx + 1) / len(paths), text=f"Signed {idx + 1}/{len(paths)}")
                    st.session_state["security_batch_manifest_root"] = str(batch_root)
                    st.success(f"Signed {signed_count} DICOM files. Manifests are in `{batch_root.relative_to(PROJECT_ROOT)}`.")
                except Exception as exc:
                    st.error(f"Batch signing stopped after {signed_count} files: {type(exc).__name__}: {exc}")
        else:
            st.info("Generate a signer keypair before batch signing.")

        batch_roots = sorted((SECURITY_UI_OUTPUT / "batch_manifests").glob("*") if
                             (SECURITY_UI_OUTPUT / "batch_manifests").is_dir() else [])
        batch_roots = [root for root in batch_roots if root.is_dir()]
        if batch_roots:
            selected_batch_root = st.selectbox("Manifest directory", batch_roots,
                                               format_func=lambda path: path.name, key="security_batch_manifest_root")
            st.markdown("#### Verify directory, then run selected models")
            batch_model_cols = st.columns(4)
            batch_if = batch_model_cols[0].checkbox("Isolation Forest", value=True, key="security_batch_if")
            batch_ae = batch_model_cols[1].checkbox("Metadata autoencoder", value=True, key="security_batch_ae")
            batch_pixel = batch_model_cols[2].checkbox("Patch pixel IF", value=False, key="security_batch_pixel")
            batch_unet = batch_model_cols[3].checkbox("U-Net (slow; optional)", value=False, key="security_batch_unet")
            if st.button("Verify directory and analyze verified files", key="security_batch_analyze"):
                try:
                    with st.spinner("Verifying manifests; study/series checks use verified files only…"):
                        result = verify_directory_then_analyze(
                            DATASET_DIR, selected_batch_root, SECURITY_TRUST_REGISTRY,
                            if_model_path=IF_MODEL if batch_if else None,
                            autoencoder_model_path=AE_MODEL if batch_ae else None,
                            pixel_model_path=PIXEL_MODEL if batch_pixel else None,
                            unet_checkpoint_path=UNET_MODEL if batch_unet and UNET_MODEL.is_file() else None,
                        )
                    # Keep the downloadable app report free of digest and DICOM UID values.
                    for record in result.get("results", {}).values():
                        record["cryptographic"] = _security_result_summary(record.get("cryptographic", {}))
                    SECURITY_UI_OUTPUT.mkdir(parents=True, exist_ok=True)
                    report_path = SECURITY_UI_OUTPUT / f"batch_analysis_{uuid.uuid4().hex[:8]}.json"
                    report_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
                    status_counts = {}
                    for record in result["results"].values():
                        status = record["cryptographic"]["status"]
                        status_counts[status] = status_counts.get(status, 0) + 1
                    st.session_state["security_last_batch_analysis"] = str(report_path)
                    st.metric("Files verified", f"{result['files_verified']} / {result['files_discovered']}")
                    st.json({"crypto_status_counts": status_counts, "ml_status": result["ml_status"]})
                    summary_rows = []
                    for token, record in result["results"].items():
                        summary_rows.append({
                            "File token": token,
                            "Crypto": record["cryptographic"]["status"],
                            "Structural finding": bool(record.get("structural", [])),
                            "Semantic finding": bool(record.get("semantic", [])),
                            "IF flag": record.get("metadata_anomaly", {}).get("isolation_forest", {}).get("flag"),
                            "AE flag": record.get("metadata_anomaly", {}).get("autoencoder", {}).get("flag"),
                        })
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                    st.download_button("Download batch analysis JSON", report_path.read_bytes(),
                                       file_name=report_path.name, mime="application/json", key="download_batch_analysis")
                except Exception as exc:
                    st.error(f"Directory analysis failed: {type(exc).__name__}: {exc}")

    with privacy_tab:
        st.markdown("#### Verify, create a research copy, and sign it again")
        st.warning("This is limited metadata pseudonymization. It does not OCR or remove burned-in pixel identifiers and is not a complete DICOM PS3.15 de-identification implementation.")
        privacy_manifests = sorted(local_manifests_dir.glob("*.manifest.json")) if local_manifests_dir.is_dir() else []
        if privacy_manifests and private_keys and pseudonym_keys:
            source_manifest = st.selectbox("Source signed manifest", privacy_manifests,
                                           format_func=lambda path: path.name, key="security_privacy_manifest")
            pseudonym_key = st.selectbox("Pseudonymization key", pseudonym_keys,
                                         format_func=lambda path: path.name, key="security_pseudonym_key")
            new_signing_key = st.selectbox("Key to sign transformed copy", private_keys,
                                           format_func=lambda path: path.name, key="security_privacy_signer")
            if st.button("Verify and create signed research copy", key="security_transform"):
                try:
                    crypto = verify_signed_manifest(dicom_path, source_manifest, _security_registry())
                    if crypto["status"] != "VERIFIED":
                        st.error(f"Source verification failed ({crypto['status']}); transformation was not performed.")
                    else:
                        output_dir = SECURITY_UI_OUTPUT / "research_copies"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        suffix = uuid.uuid4().hex[:8]
                        output_dicom = output_dir / f"research_copy_{selected_index + 1:04d}_{suffix}.dcm"
                        output_manifest = output_dir / f"research_copy_{selected_index + 1:04d}_{suffix}.manifest.json"
                        privacy_report = deidentify_research_copy(dicom_path, output_dicom, pseudonym_key.read_bytes())
                        new_envelope = create_signed_manifest(output_dicom, new_signing_key)
                        save_signed_manifest(new_envelope, output_manifest)
                        append_audit_event(
                            SECURITY_AUDIT_LOG, "verify_deidentify_resign", "TRANSFORMED_AND_RESIGNED",
                            actor_id="streamlit-local-transformer", source_sha256=privacy_report["source_sha256"],
                            output_sha256=privacy_report["output_sha256"],
                            details={"new_signer_id": new_envelope["manifest"]["signer_id"],
                                     "pixel_review_required": privacy_report["pixel_review_required"]},
                        )
                        if privacy_report["pixel_review_required"]:
                            st.warning("Manual pixel review is required before using this copy because BurnedInAnnotation is not NO.")
                        st.success("Source signature verified; a separate transformed copy was created and freshly signed.")
                        st.json({"original_verification": crypto,
                                 "output_file": output_dicom.name,
                                 "new_signer_id": new_envelope["manifest"]["signer_id"],
                                 "pixel_review_required": privacy_report["pixel_review_required"],
                                 "warning": privacy_report["warning"]})
                        st.download_button("Download transformed DICOM", output_dicom.read_bytes(),
                                           file_name=output_dicom.name, mime="application/dicom", key="download_research_copy")
                        st.download_button("Download new signed manifest", output_manifest.read_bytes(),
                                           file_name=output_manifest.name, mime="application/json", key="download_research_manifest")
                except Exception as exc:
                    st.error(f"Research-copy transformation failed: {type(exc).__name__}: {exc}")
        else:
            st.info("Create a signed manifest, an Ed25519 keypair, and a pseudonymization key before this workflow is available.")

    with storage_tab:
        st.markdown("#### AES-256-GCM encryption at rest")
        st.info("This encrypts a file stored on disk. It does not provide encryption for a network connection or replace TLS.")
        if not aes_keys:
            st.caption("Generate an AES key in the Keys and trust tab first.")
        else:
            aes_key = st.selectbox("AES key", aes_keys, format_func=lambda path: path.name, key="security_aes_key")
            if st.button("Encrypt selected CPAISD file", key="security_encrypt_selected"):
                try:
                    output_dir = SECURITY_UI_OUTPUT / "encrypted"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    encrypted_path = output_dir / f"slice_{selected_index + 1:04d}_{uuid.uuid4().hex[:8]}.dsf"
                    report = encrypt_file(dicom_path, encrypted_path, aes_key.read_bytes())
                    st.success(f"Encrypted file created ({report['ciphertext_bytes']:,} bytes).")
                    st.json({"plaintext_bytes": report["plaintext_bytes"],
                             "ciphertext_bytes": report["ciphertext_bytes"],
                             "source_sha256": report["source_sha256"]})
                    st.download_button("Download AES-GCM encrypted file", encrypted_path.read_bytes(),
                                       file_name=encrypted_path.name, mime="application/octet-stream",
                                       key="download_encrypted_dicom")
                except Exception as exc:
                    st.error(f"Encryption failed: {type(exc).__name__}: {exc}")
            encrypted_upload = st.file_uploader("Upload a `.dsf` file to decrypt with this key", type=None,
                                                key="security_encrypted_upload")
            if encrypted_upload is not None and st.button("Decrypt uploaded file in memory", key="security_decrypt_upload"):
                try:
                    with tempfile.TemporaryDirectory(prefix="dicom_secure_decrypt_") as temp_dir:
                        encrypted_temp = Path(temp_dir) / "incoming.dsf"
                        output_temp = Path(temp_dir) / "recovered.dcm"
                        encrypted_temp.write_bytes(encrypted_upload.getvalue())
                        decrypt_file(encrypted_temp, output_temp, aes_key.read_bytes())
                        plaintext = output_temp.read_bytes()
                    st.success("AES-GCM authentication passed; decrypted bytes are ready to download.")
                    st.download_button("Download decrypted DICOM", plaintext,
                                       file_name="recovered.dcm", mime="application/dicom",
                                       key="download_decrypted_dicom")
                except Exception as exc:
                    st.error(f"Decryption/authentication failed: {type(exc).__name__}: {exc}")

    with results_tab:
        st.markdown("#### Saved 809-file security experiment")
        summary_path = SECURITY_EXPERIMENT_OUTPUT / "security_summary.json"
        scenarios_path = SECURITY_EXPERIMENT_OUTPUT / "security_scenarios.csv"
        performance_path = SECURITY_EXPERIMENT_OUTPUT / "security_performance.csv"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = st.columns(4)
            metrics[0].metric("Test files", summary.get("input_files", "—"))
            metrics[1].metric("Scenario checks", f"{summary.get('scenario_rows', 0):,}")
            metrics[2].metric("Mean sign time", f"{summary.get('mean_signing_latency_ms', 0):.2f} ms")
            metrics[3].metric("Mean verify time", f"{summary.get('mean_verification_latency_ms', 0):.2f} ms")
            st.caption(f"Mean signed envelope: {summary.get('mean_manifest_bytes', 0):.1f} bytes · Relative size: {summary.get('mean_manifest_overhead_fraction', 0) * 100:.4f}%")
            counts = summary.get("scenario_counts", {})
            if counts:
                st.dataframe(pd.DataFrame([
                    {"Scenario outcome": name.replace(":", " · "), "Files": count}
                    for name, count in sorted(counts.items())
                ]), use_container_width=True, hide_index=True)
            if scenarios_path.is_file():
                with st.expander("Per-file security scenario records"):
                    st.dataframe(pd.read_csv(scenarios_path), use_container_width=True, hide_index=True)
            if performance_path.is_file():
                with st.expander("Cryptographic operation timings"):
                    st.dataframe(pd.read_csv(performance_path), use_container_width=True, hide_index=True)
        else:
            st.info("No saved full-fold security run is present yet.")

        st.markdown("#### Run full-fold security scenarios")
        st.caption("This runs signing and verification scenarios over the 809-file test fold. It does not train models or modify source DICOM files; it may take several minutes.")
        if st.button("Run full 809-file security experiment", key="security_rerun_full_experiment"):
            output_dir = SECURITY_UI_OUTPUT / "security_experiment_rerun"
            command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_crypto_security_experiment.py"),
                       "--input", str(DATASET_DIR), "--output", str(output_dir)]
            with st.spinner("Running 13 security scenarios per test file…"):
                try:
                    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                               text=True, timeout=1800, check=False)
                    if completed.returncode == 0:
                        st.success(f"Completed. Results saved under `{output_dir.relative_to(PROJECT_ROOT)}`.")
                        if completed.stdout.strip():
                            try:
                                st.json(json.loads(completed.stdout))
                            except json.JSONDecodeError:
                                st.code(completed.stdout[-5000:])
                    else:
                        st.error(f"Experiment failed with exit code {completed.returncode}.")
                        st.code((completed.stderr or completed.stdout)[-5000:])
                except subprocess.TimeoutExpired:
                    st.error("The experiment exceeded the 30-minute UI limit. Check the output directory for partial records.")
                except Exception as exc:
                    st.error(f"Could not launch the experiment: {type(exc).__name__}: {exc}")

        if SECURITY_AUDIT_LOG.is_file():
            with st.expander("Local privacy-minimized audit events"):
                try:
                    audit_rows = [json.loads(line) for line in SECURITY_AUDIT_LOG.read_text(encoding="utf-8").splitlines() if line]
                    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
                    st.download_button("Download audit JSONL", SECURITY_AUDIT_LOG.read_bytes(),
                                       file_name="security_audit.jsonl", mime="application/x-ndjson",
                                       key="download_security_audit")
                except Exception as exc:
                    st.warning(f"Could not display audit log: {type(exc).__name__}: {exc}")


st.title("DICOM Security Demo Lab")
st.caption("Preloaded CPAISD brain CT · Metadata and pixel anomaly checks · Signed-manifest exchange workflow")

if not DATASET_DIR.is_dir():
    st.error(f"Preloaded CPAISD test folder was not found: {DATASET_DIR}")
    st.stop()
missing = [path for path in (IF_MODEL, AE_MODEL, PIXEL_MODEL, PIXEL_CALIBRATION) if not path.is_file()]
if missing:
    st.warning("Some preloaded anomaly-model assets are missing. The signed-manifest workspace remains available; dependent attack-lab checks will be limited.")
    for path in missing:
        st.write(f"- `{path.relative_to(PROJECT_ROOT)}`")

dicom_paths = [Path(path) for path in get_dicom_paths(str(DATASET_DIR))]
if not dicom_paths:
    st.error("No readable DICOM files were found in the preloaded CPAISD test folder.")
    st.stop()

with st.sidebar:
    st.header("Preloaded project data")
    st.success(f"CPAISD test fold ready · {len(dicom_paths)} DICOM slices")
    st.caption("No upload or path entry is needed. Edits run on temporary copies.")
    st.caption("Dataset: data/cpaisd/test")
    st.divider()
    st.metric("Studies in test fold", "10")
    st.metric("Available checks", "11" if UNET_MODEL.is_file() else "10")
    if UNET_MODEL.is_file():
        st.caption(f"U-Net checkpoint: {UNET_MODEL.parent.relative_to(PROJECT_ROOT)}")
    st.caption("Secure exchange keys are stored in the app host's local user profile, not in the repository.")

overview_tab, live_tab, results_tab, security_tab = st.tabs([
    "Project overview", "Interactive attack lab", "Experiments completed", "Secure DICOM exchange",
])

with overview_tab:
    st.subheader("What this demo does")
    st.write("Explore controlled metadata and pixel edits, compare the existing integrity/anomaly checks, and use the signed-manifest tab to sign and verify individual files or the full test fold before analysis.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preloaded CT slices", len(dicom_paths))
    c2.metric("Metadata checks", "5")
    c3.metric("Pixel checks", "5" if UNET_MODEL.is_file() else "4")
    c4.metric("Saved experiment groups", "8")
    preview_ds = read_dataset(dicom_paths[0], stop_before_pixels=False)
    preview = _normalized_image(preview_ds)
    st.image(preview, caption="Preloaded CPAISD test slice · normalized for display", width=430)
    st.info("A detector alert means that a programmed check fired. It does not determine intent, clinical harm, or whether a real-world workflow change is legitimate.")
    with st.expander("Checks available in the live lab"):
        st.markdown("""
        - **Metadata SHA-256:** exact header change against the clean slice.
        - **PixelData SHA-256:** exact stored-pixel change against the clean slice.
        - **Structural rules:** basic metadata constraints, including CT modality.
        - **Semantic consistency:** checks whether patient/study and stable series fields agree across peer slices.
        - **Isolation Forest and autoencoder:** metadata anomaly scores learned from clean training data.
        - **No-reference pixel Isolation Forest and median residual:** image-only anomaly baselines that do not compare against a clean image.
        - **Supervised U-Net pixel localizer:** trained on synthetic edit masks; shown when the trained checkpoint is present. Its output is a research baseline, not a clinical assessment.
        - **Pixel/header consistency:** checks whether image payload layout agrees with DICOM header declarations.
        - **All-checks OR summary:** combines whether any of the above checks alert.
        - **Ed25519 signed manifests:** binds a DICOM object to a trusted sender key and scoped SHA-256 digests.
        - **HMAC-SHA256, replay cache, audit, and AES-GCM storage:** optional security demonstrations. AES-GCM covers files at rest, not network traffic.
        - **Verify-first analysis:** cryptographic failures block the AI analysis path by default; study-level checks run only on verified files in batch mode.
        """)

with live_tab:
    st.subheader("Choose an edit; see the full detector response")
    st.warning("Research demonstration only: edits are synthetic, and detector alerts do not establish clinical harm or intent.")
    left, right = st.columns([1, 2])
    with left:
        sample_index = st.selectbox(
            "Preloaded CT slice", range(len(dicom_paths)), index=0,
            format_func=lambda index: f"Slice {index + 1:03d} of {len(dicom_paths)}",
        )
        mode = st.radio("Edit layer", ["Metadata only", "Pixels only", "Metadata + pixels"], horizontal=False)
        metadata_edit = None
        pixel_edit = None
        if mode in {"Metadata only", "Metadata + pixels"}:
            metadata_edit = st.selectbox("Metadata change", [
                "Modality to MR", "PixelSpacing x 1.7",
                "ProtocolName same-length substitution",
                "Manufacturer same-length substitution", "Remove ProtocolName",
            ])
        if mode in {"Pixels only", "Metadata + pixels"}:
            pixel_edit = st.selectbox("Pixel change", [
                "Localized patch (5x5, +32)", "Sparse low-amplitude (1%, +1)",
                "Copy-move patch (withheld U-Net family)",
            ])
        st.caption("The selected clean slice supplies a temporary reference. Your preloaded dataset files are never overwritten.")
        run_demo = st.button("Apply change and show everything", type="primary", use_container_width=True,
                             disabled=bool(missing))

    if run_demo:
        try:
            with st.spinner("Applying the edit to a temporary copy and running all checks..."):
                st.session_state["live_demo_result"] = run_live_demo(
                    dicom_paths[sample_index], mode, metadata_edit, pixel_edit
                )
                st.session_state["live_demo_mode"] = mode
        except Exception as exc:
            st.error(f"Demo could not run: {exc}")

    result = st.session_state.get("live_demo_result")
    if result:
        with right:
            st.success(f"Edit applied: {result['edit_description']}")
            unet_result = result.get("unet")
            selected_mask = None
            clean_selected_mask = None
            selected_threshold = None
            if unet_result:
                calibrated_threshold = float(unet_result["pixel_threshold"])
                log_multiplier = st.slider(
                    "U-Net mask sensitivity (log10 multiplier; 0 = validation-calibrated)",
                    min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                    help="Adjusts the pixel-probability threshold for this demo only. It does not retrain the model or change the saved experiment metrics.",
                )
                selected_threshold = min(1.0, max(1e-12, calibrated_threshold * (10.0 ** log_multiplier)))
                clean_selected_mask = unet_result["clean_probability"] >= selected_threshold
                selected_mask = unet_result["edited_probability"] >= selected_threshold
                clean_pixel_fraction = float(np.mean(clean_selected_mask))
                edited_alert = bool(np.any(selected_mask))
                clean_alert = bool(np.any(clean_selected_mask))
                detector_frame = result["detectors"].copy()
                unet_row = detector_frame["Check"] == "Supervised U-Net pixel localizer"
                detector_frame.loc[unet_row, "Clean scan"] = "ALERT" if clean_alert else "Clear"
                detector_frame.loc[unet_row, "Edited scan"] = "ALERT" if edited_alert else "Clear"
                detector_frame.loc[unet_row, "New alert"] = "Yes" if edited_alert and not clean_alert else "No"
                detector_frame.loc[unet_row, "What it checks"] = (
                    f"Maximum probability {float(np.max(unet_result['edited_probability'])):.4f}; "
                    f"selected mask threshold {selected_threshold:.6g}; "
                    f"{int(np.count_nonzero(selected_mask)):,} edited-slice pixels above threshold."
                )
                summary_row = detector_frame["Check"] == "All-checks OR summary"
                component_rows = detector_frame[~detector_frame["Check"].isin(
                    ["All-checks OR summary"]
                )]
                clean_any = bool((component_rows["Clean scan"] == "ALERT").any())
                edited_any = bool((component_rows["Edited scan"] == "ALERT").any())
                detector_frame.loc[summary_row, "Clean scan"] = "ALERT" if clean_any else "Clear"
                detector_frame.loc[summary_row, "Edited scan"] = "ALERT" if edited_any else "Clear"
                detector_frame.loc[summary_row, "New alert"] = "Yes" if edited_any and not clean_any else "No"
                detector_frame.loc[summary_row, "What it checks"] = (
                    "Any configured detector fired; U-Net uses the currently selected pixel threshold."
                )
                st.caption(
                    f"Selected pixel threshold: {selected_threshold:.6g} · "
                    f"clean reference score {unet_result['clean_score']:.4f} · "
                    f"edited slice score {unet_result['edited_score']:.4f}. "
                    "The threshold-based U-Net alert is raised when at least one pixel exceeds the selected threshold."
                )
                if result.get("synthetic_mask") is not None and np.any(result["synthetic_mask"]):
                    from PIL import Image
                    truth = np.asarray(Image.fromarray(result["synthetic_mask"].astype(np.uint8)).resize(
                        (selected_mask.shape[1], selected_mask.shape[0]), resample=Image.Resampling.NEAREST
                    ), dtype=bool)
                    tp = int(np.count_nonzero(selected_mask & truth))
                    fp = int(np.count_nonzero(selected_mask & ~truth))
                    fn = int(np.count_nonzero(~selected_mask & truth))
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
                    sensitivity_metrics = st.columns(5)
                    sensitivity_metrics[0].metric("Clean-slice pixel alarm fraction", f"{clean_pixel_fraction:.2%}")
                    sensitivity_metrics[1].metric("Edited pixel precision", f"{precision:.2%}")
                    sensitivity_metrics[2].metric("Edited pixel recall", f"{recall:.2%}")
                    sensitivity_metrics[3].metric("Edited mask IoU", f"{iou:.3f}")
                    sensitivity_metrics[4].metric("Thresholded U-Net alert", "ALERT" if edited_alert else "CLEAR")
                    st.caption("Single edited-slice metrics use the synthetic mask generated for this demo. The clean-slice alarm fraction is not a test-set false-positive rate.")
                    with st.expander("Compare several thresholds on this demo slice"):
                        sweep_rows = []
                        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0):
                            sweep_threshold = min(1.0, max(1e-12, calibrated_threshold * (10.0 ** multiplier)))
                            sweep_clean = unet_result["clean_probability"] >= sweep_threshold
                            sweep_edited = unet_result["edited_probability"] >= sweep_threshold
                            sweep_tp = int(np.count_nonzero(sweep_edited & truth))
                            sweep_fp = int(np.count_nonzero(sweep_edited & ~truth))
                            sweep_fn = int(np.count_nonzero(~sweep_edited & truth))
                            sweep_rows.append({
                                "log10 multiplier": multiplier,
                                "pixel threshold": sweep_threshold,
                                "clean-slice alarm fraction": float(np.mean(sweep_clean)),
                                "edited precision": sweep_tp / (sweep_tp + sweep_fp) if sweep_tp + sweep_fp else 0.0,
                                "edited recall": sweep_tp / (sweep_tp + sweep_fn) if sweep_tp + sweep_fn else 0.0,
                                "edited IoU": sweep_tp / (sweep_tp + sweep_fp + sweep_fn) if sweep_tp + sweep_fp + sweep_fn else 1.0,
                                "threshold alert": "ALERT" if np.any(sweep_edited) else "CLEAR",
                            })
                        sweep_frame = pd.DataFrame(sweep_rows)
                        st.dataframe(sweep_frame, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download this slice's threshold sweep CSV",
                            data=sweep_frame.to_csv(index=False),
                            file_name="unet_demo_threshold_sweep.csv",
                            mime="text/csv",
                        )
                else:
                    st.metric("Thresholded U-Net alert", "ALERT" if edited_alert else "CLEAR")
                    st.caption("No synthetic pixel mask is available for this metadata-only edit; the U-Net map still runs on the edited slice.")
            st.subheader("Every check, together")
            st.dataframe(detector_frame if unet_result else result["detectors"], use_container_width=True, hide_index=True)
            metrics = st.columns(4)
            changed_pixels = int(np.count_nonzero(result["difference"]))
            metrics[0].metric("Changed stored pixels", f"{changed_pixels:,}")
            metrics[1].metric("Pixel MAE", f"{float(result['difference'].mean()):.5f}")
            metrics[2].metric("Metadata digest changed", "Yes" if result["clean_metadata_hash"] != result["edited_metadata_hash"] else "No")
            metrics[3].metric("Pixel digest changed", "Yes" if result["clean_pixel_hash"] != result["edited_pixel_hash"] else "No")

            st.subheader("Visual comparison")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for axis, image, title in (
                (axes[0], result["clean_pixels"], "Clean CT slice"),
                (axes[1], result["edited_pixels"], "Edited copy"),
            ):
                low, high = np.percentile(image, [1, 99])
                axis.imshow(image, cmap="gray", vmin=low, vmax=high)
                axis.set_title(title)
                axis.axis("off")
            maximum = max(1.0, float(result["difference"].max()))
            heat = axes[2].imshow(result["difference"], cmap="inferno", vmin=0, vmax=maximum)
            axes[2].set_title("Reference difference map")
            axes[2].axis("off")
            fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04, label="Stored-value difference")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            clean_norm, clean_mask = result["clean_norm"], result["clean_mask"]
            edited_norm, edited_mask = result["edited_norm"], result["edited_mask"]
            overlay = np.repeat(edited_norm[:, :, None], 3, axis=2)
            overlay[edited_mask] = [1.0, 0.15, 0.05]
            st.image(overlay, caption="No-reference patch-Isolation-Forest alert map (red patches)", use_container_width=False)
            if unet_result:
                unet_columns = st.columns(3)
                for column, image, mask, title, score in (
                    (unet_columns[0], unet_result["clean_image"], clean_selected_mask, "U-Net: clean slice", unet_result["clean_score"]),
                    (unet_columns[1], unet_result["edited_image"], selected_mask, "U-Net: edited mask", unet_result["edited_score"]),
                ):
                    unet_overlay = np.repeat(np.asarray(image)[:, :, None], 3, axis=2)
                    mask_on_display = mask
                    if mask_on_display.shape != image.shape:
                        from PIL import Image
                        mask_on_display = np.asarray(Image.fromarray(mask_on_display.astype(np.uint8)).resize(
                            (image.shape[1], image.shape[0]), resample=Image.Resampling.NEAREST
                        ), dtype=bool)
                    unet_overlay[mask_on_display] = [1.0, 0.15, 0.05]
                    column.image(unet_overlay, caption=f"{title} · max score {score:.3f}", use_container_width=True)
                heat_figure, heat_axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
                heat = heat_axis.imshow(unet_result["edited_probability"], cmap="magma", vmin=0, vmax=1)
                if np.any(selected_mask) and not np.all(selected_mask):
                    heat_axis.contour(selected_mask.astype(np.uint8), levels=[0.5],
                                      colors=["cyan"], linewidths=0.7)
                heat_axis.set_title("U-Net probability heatmap")
                heat_axis.axis("off")
                heat_figure.colorbar(heat, ax=heat_axis, fraction=0.046, pad=0.04,
                                     label="Predicted edit probability")
                unet_columns[2].pyplot(heat_figure, clear_figure=True)
                plt.close(heat_figure)
                st.caption(f"U-Net calibrated image-score threshold: {unet_result['image_threshold']:.4f}. The slider changes the pixel mask and demo alert threshold only. Masks and heatmap are on the {selected_mask.shape[0]}×{selected_mask.shape[1]} model grid. This supervised model may miss edits and may alert on normal anatomy.")
            else:
                st.info("The supervised U-Net checkpoint is not present yet. The other pixel checks are available; train it with the command in README.md to enable U-Net scoring and overlays.")
            st.caption("The difference map uses the clean reference. The red patch and median-residual maps are no-reference baselines. All may miss edits or flag normal image structure. Display scaling is for visualization, not Hounsfield-unit measurement.")

with results_tab:
    st.subheader("Previously completed research experiments")
    st.write("These are saved outputs from the project's experiment runs, alongside the interactive one-slice demonstration.")
    show_saved_results()

with security_tab:
    _show_secure_exchange(dicom_paths)
