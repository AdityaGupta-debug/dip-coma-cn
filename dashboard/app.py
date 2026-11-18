"""Preloaded interactive research demo for CPAISD DICOM metadata and pixels."""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
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


DATASET_DIR = PROJECT_ROOT / "data" / "cpaisd" / "test"
IF_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "isolation_forest.joblib"
AE_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "metadata_autoencoder.joblib"
PIXEL_MODEL = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "no_reference_pixel_model.joblib"
PIXEL_CALIBRATION = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "calibration_and_clean_summary.json"
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
    from src.supervised_localization import UNetLocalizer

    checkpoint = torch.load(path_string, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = UNetLocalizer(base_channels=int(config["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
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
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    pixel_mask = probability >= float(config["pixel_threshold"])
    return normalized, probability, pixel_mask


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
            clean_norm_unet, clean_probability, clean_unet_mask = unet_probability_map(
                clean_ds, unet_model, unet_config
            )
            edited_norm_unet, edited_probability, edited_unet_mask = unet_probability_map(
                edited_ds, unet_model, unet_config
            )
            image_threshold = float(unet_config["image_threshold"])
            clean_score = float(np.max(clean_probability))
            edited_score = float(np.max(edited_probability))
            no_reference_rows.append({
                "name": "Supervised U-Net pixel localizer",
                "clean_flag": int(clean_score >= image_threshold),
                "edited_flag": int(edited_score >= image_threshold),
                "detail": (f"Edited max pixel probability {edited_score:.4f}; image threshold "
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


st.title("DICOM Security Demo Lab")
st.caption("Preloaded CPAISD brain CT · Interactive metadata and pixel edits · All detector outputs in one view")

if not DATASET_DIR.is_dir():
    st.error(f"Preloaded CPAISD test folder was not found: {DATASET_DIR}")
    st.stop()
missing = [path for path in (IF_MODEL, AE_MODEL, PIXEL_MODEL, PIXEL_CALIBRATION) if not path.is_file()]
if missing:
    st.error("A trained model or calibration file is missing from the project folder:")
    for path in missing:
        st.write(f"- `{path.relative_to(PROJECT_ROOT)}`")
    st.stop()

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

overview_tab, live_tab, results_tab = st.tabs(["Project overview", "Interactive attack lab", "Experiments completed"])

with overview_tab:
    st.subheader("What this demo does")
    st.write("Pick one preloaded CT slice, apply a controlled metadata edit, pixel edit, or both, and compare the clean and edited copies across the project's implemented integrity and anomaly checks.")
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
        run_demo = st.button("Apply change and show everything", type="primary", use_container_width=True)

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
