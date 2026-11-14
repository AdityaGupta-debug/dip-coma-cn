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


DATASET_DIR = PROJECT_ROOT / "data" / "cpaisd" / "test"
IF_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "isolation_forest.joblib"
AE_MODEL = PROJECT_ROOT / "experiment_cpaisd" / "models" / "metadata_autoencoder.joblib"
PIXEL_MODEL = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "no_reference_pixel_model.joblib"
PIXEL_CALIBRATION = PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "calibration_and_clean_summary.json"

st.set_page_config(page_title="DICOM Security Demo Lab", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner="Loading the preloaded CPAISD test fold...")
def get_dicom_paths(root_string: str) -> tuple[str, ...]:
    return tuple(str(path) for path in find_dicom_files(root_string))


@st.cache_resource(show_spinner=False)
def load_pixel_bundle(path_string: str):
    return joblib.load(path_string)


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
                  pixel_model_rows: list[dict], clean_layout: list[str], edited_layout: list[str]):
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
    add("Isolation Forest", int(clean_row.isolation_forest_flag), int(edited_row.isolation_forest_flag),
        f"Score {edited_row.isolation_forest_score:.5f}; threshold {edited_row.isolation_forest_threshold:.5f}")
    add("Metadata autoencoder", int(clean_row.autoencoder_flag), int(edited_row.autoencoder_flag),
        f"Reconstruction error {edited_row.autoencoder_reconstruction_error:.6g}; threshold {edited_row.autoencoder_threshold:.6g}")
    for item in pixel_model_rows:
        add(item["name"], item["clean_flag"], item["edited_flag"], item["detail"])
    add("Pixel/header consistency", int(bool(clean_layout)), int(bool(edited_layout)),
        "; ".join(edited_layout) if edited_layout else "Do pixel layout and header dimensions agree?")
    clean_or = int(bool(clean_row.sha256_flag or clean_pixel_flag or clean_row.rule_flag
                         or clean_row.isolation_forest_flag or clean_row.autoencoder_flag
                         or any(x["clean_flag"] for x in pixel_model_rows) or clean_layout))
    edited_or = int(bool(edited_row.sha256_flag or pixel_hash_changed or edited_row.rule_flag
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
        if edit_mode in {"Pixels only", "Metadata + pixels"}:
            attack_name = "localized_patch" if pixel_edit == "Localized patch (5x5, +32)" else "low_amplitude"
            changed_pixel_count, _ = _edit_pixels(
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
        rows = detector_rows(
            clean_detection, edited_detection, pixel_hash_changed,
            int(clean_detection.pixel_integrity_flag or 0), int(edited_detection.pixel_integrity_flag or 0),
            no_reference_rows, clean_layout, edited_layout,
        )
        clean_norm, clean_mask = image_masks[0]
        edited_norm, edited_mask = image_masks[1]
        return {
            "edit_description": edit_description,
            "changed_pixel_count": changed_pixel_count,
            "difference": difference,
            "clean_pixels": clean_pixels,
            "edited_pixels": edited_pixels,
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
        }


def show_saved_results():
    results = [
        ("Metadata experiment", PROJECT_ROOT / "experiment_cpaisd" / "reports" / "macro_average_metrics.csv"),
        ("Metadata clean-test alarms", PROJECT_ROOT / "experiment_cpaisd" / "reports" / "clean_test_false_positive_rates.csv"),
        ("Pixel integrity and masks", PROJECT_ROOT / "experiment_cpaisd_pixel_masks" / "pixel_metrics.csv"),
        ("No-reference pixel detector", PROJECT_ROOT / "experiment_cpaisd_no_reference_v2" / "no_reference_pixel_metrics.csv"),
        ("Header-to-pixel consistency", PROJECT_ROOT / "experiment_cpaisd_cross_layer" / "cross_layer_summary.csv"),
        ("Adaptive feature evasion", PROJECT_ROOT / "experiment_cpaisd_adaptive_evasion" / "adaptive_evasion_summary.csv"),
        ("Benign changes versus attacks", PROJECT_ROOT / "experiment_cpaisd_benign_attack_v3" / "benign_vs_attack_summary.csv"),
        ("Combined metadata + pixel checks", PROJECT_ROOT / "experiment_cpaisd_combined_independent" / "combined_metrics.csv"),
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
    st.metric("Available checks", "9")

overview_tab, live_tab, results_tab = st.tabs(["Project overview", "Interactive attack lab", "Experiments completed"])

with overview_tab:
    st.subheader("What this demo does")
    st.write("Pick one preloaded CT slice, apply a controlled metadata edit, pixel edit, or both, and compare the clean and edited copies across the project's implemented integrity and anomaly checks.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preloaded CT slices", len(dicom_paths))
    c2.metric("Metadata checks", "4")
    c3.metric("Pixel checks", "4")
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
        - **Isolation Forest and autoencoder:** metadata anomaly scores learned from clean training data.
        - **No-reference pixel Isolation Forest and median residual:** image-only anomaly baselines that do not compare against a clean image.
        - **Pixel/header consistency:** checks whether image payload layout agrees with DICOM header declarations.
        - **All-checks OR summary:** combines whether any of the above checks alert.
        """)

with live_tab:
    st.subheader("Choose an edit; see the full detector response")
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
            st.subheader("Every check, together")
            st.dataframe(result["detectors"], use_container_width=True, hide_index=True)
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
            st.caption("The difference map uses the clean reference. The red map is a separate no-reference baseline; it may miss edits and can flag normal image structure. Display scaling is for visualization, not Hounsfield-unit measurement.")

with results_tab:
    st.subheader("Previously completed research experiments")
    st.write("These are saved outputs from the project's experiment runs, alongside the interactive one-slice demonstration.")
    show_saved_results()
