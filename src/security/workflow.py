"""Cryptographic-first single-file and directory verification workflows."""
from __future__ import annotations

import json
from pathlib import Path

import pydicom

from .manifest import verify_signed_manifest


def _image_threshold_for_checkpoint(model, config: dict) -> tuple[float, str]:
    """Read the configured score, recalibrating legacy probability thresholds when possible."""
    score_type = config.get("image_score_type")
    if score_type == "maximum_pre_sigmoid_logit":
        return float(config.get("image_threshold", 0.0)), score_type
    validation_dir = Path(__file__).resolve().parents[2] / "data" / "cpaisd" / "val"
    if validation_dir.is_dir():
        from ..dicom_io import find_dicom_files
        from ..supervised_localization import calibrate_clean_image_threshold
        paths = find_dicom_files(validation_dir)
        if paths:
            threshold = calibrate_clean_image_threshold(
                model, paths, int(config.get("image_size", 256)), quantile=99.0,
            )
            return threshold, "maximum_pre_sigmoid_logit"
    return float(config.get("image_threshold", 1.0)), "legacy_max_probability"


def verify_then_analyze(dicom_path: str | Path, manifest_path: str | Path,
                        trust_registry_path: str | Path, *, if_model_path: str | Path | None = None,
                        autoencoder_model_path: str | Path | None = None,
                        pixel_model_path: str | Path | None = None,
                        unet_checkpoint_path: str | Path | None = None,
                        run_ml_on_unverified: bool = False) -> dict:
    """Return separate cryptographic, rules, semantic, IF, and AE results.

    ML is skipped after a cryptographic failure by default. U-Net and patch
    pixel inference are invoked only when their saved model paths are supplied.
    """
    registry = json.loads(Path(trust_registry_path).read_text(encoding="utf-8"))
    crypto = verify_signed_manifest(dicom_path, manifest_path, registry)
    result = {"cryptographic": crypto, "structural": None, "semantic": None,
              "metadata_anomaly": {}, "pixel_anomaly": {"status": "NOT_CONFIGURED"},
              "segmentation": {"status": "NOT_CONFIGURED"}}
    if crypto["status"] != "VERIFIED" and not run_ml_on_unverified:
        result["ml_status"] = "SKIPPED_CRYPTOGRAPHIC_POLICY"
        return result

    from ..validation import validate_datasets
    from ..semantic_consistency import validate_semantic_consistency
    from ..pixel_metadata_consistency import check_pixel_metadata_consistency
    from ..anomaly import load_model, predict_anomalies
    from ..deep_anomaly import predict_autoencoder_with_explanations

    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False, force=False)
    key = Path(dicom_path).name
    items = [(key, ds)]
    result["structural"] = validate_datasets(items)[key]
    result["semantic"] = {"status": "REQUIRES_STUDY_CONTEXT",
                           "reason": "single-file analysis cannot assess peer-slice agreement"}
    result["cross_layer"] = check_pixel_metadata_consistency(ds)

    if if_model_path and Path(if_model_path).is_file():
        try:
            bundle = load_model(str(if_model_path))
            flag, score = predict_anomalies(items, bundle)[key]
            result["metadata_anomaly"]["isolation_forest"] = {"status": "COMPLETED", "flag": int(flag), "score": float(score),
                "threshold": float(bundle.get("threshold", 0.0))}
        except Exception as exc:
            result["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_UNAVAILABLE"}
    if autoencoder_model_path and Path(autoencoder_model_path).is_file():
        try:
            flag, score, features = predict_autoencoder_with_explanations(items, str(autoencoder_model_path))[key]
            result["metadata_anomaly"]["autoencoder"] = {"status": "COMPLETED", "flag": int(flag), "reconstruction_error": float(score),
                                                             "top_features": features}
        except Exception as exc:
            result["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_UNAVAILABLE"}

    import numpy as np
    import torch
    import torch.nn.functional as F
    from ..no_reference_pixel import _normalized_image, _patch_feature_grid, _mask_from_patch_scores

    def mask_summary(mask):
        flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
        counts, current, run = [], 0, 0
        for value in flat:
            if int(value) == current:
                run += 1
            else:
                counts.append(run); run = 1; current = int(value)
        counts.append(run)
        return {"shape": list(np.asarray(mask).shape), "positive_pixels": int(flat.sum()), "rle_counts": counts}

    if pixel_model_path and Path(pixel_model_path).is_file():
        try:
            import joblib
            bundle = joblib.load(pixel_model_path)
            image = _normalized_image(ds)
            features, grid_shape, image_shape = _patch_feature_grid(image, int(bundle.get("patch_size", 16)))
            scores = bundle["model"].score_samples(features)
            threshold = float(bundle["patch_threshold"])
            mask = _mask_from_patch_scores(scores, grid_shape, image_shape, threshold)
            image_score = float(np.min(scores))
            result["pixel_anomaly"] = {"status": "COMPLETED", "flag": int(image_score < float(bundle["file_threshold"])),
                                        "score": image_score, "image_threshold": float(bundle["file_threshold"]),
                                        "mask": mask_summary(mask)}
        except Exception as exc:
            result["pixel_anomaly"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["pixel_anomaly"] = {"status": "MODEL_UNAVAILABLE"}

    if unet_checkpoint_path and Path(unet_checkpoint_path).is_file():
        try:
            from ..supervised_localization import build_localizer
            checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=True)
            config = checkpoint.get("config", {})
            architecture = config.get("architecture", "unet")
            model = build_localizer(architecture, base_channels=int(config.get("base_channels", 24)))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            image = _normalized_image(ds)
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
            tensor = F.interpolate(tensor, size=(int(config.get("image_size", 256)),) * 2,
                                   mode="bilinear", align_corners=False)
            with torch.no_grad():
                logits = model(tensor)
                probability = torch.sigmoid(logits)[0, 0].numpy()
                image_score = float(logits.flatten(start_dim=1).amax(dim=1)[0].item())
            pixel_threshold = float(config.get("pixel_threshold", 0.5))
            image_threshold, image_score_type = _image_threshold_for_checkpoint(model, config)
            mask = probability >= pixel_threshold
            if image_score_type != "maximum_pre_sigmoid_logit":
                image_score = float(probability.max())
            result["segmentation"] = {"status": "COMPLETED", "flag": int(image_score >= image_threshold),
                                      "architecture": architecture,
                                      "image_score": image_score,
                                      "image_score_type": image_score_type,
                                      "image_threshold": image_threshold,
                                      "pixel_threshold": pixel_threshold, "mask": mask_summary(mask),
                                      "model_input_size": list(probability.shape)}
        except Exception as exc:
            result["segmentation"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        result["segmentation"] = {"status": "MODEL_UNAVAILABLE"}
    result["ml_status"] = "COMPLETED"
    return result


def verify_directory_then_analyze(dicom_root: str | Path, manifest_root: str | Path,
                                  trust_registry_path: str | Path, *,
                                  if_model_path: str | Path | None = None,
                                  autoencoder_model_path: str | Path | None = None,
                                  pixel_model_path: str | Path | None = None,
                                  unet_checkpoint_path: str | Path | None = None) -> dict:
    """Verify per-file sidecars first, then analyze verified objects as groups.

    Manifest sidecars are resolved as ``manifest_root / (relative_path +
    '.manifest.json')``. Public UIDs and source paths are not copied into the
    returned records; each result is keyed by a truncated hash of its relative
    path. Group consistency checks operate only on cryptographically verified
    files.
    """
    import hashlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .manifest import verify_signed_manifest
    from ..dicom_io import find_dicom_files, read_dataset, relative_key
    from ..validation import validate_datasets
    from ..semantic_consistency import validate_semantic_consistency
    from ..pixel_metadata_consistency import check_pixel_metadata_consistency
    from ..no_reference_pixel import _normalized_image, _patch_feature_grid, _mask_from_patch_scores

    root, manifest_root = Path(dicom_root).resolve(), Path(manifest_root).resolve()
    registry = json.loads(Path(trust_registry_path).read_text(encoding="utf-8"))
    paths = find_dicom_files(root)
    if not paths:
        raise ValueError(f"No readable DICOM files found under {root}")
    records, verified_items = {}, []
    for path in paths:
        rel = relative_key(path, root)
        token = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:20]
        sidecar = manifest_root / (rel + ".manifest.json")
        crypto = verify_signed_manifest(path, sidecar, registry)
        record = {"cryptographic": crypto, "structural": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "semantic": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "cross_layer": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "metadata_anomaly": {}, "pixel_anomaly": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"},
                  "segmentation": {"status": "SKIPPED_CRYPTOGRAPHIC_POLICY"}}
        records[token] = record
        if crypto["status"] == "VERIFIED":
            ds = read_dataset(path, stop_before_pixels=False)
            verified_items.append((token, ds, path))

    if not verified_items:
        return {"files_discovered": len(paths), "files_verified": 0, "results": records,
                "ml_status": "SKIPPED_NO_VERIFIED_FILES"}

    datasets = [(token, ds) for token, ds, _ in verified_items]
    structural = validate_datasets(datasets)
    semantic = validate_semantic_consistency(datasets)
    for token, ds, _ in verified_items:
        records[token]["structural"] = structural[token]
        records[token]["semantic"] = semantic[token]
        records[token]["cross_layer"] = check_pixel_metadata_consistency(ds)

    if if_model_path and Path(if_model_path).is_file():
        try:
            from ..anomaly import load_model, predict_anomalies
            bundle = load_model(str(if_model_path))
            predictions = predict_anomalies(datasets, bundle)
            for token, _ in datasets:
                flag, score = predictions[token]
                records[token]["metadata_anomaly"]["isolation_forest"] = {
                    "status": "COMPLETED", "flag": int(flag), "score": float(score),
                    "threshold": float(bundle.get("threshold", 0.0))}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["metadata_anomaly"]["isolation_forest"] = {"status": "MODEL_UNAVAILABLE"}

    if autoencoder_model_path and Path(autoencoder_model_path).is_file():
        try:
            from ..deep_anomaly import predict_autoencoder_with_explanations
            predictions = predict_autoencoder_with_explanations(datasets, str(autoencoder_model_path))
            for token, _ in datasets:
                flag, score, features = predictions[token]
                records[token]["metadata_anomaly"]["autoencoder"] = {
                    "status": "COMPLETED", "flag": int(flag), "reconstruction_error": float(score),
                    "top_features": features}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["metadata_anomaly"]["autoencoder"] = {"status": "MODEL_UNAVAILABLE"}

    def mask_summary(mask):
        flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
        counts, current, run = [], 0, 0
        for value in flat:
            if int(value) == current:
                run += 1
            else:
                counts.append(run); run = 1; current = int(value)
        counts.append(run)
        return {"shape": list(np.asarray(mask).shape), "positive_pixels": int(flat.sum()), "rle_counts": counts}

    if pixel_model_path and Path(pixel_model_path).is_file():
        try:
            import joblib
            bundle = joblib.load(pixel_model_path)
            for token, ds, _ in verified_items:
                image = _normalized_image(ds)
                features, grid, shape = _patch_feature_grid(image, int(bundle.get("patch_size", 16)))
                scores = bundle["model"].score_samples(features)
                mask = _mask_from_patch_scores(scores, grid, shape, float(bundle["patch_threshold"]))
                image_score = float(np.min(scores))
                records[token]["pixel_anomaly"] = {"status": "COMPLETED",
                    "flag": int(image_score < float(bundle["file_threshold"])), "score": image_score,
                    "image_threshold": float(bundle["file_threshold"]), "mask": mask_summary(mask)}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["pixel_anomaly"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["pixel_anomaly"] = {"status": "MODEL_UNAVAILABLE"}

    if unet_checkpoint_path and Path(unet_checkpoint_path).is_file():
        try:
            from ..supervised_localization import build_localizer
            checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=True)
            config = checkpoint.get("config", {})
            architecture = config.get("architecture", "unet")
            model = build_localizer(architecture, base_channels=int(config.get("base_channels", 24)))
            model.load_state_dict(checkpoint["state_dict"]); model.eval()
            image_size = int(config.get("image_size", 256))
            pixel_threshold = float(config.get("pixel_threshold", 0.5))
            image_threshold, image_score_type = _image_threshold_for_checkpoint(model, config)
            for token, ds, _ in verified_items:
                image = _normalized_image(ds)
                tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
                tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    logits = model(tensor)
                    probability = torch.sigmoid(logits)[0, 0].numpy()
                    image_score = float(logits.flatten(start_dim=1).amax(dim=1)[0].item())
                mask = probability >= pixel_threshold
                if image_score_type != "maximum_pre_sigmoid_logit":
                    image_score = float(probability.max())
                records[token]["segmentation"] = {"status": "COMPLETED", "architecture": architecture,
                    "flag": int(image_score >= image_threshold),
                    "image_score": image_score,
                    "image_score_type": image_score_type,
                    "image_threshold": image_threshold,
                    "pixel_threshold": pixel_threshold, "mask": mask_summary(mask),
                    "model_input_size": list(probability.shape)}
        except Exception as exc:
            for token, _ in datasets:
                records[token]["segmentation"] = {"status": "MODEL_ERROR", "reason": type(exc).__name__}
    else:
        for token, _ in datasets:
            records[token]["segmentation"] = {"status": "MODEL_UNAVAILABLE"}

    return {"files_discovered": len(paths), "files_verified": len(verified_items),
            "files_not_verified": len(paths) - len(verified_items), "results": records,
            "ml_status": "COMPLETED"}
