"""Persist provenance-bound AutoDex camera frames without requiring FoundPose.

The camera-side producer must supply trustworthy exposure timestamps and frame
IDs. Saving/verifying these bytes does not establish object pose or motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .frame_provenance import verify_frame_provenance
from .session_bootstrap import _safe_id


PHASES = frozenset({"after_lift", "preinsert", "post_lateral_hold",
                    "final_or_abort", "final_clear_view"})


@dataclass(frozen=True)
class RawCameraCapture:
    capture_id: str
    request_id: int
    images_bgr: Mapping[str, np.ndarray]
    frame_ids: Mapping[str, int]
    acquisition_metadata: Mapping
    source_images_bgr: Mapping[str, np.ndarray] | None = None
    source_acquisition_metadata: Mapping | None = None
    undistort_calibration: Mapping[str, Mapping] | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_raw_camera_capture(
    capture: RawCameraCapture, output_dir: Path, *, phase: str,
) -> Path:
    """Save one same-request set of full-frame, undistorted camera images."""
    if phase not in PHASES:
        raise ValueError("unknown raw camera capture phase")
    if not isinstance(capture, RawCameraCapture):
        raise TypeError("capture must be a RawCameraCapture")
    _safe_id(capture.capture_id, "raw capture ID")
    if type(capture.request_id) is not int or capture.request_id <= 0:
        raise ValueError("raw capture needs a positive request ID")
    if len(capture.images_bgr) < 2:
        raise ValueError("raw capture needs at least two calibrated views")
    for camera in capture.images_bgr:
        _safe_id(camera, "camera ID")
    evidence = verify_frame_provenance(
        capture.acquisition_metadata, request_id=capture.request_id,
        images_bgr=capture.images_bgr, frame_ids=capture.frame_ids)
    derived = (capture.source_images_bgr is not None or
               capture.source_acquisition_metadata is not None or
               capture.undistort_calibration is not None)
    source_evidence = None
    calibrations = None
    if derived:
        if (capture.source_images_bgr is None or
                capture.source_acquisition_metadata is None or
                capture.undistort_calibration is None or
                set(capture.source_images_bgr) != set(capture.images_bgr) or
                set(capture.undistort_calibration) != set(capture.images_bgr)):
            raise ValueError("undistorted capture needs all original frames and calibration")
        source_evidence = verify_frame_provenance(
            capture.source_acquisition_metadata,
            request_id=capture.request_id,
            images_bgr=capture.source_images_bgr,
            frame_ids=capture.frame_ids)
        calibrations = {}
        for camera, image in capture.images_bgr.items():
            if any(evidence[camera][key] != source_evidence[camera][key]
                   for key in ("frame_id", "timestamp_s", "max_error_s",
                               "timestamp_method", "clock_domain")):
                raise ValueError("derived frame changed source exposure identity")
            row = capture.undistort_calibration[camera]
            original = np.asarray(row["K_orig"], dtype=float)
            target = np.asarray(row["K_undist"], dtype=float)
            distortion = np.asarray(row["dist_params"], dtype=float).reshape(-1)
            if (original.shape != (3, 3) or target.shape != (3, 3) or
                    not np.all(np.isfinite(original)) or
                    not np.all(np.isfinite(target)) or
                    not np.all(np.isfinite(distortion)) or
                    not np.array_equal(cv2.undistort(
                        capture.source_images_bgr[camera], original,
                        distortion, None, target), image)):
                raise ValueError("undistorted frame differs from original snapshot")
            calibrations[camera] = {
                "K_orig": original.tolist(), "K_undist": target.tolist(),
                "dist_params": distortion.tolist()}
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    image_dir = target / "images"
    image_dir.mkdir()
    hashes = {}
    for camera, image in sorted(capture.images_bgr.items()):
        path = image_dir / f"{camera}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not save raw camera frame: {path}")
        hashes[camera] = _sha(path)
    source_hashes = None
    if derived:
        source_dir = target / "source_original"
        source_dir.mkdir()
        source_hashes = {}
        for camera, image in sorted(capture.source_images_bgr.items()):
            path = source_dir / f"{camera}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"could not save original snapshot: {path}")
            source_hashes[camera] = _sha(path)
    manifest = {
        "schema": "precision_insertion_raw_capture_v1",
        "phase": phase,
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "image_space": "autodex_undistorted_full_frame",
        "frame_evidence": evidence,
        "image_file_sha256": hashes,
        "scope": "raw_camera_pixels_not_object_pose_or_motion",
        "robot_ready": False,
    }
    if derived:
        manifest["undistort_derivation"] = {
            "source_frame_evidence": source_evidence,
            "source_image_file_sha256": source_hashes,
            "calibration": calibrations,
        }
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return target


def verify_raw_camera_capture(output_dir: Path, *, phase: str) -> dict:
    """Recheck source PNG bytes, decoded pixels, frame IDs and exposure data."""
    if phase not in PHASES:
        raise ValueError("unknown raw camera capture phase")
    root = Path(output_dir).expanduser().resolve()
    report = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or
            report.get("schema") != "precision_insertion_raw_capture_v1" or
            report.get("phase") != phase or
            report.get("image_space") != "autodex_undistorted_full_frame" or
            not isinstance(report.get("frame_evidence"), dict) or
            len(report["frame_evidence"]) < 2 or
            set(report["frame_evidence"]) !=
            set(report.get("image_file_sha256", {}))):
        raise ValueError("invalid raw capture manifest")
    images, frame_ids = {}, {}
    for camera, source in report["frame_evidence"].items():
        path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        if not path.is_file() or _sha(path) != report["image_file_sha256"][camera]:
            raise ValueError(("final " if phase == "final_or_abort" else "") +
                             "raw camera PNG changed")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("raw camera PNG cannot be decoded")
        images[camera] = image
        frame_ids[camera] = source["frame_id"]
    expected = verify_frame_provenance(
        {"request_id": report["request_id"],
         "source": "camera_acquisition", "frames": report["frame_evidence"]},
        request_id=report["request_id"], images_bgr=images,
        frame_ids=frame_ids)
    if expected != report["frame_evidence"]:
        raise ValueError("raw frame evidence changed")
    derivation = report.get("undistort_derivation")
    if derivation is not None:
        if (not isinstance(derivation, dict) or
                set(derivation) != {"source_frame_evidence",
                                    "source_image_file_sha256", "calibration"} or
                any(set(derivation[key]) != set(images) for key in derivation)):
            raise ValueError("invalid original snapshot derivation")
        originals = {}
        for camera, image in images.items():
            source = root / "source_original" / f"{camera}.png"
            if (not source.is_file() or _sha(source) !=
                    derivation["source_image_file_sha256"][camera]):
                raise ValueError("original snapshot PNG changed")
            original = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError("original snapshot PNG cannot be decoded")
            originals[camera] = original
            row = derivation["calibration"][camera]
            rebuilt = cv2.undistort(
                original, np.asarray(row["K_orig"], dtype=float),
                np.asarray(row["dist_params"], dtype=float), None,
                np.asarray(row["K_undist"], dtype=float))
            if not np.array_equal(rebuilt, image):
                raise ValueError("raw frame differs from original undistortion")
        source_evidence = verify_frame_provenance(
            {"request_id": report["request_id"],
             "source": "camera_acquisition",
             "frames": derivation["source_frame_evidence"]},
            request_id=report["request_id"], images_bgr=originals,
            frame_ids=frame_ids)
        if source_evidence != derivation["source_frame_evidence"]:
            raise ValueError("original snapshot exposure identity changed")
        for camera in images:
            if any(source_evidence[camera][key] != expected[camera][key]
                   for key in ("frame_id", "timestamp_s", "max_error_s",
                               "timestamp_method", "clock_domain")):
                raise ValueError("raw exposure differs from source snapshot")
    return report
