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
                    "final_or_abort"})


@dataclass(frozen=True)
class RawCameraCapture:
    capture_id: str
    request_id: int
    images_bgr: Mapping[str, np.ndarray]
    frame_ids: Mapping[str, int]
    acquisition_metadata: Mapping


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
    return report
