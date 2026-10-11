"""Reuse the open AutoDex cameras for raw multi-view VLM checkpoints.

The default snapshot path requires no post-grasp key pose or segmentation.
It retains the original decoded JPEG frames and their calibrated undistorted
derivatives. Robot state must still be sampled at frame exposure by the
robot-PC feedback buffer. The older key-init request remains opt-in.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import cv2
import numpy as np

from .frame_provenance import image_sha256
from .live_capture import collect_board_snapshot, collect_key_capture
from .raw_camera_capture import PHASES, RawCameraCapture


def capture_snapshot_checkpoint_frames(
    live: object, *, phase: str, timeout_s: float,
    capture_id: str | None = None,
    snapshot_collector: Callable = collect_board_snapshot,
) -> RawCameraCapture:
    """Get raw VLM frames without asking SAM/FoundPose to recognize a held key.

    The stock snapshot returns distorted JPEG pixels. Replay the session's
    calibrated OpenCV undistortion and retain the original images, exposure
    metadata and matrices so the saved VLM bundle can recheck derivation.
    """
    if phase not in PHASES:
        raise ValueError("unknown AutoDex checkpoint phase")
    cameras = live.cameras
    launch = cameras.launch
    source = snapshot_collector(
        snapshot_orchestrator=cameras.snapshot_orchestrator,
        calibrated_camera_ids=set(launch.camera_ids),
        acquisition_metadata_for_request=(
            cameras.acquisition_metadata_for_request),
        timeout_s=timeout_s)
    if (set(source.images_bgr) != set(launch.camera_ids) or
            set(source.frame_evidence) != set(launch.camera_ids)):
        raise ValueError("snapshot differs from the open AutoDex cameras")
    undistorted = {}
    calibration = {}
    derived_evidence = {}
    for camera, original in source.images_bgr.items():
        row = launch.intrinsics_full[camera]
        K_orig = np.asarray(row["K_orig"], dtype=np.float64)
        K_undist = np.asarray(row["K_undist"], dtype=np.float64)
        distortion = np.asarray(row["dist_params"], dtype=np.float64)
        image = cv2.undistort(original, K_orig, distortion, None, K_undist)
        undistorted[camera] = image
        calibration[camera] = {
            "K_orig": K_orig, "K_undist": K_undist,
            "dist_params": distortion}
        derived_evidence[camera] = {
            **source.frame_evidence[camera],
            "image_sha256": image_sha256(image)}
    source_metadata = {
        "request_id": source.request_id, "source": "camera_acquisition",
        "frames": source.frame_evidence}
    return RawCameraCapture(
        capture_id=capture_id or f"{phase}_{uuid4().hex}",
        request_id=source.request_id, images_bgr=undistorted,
        frame_ids={camera: row["frame_id"]
                   for camera, row in source.frame_evidence.items()},
        acquisition_metadata={
            "request_id": source.request_id,
            "source": "camera_acquisition", "frames": derived_evidence},
        source_images_bgr=source.images_bgr,
        source_acquisition_metadata=source_metadata,
        undistort_calibration=calibration)


def capture_checkpoint_frames(
    live: object, *, phase: str, key_prompt: str,
    timeout_s: float, image_write_timeout_s: float = 5.0,
    capture_id: str | None = None,
    collector: Callable = collect_key_capture,
) -> RawCameraCapture:
    """Acquire one new raw multi-view checkpoint from the live camera rig."""
    if phase not in PHASES:
        raise ValueError("unknown AutoDex checkpoint phase")
    if not isinstance(key_prompt, str) or not key_prompt.strip():
        raise ValueError("checkpoint capture needs the selected key prompt")
    cameras = live.cameras
    launch = cameras.launch
    capture = collector(
        init_orchestrator=cameras.init_orchestrator,
        key_object=launch.mode.key_object,
        capture_id=capture_id or f"{phase}_{uuid4().hex}",
        key_prompt=key_prompt, capture_root=launch.capture_root,
        calibrated_camera_ids=set(launch.camera_ids),
        acquisition_metadata_for_request=(
            cameras.acquisition_metadata_for_request),
        timeout_s=timeout_s,
        image_write_timeout_s=image_write_timeout_s)
    metadata = cameras.acquisition_metadata_for_request(capture.request_id)
    if (set(capture.images_bgr) != set(launch.camera_ids) or
            set(capture.frame_evidence) != set(launch.camera_ids) or
            metadata.get("frames") != capture.frame_evidence):
        raise ValueError("checkpoint frames do not match the open camera session")
    return RawCameraCapture(
        capture_id=capture.capture_id,
        request_id=capture.request_id,
        images_bgr=capture.images_bgr,
        frame_ids={camera: row["frame_id"]
                   for camera, row in capture.frame_evidence.items()},
        acquisition_metadata=metadata)


def make_checkpoint_capture(
    *, phase: str, timeout_s: float, key_prompt: str | None = None,
    image_write_timeout_s: float = 5.0,
    transport: str = "snapshot",
) -> Callable:
    """Return the `(live, runner, decision)` callback used by live handlers."""
    if phase not in PHASES:
        raise ValueError("unknown AutoDex checkpoint phase")
    if transport not in {"snapshot", "init_key"}:
        raise ValueError("checkpoint transport must be snapshot or init_key")
    if transport == "init_key" and (
            not isinstance(key_prompt, str) or not key_prompt.strip()):
        raise ValueError("init_key checkpoint needs a key-specific prompt")

    def capture(live, _runner, _decision) -> RawCameraCapture:
        if transport == "snapshot":
            return capture_snapshot_checkpoint_frames(
                live, phase=phase, timeout_s=timeout_s)
        return capture_checkpoint_frames(
            live, phase=phase, key_prompt=key_prompt,
            timeout_s=timeout_s,
            image_write_timeout_s=image_write_timeout_s)

    return capture
