"""Qualify AutoDex per-view FoundPose socket evidence before calibration.

``InitOrchestrator.collect_payloads`` already returns per-camera SAM masks and
FoundPose poses. Its payload ``ts`` is *publication* time after inference,
not image acquisition time (see ``src/execution/daemon/init_daemon.py``).
This adapter therefore requires independently supplied frame timestamps and
never treats publication time or one request ID as measured synchronization.
It performs no camera I/O or robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

from .calibration import SocketObservation
from .geometry import validate_se3


@dataclass(frozen=True)
class SocketViewLimits:
    minimum_mask_pixels: int
    minimum_foundpose_quality: float
    minimum_foundpose_inliers: int
    minimum_border_clearance_px: int
    maximum_capture_skew_s: float
    minimum_accepted_views: int = 2

    def validate(self) -> None:
        if (self.minimum_mask_pixels < 1 or
                self.minimum_foundpose_inliers < 1 or
                self.minimum_border_clearance_px < 0 or
                self.minimum_accepted_views < 2 or
                not math.isfinite(self.minimum_foundpose_quality) or
                self.minimum_foundpose_quality <= 0 or
                not math.isfinite(self.maximum_capture_skew_s) or
                self.maximum_capture_skew_s <= 0):
            raise ValueError("socket view limits must be positive and commissioned")


@dataclass(frozen=True)
class SocketCaptureEvidence:
    capture_id: str
    observations: tuple[SocketObservation, ...]
    per_view: dict[str, dict]
    frame_timestamp_source: str

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_socket_capture_admission_v1",
            "capture_id": self.capture_id,
            "frame_timestamp_source": self.frame_timestamp_source,
            "accepted_camera_ids": [view.camera_id for view in self.observations],
            "per_view": self.per_view,
            "scope": "per_view_quality_gate_not_absolute_pose_accuracy",
            "robot_ready": False,
        }


def _positive_time(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive capture time") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive capture time")
    return result


def admit_socket_capture(
    *, capture_id: str, masks: Mapping[str, dict],
    poses: Mapping[str, dict], frame_timestamps_s: Mapping[str, float],
    frame_timestamp_source: str, calibrated_camera_ids: set[str],
    limits: SocketViewLimits,
) -> SocketCaptureEvidence:
    """Accept only well-observed per-camera socket poses for one capture.

    The timestamp mapping must come from actual acquisition metadata on the
    same clock across cameras. Existing AutoDex ``ts`` values are rejected as
    a source because processing can finish seconds apart. A good per-view
    result still needs multi-capture fixture consistency in ``calibrate_session``.
    """
    limits.validate()
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ValueError("nonempty capture_id is required")
    if frame_timestamp_source != "camera_acquisition":
        raise ValueError("socket calibration requires camera acquisition timestamps")
    if not isinstance(masks, Mapping) or not isinstance(poses, Mapping):
        raise TypeError("AutoDex masks and poses must be per-camera mappings")
    if not calibrated_camera_ids or not all(
            isinstance(serial, str) and serial for serial in calibrated_camera_ids):
        raise ValueError("calibrated camera IDs are required")
    if not isinstance(frame_timestamps_s, Mapping) or not frame_timestamps_s:
        raise ValueError("per-camera acquisition timestamps are required")
    if set(frame_timestamps_s) - calibrated_camera_ids:
        raise ValueError("acquisition timestamp contains an uncalibrated camera")
    frame_times = {
        serial: _positive_time(value, f"capture time {serial}")
        for serial, value in frame_timestamps_s.items()
    }
    per_view: dict[str, dict] = {}
    admitted: list[SocketObservation] = []
    for serial in sorted(calibrated_camera_ids):
        reasons: list[str] = []
        mask_entry = masks.get(serial)
        pose_entry = poses.get(serial)
        mask_pixels = None
        quality = None
        inliers = None
        border_clearance = None
        if serial not in frame_times:
            reasons.append("missing_camera_acquisition_time")
        if not isinstance(mask_entry, Mapping):
            reasons.append("missing_sam_mask")
        else:
            mask = mask_entry.get("mask")
            if not isinstance(mask, np.ndarray) or mask.ndim != 2:
                reasons.append("invalid_sam_mask")
            else:
                binary = np.asarray(mask, dtype=bool)
                mask_pixels = int(np.count_nonzero(binary))
                if mask_pixels < limits.minimum_mask_pixels:
                    reasons.append("mask_too_small")
                if mask_pixels:
                    y, x = np.nonzero(binary)
                    border_clearance = int(min(
                        x.min(), y.min(), binary.shape[1] - 1 - x.max(),
                        binary.shape[0] - 1 - y.max()))
                    if border_clearance < limits.minimum_border_clearance_px:
                        reasons.append("mask_touches_image_border")
        pose = None
        if not isinstance(pose_entry, Mapping) or pose_entry.get("ok") is not True:
            reasons.append("missing_foundpose_pose")
        else:
            try:
                pose = validate_se3(pose_entry.get("pose_world"),
                                    name=f"FoundPose {serial} pose_world")
            except (TypeError, ValueError):
                reasons.append("invalid_foundpose_pose")
            try:
                quality = float(pose_entry["quality"])
                inliers = int(pose_entry["inliers"])
            except (KeyError, TypeError, ValueError, OverflowError):
                reasons.append("missing_foundpose_quality")
            else:
                if (not math.isfinite(quality) or
                        quality < limits.minimum_foundpose_quality):
                    reasons.append("low_foundpose_quality")
                if inliers < limits.minimum_foundpose_inliers:
                    reasons.append("too_few_foundpose_inliers")
            if mask_pixels is not None and pose_entry.get("mask_pixels") is not None:
                try:
                    declared_pixels = int(pose_entry["mask_pixels"])
                except (TypeError, ValueError, OverflowError):
                    reasons.append("invalid_pose_mask_pixel_count")
                else:
                    if declared_pixels != mask_pixels:
                        reasons.append("mask_pose_pixel_count_mismatch")
        accepted = not reasons
        per_view[serial] = {
            "accepted": accepted, "reasons": reasons,
            "frame_capture_timestamp_s": frame_times.get(serial),
            "mask_publish_timestamp_s": (
                mask_entry.get("ts") if isinstance(mask_entry, Mapping) else None),
            "pose_publish_timestamp_s": (
                pose_entry.get("ts") if isinstance(pose_entry, Mapping) else None),
            "mask_pixels": mask_pixels, "border_clearance_px": border_clearance,
            "foundpose_quality": quality, "foundpose_inliers": inliers,
        }
        if accepted:
            admitted.append(SocketObservation(
                capture_id, serial, frame_times[serial], pose.copy(),
                "camera_acquisition"))
    if len(admitted) < limits.minimum_accepted_views:
        raise ValueError(
            f"socket capture {capture_id}: {len(admitted)} accepted views < "
            f"{limits.minimum_accepted_views}; per-view reasons: "
            + repr({serial: row["reasons"] for serial, row in per_view.items()}))
    accepted_times = [item.timestamp_s for item in admitted]
    if max(accepted_times) - min(accepted_times) > limits.maximum_capture_skew_s:
        raise ValueError("accepted socket camera frames exceed acquisition skew limit")
    return SocketCaptureEvidence(
        capture_id, tuple(admitted), per_view, frame_timestamp_source)


def collect_and_admit_socket_capture(
    *, orchestrator, socket_object: str, capture_id: str, prompt: str,
    calibrated_camera_ids: set[str], limits: SocketViewLimits,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    timeout_s: float, save_capture_dir: str | None = None,
) -> tuple[SocketCaptureEvidence, dict]:
    """Reuse the unchanged AutoDex collector with an external timing source.

    The provider is a demo-local camera acquisition/frame-ID side channel;
    it must return ``{request_id, source, camera_times_s}``. Existing AutoDex
    mask/pose ``ts`` fields do not satisfy this contract. The collector is
    already live and initialized for the *socket*, not the trial key. This
    function only asks cameras for evidence; it does not calibrate or move.
    """
    if getattr(orchestrator, "obj_name", None) != socket_object:
        raise ValueError("FoundPose orchestrator is not initialized for this socket")
    if not isinstance(prompt, str) or not prompt.strip() or prompt == "object":
        raise ValueError("a socket-specific segmentation prompt is required")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("capture timeout must be positive")
    if (set(getattr(orchestrator, "intrinsics_undist", {})) !=
            calibrated_camera_ids or
            set(getattr(orchestrator, "extrinsics", {})) !=
            calibrated_camera_ids):
        raise ValueError("orchestrator camera IDs differ from calibration")
    masks, poses, timing = orchestrator.collect_payloads(
        prompt=prompt, n_expected_serials=len(calibrated_camera_ids),
        timeout_s=timeout_s, save_capture_dir=save_capture_dir)
    if not isinstance(timing, Mapping) or "request_id" not in timing:
        raise ValueError("AutoDex collector returned no request ID")
    request_id = int(timing["request_id"])
    metadata = acquisition_metadata_for_request(request_id)
    if (not isinstance(metadata, Mapping) or
            metadata.get("request_id") != request_id):
        raise ValueError("acquisition metadata does not match FoundPose request")
    admitted = admit_socket_capture(
        capture_id=capture_id, masks=masks, poses=poses,
        frame_timestamps_s=metadata.get("camera_times_s"),
        frame_timestamp_source=metadata.get("source"),
        calibrated_camera_ids=calibrated_camera_ids, limits=limits)
    return admitted, dict(timing)
