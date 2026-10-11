"""Optional same-frame registered-ROI SAM3/FoundPose tabletop key producer.

Stock AutoDex still acquires and saves each key frame. This producer derives
new masks and poses from those *same decoded pixels* using a socket-only
baseline bound to the opened session. A failed ROI never falls back to a
stock full-frame mask: the downstream AutoDex quality/refinement gate must
see only one clearly named perception source for a trial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np

from .frame_provenance import image_sha256, verify_frame_provenance
from .live_capture import KeyCaptureInput
from .live_session_runner import OpenedPrecisionSession
from .registered_key_roi import (
    RegisteredKeyROILimits, detector_from_board_config,
    propose_registered_key_roi, segment_registered_key_roi,
)
from .session_key_baseline import (
    SessionKeyBaseline, load_bound_session_key_baseline,
)


_METHOD = "same_session_registered_roi_sam3_foundpose"


@dataclass(frozen=True)
class RegisteredKeyAssistance:
    baseline: SessionKeyBaseline
    board_detector: Callable
    board_id: str
    limits: RegisteredKeyROILimits
    sam_segmentor: object
    foundpose_model: object
    board_config_path: Path
    board_config_sha256: str


def prepare_registered_key_assistance(
    *, opened: OpenedPrecisionSession, socket_capture_id: str,
    calibrated_camera_ids: set[str], board_config_path: Path,
    board_id: str, limits: RegisteredKeyROILimits,
    sam_segmentor, foundpose_model,
) -> RegisteredKeyAssistance:
    """Reuse already-loaded AutoDex SAM3/FoundPose objects, never reload them.

    The caller owns model lifetime (usually one model per independent demo
    process). This avoids shadow implementations of either model or of the
    stock multi-view IoU/silhouette selector. The model object must be for
    the selected key; the robot's GPU/process placement remains a deployment
    concern and is not inferred here.
    """
    if not isinstance(limits, RegisteredKeyROILimits):
        raise TypeError("registered key assistance needs explicit ROI limits")
    limits.validate()
    if (not callable(getattr(sam_segmentor, "segment", None)) or
            not callable(getattr(foundpose_model, "estimate_one_view", None)) or
            getattr(foundpose_model, "obj_name", None) !=
            opened.runner.mode.key_object):
        raise ValueError("assistance needs loaded SAM3 and matching key FoundPose")
    board_config = Path(board_config_path).expanduser().resolve()
    board_bytes = board_config.read_bytes()
    detector = detector_from_board_config(
        board_config_path=board_config, board_id=board_id)
    if board_config.read_bytes() != board_bytes:
        raise ValueError("ChArUco board config changed during detector creation")
    baseline = load_bound_session_key_baseline(
        opened=opened, socket_capture_id=socket_capture_id,
        calibrated_camera_ids=calibrated_camera_ids,
        minimum_views=2)
    return RegisteredKeyAssistance(
        baseline, detector, board_id, limits, sam_segmentor, foundpose_model,
        board_config, hashlib.sha256(board_bytes).hexdigest())


def _stock_summary(capture: KeyCaptureInput, serial: str) -> dict:
    entry = capture.masks.get(serial, {})
    mask = entry.get("mask")
    pose = capture.poses.get(serial, {})
    return {
        "mask_pixels": (None if mask is None else int(np.count_nonzero(mask))),
        "pose_ok": (None if pose.get("ok") is None else bool(pose["ok"])),
        "pose_quality": (None if pose.get("quality") is None else
                         float(pose["quality"])),
        "source": "stock_full_frame_sam3_foundpose_not_selected",
    }


def derive_registered_key_capture(
    *, capture: KeyCaptureInput, assistance: RegisteredKeyAssistance,
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_full: Mapping[str, np.ndarray], key_object: str,
) -> tuple[KeyCaptureInput, dict]:
    """Replace stock model outputs on an immutable same-exposure key capture.

    The full undistorted frame is passed to FoundPose with a full-frame ROI
    mask; no crop-local camera matrix or hidden pixel coordinate transform is
    introduced. Per-view ROI/SAM/FoundPose failures become empty rejected
    views, never stock fallback poses.
    """
    if (not isinstance(capture, KeyCaptureInput) or
            not isinstance(assistance, RegisteredKeyAssistance) or
            getattr(assistance.foundpose_model, "obj_name", None) != key_object or
            not isinstance(key_object, str) or not key_object):
        raise ValueError("registered capture needs the selected key model")
    if (not capture.images_bgr or
            not set(capture.images_bgr) <= set(intrinsics_undist) or
            set(intrinsics_undist) != set(extrinsics_full)):
        raise ValueError("registered capture cameras differ from calibrated rig")
    frame_ids = {}
    for serial, image in capture.images_bgr.items():
        mask = capture.masks.get(serial, {})
        pose = capture.poses.get(serial, {})
        if (mask.get("frame_id") != pose.get("frame_id") or
                any(row.get("image_sha256") != image_sha256(image) or
                    row.get("image_space") !=
                    "autodex_undistorted_full_frame"
                    for row in (mask, pose))):
            raise ValueError("stock capture source frame metadata is inconsistent")
        frame_ids[serial] = mask.get("frame_id")
    verified = verify_frame_provenance(
        {"request_id": capture.request_id,
         "source": capture.frame_timestamp_source,
         "frames": capture.frame_evidence},
        request_id=capture.request_id, images_bgr=capture.images_bgr,
        frame_ids=frame_ids)
    if capture.frame_timestamps_s != {
            serial: row["timestamp_s"] for serial, row in verified.items()}:
        raise ValueError("registered capture timestamps differ from source frames")
    masks, poses, per_view = {}, {}, {}
    for serial, image in sorted(capture.images_bgr.items()):
        current = verified[serial]
        source = {
            "frame_id": current["frame_id"],
            "image_sha256": current["image_sha256"],
            "image_space": "autodex_undistorted_full_frame",
            "perception_method": _METHOD,
            "baseline_session_manifest_sha256": (
                assistance.baseline.session_manifest_sha256),
            "baseline_socket_capture_id": (
                assistance.baseline.socket_capture_id),
        }
        mask = np.zeros(image.shape[:2], dtype=bool)
        pose = {**source, "ok": False}
        diagnosis = {"stock_full_frame": _stock_summary(capture, serial)}
        if serial not in assistance.baseline.images_bgr:
            diagnosis["status"] = "no_admitted_same_camera_baseline"
        else:
            baseline = assistance.baseline.images_bgr[serial]
            if baseline.shape != image.shape:
                raise ValueError("key and socket baseline have different image geometry")
            if (image_sha256(baseline) !=
                    assistance.baseline.source[serial]["image_sha256"]):
                raise ValueError("socket baseline image changed in memory")
            if (hashlib.sha256(np.ascontiguousarray(
                    assistance.baseline.socket_masks[serial]).tobytes()
                    ).hexdigest() != assistance.baseline.source[serial][
                        "mask_pixels_sha256"]):
                raise ValueError("socket baseline mask changed in memory")
            proposal = propose_registered_key_roi(
                current_bgr=image, socket_only_bgr=baseline,
                socket_only_mask=assistance.baseline.socket_masks[serial],
                board_detector=assistance.board_detector,
                board_id=assistance.board_id, limits=assistance.limits)
            diagnosis["roi"] = proposal.to_record()
            if proposal.status == "roi_available":
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                segmented = segment_registered_key_roi(
                    current_rgb=rgb, proposal=proposal,
                    sam_segmentor=assistance.sam_segmentor,
                    prompt=capture.prompt, limits=assistance.limits)
                diagnosis["mask_qa"] = segmented.to_record()
                if segmented.status == "mask_available":
                    mask = segmented.mask
                    result = assistance.foundpose_model.estimate_one_view(
                        image_rgb=rgb, mask_bool=mask,
                        K=np.asarray(intrinsics_undist[serial]),
                        ext_cw=np.asarray(extrinsics_full[serial]))
                    if result is not None:
                        pose = {
                            **source, "ok": True,
                            "pose_world": result["pose_world"],
                            "quality": float(result["quality"]),
                            "inliers": int(result["inliers"]),
                            "template_id": int(result.get("template_id", -1)),
                            "mask_pixels": int(mask.sum()),
                        }
            diagnosis["status"] = (
                "foundpose_available" if pose["ok"] else
                "registered_roi_view_rejected")
            diagnosis["baseline"] = assistance.baseline.source[serial]
        masks[serial] = {**source, "mask": mask}
        poses[serial] = pose
        per_view[serial] = diagnosis
    derived = KeyCaptureInput(
        capture.capture_id, capture.request_id, capture.prompt,
        dict(capture.images_bgr), masks, poses,
        dict(capture.frame_timestamps_s), capture.frame_timestamp_source,
        dict(capture.frame_evidence), capture.capture_dir)
    report = {
        "schema": "precision_insertion_registered_key_capture_derivation_v1",
        "perception_method": _METHOD,
        "key_object": key_object,
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "baseline": assistance.baseline.to_record(),
        "board_id": assistance.board_id,
        "board_config_path": str(assistance.board_config_path),
        "board_config_sha256": assistance.board_config_sha256,
        "roi_limits": asdict(assistance.limits),
        "per_view": per_view,
        "scope": "assisted_same_frame_masks_and_poses_not_robot_permission",
        "robot_ready": False,
    }
    return derived, report
