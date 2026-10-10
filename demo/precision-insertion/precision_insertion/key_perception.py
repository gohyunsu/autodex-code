"""Admit one fresh multi-view key pose without changing AutoDex FoundPose.

The socket is measured once per session; the key is remeasured every trial.
This module reuses the same per-view SAM/FoundPose quality gate and AutoDex's
IoU/silhouette refinement, then rejects disagreeing key poses. It performs
no camera capture, planning, VLM call, or robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import TaskMode
from .calibration import validate_session_camera_calibration
from .frame_provenance import bounded_capture_skew_s, verify_frame_provenance
from .geometry import pose_angle_deg, validate_se3
from .live_capture import KeyCaptureInput
from .perception_evidence import SocketViewLimits, admit_socket_capture
from .session_bootstrap import _safe_id, _write_json, _write_png
from .socket_exclusion import (
    key_mask_socket_overlap_fraction, project_frozen_socket_exclusion,
)
from .symmetry import load_axial_symmetry


@dataclass(frozen=True)
class KeyPoseObservation:
    capture_id: str
    request_id: int
    key_object: str
    family: str
    pose_world: np.ndarray
    selected_camera_id: str
    selected_acquisition_timestamp_s: float
    acquisition_interval_s: tuple[float, float]
    frame_evidence: dict[str, dict]
    per_view: dict[str, dict]
    consistency: dict
    selection: dict
    source_capture_dir: Path
    phase: str = "tabletop"

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_fresh_key_observation_v1",
            "capture_id": self.capture_id,
            "request_id": self.request_id,
            "key_object": self.key_object,
            "family": self.family,
            "phase": self.phase,
            "pose_world": self.pose_world.tolist(),
            "selected_camera_id": self.selected_camera_id,
            "selected_acquisition_timestamp_s": (
                self.selected_acquisition_timestamp_s),
            "acquisition_interval_s": list(self.acquisition_interval_s),
            "frame_evidence": self.frame_evidence,
            "per_view": self.per_view,
            "consistency": self.consistency,
            "selection": self.selection,
            "source_capture_dir": str(self.source_capture_dir),
            "scope": "fresh_multiview_key_pose_not_absolute_accuracy_or_motion",
            "robot_ready": False,
        }

    def require_state_alignment(
        self, *, state_timestamp_s: float, maximum_skew_s: float,
    ) -> None:
        """Check the measured robot state against *all* camera time bounds."""
        state = float(state_timestamp_s)
        limit = float(maximum_skew_s)
        if (not math.isfinite(state) or not math.isfinite(limit) or limit <= 0 or
                max(abs(state - endpoint)
                    for endpoint in self.acquisition_interval_s) > limit):
            raise ValueError("robot state is not aligned to every key view")


def _cylinder_symmetry(mode: TaskMode, shared_root: Path):
    info = (Path(shared_root).expanduser().resolve() / "object_processing" /
            mode.key_object / "processed_data" / "info")
    symmetry = load_axial_symmetry(info.parent.parent.parent, mode.key_object)
    if not symmetry.end_exchange:
        raise ValueError("cylindrical key must have Dinf end symmetry")
    data = json.loads((info / "symmetry.json").read_text(encoding="utf-8"))
    center = np.asarray(data.get("center"), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("cylinder symmetry center is missing or invalid")
    return center, symmetry.axis_local


def _pose_residual(
    first: np.ndarray, second: np.ndarray,
    *, cylinder_center: np.ndarray | None,
    cylinder_axis: np.ndarray | None,
) -> tuple[float, float]:
    if cylinder_center is None:
        distance = np.linalg.norm(first[:3, 3] - second[:3, 3])
        angle = pose_angle_deg(first, second)
    else:
        first_center = first[:3, :3] @ cylinder_center + first[:3, 3]
        second_center = second[:3, :3] @ cylinder_center + second[:3, 3]
        distance = np.linalg.norm(first_center - second_center)
        first_axis = first[:3, :3] @ cylinder_axis
        second_axis = second[:3, :3] @ cylinder_axis
        cosine = abs(float(np.dot(first_axis, second_axis)))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return float(distance * 1000.0), float(angle)


def _admit_key_capture_common(
    *, capture: KeyCaptureInput, init_orchestrator, mode: TaskMode,
    shared_root: Path, calibration, calibrated_camera_ids: set[str],
    view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
    held_pose_prior_world: np.ndarray | None = None,
    held_pose_prior_timestamp_s: float | None = None,
    held_pose_prior_source: str | None = None,
    maximum_held_prior_center_error_mm: float | None = None,
    maximum_held_prior_angle_error_deg: float | None = None,
    maximum_held_prior_time_skew_s: float | None = None,
    minimum_held_refinement_iou: float | None = None,
) -> KeyPoseObservation:
    """Shared per-view FoundPose gate for tabletop and held-key captures.

    Square poses use full rotation. The D-infinity cylinder compares physical
    centers and *unoriented* axes, so a yaw or identical-end frame flip is not
    mistaken for a new tabletop state. Limits are commissioning inputs.
    """
    held_phase = held_pose_prior_world is not None
    if not held_phase and any(value is not None for value in (
            held_pose_prior_timestamp_s, held_pose_prior_source,
            maximum_held_prior_center_error_mm,
            maximum_held_prior_angle_error_deg,
            maximum_held_prior_time_skew_s,
            minimum_held_refinement_iou)):
        raise ValueError("held-key prior fields require a held pose prior")
    if held_phase:
        prior = validate_se3(held_pose_prior_world,
                             name="held key pose prior_world")
        held_limits = (
            held_pose_prior_timestamp_s,
            maximum_held_prior_center_error_mm,
            maximum_held_prior_angle_error_deg,
            maximum_held_prior_time_skew_s,
            minimum_held_refinement_iou)
        if (held_pose_prior_source !=
                "measured_wrist_plus_observed_held_relation" or
                any(value is None or not math.isfinite(float(value))
                    for value in held_limits) or
                maximum_held_prior_center_error_mm <= 0 or
                maximum_held_prior_angle_error_deg <= 0 or
                maximum_held_prior_time_skew_s <= 0 or
                not 0 < minimum_held_refinement_iou <= 1):
            raise ValueError("held-key prior needs a timed measured-wrist source and limits")
    if not isinstance(capture, KeyCaptureInput):
        raise TypeError("fresh key capture must be KeyCaptureInput")
    if mode.key_object != getattr(init_orchestrator, "obj_name", None):
        raise ValueError("FoundPose is not initialized for selected v8 key")
    if (not capture.images_bgr or
            not set(capture.images_bgr) <= calibrated_camera_ids or
            set(getattr(init_orchestrator, "intrinsics_undist", {})) !=
            calibrated_camera_ids or
            set(getattr(init_orchestrator, "extrinsics", {})) !=
            calibrated_camera_ids):
        raise ValueError("key camera IDs differ from session calibration")
    validate_session_camera_calibration(
        calibration, intrinsics_undist=init_orchestrator.intrinsics_undist,
        extrinsics_full=init_orchestrator.extrinsics,
        calibrated_camera_ids=calibrated_camera_ids)
    if (not math.isfinite(maximum_multiview_center_error_mm) or
            maximum_multiview_center_error_mm <= 0 or
            not math.isfinite(maximum_multiview_angle_error_deg) or
            maximum_multiview_angle_error_deg <= 0 or
            not math.isfinite(maximum_socket_mask_overlap_fraction) or
            not 0 <= maximum_socket_mask_overlap_fraction < 1 or
            type(silhouette_iterations) is not int or
            silhouette_iterations < 0 or
            not math.isfinite(silhouette_loss_threshold) or
            silhouette_loss_threshold <= 0):
        raise ValueError("key consistency/refinement limits must be commissioned")
    frame_ids = {}
    for serial in capture.images_bgr:
        mask_fid = capture.masks.get(serial, {}).get("frame_id")
        pose_fid = capture.poses.get(serial, {}).get("frame_id")
        if mask_fid != pose_fid:
            raise ValueError("key SAM/FoundPose frame IDs differ")
        frame_ids[serial] = mask_fid
    verified = verify_frame_provenance(
        {"request_id": capture.request_id, "source": capture.frame_timestamp_source,
         "frames": capture.frame_evidence},
        request_id=capture.request_id, images_bgr=capture.images_bgr,
        frame_ids=frame_ids)
    if (capture.frame_timestamps_s !=
            {serial: row["timestamp_s"] for serial, row in verified.items()}):
        raise ValueError("key frame times differ from bound images")
    view_limits.validate()
    if bounded_capture_skew_s(verified) > view_limits.maximum_capture_skew_s:
        raise ValueError("key views exceed bounded camera acquisition skew")
    admitted = admit_socket_capture(
        capture_id=capture.capture_id, masks=capture.masks,
        poses=capture.poses, frame_timestamps_s=capture.frame_timestamps_s,
        frame_timestamp_source=capture.frame_timestamp_source,
        calibrated_camera_ids=calibrated_camera_ids, limits=view_limits)
    if mode.family == "cylinder":
        center, axis = _cylinder_symmetry(mode, shared_root)
    elif mode.family == "square":
        center, axis = None, None
    else:
        raise ValueError("unknown precision key family")
    per_view = {serial: dict(row) for serial, row in admitted.per_view.items()}
    admitted_ids = []
    exclusion = None
    max_prior_center = 0.0
    max_prior_angle = 0.0
    if held_phase:
        if max(abs(held_pose_prior_timestamp_s - (
                verified[row.camera_id]["timestamp_s"] + signed_error))
                for row in admitted.observations
                for signed_error in (
                    -verified[row.camera_id]["max_error_s"],
                    verified[row.camera_id]["max_error_s"])) > (
                    maximum_held_prior_time_skew_s):
            raise ValueError("held-key motion prior is stale relative to camera views")
        for row in admitted.observations:
            serial = row.camera_id
            distance, angle = _pose_residual(
                prior, validate_se3(row.pose_world),
                cylinder_center=center, cylinder_axis=axis)
            per_view[serial]["held_prior_center_residual_mm"] = distance
            per_view[serial]["held_prior_angle_residual_deg"] = angle
            if (distance > maximum_held_prior_center_error_mm or
                    angle > maximum_held_prior_angle_error_deg):
                per_view[serial]["accepted"] = False
                per_view[serial]["reasons"] = [
                    *per_view[serial]["reasons"],
                    "held_key_pose_disagrees_with_measured_wrist_prior"]
            else:
                admitted_ids.append(serial)
            max_prior_center = max(max_prior_center, distance)
            max_prior_angle = max(max_prior_angle, angle)
    else:
        exclusion = project_frozen_socket_exclusion(
            mode=mode, shared_root=shared_root, calibration=calibration,
            images_bgr=capture.images_bgr,
            intrinsics_undist=init_orchestrator.intrinsics_undist,
            extrinsics_full=init_orchestrator.extrinsics,
            dilation_px=socket_projection_dilation_px)
        for row in admitted.observations:
            serial = row.camera_id
            overlap = key_mask_socket_overlap_fraction(
                capture.masks[serial]["mask"], exclusion.masks[serial])
            per_view[serial]["key_mask_socket_overlap_fraction"] = overlap
            if overlap > maximum_socket_mask_overlap_fraction:
                per_view[serial]["accepted"] = False
                per_view[serial]["reasons"] = [
                    *per_view[serial]["reasons"],
                    "key_mask_overlaps_frozen_socket_projection"]
            else:
                admitted_ids.append(serial)
    if len(admitted_ids) < view_limits.minimum_accepted_views:
        raise ValueError(
            "too few key views remain after " + (
                "held-wrist pose-prior gate" if held_phase else
                "fixed-socket mask exclusion"))
    poses = {row.camera_id: validate_se3(row.pose_world)
             for row in admitted.observations if row.camera_id in admitted_ids}
    max_center = 0.0
    max_angle = 0.0
    for index, camera in enumerate(admitted_ids):
        for other in admitted_ids[index + 1:]:
            distance, angle = _pose_residual(
                poses[camera], poses[other], cylinder_center=center,
                cylinder_axis=axis)
            max_center = max(max_center, distance)
            max_angle = max(max_angle, angle)
    if (max_center > maximum_multiview_center_error_mm or
            max_angle > maximum_multiview_angle_error_deg):
        raise ValueError("key multiview FoundPose estimates disagree")
    if not callable(getattr(init_orchestrator, "refine_from_payloads", None)):
        raise ValueError("AutoDex IoU/silhouette key selector is unavailable")
    pose, diagnostics = init_orchestrator.refine_from_payloads(
        capture.masks, capture.poses, subset_serials=admitted_ids,
        sil_iters=silhouette_iterations,
        sil_loss_threshold=silhouette_loss_threshold,
        selection_mode="iou")
    if pose is None or not isinstance(diagnostics, dict):
        raise ValueError("AutoDex key pose refinement failed")
    selected = diagnostics.get("best_serial")
    if selected not in admitted_ids:
        raise ValueError("AutoDex selected a non-admitted key camera")
    if held_phase:
        best_iou = diagnostics.get("best_iou")
        if (best_iou is None or not math.isfinite(float(best_iou)) or
                float(best_iou) < minimum_held_refinement_iou):
            raise ValueError("held-key mask/silhouette IoU is below commissioned limit")
    refined = validate_se3(pose, name="refined key pose_world")
    for camera in admitted_ids:
        distance, angle = _pose_residual(
            refined, poses[camera], cylinder_center=center,
            cylinder_axis=axis)
        max_center = max(max_center, distance)
        max_angle = max(max_angle, angle)
        if (distance > maximum_multiview_center_error_mm or
                angle > maximum_multiview_angle_error_deg):
            raise ValueError("refined key pose conflicts with admitted views")
    if held_phase:
        distance, angle = _pose_residual(
            prior, refined, cylinder_center=center, cylinder_axis=axis)
        max_prior_center = max(max_prior_center, distance)
        max_prior_angle = max(max_prior_angle, angle)
        if (distance > maximum_held_prior_center_error_mm or
                angle > maximum_held_prior_angle_error_deg):
            raise ValueError("refined held key pose conflicts with measured wrist prior")
    lower = min(verified[serial]["timestamp_s"] -
                verified[serial]["max_error_s"] for serial in admitted_ids)
    upper = max(verified[serial]["timestamp_s"] +
                verified[serial]["max_error_s"] for serial in admitted_ids)
    selection = {
        "method": "AutoDex_refine_from_payloads_iou",
        "best_serial": selected,
        "best_iou": diagnostics.get("best_iou"),
        "sil_loss": diagnostics.get("sil_loss"),
        "sil_skipped": diagnostics.get("sil_skipped", False),
    }
    return KeyPoseObservation(
        capture.capture_id, capture.request_id, mode.key_object, mode.family,
        refined, selected,
        verified[selected]["timestamp_s"], (lower, upper), verified,
        per_view,
        {"accepted_views": admitted_ids,
         "max_center_residual_mm": max_center,
         "max_angle_residual_deg": max_angle,
         "center_limit_mm": float(maximum_multiview_center_error_mm),
         "angle_limit_deg": float(maximum_multiview_angle_error_deg),
         "cylinder_symmetry_quotient": mode.family == "cylinder",
         "maximum_socket_mask_overlap_fraction": (
             None if held_phase else float(maximum_socket_mask_overlap_fraction)),
         "socket_exclusion": (
             None if exclusion is None else exclusion.to_record()),
         "held_pose_prior": (None if not held_phase else {
             "source": held_pose_prior_source,
             "timestamp_s": float(held_pose_prior_timestamp_s),
             "pose_world": prior.tolist(),
             "maximum_center_error_mm": float(
                 maximum_held_prior_center_error_mm),
             "maximum_angle_error_deg": float(
                 maximum_held_prior_angle_error_deg),
             "maximum_time_skew_s": float(
                 maximum_held_prior_time_skew_s),
             "minimum_refinement_iou": float(minimum_held_refinement_iou),
             "max_center_residual_mm": max_prior_center,
             "max_angle_residual_deg": max_prior_angle,
         })},
        selection, capture.capture_dir,
        phase="held_preinsert" if held_phase else "tabletop")


def admit_key_capture(
    *, capture: KeyCaptureInput, init_orchestrator, mode: TaskMode,
    shared_root: Path, calibration, calibrated_camera_ids: set[str],
    view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
) -> KeyPoseObservation:
    """Admit a tabletop key; reject masks containing the frozen socket."""
    return _admit_key_capture_common(
        capture=capture, init_orchestrator=init_orchestrator, mode=mode,
        shared_root=shared_root, calibration=calibration,
        calibrated_camera_ids=calibrated_camera_ids,
        view_limits=view_limits,
        maximum_multiview_center_error_mm=(
            maximum_multiview_center_error_mm),
        maximum_multiview_angle_error_deg=(
            maximum_multiview_angle_error_deg),
        maximum_socket_mask_overlap_fraction=(
            maximum_socket_mask_overlap_fraction),
        socket_projection_dilation_px=socket_projection_dilation_px,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold)


def admit_held_key_capture(
    *, capture: KeyCaptureInput, init_orchestrator, mode: TaskMode,
    shared_root: Path, calibration, calibrated_camera_ids: set[str],
    view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    held_pose_prior_world: np.ndarray,
    held_pose_prior_timestamp_s: float,
    held_pose_prior_source: str,
    maximum_held_prior_center_error_mm: float,
    maximum_held_prior_angle_error_deg: float,
    maximum_held_prior_time_skew_s: float,
    minimum_held_refinement_iou: float,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
) -> KeyPoseObservation:
    """Admit a held key near the socket using an independent wrist pose prior.

    Socket-mask overlap is expected at pre-insertion hold. Per-view FoundPose,
    multi-view agreement, refinement IoU and a bounded *measured-wrist* prior
    replace that tabletop-only veto. A caller must derive the prior from live
    wrist feedback plus an already observed key/hand relation; this function
    verifies the declared source and timing but cannot prove its provenance.
    """
    return _admit_key_capture_common(
        capture=capture, init_orchestrator=init_orchestrator, mode=mode,
        shared_root=shared_root, calibration=calibration,
        calibrated_camera_ids=calibrated_camera_ids,
        view_limits=view_limits,
        maximum_multiview_center_error_mm=(
            maximum_multiview_center_error_mm),
        maximum_multiview_angle_error_deg=(
            maximum_multiview_angle_error_deg),
        maximum_socket_mask_overlap_fraction=0.0,
        socket_projection_dilation_px=0,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold,
        held_pose_prior_world=held_pose_prior_world,
        held_pose_prior_timestamp_s=held_pose_prior_timestamp_s,
        held_pose_prior_source=held_pose_prior_source,
        maximum_held_prior_center_error_mm=(
            maximum_held_prior_center_error_mm),
        maximum_held_prior_angle_error_deg=(
            maximum_held_prior_angle_error_deg),
        maximum_held_prior_time_skew_s=(
            maximum_held_prior_time_skew_s),
        minimum_held_refinement_iou=minimum_held_refinement_iou)


def write_key_capture_artifacts(
    capture: KeyCaptureInput, observation: KeyPoseObservation,
    output_dir: Path,
) -> Path:
    """Save the selected pose and its exact raw image/mask/pose evidence.

    Create one new directory per trial observation; never overwrite a prior
    camera record. This is evidence persistence, not a live success label.
    """
    if (not isinstance(capture, KeyCaptureInput) or
            not isinstance(observation, KeyPoseObservation) or
            capture.capture_id != observation.capture_id or
            capture.request_id != observation.request_id or
            capture.frame_evidence != observation.frame_evidence):
        raise ValueError("key capture and admitted observation do not match")
    frame_ids = {}
    for serial in capture.images_bgr:
        mask_fid = capture.masks.get(serial, {}).get("frame_id")
        pose_fid = capture.poses.get(serial, {}).get("frame_id")
        if mask_fid != pose_fid:
            raise ValueError("key capture SAM/FoundPose frame IDs differ")
        frame_ids[serial] = mask_fid
    verified = verify_frame_provenance(
        {"request_id": capture.request_id,
         "source": capture.frame_timestamp_source,
         "frames": capture.frame_evidence},
        request_id=capture.request_id, images_bgr=capture.images_bgr,
        frame_ids=frame_ids)
    if (capture.frame_timestamps_s !=
            {serial: row["timestamp_s"] for serial, row in verified.items()} or
            observation.selected_camera_id not in verified):
        raise ValueError("key observation and frame acquisition differ")
    used_ids = observation.consistency.get("accepted_views")
    if (not isinstance(used_ids, list) or len(used_ids) < 2 or
            len(set(used_ids)) != len(used_ids) or
            not set(used_ids) <= set(verified) or
            observation.selected_camera_id not in used_ids):
        raise ValueError("key observation has no valid admitted view set")
    expected_interval = (
        min(verified[serial]["timestamp_s"] -
            verified[serial]["max_error_s"] for serial in used_ids),
        max(verified[serial]["timestamp_s"] +
            verified[serial]["max_error_s"] for serial in used_ids),
    )
    if (observation.selected_acquisition_timestamp_s !=
            verified[observation.selected_camera_id]["timestamp_s"] or
            observation.acquisition_interval_s != expected_interval):
        raise ValueError("key observation timing differs from source frames")
    validate_se3(observation.pose_world, name="key observation pose")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    image_dir = target / "images"
    mask_dir = target / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    files = {}
    payloads = {"request_id": capture.request_id,
                "capture_id": capture.capture_id,
                "sam_prompt": capture.prompt,
                "frame_timestamp_source": capture.frame_timestamp_source,
                "frame_evidence": capture.frame_evidence,
                "mask_metadata": {}, "pose_payloads": {}}
    for serial, image in sorted(capture.images_bgr.items()):
        safe = _safe_id(serial, "key camera ID")
        image_path = image_dir / f"{safe}.png"
        files[str(image_path.relative_to(target))] = _write_png(image_path, image)
        if serial in capture.masks:
            mask_entry = capture.masks[serial]
            mask = np.asarray(mask_entry["mask"], dtype=np.uint8) * 255
            mask_path = mask_dir / f"{safe}.png"
            files[str(mask_path.relative_to(target))] = _write_png(mask_path, mask)
            payloads["mask_metadata"][serial] = {
                key: value for key, value in mask_entry.items() if key != "mask"}
        if serial in capture.poses:
            pose = dict(capture.poses[serial])
            if "pose_world" in pose:
                pose["pose_world"] = np.asarray(pose["pose_world"],
                                                dtype=float).tolist()
            payloads["pose_payloads"][serial] = pose
    payload_file = target / "payloads.json"
    files["payloads.json"] = _write_json(payload_file, payloads)
    report_file = target / "key_observation.json"
    files["key_observation.json"] = _write_json(
        report_file, observation.to_record())
    _write_json(target / "evidence_manifest.json", {
        "schema": "precision_insertion_key_capture_evidence_v1",
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "files_sha256": files,
        "scope": "saved_fresh_key_pose_not_physical_grasp_or_motion",
        "robot_ready": False,
    })
    return target


def verify_key_capture_artifacts(output_dir: Path) -> dict:
    """Recheck a saved key bundle's bytes and frame-to-pose binding."""
    root = Path(output_dir).expanduser().resolve()
    manifest = json.loads((root / "evidence_manifest.json").read_text(
        encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema") !=
            "precision_insertion_key_capture_evidence_v1"):
        raise ValueError("unknown key evidence bundle schema")
    hashes = manifest.get("files_sha256")
    if (not isinstance(hashes, dict) or not hashes or
            "payloads.json" not in hashes or
            "key_observation.json" not in hashes):
        raise ValueError("key evidence bundle lacks required file hashes")
    for relative, expected in hashes.items():
        if (not isinstance(relative, str) or not relative or
                not isinstance(expected, str) or len(expected) != 64 or
                any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("invalid key evidence digest entry")
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("missing or out-of-root key evidence file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"key evidence changed: {relative}")
    payloads = json.loads((root / "payloads.json").read_text(encoding="utf-8"))
    report = json.loads((root / "key_observation.json").read_text(
        encoding="utf-8"))
    phase = report.get("phase", "tabletop")
    if phase not in {"tabletop", "held_preinsert"}:
        raise ValueError("saved key observation has an unknown capture phase")
    if phase == "held_preinsert" and (
            report.get("consistency", {}).get("held_pose_prior") is None or
            report.get("consistency", {}).get("socket_exclusion") is not None):
        raise ValueError("held key evidence lacks its measured-wrist prior gate")
    if (payloads.get("capture_id") != manifest.get("capture_id") or
            payloads.get("request_id") != manifest.get("request_id") or
            report.get("capture_id") != manifest.get("capture_id") or
            report.get("request_id") != manifest.get("request_id") or
            payloads.get("frame_evidence") != report.get("frame_evidence")):
        raise ValueError("key evidence identities conflict")
    validate_se3(report.get("pose_world"), name="saved key observation pose")
    evidence = payloads["frame_evidence"]
    if report.get("selected_camera_id") not in evidence:
        raise ValueError("saved key selection lacks source camera")
    images = {}
    frame_ids = {}
    for serial in evidence:
        safe = _safe_id(serial, "key camera ID")
        image = cv2.imread(str(root / "images" / f"{safe}.png"),
                           cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"saved key frame is missing: {serial}")
        images[serial] = image
        mask_fid = payloads["mask_metadata"].get(serial, {}).get("frame_id")
        pose_fid = payloads["pose_payloads"].get(serial, {}).get("frame_id")
        if mask_fid != pose_fid:
            raise ValueError("saved key SAM/FoundPose frame IDs differ")
        frame_ids[serial] = mask_fid
    verified = verify_frame_provenance(
        {"request_id": manifest["request_id"],
         "source": payloads.get("frame_timestamp_source"),
         "frames": evidence},
        request_id=manifest["request_id"], images_bgr=images,
        frame_ids=frame_ids)
    selected = report["selected_camera_id"]
    used_ids = report.get("consistency", {}).get("accepted_views")
    if (not isinstance(used_ids, list) or len(used_ids) < 2 or
            len(set(used_ids)) != len(used_ids) or
            not set(used_ids) <= set(verified) or selected not in used_ids):
        raise ValueError("saved key admission lacks valid camera IDs")
    expected_interval = [
        min(verified[serial]["timestamp_s"] -
            verified[serial]["max_error_s"] for serial in used_ids),
        max(verified[serial]["timestamp_s"] +
            verified[serial]["max_error_s"] for serial in used_ids),
    ]
    if (report.get("selected_acquisition_timestamp_s") !=
            verified[selected]["timestamp_s"] or
            report.get("acquisition_interval_s") != expected_interval):
        raise ValueError("saved key timing differs from source frames")
    return manifest
