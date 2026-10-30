"""Admit one fresh multi-view key pose without changing AutoDex FoundPose.

The socket is measured once per session; the key is remeasured every trial.
This module reuses the same per-view SAM/FoundPose quality gate and AutoDex's
IoU/silhouette refinement. Cylinder tabletop raw pose disagreement is recorded
while stock refinement selects the key; held and square captures retain their
raw pose agreement gate. It performs no camera capture, planning, VLM call,
or robot motion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import TaskMode
from .calibration import validate_session_camera_calibration
from .frame_provenance import (
    exceeds_static_capture_skew_limit, verify_frame_metadata,
    verify_frame_provenance,
)
from .geometry import pose_angle_deg, validate_se3
from .live_capture import KeyCaptureInput
from .perception_evidence import SocketViewLimits, admit_socket_capture
from .runtime_evidence_cache import recalled, remember
from .raw_camera_capture import configured_capture_io_workers
from .session_bootstrap import _safe_id, _write_json
from .timing_profile import timed
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
        """Check state time without treating inference time as exposure skew."""
        state = float(state_timestamp_s)
        limit = float(maximum_skew_s)
        if not math.isfinite(state) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("robot state is not aligned to every key view")
        lower, upper = self.acquisition_interval_s
        bounded_request = bool(self.frame_evidence) and all(
            row.get("timestamp_method") == "bounded_request_interval"
            for row in self.frame_evidence.values())
        aligned = (lower <= state <= upper if bounded_request else
                   max(abs(state - endpoint)
                       for endpoint in self.acquisition_interval_s) <= limit)
        if not aligned:
            raise ValueError("robot state is not aligned to every key view")


def _cylinder_symmetry(mode: TaskMode, shared_root: Path):
    info = (Path(shared_root).expanduser().resolve() / "object_processing" /
            mode.key_object / "processed_data" / "info")
    symmetry = load_axial_symmetry(info.parent.parent.parent, mode.key_object)
    if not symmetry.end_exchange:
        raise ValueError("cylindrical key must have Dinf end symmetry")
    return symmetry.center_local, symmetry.axis_local


def _tabletop_cylinder_axis_layout(
    poses: dict[str, np.ndarray], *, calibration,
    axis_local: np.ndarray,
) -> str:
    """Choose the upright or side tabletop geometry from the view majority."""
    board = validate_se3(calibration.board["T_robot_board"],
                         name="measured board pose")
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    up_world = c2r[:3, :3] @ -board[:3, 2]
    axial_cosines = [abs(float(np.dot(
        pose[:3, :3] @ axis_local, up_world))) for pose in poses.values()]
    # The two v8 tabletop classes are separated by the geometric 45-degree
    # bisector. This only selects the pose model; it is not a failure gate.
    return ("upright" if np.median(axial_cosines) >= np.sqrt(0.5)
            else "side_down")


def _support_side_down_key_on_board(
    pose_world: np.ndarray, *, calibration, shared_root: Path,
    mode: TaskMode,
) -> tuple[np.ndarray, float]:
    """Condition a refined side-down key on its measured tabletop plane."""
    from .board_socket_fit import _mesh_vertices

    board = validate_se3(calibration.board["T_robot_board"],
                         name="measured board pose")
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    robot_pose = np.linalg.inv(c2r) @ validate_se3(pose_world)
    up = -board[:3, 2]
    axis = robot_pose[:3, 2]
    horizontal = axis - float(np.dot(axis, up)) * up
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1e-12:
        raise ValueError("side-down key refinement has no horizontal axis")
    horizontal /= horizontal_norm
    center_local, _ = _cylinder_symmetry(mode, shared_root)
    measured_center = robot_pose[:3, :3] @ center_local + robot_pose[:3, 3]
    cross = np.cross(axis, horizontal)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(axis, horizontal))
    if sine > 1e-12:
        skew = np.array([[0., -cross[2], cross[1]],
                         [cross[2], 0., -cross[0]],
                         [-cross[1], cross[0], 0.]])
        rotation = np.eye(3) + skew + skew @ skew * ((1. - cosine) / sine**2)
        robot_pose[:3, :3] = rotation @ robot_pose[:3, :3]
        robot_pose[:3, 3] = measured_center - robot_pose[:3, :3] @ center_local
    raw_mesh = (Path(shared_root) / "object_processing" / mode.key_object /
                "raw_mesh" / f"{mode.key_object}.obj")
    vertices = _mesh_vertices(raw_mesh)
    transformed = vertices @ robot_pose[:3, :3].T + robot_pose[:3, 3]
    offset = float(np.dot(board[:3, 3], up) -
                   np.min(transformed @ up))
    robot_pose[:3, 3] += offset * up
    return validate_se3(c2r @ robot_pose), offset


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
    minimum_tabletop_refinement_iou: float | None = None,
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
        if minimum_tabletop_refinement_iou is not None:
            raise ValueError("tabletop IoU limit cannot be used for a held key")
        prior = validate_se3(held_pose_prior_world,
                             name="held key pose prior_world")
        held_limits = (
            held_pose_prior_timestamp_s,
            maximum_held_prior_center_error_mm,
            maximum_held_prior_angle_error_deg,
            maximum_held_prior_time_skew_s,
            minimum_held_refinement_iou)
        if (held_pose_prior_source not in {
                "measured_wrist_plus_observed_held_relation",
                "measured_wrist_plus_bounded_held_relation",
                "measured_wrist_plus_candidate_grasp"} or
                any(value is None or not math.isfinite(float(value))
                    for value in held_limits) or
                maximum_held_prior_center_error_mm <= 0 or
                maximum_held_prior_angle_error_deg <= 0 or
                maximum_held_prior_time_skew_s <= 0 or
                not 0 < minimum_held_refinement_iou <= 1):
            raise ValueError("held-key prior needs a timed measured-wrist source and limits")
    elif (type(minimum_tabletop_refinement_iou) not in (int, float) or
          not math.isfinite(minimum_tabletop_refinement_iou) or
          not 0 < minimum_tabletop_refinement_iou <= 1):
        raise ValueError("tabletop key needs a commissioned minimum refinement IoU")
    if not isinstance(capture, KeyCaptureInput):
        raise TypeError("fresh key capture must be KeyCaptureInput")
    if mode.key_object != getattr(init_orchestrator, "obj_name", None):
        raise ValueError("FoundPose is not initialized for selected v8 key")
    pose_only = not capture.images_bgr
    camera_ids = set(capture.masks) if pose_only else set(capture.images_bgr)
    if (not camera_ids or not camera_ids <= calibrated_camera_ids or
            (pose_only and held_phase) or
            set(getattr(init_orchestrator, "intrinsics_undist", {})) !=
            calibrated_camera_ids or
            set(getattr(init_orchestrator, "extrinsics", {})) !=
            calibrated_camera_ids):
        raise ValueError("key camera IDs differ from session calibration")
    validate_session_camera_calibration(
        calibration, intrinsics_undist=init_orchestrator.intrinsics_undist,
        extrinsics_full=init_orchestrator.extrinsics,
        calibrated_camera_ids=calibrated_camera_ids)
    perception_inputs = {
        (capture.masks.get(serial, {}).get(
            "perception_method", "stock_full_frame_sam3_foundpose"),
         capture.poses.get(serial, {}).get(
            "perception_method", "stock_full_frame_sam3_foundpose"))
        for serial in camera_ids}
    if (len(perception_inputs) != 1 or
            next(iter(perception_inputs))[0] !=
            next(iter(perception_inputs))[1]):
        raise ValueError("key views mix incompatible perception producers")
    perception_input = next(iter(perception_inputs))[0]
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
    for serial in camera_ids:
        mask_fid = capture.masks.get(serial, {}).get("frame_id")
        pose_fid = capture.poses.get(serial, {}).get("frame_id")
        if mask_fid != pose_fid:
            raise ValueError("key SAM/FoundPose frame IDs differ")
        frame_ids[serial] = mask_fid
    metadata = {
        "request_id": capture.request_id,
        "source": capture.frame_timestamp_source,
        "frames": capture.frame_evidence}
    if pose_only:
        verified = verify_frame_metadata(
            metadata, request_id=capture.request_id, frame_ids=frame_ids)
    else:
        verified = verify_frame_provenance(
            metadata, request_id=capture.request_id,
            images_bgr=capture.images_bgr, frame_ids=frame_ids)
    if (capture.frame_timestamps_s !=
            {serial: row["timestamp_s"] for serial, row in verified.items()}):
        raise ValueError("key frame times differ from bound images")
    view_limits.validate()
    if exceeds_static_capture_skew_limit(
            verified, view_limits.maximum_capture_skew_s):
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
            dilation_px=socket_projection_dilation_px,
            image_shapes=(
                {serial: tuple(np.asarray(capture.masks[serial]["mask"]).shape)
                 for serial in camera_ids} if pose_only else None))
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
    # Stock AutoDex chooses one FoundPose candidate by rendered IoU and then
    # refines it against the accepted masks. In the live cylinder tabletop
    # scene, other raw candidates can disagree widely even when that stock
    # refinement converges. Retain their spread as evidence, not a veto.
    stock_tabletop_cylinder = mode.family == "cylinder" and not held_phase
    if (not stock_tabletop_cylinder and
            (max_center > maximum_multiview_center_error_mm or
             max_angle > maximum_multiview_angle_error_deg)):
        raise ValueError("key multiview FoundPose estimates disagree")
    if not callable(getattr(init_orchestrator, "refine_from_payloads", None)):
        raise ValueError("AutoDex IoU/silhouette key selector is unavailable")
    tabletop_layout = None
    if stock_tabletop_cylinder and "T_robot_board" in calibration.board:
        tabletop_layout = _tabletop_cylinder_axis_layout(
            poses, calibration=calibration, axis_local=axis)
    stock_tabletop_layout = tabletop_layout
    board_fit = None
    if stock_tabletop_cylinder and "T_robot_board" in calibration.board:
        # Raw FoundPose axes can disagree even when the key is visibly
        # upright. Fit the board-supported upright body for either raw class,
        # then select the shape that explains the acquired masks better.
        from .board_socket_fit import fit_board_socket_body
        try:
            fitted_pose, fit = fit_board_socket_body(
                masks={serial: capture.masks[serial]["mask"]
                       for serial in admitted_ids},
                intrinsics_undist={serial: init_orchestrator.intrinsics_undist[serial]
                                   for serial in admitted_ids},
                extrinsics_camera_world={serial: init_orchestrator.extrinsics[serial]
                                         for serial in admitted_ids},
                c2r=np.asarray(calibration.record["c2r"], dtype=np.float64),
                board_pose_robot=np.asarray(
                    calibration.board["T_robot_board"], dtype=np.float64),
                raw_mesh=Path(shared_root) / "object_processing" /
                    mode.key_object / "raw_mesh" / f"{mode.key_object}.obj")
            board_fit = (fitted_pose, fit)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if board_fit is not None:
        _, raw_diagnostics = init_orchestrator.refine_from_payloads(
            capture.masks, capture.poses, subset_serials=admitted_ids,
            sil_iters=0, selection_mode="iou")
        raw_iou = raw_diagnostics.get("best_iou")
        raw_iou = float(raw_iou) if raw_iou is not None else float("-inf")
        if board_fit[1]["mean_body_iou"] >= raw_iou:
            tabletop_layout = "upright"
            pose = board_fit[0]
            per_view_iou = board_fit[1]["per_view_iou"]
            board_selected = max(per_view_iou, key=per_view_iou.__getitem__)
            diagnostics = {
                "best_serial": board_selected,
                "best_iou": board_fit[1]["mean_body_iou"],
                "sil_skipped": True,
                "board_body_fit": board_fit[1],
                "stock_initial_iou": raw_iou,
                "stock_axis_layout": stock_tabletop_layout,
            }
        else:
            pose, diagnostics = init_orchestrator.refine_from_payloads(
                capture.masks, capture.poses, subset_serials=admitted_ids,
                sil_iters=silhouette_iterations,
                sil_loss_threshold=silhouette_loss_threshold,
                selection_mode="iou")
    else:
        pose, diagnostics = init_orchestrator.refine_from_payloads(
            capture.masks, capture.poses, subset_serials=admitted_ids,
            sil_iters=silhouette_iterations,
            sil_loss_threshold=silhouette_loss_threshold,
            selection_mode="iou")
    if pose is None or not isinstance(diagnostics, dict):
        raise ValueError("AutoDex key pose refinement failed")
    if tabletop_layout == "side_down":
        pose, support_offset = _support_side_down_key_on_board(
            pose, calibration=calibration, shared_root=shared_root,
            mode=mode)
        diagnostics["board_support_offset_m"] = support_offset
    diagnostics["tabletop_axis_layout"] = tabletop_layout
    selected = diagnostics.get("best_serial")
    if selected not in admitted_ids:
        raise ValueError("AutoDex selected a non-admitted key camera")
    best_iou = diagnostics.get("best_iou")
    minimum_iou = (minimum_held_refinement_iou if held_phase else
                   minimum_tabletop_refinement_iou)
    if (type(best_iou) not in (int, float) or
            not math.isfinite(best_iou) or not 0 <= best_iou <= 1):
        raise ValueError("AutoDex key IoU selector returned an invalid score")
    # Stock AutoDex uses IoU to rank tabletop FoundPose candidates and the
    # silhouette loss to accept the refined pose. Keep the IoU score as
    # evidence, without an extra absolute cutoff on any tabletop key.
    if held_phase and best_iou < minimum_iou:
        raise ValueError("key mask/silhouette IoU is below commissioned limit")
    refined = validate_se3(pose, name="refined key pose_world")
    for camera in admitted_ids:
        distance, angle = _pose_residual(
            refined, poses[camera], cylinder_center=center,
            cylinder_axis=axis)
        max_center = max(max_center, distance)
        max_angle = max(max_angle, angle)
        if (not stock_tabletop_cylinder and
                (distance > maximum_multiview_center_error_mm or
                 angle > maximum_multiview_angle_error_deg)):
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
        "method": ("board_constrained_key_body" if "board_body_fit" in diagnostics
                   else "AutoDex_refine_from_payloads_iou"),
        "perception_input": perception_input,
        "best_serial": selected,
        "best_iou": diagnostics.get("best_iou"),
        "minimum_refinement_iou": float(minimum_iou),
        "minimum_refinement_iou_gate_applied": held_phase,
        "sil_loss": diagnostics.get("sil_loss"),
        "sil_skipped": diagnostics.get("sil_skipped", False),
        "board_body_fit": diagnostics.get("board_body_fit"),
        "stock_initial_iou": diagnostics.get("stock_initial_iou"),
        "stock_axis_layout": diagnostics.get("stock_axis_layout"),
        "tabletop_axis_layout": diagnostics.get("tabletop_axis_layout"),
        "board_support_offset_m": diagnostics.get("board_support_offset_m"),
    }
    return KeyPoseObservation(
        capture.capture_id, capture.request_id, mode.key_object, mode.family,
        refined, selected,
        verified[selected]["timestamp_s"], (lower, upper), verified,
        per_view,
        {"accepted_views": admitted_ids,
         "max_center_residual_mm": max_center,
         "max_angle_residual_deg": max_angle,
         "raw_foundpose_gate_applied": not stock_tabletop_cylinder,
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
        phase=("held_postlift" if held_pose_prior_source ==
               "measured_wrist_plus_candidate_grasp" else
               "held_preinsert" if held_phase else "tabletop"))


def admit_key_capture(
    *, capture: KeyCaptureInput, init_orchestrator, mode: TaskMode,
    shared_root: Path, calibration, calibrated_camera_ids: set[str],
    view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int,
    minimum_refinement_iou: float,
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
        minimum_tabletop_refinement_iou=minimum_refinement_iou,
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
    wrist feedback plus a previously observed or independently calibrated
    key/hand relation; this function verifies the declared source and timing
    but cannot prove its provenance.
    """
    if held_pose_prior_source not in {
            "measured_wrist_plus_observed_held_relation",
            "measured_wrist_plus_bounded_held_relation"}:
        raise ValueError("preinsert held-key prior needs a held relation")
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


def admit_postlift_key_capture(
    *, capture: KeyCaptureInput, init_orchestrator, mode: TaskMode,
    shared_root: Path, calibration, calibrated_camera_ids: set[str],
    view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    candidate_pose_prior_world: np.ndarray,
    measured_wrist_timestamp_s: float,
    maximum_candidate_prior_center_error_mm: float,
    maximum_candidate_prior_angle_error_deg: float,
    maximum_prior_time_skew_s: float,
    minimum_refinement_iou: float,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
) -> KeyPoseObservation:
    """Admit the first held-key view after lift, before its relation is known.

    The measured wrist plus selected BODex grasp is only a loose search prior:
    squeeze may move the key. The runner must verify that provenance and then
    replan from the *observed* relation. It is not a retry/preinsert capture.
    """
    return _admit_key_capture_common(
        capture=capture, init_orchestrator=init_orchestrator, mode=mode,
        shared_root=shared_root, calibration=calibration,
        calibrated_camera_ids=calibrated_camera_ids, view_limits=view_limits,
        maximum_multiview_center_error_mm=maximum_multiview_center_error_mm,
        maximum_multiview_angle_error_deg=maximum_multiview_angle_error_deg,
        maximum_socket_mask_overlap_fraction=0.0,
        socket_projection_dilation_px=0,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold,
        held_pose_prior_world=candidate_pose_prior_world,
        held_pose_prior_timestamp_s=measured_wrist_timestamp_s,
        held_pose_prior_source="measured_wrist_plus_candidate_grasp",
        maximum_held_prior_center_error_mm=(
            maximum_candidate_prior_center_error_mm),
        maximum_held_prior_angle_error_deg=(
            maximum_candidate_prior_angle_error_deg),
        maximum_held_prior_time_skew_s=maximum_prior_time_skew_s,
        minimum_held_refinement_iou=minimum_refinement_iou)


def write_key_capture_artifacts(
    capture: KeyCaptureInput, observation: KeyPoseObservation,
    output_dir: Path, *, derivation_report: dict | None = None,
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
    _verify_perception_derivation(
        capture.masks, capture.poses, capture.frame_evidence,
        derivation_report, capture.capture_id, capture.request_id)
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
                "mask_metadata": {}, "pose_payloads": {},
                "perception_derivation": derivation_report}
    if derivation_report is not None:
        source = Path(derivation_report["board_config_path"])
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != derivation_report[
                "board_config_sha256"]:
            raise ValueError("ROI board config changed before evidence save")
        config_path = target / "board_config.json"
        with config_path.open("xb") as stream:
            stream.write(raw)
        files["board_config.json"] = hashlib.sha256(raw).hexdigest()
    from .raw_camera_capture import write_png_jobs

    png_jobs = []
    for serial, image in sorted(capture.images_bgr.items()):
        safe = _safe_id(serial, "key camera ID")
        image_path = image_dir / f"{safe}.png"
        png_jobs.append((serial, image_path, image, "undistorted"))
        if serial in capture.masks:
            mask_entry = capture.masks[serial]
            mask = np.asarray(mask_entry["mask"], dtype=np.uint8) * 255
            mask_path = mask_dir / f"{safe}.png"
            png_jobs.append((serial, mask_path, mask, "mask"))
            payloads["mask_metadata"][serial] = {
                key: value for key, value in mask_entry.items() if key != "mask"}
        if serial in capture.poses:
            pose = dict(capture.poses[serial])
            if "pose_world" in pose:
                pose["pose_world"] = np.asarray(pose["pose_world"],
                                                dtype=float).tolist()
            payloads["pose_payloads"][serial] = pose
    png_results, _ = write_png_jobs(png_jobs, phase="key")
    for serial, source, digest in png_results:
        folder = "images" if source == "undistorted" else "masks"
        files[f"{folder}/{serial}.png"] = digest
    payload_file = target / "payloads.json"
    files["payloads.json"] = _write_json(payload_file, payloads)
    report_file = target / "key_observation.json"
    files["key_observation.json"] = _write_json(
        report_file, observation.to_record())
    manifest = {
        "schema": "precision_insertion_key_capture_evidence_v1",
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "files_sha256": files,
        "scope": "saved_fresh_key_pose_not_physical_grasp_or_motion",
        "robot_ready": False,
    }
    _write_json(target / "evidence_manifest.json", manifest)
    remember("key_capture", target, "key", manifest)
    return target


def write_key_pose_artifacts(
    capture: KeyCaptureInput, observation: KeyPoseObservation,
    output_dir: Path,
) -> Path:
    """Save tabletop FoundPose inputs without waiting for daemon image PNGs.

    A VLM checkpoint uses its own snapshot and frame evidence when needed.
    Recovery only needs the observed pose to plan the next action.
    """
    if (capture.images_bgr or observation.phase != "tabletop" or
            capture.capture_id != observation.capture_id or
            capture.request_id != observation.request_id or
            capture.frame_evidence != observation.frame_evidence or
            set(capture.masks) != set(capture.poses) or
            set(capture.masks) != set(capture.frame_evidence)):
        raise ValueError("pose-only key evidence has mismatched inputs")
    frame_ids = {}
    for serial in capture.masks:
        mask_fid = capture.masks[serial].get("frame_id")
        if mask_fid != capture.poses[serial].get("frame_id"):
            raise ValueError("key SAM/FoundPose frame IDs differ")
        frame_ids[serial] = mask_fid
    verified = verify_frame_metadata(
        {"request_id": capture.request_id,
         "source": capture.frame_timestamp_source,
         "frames": capture.frame_evidence},
        request_id=capture.request_id, frame_ids=frame_ids)
    if (capture.frame_timestamps_s !=
            {serial: row["timestamp_s"] for serial, row in verified.items()}):
        raise ValueError("key pose timestamps differ from source frames")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    mask_dir = target / "masks"
    mask_dir.mkdir()
    payloads = {
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "sam_prompt": capture.prompt,
        "frame_timestamp_source": capture.frame_timestamp_source,
        "frame_evidence": verified,
        "mask_metadata": {}, "pose_payloads": {},
        "perception_derivation": None,
    }
    def save_mask(serial: str) -> None:
        safe = _safe_id(serial, "key camera ID")
        entry = capture.masks[serial]
        mask = np.asarray(entry["mask"], dtype=np.uint8) * 255
        if mask.ndim != 2 or not mask.size:
            raise ValueError(f"invalid key mask: {serial}")
        encoded, data = cv2.imencode(".png", mask)
        if not encoded:
            raise OSError(f"could not save key mask: {serial}")
        with (mask_dir / f"{safe}.png").open("xb") as stream:
            stream.write(data.tobytes())

    serials = sorted(capture.masks)
    with ThreadPoolExecutor(max_workers=configured_capture_io_workers(),
                            thread_name_prefix="key-mask-save") as pool:
        list(pool.map(save_mask, serials))
    for serial in serials:
        entry = capture.masks[serial]
        payloads["mask_metadata"][serial] = {
            key: value for key, value in entry.items() if key != "mask"}
        pose = dict(capture.poses[serial])
        if "pose_world" in pose:
            pose["pose_world"] = np.asarray(pose["pose_world"],
                                            dtype=float).tolist()
        payloads["pose_payloads"][serial] = pose
    _write_json(target / "payloads.json", payloads)
    _write_json(target / "key_observation.json", observation.to_record())
    manifest = {
        "schema": "precision_insertion_key_pose_evidence_v2",
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "camera_ids": sorted(verified),
        "image_source": "not_saved_pose_only",
        "scope": "foundpose_pose_and_masks_no_saved_source_image_gate",
        "robot_ready": False,
    }
    _write_json(target / "evidence_manifest.json", manifest)
    remember("key_capture", target, "key", manifest)
    return target


def write_failed_key_pose_capture(
    capture: KeyCaptureInput, output_dir: Path, reason: str,
) -> Path:
    """Keep the same-request SAM masks when tabletop admission fails.

    Pose-only capture normally avoids full-frame PNG transfer. Keeping these
    small masks and payload summaries costs no time on successful trials and
    identifies a socket selected as the key on a failed trial.
    """
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    masks_dir = target / "masks"
    masks_dir.mkdir()
    views = {}
    for serial in sorted(capture.masks):
        entry = capture.masks[serial]
        raw_mask = entry.get("mask")
        mask = (np.asarray(raw_mask, dtype=np.uint8) if raw_mask is not None
                else np.empty(0, dtype=np.uint8))
        if mask.ndim == 2 and mask.size:
            if not cv2.imwrite(str(masks_dir / f"{_safe_id(serial, 'key camera ID')}.png"),
                               mask * 255):
                raise OSError(f"failed to write key mask for {serial}")
        pose = capture.poses.get(serial, {})
        views[serial] = {
            "mask_pixels": int(np.count_nonzero(mask)) if mask.ndim == 2 else 0,
            "frame_id": entry.get("frame_id"),
            "pose_ok": bool(pose.get("ok")),
            "quality": float(pose["quality"])
            if pose.get("quality") is not None else None,
            "inliers": int(pose["inliers"])
            if pose.get("inliers") is not None else None,
            "pose_world": np.asarray(pose["pose_world"]).tolist()
            if pose.get("pose_world") is not None else None,
        }
    (target / "report.json").write_text(json.dumps({
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "prompt": capture.prompt,
        "reason": reason,
        "frame_evidence": capture.frame_evidence,
        "views": views,
    }, indent=2), encoding="utf-8")
    return target


def verify_key_capture_artifacts(output_dir: Path) -> dict:
    """Recheck a saved key bundle's bytes and frame-to-pose binding."""
    root = Path(output_dir).expanduser().resolve()
    cached = recalled("key_capture", root, "key")
    if cached is not None:
        with timed("evidence.live_manifest_reuse", phase="evidence",
                   kind="io", capture_phase="key"):
            return cached
    manifest = json.loads((root / "evidence_manifest.json").read_text(
        encoding="utf-8"))
    if isinstance(manifest, dict) and manifest.get("schema") == (
            "precision_insertion_key_pose_evidence_v2"):
        _verify_key_pose_artifacts(root, manifest)
        return manifest
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
    if phase not in {"tabletop", "held_postlift", "held_preinsert"}:
        raise ValueError("saved key observation has an unknown capture phase")
    prior = report.get("consistency", {}).get("held_pose_prior")
    expected_sources = {
        "held_postlift": {"measured_wrist_plus_candidate_grasp"},
        "held_preinsert": {
            "measured_wrist_plus_observed_held_relation",
            "measured_wrist_plus_bounded_held_relation"},
    }.get(phase)
    if expected_sources is not None and (
            not isinstance(prior, dict) or
            prior.get("source") not in expected_sources or
            report.get("consistency", {}).get("socket_exclusion") is not None):
        raise ValueError("held key evidence lacks its measured-wrist prior gate")
    if (payloads.get("capture_id") != manifest.get("capture_id") or
            payloads.get("request_id") != manifest.get("request_id") or
            report.get("capture_id") != manifest.get("capture_id") or
            report.get("request_id") != manifest.get("request_id") or
            payloads.get("frame_evidence") != report.get("frame_evidence")):
        raise ValueError("key evidence identities conflict")
    _verify_perception_derivation(
        payloads.get("mask_metadata", {}), payloads.get("pose_payloads", {}),
        payloads.get("frame_evidence", {}),
        payloads.get("perception_derivation"), manifest["capture_id"],
        manifest["request_id"])
    derivation = payloads.get("perception_derivation")
    if derivation is not None and (
            "board_config.json" not in hashes or
            hashlib.sha256((root / "board_config.json").read_bytes()
                           ).hexdigest() != derivation.get(
                               "board_config_sha256")):
        raise ValueError("saved ROI board config differs from derivation")
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


def _verify_key_pose_artifacts(root: Path, manifest: dict) -> None:
    payloads = json.loads((root / "payloads.json").read_text(encoding="utf-8"))
    report = json.loads((root / "key_observation.json").read_text(
        encoding="utf-8"))
    camera_ids = set(manifest.get("camera_ids", []))
    if (report.get("phase") != "tabletop" or
            report.get("capture_id") != manifest.get("capture_id") or
            report.get("request_id") != manifest.get("request_id") or
            payloads.get("capture_id") != manifest.get("capture_id") or
            payloads.get("request_id") != manifest.get("request_id") or
            camera_ids != set(report.get("frame_evidence", {})) or
            camera_ids != set(payloads.get("frame_evidence", {})) or
            camera_ids != set(payloads.get("mask_metadata", {})) or
            camera_ids != set(payloads.get("pose_payloads", {})) or
            payloads["frame_evidence"] != report["frame_evidence"] or
            payloads.get("perception_derivation") is not None):
        raise ValueError("pose-only key evidence identities conflict")
    _verify_perception_derivation(
        payloads["mask_metadata"], payloads["pose_payloads"],
        payloads["frame_evidence"], None,
        manifest["capture_id"], manifest["request_id"])
    frame_ids = {}
    for serial in camera_ids:
        mask = payloads["mask_metadata"][serial]
        pose = payloads["pose_payloads"][serial]
        if mask.get("frame_id") != pose.get("frame_id"):
            raise ValueError("saved key pose frame IDs differ")
        if not (root / "masks" / f"{_safe_id(serial, 'camera ID')}.png").is_file():
            raise ValueError("saved key pose mask is missing")
        frame_ids[serial] = mask["frame_id"]
    verified = verify_frame_metadata(
        {"request_id": manifest["request_id"],
         "source": payloads.get("frame_timestamp_source"),
         "frames": payloads["frame_evidence"]},
        request_id=manifest["request_id"], frame_ids=frame_ids)
    selected = report.get("selected_camera_id")
    used = report.get("consistency", {}).get("accepted_views")
    if (not isinstance(used, list) or len(used) < 2 or
            not set(used) <= camera_ids or selected not in used or
            report.get("selected_acquisition_timestamp_s") !=
            verified[selected]["timestamp_s"] or
            report.get("acquisition_interval_s") != [
                min(verified[c]["timestamp_s"] - verified[c]["max_error_s"]
                    for c in used),
                max(verified[c]["timestamp_s"] + verified[c]["max_error_s"]
                    for c in used)]):
        raise ValueError("saved key pose timing differs from admission")
    validate_se3(report.get("pose_world"), name="saved key pose")


def _verify_perception_derivation(
    masks: dict, poses: dict, frame_evidence: dict,
    derivation: dict | None, capture_id: str, request_id: int,
) -> None:
    """Reject mixed stock/ROI payloads and unbound socket references."""
    assisted = "same_session_registered_roi_sam3_foundpose"
    methods = {entry.get("perception_method") for entry in masks.values()}
    pose_methods = {entry.get("perception_method") for entry in poses.values()}
    if derivation is None:
        if methods - {None} or pose_methods - {None}:
            raise ValueError("assisted key payload lacks derivation evidence")
        return
    if (not isinstance(derivation, dict) or derivation.get("schema") !=
            "precision_insertion_registered_key_capture_derivation_v1" or
            derivation.get("perception_method") != assisted or
            derivation.get("capture_id") != capture_id or
            derivation.get("request_id") != request_id or
            set(masks) != set(frame_evidence) or
            set(poses) != set(frame_evidence) or
            set(derivation.get("per_view", {})) != set(frame_evidence) or
            methods != {assisted} or pose_methods != {assisted}):
        raise ValueError("registered key derivation is incomplete or mixed")
    baseline_record = derivation.get("baseline")
    if not isinstance(baseline_record, dict):
        raise ValueError("registered key derivation lacks socket baseline")
    # Local import avoids a module cycle through the opened SessionRunner.
    from .session_key_baseline import load_session_key_baseline

    source = baseline_record.get("source")
    if not isinstance(source, dict):
        raise ValueError("registered key baseline has no source cameras")
    baseline = load_session_key_baseline(
        session_evidence_dir=Path(baseline_record["session_evidence_dir"]),
        socket_capture_id=baseline_record["socket_capture_id"],
        expected_session_manifest_sha256=baseline_record[
            "session_manifest_sha256"],
        calibrated_camera_ids=set(frame_evidence), minimum_views=2)
    if baseline.to_record() != baseline_record:
        raise ValueError("registered key baseline differs from frozen session")
    for serial, row in frame_evidence.items():
        mask, pose = masks[serial], poses[serial]
        for entry in (mask, pose):
            if (entry.get("frame_id") != row.get("frame_id") or
                    entry.get("image_sha256") != row.get("image_sha256") or
                    entry.get("image_space") !=
                    "autodex_undistorted_full_frame" or
                    entry.get("baseline_session_manifest_sha256") !=
                    baseline.session_manifest_sha256 or
                    entry.get("baseline_socket_capture_id") !=
                    baseline.socket_capture_id):
                raise ValueError("registered key payload source frame differs")
