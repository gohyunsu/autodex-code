"""Reobserve a cylinder after the post-shift transfer reaches pre-insertion.

This is a fresh, multi-view tip/axis and measured-wrist check. It consumes a
verified non-contact transfer log but never executes axial contact, records
task success or treats a nominal held-key overlay as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .calibration import validate_session_camera_calibration
from .config import TaskMode
from .frame_provenance import bounded_capture_skew_s
from .geometry import pose_angle_deg, validate_se3
from .grounded_alignment import (
    AlignmentLimits, estimate_grounded_line_alignment,
    observe_grounded_cylinder_axis,
)
from .grounded_lateral import GroundedLateralPreflight
from .live_robot_state import LiveRobotState
from .observer import ImageVLM, VLMObservation, _parse_object
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .postshift_path_handoff import verify_postshift_path_handoff
from .postshift_transfer_execution import verify_postshift_transfer_execution
from .raw_camera_capture import verify_raw_camera_capture
from .session_bootstrap import _safe_id
from .xy_overlay import CalibratedXYFrame, camera_socket_transform
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PostShiftArrivalCheckpoint:
    status: str
    attempt_id: str
    candidate_id: str
    handoff_report_path: Path
    handoff_report_sha256: str
    transfer_execution_path: Path
    transfer_execution_sha256: str
    capture_dir: Path
    capture_manifest_sha256: str
    joint_sample: LiveRobotState
    decision_timestamp_s: float
    measured_goal_translation_error_m: float
    measured_goal_rotation_error_deg: float
    alignment: dict
    vlm_observations: tuple
    visual_lateral_budget_m: float

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_postshift_arrival_checkpoint_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "handoff_report_path": str(self.handoff_report_path),
            "handoff_report_sha256": self.handoff_report_sha256,
            "transfer_execution_path": str(self.transfer_execution_path),
            "transfer_execution_sha256": self.transfer_execution_sha256,
            "capture_dir": str(self.capture_dir),
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "joint_sample": self.joint_sample.to_record(),
            "decision_timestamp_s": self.decision_timestamp_s,
            "measured_goal_translation_error_m": (
                self.measured_goal_translation_error_m),
            "measured_goal_rotation_error_deg": (
                self.measured_goal_rotation_error_deg),
            "alignment": self.alignment,
            "vlm_observations": [row.to_record()
                                 for row in self.vlm_observations],
            "visual_lateral_budget_m": self.visual_lateral_budget_m,
            "scope": "fresh_transfer_arrival_diagnostic_not_contact_permission",
            "axial_retry_allowed": False,
            "robot_ready": False,
        }


def assess_postshift_arrival(
    *, preflight: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    handoff_report_path: Path, transfer_execution_path: Path,
    capture_dir: Path, joint_sample: LiveRobotState,
    decision_timestamp_s: float, planner,
    mode: TaskMode, shared_root: Path, calibration,
    intrinsics_full: Mapping, extrinsics_full: Mapping,
    backend: ImageVLM, alignment_limits: AlignmentLimits,
    max_capture_skew_s: float, max_execution_observation_gap_s: float,
    max_frame_age_s: float, max_joint_frame_skew_s: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float, max_joint_goal_error_rad: float,
    max_goal_translation_error_m: float,
    max_goal_rotation_error_deg: float,
    max_visual_lateral_error_m: float,
    max_visual_axis_tilt_deg: float,
    max_grounded_tip_error_m: float,
) -> PostShiftArrivalCheckpoint:
    """Require fresh raw frames and measured joints after completed transfer."""
    alignment_limits.validate()
    limits = (max_capture_skew_s, max_execution_observation_gap_s,
              max_frame_age_s, max_joint_frame_skew_s, max_arm_hand_skew_s,
              max_hand_command_error_raw, max_arm_velocity_rad_s,
              max_joint_goal_error_rad, max_goal_translation_error_m,
              max_goal_rotation_error_deg, max_visual_lateral_error_m,
              max_visual_axis_tilt_deg, max_grounded_tip_error_m)
    if not all(math.isfinite(float(value)) and value > 0 for value in limits):
        raise ValueError("post-shift arrival needs positive commissioned limits")
    if (mode.family != "cylinder" or
            preflight.status != "sampled_postshift_20mm_preflight_pass" or
            checkpoint.status != "visual_alignment_within_budget" or
            preflight.attempt_id != checkpoint.attempt_id or
            preflight.candidate_id != checkpoint.candidate_id or
            preflight.attempt_id != shift_plan.attempt_id or
            preflight.candidate_id != shift_plan.candidate_id):
        raise ValueError("arrival needs the same aligned held cylinder")
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    handoff = verify_postshift_path_handoff(
        handoff_path, expected=preflight, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=shared_root,
        calibration=calibration)
    if handoff["transfer_required"] is not True:
        raise ValueError("arrival transfer checkpoint needs a nonzero transfer")
    execution_path = Path(transfer_execution_path).expanduser().resolve()
    execution = verify_postshift_transfer_execution(
        execution_path, handoff_report_path=handoff_path,
        expected=preflight, checkpoint=checkpoint, shift_plan=shift_plan,
        mode=mode, shared_root=shared_root, calibration=calibration)
    completed = float(execution["completed_at_s"])
    bundle = Path(capture_dir).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="preinsert")
    earlier = verify_raw_camera_capture(
        checkpoint.capture_dir, phase="post_lateral_hold")
    rows = capture["frame_evidence"]
    earlier_rows = earlier["frame_evidence"]
    if (capture["request_id"] <= earlier["request_id"] or
            len(rows) < alignment_limits.minimum_views or
            not set(rows) <= set(earlier_rows) or
            bounded_capture_skew_s(rows) > max_capture_skew_s or
            any(row["frame_id"] <= earlier_rows[camera]["frame_id"]
                for camera, row in rows.items())):
        raise ValueError("arrival needs new synchronized camera frames")
    first = min(row["timestamp_s"] - row["max_error_s"]
                for row in rows.values())
    last = max(row["timestamp_s"] + row["max_error_s"]
               for row in rows.values())
    decision = float(decision_timestamp_s)
    if (not math.isfinite(decision) or
            completed >= first or
            first - completed > max_execution_observation_gap_s or
            decision < last or decision - first > max_frame_age_s or
            not isinstance(joint_sample, LiveRobotState)):
        raise ValueError("arrival frames/state must follow the transfer")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    if (joint_sample.sample_timestamp_s < completed or
            joint_sample.sample_timestamp_s < first - max_joint_frame_skew_s or
            joint_sample.sample_timestamp_s > last + max_joint_frame_skew_s or
            np.max(np.abs(joint_sample.full_q -
                          handoff["transfer_end_q"])) >
                max_joint_goal_error_rad or
            np.max(np.abs(joint_sample.hand_raw_measured -
                          checkpoint.joint_sample.hand_raw_measured)) >
                handoff["limits"]["max_hand_drift_raw"]):
        raise ValueError("arrival joints disagree with completed transfer")
    wrist = validate_se3(planner.fk_wrist(joint_sample.full_q),
                         name="arrival measured wrist FK")
    target = validate_se3(
        preflight.targets.T_robot_hand_preinsert,
        name="planned post-shift preinsert hand target")
    translation_error = float(np.linalg.norm(
        wrist[:3, 3] - target[:3, 3]))
    rotation_error = pose_angle_deg(wrist, target)
    if (translation_error > max_goal_translation_error_m or
            rotation_error > max_goal_rotation_error_deg):
        raise ValueError("measured arrival wrist missed the preinsert target")
    root = Path(shared_root).expanduser().resolve()
    geometry_path = AssetPaths(root, mode).task_geometry
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (_sha(geometry_path) !=
            preflight.targets.task_geometry_sha256 or
            geometry.get("key_object") != mode.key_object or
            geometry.get("socket_pose_object") != mode.socket_object):
        raise ValueError("arrival geometry changed since post-shift replan")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    clearance = float(geometry["socket_bore_radius_m"]) - float(
        geometry["key_radius_m"])
    future_bound = preflight.bounds.key_surface_m
    if (not math.isfinite(clearance) or
            clearance <= future_bound or
            max_visual_lateral_error_m >= clearance - future_bound):
        raise ValueError("arrival visual budget exceeds cylinder CAD clearance")
    cameras = set(intrinsics_full)
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={camera: row["K_undist"]
                           for camera, row in intrinsics_full.items()},
        extrinsics_full=extrinsics_full,
        calibrated_camera_ids=cameras)
    if not set(rows) <= cameras:
        raise ValueError("arrival capture uses an uncalibrated camera")
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    frames = []
    for camera in sorted(rows):
        image_path = bundle / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("arrival raw image cannot be decoded")
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append(CalibratedXYFrame(
            camera, rows[camera]["timestamp_s"], rgb,
            camera_socket_transform(
                T_camera_world=extrinsics_full[camera], T_world_robot=c2r,
                T_robot_socket=socket),
            np.asarray(intrinsics_full[camera]["K_undist"], dtype=float)))
    grounded, observations = observe_grounded_cylinder_axis(backend, frames)
    if (len(observations) != len(frames) or
            any(not isinstance(row, VLMObservation) or
                row.stage != "cylinder_tip_axis_line_grounding" or
                row.image_order != (
                    f"raw_preinsert_hold/{frame.camera_id}@"
                    f"{frame.timestamp_s:.6f}",) or
                not isinstance(row.raw_answer, str)
                for row, frame in zip(observations, frames))):
        raise ValueError("arrival VLM responses are not bound to raw views")
    alignment = estimate_grounded_line_alignment(
        grounded, socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m, limits=alignment_limits)
    status = "visual_abstain"
    inliers = alignment.get("inlier_cameras")
    has_metric_axis = (
        alignment.get("status") == "diagnostic_metric_xy_correction" or
        (alignment.get("status") == "abstain" and
         alignment.get("reason") ==
         "continuous_xy_correction_not_confident"))
    if (has_metric_axis and isinstance(inliers, list) and
            len(set(inliers)) >= alignment_limits.minimum_views and
            set(inliers) <= set(rows)):
        grounded_rows = {frame.camera_id: observation for frame, observation
                         in zip(frames, observations)}
        if any(grounded_rows[camera].parse_error is not None or
               grounded_rows[camera].parsed.get("tip_px") is None or
               grounded_rows[camera].parsed.get("axis_line_px") is None
               for camera in inliers):
            raise ValueError("metric arrival uses a malformed VLM view")
        try:
            if any(_parse_object(grounded_rows[camera].raw_answer) !=
                   grounded_rows[camera].parsed or
                   not grounded_rows[camera].parsed["evidence"].strip()
                   for camera in inliers):
                raise ValueError("metric arrival VLM text differs from parsed marks")
        except (KeyError, TypeError) as exc:
            raise ValueError("metric arrival VLM marks are malformed") from exc
        tip = np.asarray(alignment["tip_socket_m"], dtype=float)
        expected_key = (np.linalg.inv(socket) @ wrist @
                        np.linalg.inv(preflight.hypothesis.T_key_hand))
        predicted_tip = (expected_key @ np.array([
            0., 0., float(geometry["key_frame"]["tip_z_m"]), 1.]))[:3]
        if np.linalg.norm(tip - predicted_tip) > (
                future_bound + max_grounded_tip_error_m):
            status = "held_relation_inconsistent"
        elif (max(np.linalg.norm(alignment["rim_error_xy_m"]),
                  np.linalg.norm(alignment["depth_error_xy_m"])) +
              float(alignment["lateral_uncertainty_95_m"]) <=
              max_visual_lateral_error_m and
              float(alignment["axis_tilt_deg"]) <=
              max_visual_axis_tilt_deg):
            status = "visual_alignment_within_budget"
        else:
            status = "residual_requires_new_shift"
    return PostShiftArrivalCheckpoint(
        status, preflight.attempt_id, preflight.candidate_id,
        handoff_path, _sha(handoff_path), execution_path,
        _sha(execution_path), bundle, _sha(bundle / "manifest.json"),
        joint_sample, decision, translation_error, rotation_error,
        alignment, tuple(observations), max_visual_lateral_error_m)


def write_postshift_arrival_checkpoint(
    result: PostShiftArrivalCheckpoint, output_dir: Path,
) -> Path:
    """Save a new read-only arrival observation."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result.to_record(), stream, indent=2,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def verify_postshift_arrival_checkpoint(
    result: PostShiftArrivalCheckpoint, report_path: Path, *,
    preflight: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Recheck saved evidence, not VLM accuracy or physical key penetration."""
    path = Path(report_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(result, PostShiftArrivalCheckpoint) or
            not isinstance(saved, dict) or saved != result.to_record() or
            saved.get("robot_ready") is not False or
            saved.get("axial_retry_allowed") is not False or
            result.status not in {
                "visual_alignment_within_budget", "visual_abstain",
                "held_relation_inconsistent", "residual_requires_new_shift",
            } or result.attempt_id != preflight.attempt_id or
            result.candidate_id != preflight.candidate_id):
        raise ValueError("saved post-shift arrival report changed")
    handoff = result.handoff_report_path.resolve()
    if not handoff.is_file() or _sha(handoff) != result.handoff_report_sha256:
        raise ValueError("arrival handoff source changed")
    verify_postshift_path_handoff(
        handoff, expected=preflight, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=shared_root,
        calibration=calibration)
    execution = result.transfer_execution_path.resolve()
    if (not execution.is_file() or
            _sha(execution) != result.transfer_execution_sha256):
        raise ValueError("arrival transfer execution source changed")
    verify_postshift_transfer_execution(
        execution, handoff_report_path=handoff,
        expected=preflight, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=shared_root,
        calibration=calibration)
    bundle = result.capture_dir.resolve()
    manifest = bundle / "manifest.json"
    if (not manifest.is_file() or
            _sha(manifest) != result.capture_manifest_sha256):
        raise ValueError("arrival capture manifest changed")
    verify_raw_camera_capture(bundle, phase="preinsert")
    return saved
