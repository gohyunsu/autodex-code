"""Read-only visual re-observation after a logged lateral hold shift.

An external commissioned controller, not this module, must move the robot.
Fresh AutoDex camera frames and measured joints are required after that
motion. Even a visually aligned result is not a 20 mm insertion endpoint,
contact-controller, or physical task-success authorization.
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
from .bounded_postlift import verify_bounded_postlift_preflight
from .calibration import validate_session_camera_calibration
from .config import TaskMode
from .frame_provenance import bounded_capture_skew_s
from .geometry import pose_angle_deg, validate_se3
from .grounded_alignment import (
    AlignmentLimits, estimate_grounded_line_alignment,
    observe_grounded_cylinder_axis,
)
from .grounded_lateral import (
    GroundedLateralPreflight, verify_grounded_lateral_preflight,
)
from .live_robot_state import LiveRobotState
from .observer import ImageVLM
from .raw_camera_capture import verify_raw_camera_capture
from .session_bootstrap import _safe_id
from .xy_overlay import CalibratedXYFrame, camera_socket_transform
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_log(
    path: Path, *, plan: GroundedLateralPreflight,
    plan_report_path: Path,
) -> tuple[dict, float, float]:
    log_path = Path(path).expanduser().resolve()
    record = json.loads(log_path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or
            record.get("schema") !=
            "precision_insertion_lateral_hold_execution_v1" or
            record.get("source") != "commissioned_lateral_controller" or
            record.get("attempt_id") != plan.attempt_id or
            record.get("candidate_id") != plan.candidate_id or
            record.get("preflight_report_path") != str(
                Path(plan_report_path).expanduser().resolve()) or
            record.get("preflight_report_sha256") != _sha(
                Path(plan_report_path).expanduser().resolve()) or
            record.get("measurement") != {
                "trajectory_complete": True,
                "safety_abort": False,
                "grasp_held": True,
            }):
        raise ValueError("lateral execution log does not confirm this held shift")
    started = float(record["started_at_s"])
    completed = float(record["completed_at_s"])
    if (not math.isfinite(started) or not math.isfinite(completed) or
            not plan.decision_timestamp_s < started < completed):
        raise ValueError("lateral execution is not after its preflight")
    sources = record.get("source_records")
    if (not isinstance(sources, dict) or set(sources) != {
            "trajectory_feedback", "safety", "grasp_state"}):
        raise ValueError("lateral execution lacks independent source records")
    for name, source in sources.items():
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError(f"invalid lateral execution source: {name}")
        source_path = Path(source["path"])
        if (not source_path.is_absolute() or not source_path.is_file() or
                _sha(source_path) != source["sha256"]):
            raise ValueError(f"lateral execution source changed: {name}")
    return record, started, completed


@dataclass(frozen=True)
class PostShiftCheckpoint:
    status: str
    attempt_id: str
    candidate_id: str
    lateral_preflight_report_path: Path
    lateral_preflight_report_sha256: str
    lateral_execution_path: Path
    lateral_execution_sha256: str
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
            "schema": "precision_insertion_postshift_checkpoint_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "lateral_preflight_report_path": str(
                self.lateral_preflight_report_path),
            "lateral_preflight_report_sha256": (
                self.lateral_preflight_report_sha256),
            "lateral_execution_path": str(self.lateral_execution_path),
            "lateral_execution_sha256": self.lateral_execution_sha256,
            "capture_dir": str(self.capture_dir),
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "joint_sample": self.joint_sample.to_record(),
            "decision_timestamp_s": self.decision_timestamp_s,
            "measured_goal_translation_error_m": (
                self.measured_goal_translation_error_m),
            "measured_goal_rotation_error_deg": (
                self.measured_goal_rotation_error_deg),
            "alignment": self.alignment,
            "vlm_observations": [v.to_record()
                                 for v in self.vlm_observations],
            "visual_lateral_budget_m": self.visual_lateral_budget_m,
            "scope": "postshift_visual_diagnostic_not_insertion_authorization",
            "insertion_replan_allowed": False,
            "robot_ready": False,
        }


def assess_postshift_alignment(
    *, plan: GroundedLateralPreflight, plan_report_path: Path,
    execution_log_path: Path, capture_dir: Path,
    joint_sample: LiveRobotState, decision_timestamp_s: float,
    mode: TaskMode, shared_root: Path, calibration, planner,
    intrinsics_full: Mapping, extrinsics_full: Mapping,
    backend: ImageVLM, alignment_limits: AlignmentLimits,
    max_capture_skew_s: float, max_execution_observation_gap_s: float,
    max_frame_age_s: float, max_joint_frame_skew_s: float,
    max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
    max_joint_goal_error_rad: float, max_goal_translation_error_m: float,
    max_goal_rotation_error_deg: float,
    max_visual_lateral_error_m: float, max_visual_axis_tilt_deg: float,
    max_grounded_tip_error_m: float,
) -> PostShiftCheckpoint:
    """Require an executed shift and *new* images before visual assessment."""
    alignment_limits.validate()
    limits = (
        max_capture_skew_s, max_execution_observation_gap_s,
        max_frame_age_s, max_joint_frame_skew_s, max_arm_hand_skew_s,
        max_hand_command_error_raw, max_arm_velocity_rad_s,
        max_joint_goal_error_rad, max_goal_translation_error_m,
        max_goal_rotation_error_deg, max_visual_lateral_error_m,
        max_visual_axis_tilt_deg, max_grounded_tip_error_m)
    if not all(math.isfinite(float(v)) and v > 0 for v in limits):
        raise ValueError("post-shift checkpoint needs commissioned positive limits")
    if (mode.family != "cylinder" or
            not isinstance(plan, GroundedLateralPreflight) or
            plan.lateral.status != "sampled_lateral_hold_shift_pass" or
            plan.lateral.sampled_audit.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm}):
        raise ValueError("post-shift needs a passing cylinder lateral plan")
    plan_path = Path(plan_report_path).expanduser().resolve()
    verify_grounded_lateral_preflight(plan, plan_path)
    root = Path(shared_root).expanduser().resolve()
    candidate_key = plan.candidate_id.split("/")
    if len(candidate_key) != 3:
        raise ValueError("post-shift grasp candidate ID is invalid")
    verify_bounded_postlift_preflight(
        plan.postlift_report_path, mode=mode, shared_root=root,
        candidate_dir=AssetPaths(root, mode).candidate_dir /
        Path(*candidate_key))
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    geometry_path = AssetPaths(root, mode).task_geometry
    prior = json.loads(plan.diagnostic_report_path.read_text(
        encoding="utf-8"))
    if _sha(geometry_path) != prior.get("task_geometry_sha256"):
        raise ValueError("post-shift cylinder CAD changed since diagnostic")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (geometry.get("key_object") != mode.key_object or
            geometry.get("socket_pose_object") != mode.socket_object):
        raise ValueError("post-shift CAD belongs to another key or socket")
    clearance = float(geometry["socket_bore_radius_m"]) - float(
        geometry["key_radius_m"])
    future_key_bound = float(plan.lateral.sampled_audit[
        "limits"]["key_surface_bound_m"])
    if (not math.isfinite(clearance) or not math.isfinite(future_key_bound) or
            clearance <= future_key_bound or
            max_visual_lateral_error_m >= clearance - future_key_bound):
        raise ValueError("visual budget exceeds bounded cylinder CAD clearance")
    execution_path = Path(execution_log_path).expanduser().resolve()
    _log, _started, completed = _execution_log(
        execution_path, plan=plan, plan_report_path=plan_path)
    bundle = Path(capture_dir).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="post_lateral_hold")
    frame_rows = capture["frame_evidence"]
    old_frames = prior["frame_binding"]
    if (capture["request_id"] == prior["frame_request_id"] or
            not set(frame_rows) <= set(old_frames) or
            len(frame_rows) < alignment_limits.minimum_views or
            bounded_capture_skew_s(frame_rows) > max_capture_skew_s or
            any(row["frame_id"] <= old_frames[camera]["frame_id"]
                for camera, row in frame_rows.items())):
        raise ValueError("post-shift needs new synchronized camera frames")
    first = min(row["timestamp_s"] - row["max_error_s"]
                for row in frame_rows.values())
    last = max(row["timestamp_s"] + row["max_error_s"]
               for row in frame_rows.values())
    decision = float(decision_timestamp_s)
    if (not math.isfinite(decision) or
            completed >= first or
            first - completed > max_execution_observation_gap_s or
            decision < last or
            decision - first > max_frame_age_s or
            not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("post-shift frames/state do not follow completed motion")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    if (joint_sample.sample_timestamp_s < completed or
            joint_sample.sample_timestamp_s < first - max_joint_frame_skew_s or
            joint_sample.sample_timestamp_s > last + max_joint_frame_skew_s or
            not np.allclose(joint_sample.full_q[7:],
                            plan.lateral.start_q[7:], atol=1e-4, rtol=0) or
            np.max(np.abs(joint_sample.full_q -
                          plan.lateral.trajectory[-1])) >
            max_joint_goal_error_rad):
        raise ValueError("post-shift measured joints differ from completed plan")
    wrist = validate_se3(planner.fk_wrist(joint_sample.full_q),
                         name="post-shift measured wrist FK")
    goal = plan.lateral.T_robot_hand_goal
    translation_error = float(np.linalg.norm(
        wrist[:3, 3] - goal[:3, 3]))
    rotation_error = pose_angle_deg(wrist, goal)
    if (translation_error > max_goal_translation_error_m or
            rotation_error > max_goal_rotation_error_deg):
        raise ValueError("post-shift measured wrist missed the lateral target")
    cameras = set(intrinsics_full)
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={camera: row["K_undist"]
                           for camera, row in intrinsics_full.items()},
        extrinsics_full=extrinsics_full,
        calibrated_camera_ids=cameras)
    if not set(frame_rows) <= cameras:
        raise ValueError("post-shift capture uses an uncalibrated camera")
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    frames = []
    for camera in sorted(frame_rows):
        image_path = bundle / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append(CalibratedXYFrame(
            camera, frame_rows[camera]["timestamp_s"], rgb,
            camera_socket_transform(
                T_camera_world=extrinsics_full[camera], T_world_robot=c2r,
                T_robot_socket=socket),
            np.asarray(intrinsics_full[camera]["K_undist"], dtype=float)))
    grounded, observations = observe_grounded_cylinder_axis(backend, frames)
    alignment = estimate_grounded_line_alignment(
        grounded, socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m,
        limits=alignment_limits)
    status = "visual_abstain"
    if (alignment["status"] == "diagnostic_metric_xy_correction" or
            alignment.get("reason") ==
            "continuous_xy_correction_not_confident"):
        tip = np.asarray(alignment["tip_socket_m"], dtype=float)
        expected_key = (np.linalg.inv(socket) @ wrist @
                        np.linalg.inv(plan.lateral.T_key_hand))
        predicted_tip = (expected_key @ np.array([
            0., 0., float(geometry["key_frame"]["tip_z_m"]), 1.]))[:3]
        if (np.linalg.norm(tip - predicted_tip) >
                future_key_bound + max_grounded_tip_error_m):
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
    return PostShiftCheckpoint(
        status, plan.attempt_id, plan.candidate_id, plan_path,
        _sha(plan_path), execution_path, _sha(execution_path), bundle,
        _sha(bundle / "manifest.json"), joint_sample, decision,
        translation_error, rotation_error, alignment, tuple(observations),
        max_visual_lateral_error_m)


def write_postshift_checkpoint(
    result: PostShiftCheckpoint, output_dir: Path,
) -> Path:
    """Save visual status without changing the attempt's success labels."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result.to_record(), stream, indent=2,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target / "report.json"


def verify_postshift_checkpoint(
    result: PostShiftCheckpoint, report_path: Path,
    *, plan: GroundedLateralPreflight,
) -> dict:
    """Bind a saved re-observation to its shift, feedback and raw pixels.

    File integrity is rechecked before any later endpoint calculation. This
    does not certify the authenticity of external controller or camera logs.
    """
    path = Path(report_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(result, PostShiftCheckpoint) or
            not isinstance(saved, dict) or saved != result.to_record() or
            saved.get("robot_ready") is not False or
            saved.get("insertion_replan_allowed") is not False or
            result.attempt_id != plan.attempt_id or
            result.candidate_id != plan.candidate_id or
            result.status not in {
                "visual_alignment_within_budget", "visual_abstain",
                "held_relation_inconsistent", "residual_requires_new_shift",
            }):
        raise ValueError("saved post-shift checkpoint changed")
    plan_path = result.lateral_preflight_report_path.resolve()
    if (not plan_path.is_file() or
            _sha(plan_path) != result.lateral_preflight_report_sha256):
        raise ValueError("post-shift lateral plan source changed")
    verify_grounded_lateral_preflight(plan, plan_path)
    execution = result.lateral_execution_path.resolve()
    if (not execution.is_file() or
            _sha(execution) != result.lateral_execution_sha256):
        raise ValueError("post-shift execution source changed")
    _execution_log(execution, plan=plan, plan_report_path=plan_path)
    capture = result.capture_dir.resolve()
    manifest = capture / "manifest.json"
    if (not manifest.is_file() or
            _sha(manifest) != result.capture_manifest_sha256):
        raise ValueError("post-shift capture manifest changed")
    verify_raw_camera_capture(capture, phase="post_lateral_hold")
    return saved
