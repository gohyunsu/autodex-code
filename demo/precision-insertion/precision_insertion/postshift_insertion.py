"""Source-bound, read-only 20 mm replan after a verified cylinder hold shift.

The post-shift tip/axis is a VLM-assisted observation, not a motor command.
This module reuses the exact endpoint screen, frozen socket world, unchanged
FR3/Inspire cuRobo planner and sampled held-key/hand audit. It never executes
contact, labels success, or promotes a preflight to a physical retry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .assets import AssetPaths
from .candidates import select_pose_candidates
from .config import TaskMode
from .endpoint import _load_mesh, screen_grasp_endpoint
from .geometry import pose_angle_deg, validate_se3
from .grounded_lateral import GroundedLateralPreflight
from .postshift_checkpoint import (
    PostShiftCheckpoint, verify_postshift_checkpoint,
)
from .postshift_pose import (
    AxisymmetricHeldHypothesis, reconstruct_axisymmetric_held_hypothesis,
    tip_axis_visual_surface_bound,
)
from .preflight import InsertionPreflight, plan_held_transfer_and_axial
from .raw_camera_capture import verify_raw_camera_capture
from .retry_session import _validated_retry_trial_context
from .targets import InsertionTargets, build_rigid_insertion_targets
from .uncertainty_margin import (
    SurfaceDeviationBounds, audit_sampled_uncertainty_margins,
)
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PostShiftInsertionPreflight:
    status: str
    attempt_id: str
    candidate_id: str
    postshift_report_path: Path
    postshift_report_sha256: str
    postlift_report_path: Path
    postlift_report_sha256: str
    candidate_dir: Path
    yaw_gauge_socket_rad: float
    extra_visual_key_surface_bound_m: float
    required_key_surface_bound_m: float
    bounds: SurfaceDeviationBounds
    hypothesis: AxisymmetricHeldHypothesis
    endpoint: dict[str, Any]
    targets: InsertionTargets
    planning: InsertionPreflight | None
    uncertainty_margin: dict | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_postshift_20mm_preflight_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "postshift_report_path": str(self.postshift_report_path),
            "postshift_report_sha256": self.postshift_report_sha256,
            "postlift_report_path": str(self.postlift_report_path),
            "postlift_report_sha256": self.postlift_report_sha256,
            "candidate_dir": str(self.candidate_dir),
            "yaw_gauge_socket_rad": self.yaw_gauge_socket_rad,
            "extra_visual_key_surface_bound_m": (
                self.extra_visual_key_surface_bound_m),
            "required_key_surface_bound_m": self.required_key_surface_bound_m,
            "surface_bounds": {
                "key_surface_m": self.bounds.key_surface_m,
                "hand_surface_m": self.bounds.hand_surface_m,
                "source": self.bounds.source,
            },
            "held_hypothesis": self.hypothesis.to_record(),
            "endpoint": self.endpoint,
            "targets": self.targets.to_record(),
            "planning": (None if self.planning is None else
                         self.planning.to_record()),
            "uncertainty_margin": self.uncertainty_margin,
            "insertion_replan_allowed": False,
            "scope": "source_bound_sampled_plan_not_contact_or_retry_authorization",
            "robot_ready": False,
        }


def plan_postshift_insertion_preflight(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    catalog: dict, trial, attempt,
    shift_plan: GroundedLateralPreflight,
    checkpoint: PostShiftCheckpoint, checkpoint_report_path: Path,
    bounds: SurfaceDeviationBounds,
    max_visual_tip_error_m: float, max_visual_axis_error_deg: float,
    max_axis_prior_residual_deg: float,
    max_hold_height_delta_m: float,
    max_preinsert_hand_rotation_deg: float,
    screen: Callable = screen_grasp_endpoint,
) -> PostShiftInsertionPreflight:
    """Recheck saved physical sources, then plan held 20 mm geometry.

    Error inputs must be conservative commissioned *worst-case* quantities,
    not a 95% covariance or one VLM answer. A positive result only clears
    sampled geometry; no guarded force/contact motion is produced here.
    """
    limits = (max_visual_tip_error_m, max_visual_axis_error_deg,
              max_axis_prior_residual_deg, max_hold_height_delta_m,
              max_preinsert_hand_rotation_deg)
    if (not all(math.isfinite(float(value)) and value > 0 for value in limits)
            or max_visual_axis_error_deg >= 90 or
            max_axis_prior_residual_deg >= 90 or
            max_preinsert_hand_rotation_deg >= 90):
        raise ValueError("post-shift insertion needs commissioned positive limits")
    bounds.validate()
    if (mode.family != "cylinder" or
            not isinstance(shift_plan, GroundedLateralPreflight) or
            not isinstance(checkpoint, PostShiftCheckpoint) or
            checkpoint.status != "visual_alignment_within_budget" or
            checkpoint.attempt_id != shift_plan.attempt_id or
            checkpoint.candidate_id != shift_plan.candidate_id or
            checkpoint.attempt_id != getattr(attempt, "attempt_id", None) or
            checkpoint.candidate_id != getattr(attempt, "candidate_id", None)):
        raise ValueError("20 mm replan needs one visually aligned held cylinder")
    checkpoint_path = Path(checkpoint_report_path).expanduser().resolve()
    verify_postshift_checkpoint(checkpoint, checkpoint_path, plan=shift_plan)
    _insertion, postlift, postlift_path, root = (
        _validated_retry_trial_context(
            mode=mode, shared_root=shared_root, calibration=calibration,
            catalog=catalog, trial=trial, attempt=attempt,
            postlift_preflight_report_path=shift_plan.postlift_report_path))
    if (postlift_path != shift_plan.postlift_report_path.resolve() or
            postlift.get("schema") !=
            "precision_insertion_bounded_postlift_preflight_v1" or
            postlift.get("bounded_held_relation", {}).get("source") !=
            "physical_grasp_calibration_medoid_not_runtime_key_pose"):
        raise ValueError("post-shift replan lacks the verified physical medoid")
    medoid = validate_se3(postlift["bounded_held_relation"]["T_key_hand"],
                          name="verified physical grasp medoid")
    if not np.allclose(medoid, shift_plan.lateral.T_key_hand,
                       atol=1e-8, rtol=0):
        raise ValueError("shift and insertion use different held grasp medoids")
    original_bounds = postlift["bounded_held_relation"]["surface_bounds"]
    key_prior = float(original_bounds["key_surface_m"])
    hand_prior = float(original_bounds["hand_surface_m"])
    if (not math.isfinite(key_prior) or not math.isfinite(hand_prior) or
            key_prior <= 0 or hand_prior <= 0):
        raise ValueError("physical medoid has no valid surface bounds")

    capture = verify_raw_camera_capture(
        checkpoint.capture_dir, phase="post_lateral_hold")
    alignment = checkpoint.alignment
    inliers = alignment.get("inlier_cameras")
    if (alignment.get("schema") !=
            "precision_insertion_grounded_alignment_v2" or
            not (alignment.get("status") ==
                 "diagnostic_metric_xy_correction" or
                 (alignment.get("status") == "abstain" and
                  alignment.get("reason") ==
                  "continuous_xy_correction_not_confident")) or
            alignment.get("robot_ready") is not False or
            not isinstance(inliers, list) or len(inliers) < 2 or
            len(set(inliers)) != len(inliers) or
            not set(inliers) <= set(capture["frame_evidence"])):
        raise ValueError("post-shift alignment lacks grounded multi-view evidence")
    paths = AssetPaths(root, mode)
    geometry_path = paths.task_geometry
    original = json.loads(shift_plan.diagnostic_report_path.read_text(
        encoding="utf-8"))
    if _sha(geometry_path) != original.get("task_geometry_sha256"):
        raise ValueError("cylinder task geometry changed after the VLM diagnostic")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (geometry.get("key_object") != mode.key_object or
            geometry.get("socket_pose_object") != mode.socket_object):
        raise ValueError("task geometry does not match selected cylinder")
    axis_key = np.asarray(geometry["key_frame"]["insertion_axis"],
                          dtype=np.float64)
    tip_key = np.array([0., 0., float(geometry["key_frame"]["tip_z_m"])])
    key_mesh = _load_mesh(paths.raw_mesh(mode.key_object))
    visual_extra = tip_axis_visual_surface_bound(
        key_vertices_m=np.asarray(key_mesh.vertices), tip_key_m=tip_key,
        max_tip_error_m=max_visual_tip_error_m,
        max_axis_error_deg=max_visual_axis_error_deg)
    required_key = key_prior + visual_extra
    if (bounds.key_surface_m < required_key or
            bounds.hand_surface_m < hand_prior):
        raise ValueError("future surface bounds omit post-shift visual error")

    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    wrist = validate_se3(planner.fk_wrist(checkpoint.joint_sample.full_q),
                         name="post-shift measured wrist FK")
    hypothesis = reconstruct_axisymmetric_held_hypothesis(
        T_robot_socket=socket, T_robot_hand_measured=wrist,
        T_key_hand_prior=medoid, tip_key_m=tip_key,
        insertion_axis_key=axis_key,
        tip_socket_m=alignment["tip_socket_m"],
        insertion_axis_socket=alignment["insertion_axis_socket"],
        max_tip_prior_residual_m=key_prior + max_visual_tip_error_m,
        max_axis_prior_residual_deg=max_axis_prior_residual_deg)
    # Both endpoint and held-path targets use the same unobservable yaw gauge.
    heading = hypothesis.T_socket_key[:3, 0]
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    selected = select_pose_candidates(
        catalog, expected_mode=mode,
        tabletop_pose_stem=attempt.tabletop_pose_stem)
    matches = [row for row in selected.get("candidates", [])
               if tuple(row["key"]) == trial.selected_candidate_key]
    if selected.get("status") != "candidates_available" or len(matches) != 1:
        raise ValueError("post-shift grasp is absent from verified v8 catalogue")
    candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=hypothesis.T_key_hand,
        cylinder_yaw_gauge_socket_rad=yaw)
    target_socket_key = np.linalg.inv(socket) @ targets.T_robot_key_preinsert
    target_tip = (target_socket_key @ np.r_[tip_key, 1.])[:3]
    if (abs(hypothesis.observed_tip_socket_m[2] - target_tip[2]) >
            max_hold_height_delta_m or
            pose_angle_deg(wrist, targets.T_robot_hand_preinsert) >
            max_preinsert_hand_rotation_deg):
        raise ValueError("replan would change the held height or orientation")
    endpoint = screen(
        shared_root=root, mode=mode, candidate_dir=candidate_dir,
        minimum_hand_clearance_m=trial.limits.minimum_hand_clearance_m,
        T_key_hand_override=hypothesis.T_key_hand,
        hand_poses_override={"measured_postshift":
                             checkpoint.joint_sample.full_q[7:]},
        override_source="postshift_tip_axis_physical_medoid_yaw_gauge",
        cylinder_yaw_gauge_socket_rad=yaw)
    screened = validate_se3(endpoint.get("T_socket_key_tested"),
                            name="screened cylinder endpoint")
    if not np.allclose(screened, np.linalg.inv(socket) @
                       targets.T_robot_key_verification,
                       atol=1e-8, rtol=0):
        raise ValueError("endpoint screen and axial goal use different yaw/pose")

    def result(status: str, planning=None, margin=None):
        return PostShiftInsertionPreflight(
            status, attempt.attempt_id, attempt.candidate_id,
            checkpoint_path, _sha(checkpoint_path),
            postlift_path, _sha(postlift_path), candidate_dir,
            yaw, visual_extra, required_key, bounds, hypothesis,
            endpoint, targets, planning, margin)

    if endpoint.get("endpoint_pass") is not True:
        return result("postshift_20mm_endpoint_rejected")
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=root, calibration=calibration,
        targets=targets, start_q=checkpoint.joint_sample.full_q,
        held_hand_q=checkpoint.joint_sample.full_q[7:],
        held_hand_source="measured", limits=trial.limits,
        axial_waypoint_step_m=trial.axial_waypoint_step_m)
    if not planning.sampled_planning_pass:
        return result("postshift_" + planning.status, planning)
    margin = audit_sampled_uncertainty_margins(
        mode=mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    return result(
        "sampled_postshift_20mm_preflight_pass" if
        margin["sampled_margin_pass"] else
        "postshift_uncertainty_margin_rejected", planning, margin)


def write_postshift_insertion_preflight(
    result: PostShiftInsertionPreflight, output_dir: Path,
) -> Path:
    """Save provenance, exact joint paths and no physical success claim."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.planning is not None:
        arrays = {
            name: path for name, path in (
                ("transfer", result.planning.transfer_trajectory),
                ("axial", result.planning.axial_trajectory))
            if path is not None}
        if arrays:
            archive = target / "planned_trajectories.npz"
            np.savez_compressed(archive, **arrays)
            record["planned_trajectories"] = archive.name
            record["planned_trajectories_sha256"] = _sha(archive)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path
