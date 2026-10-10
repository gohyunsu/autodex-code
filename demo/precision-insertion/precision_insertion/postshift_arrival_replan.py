"""Re-screen a *newly observed* retry arrival before any axial replay.

The transfer's old axial path is never reused: a fresh multi-view key tip/axis
and measured wrist define a new held hypothesis. Exact 20 mm endpoint CAD,
axial-only cuRobo waypoints and sampled held-key/hand clearances are recomputed
from that state. A passing packet is still read-only, not contact permission.
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
from .config import TaskMode
from .endpoint import _load_mesh, screen_grasp_endpoint
from .geometry import validate_se3
from .grounded_lateral import GroundedLateralPreflight
from .postshift_arrival_checkpoint import (
    PostShiftArrivalCheckpoint, verify_postshift_arrival_checkpoint,
)
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import (
    PostShiftInsertionPreflight, verify_postshift_insertion_preflight,
)
from .postshift_pose import (
    AxisymmetricHeldHypothesis, reconstruct_axisymmetric_held_hypothesis,
    tip_axis_visual_surface_bound,
)
from .preflight import InsertionPreflight, _path, plan_held_transfer_and_axial
from .targets import InsertionTargets, build_rigid_insertion_targets
from .uncertainty_margin import (
    SurfaceDeviationBounds, audit_sampled_uncertainty_margins,
)
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PostShiftArrivalReplan:
    status: str
    attempt_id: str
    candidate_id: str
    arrival_report_path: Path
    arrival_report_sha256: str
    previous_preflight_report_path: Path
    previous_preflight_report_sha256: str
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
            "schema": "precision_insertion_postshift_arrival_replan_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "arrival_report_path": str(self.arrival_report_path),
            "arrival_report_sha256": self.arrival_report_sha256,
            "previous_preflight_report_path": str(
                self.previous_preflight_report_path),
            "previous_preflight_report_sha256": (
                self.previous_preflight_report_sha256),
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
            "old_axial_path_reusable": False,
            "axial_contact_authorized": False,
            "scope": "fresh_arrival_endpoint_and_sampled_axial_plan_not_motion",
            "robot_ready": False,
        }


def plan_postshift_arrival_axial(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    trial, previous: PostShiftInsertionPreflight,
    previous_report_path: Path, arrival: PostShiftArrivalCheckpoint,
    arrival_report_path: Path, checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight, bounds: SurfaceDeviationBounds,
    max_visual_tip_error_m: float, max_visual_axis_error_deg: float,
    max_axis_prior_residual_deg: float,
    screen: Callable = screen_grasp_endpoint,
) -> PostShiftArrivalReplan:
    """Plan only if measured arrival already meets the *new* preinsert pose.

    Visual errors are commissioned worst-case limits, not VLM confidence.
    If the refitted held relation moves the target outside the measured
    preinsert goal tolerance, this rejects instead of silently transferring
    again or replaying the previous axial trajectory.
    """
    numbers = (max_visual_tip_error_m, max_visual_axis_error_deg,
               max_axis_prior_residual_deg)
    if (not all(math.isfinite(float(x)) and x > 0 for x in numbers) or
            max_visual_axis_error_deg >= 90 or
            max_axis_prior_residual_deg >= 90):
        raise ValueError("arrival replan needs commissioned positive limits")
    bounds.validate()
    if (mode.family != "cylinder" or
            not math.isclose(mode.target_depth_m, .020, abs_tol=1e-9) or
            not isinstance(arrival, PostShiftArrivalCheckpoint) or
            arrival.status != "visual_alignment_within_budget" or
            arrival.attempt_id != previous.attempt_id or
            arrival.candidate_id != previous.candidate_id or
            previous.status != "sampled_postshift_20mm_preflight_pass"):
        raise ValueError("arrival axial replan needs a newly aligned 20 mm hold")
    root = Path(shared_root).expanduser().resolve()
    old_path = Path(previous_report_path).expanduser().resolve()
    verify_postshift_insertion_preflight(
        old_path, expected=previous, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=root,
        calibration=calibration)
    arrival_path = Path(arrival_report_path).expanduser().resolve()
    verify_postshift_arrival_checkpoint(
        arrival, arrival_path, preflight=previous,
        checkpoint=checkpoint, shift_plan=shift_plan,
        mode=mode, shared_root=root, calibration=calibration)
    handoff = json.loads(arrival.handoff_report_path.read_text(
        encoding="utf-8"))
    if (Path(handoff.get("postshift_preflight_report_path", "")).resolve() !=
            old_path or
            handoff.get("postshift_preflight_report_sha256") != _sha(old_path)):
        raise ValueError("arrival transfer did not use this previous preflight")

    paths = AssetPaths(root, mode)
    geometry_path = paths.task_geometry
    if _sha(geometry_path) != previous.targets.task_geometry_sha256:
        raise ValueError("arrival task geometry changed since prior preflight")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (geometry.get("key_object") != mode.key_object or
            geometry.get("socket_pose_object") != mode.socket_object):
        raise ValueError("arrival task geometry differs from selected cylinder")
    tip_key = np.array([0., 0., float(geometry["key_frame"]["tip_z_m"])])
    axis_key = np.asarray(geometry["key_frame"]["insertion_axis"], dtype=float)
    visual_extra = tip_axis_visual_surface_bound(
        key_vertices_m=np.asarray(_load_mesh(paths.raw_mesh(
            mode.key_object)).vertices), tip_key_m=tip_key,
        max_tip_error_m=max_visual_tip_error_m,
        max_axis_error_deg=max_visual_axis_error_deg)
    postlift = json.loads(previous.postlift_report_path.read_text(
        encoding="utf-8"))
    physical_bounds = postlift["bounded_held_relation"]["surface_bounds"]
    physical_key = float(physical_bounds["key_surface_m"])
    physical_hand = float(physical_bounds["hand_surface_m"])
    required_key = max(previous.bounds.key_surface_m,
                       physical_key + visual_extra)
    if (bounds.key_surface_m < required_key or
            bounds.hand_surface_m < max(previous.bounds.hand_surface_m,
                                        physical_hand)):
        raise ValueError("arrival surface bounds omit prior or fresh visual error")

    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    wrist = validate_se3(planner.fk_wrist(arrival.joint_sample.full_q),
                         name="measured post-transfer wrist FK")
    alignment = arrival.alignment
    if (alignment.get("schema") !=
            "precision_insertion_grounded_alignment_v2" or
            not isinstance(alignment.get("inlier_cameras"), list) or
            len(set(alignment["inlier_cameras"])) < 2 or
            alignment.get("robot_ready") is not False):
        raise ValueError("arrival has no grounded multi-view tip and axis")
    hypothesis = reconstruct_axisymmetric_held_hypothesis(
        T_robot_socket=socket, T_robot_hand_measured=wrist,
        T_key_hand_prior=previous.hypothesis.T_key_hand,
        tip_key_m=tip_key, insertion_axis_key=axis_key,
        tip_socket_m=alignment["tip_socket_m"],
        insertion_axis_socket=alignment["insertion_axis_socket"],
        max_tip_prior_residual_m=(previous.bounds.key_surface_m +
                                  max_visual_tip_error_m),
        max_axis_prior_residual_deg=max_axis_prior_residual_deg)
    heading = hypothesis.T_socket_key[:3, 0]
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=hypothesis.T_key_hand,
        cylinder_yaw_gauge_socket_rad=yaw)
    endpoint = screen(
        shared_root=root, mode=mode,
        candidate_dir=previous.candidate_dir,
        minimum_hand_clearance_m=trial.limits.minimum_hand_clearance_m,
        T_key_hand_override=hypothesis.T_key_hand,
        hand_poses_override={"measured_arrival":
                             arrival.joint_sample.full_q[7:]},
        override_source="fresh_arrival_tip_axis_physical_medoid_yaw_gauge",
        cylinder_yaw_gauge_socket_rad=yaw)
    tested = validate_se3(endpoint.get("T_socket_key_tested"),
                          name="arrival screened cylinder endpoint")
    if not np.allclose(tested, np.linalg.inv(socket) @
                       targets.T_robot_key_verification,
                       atol=1e-8, rtol=0):
        raise ValueError("arrival endpoint and axial goal use different poses")

    def result(status: str, planning=None, margin=None):
        return PostShiftArrivalReplan(
            status, arrival.attempt_id, arrival.candidate_id,
            arrival_path, _sha(arrival_path), old_path, _sha(old_path),
            previous.candidate_dir, yaw, visual_extra, required_key,
            bounds, hypothesis, endpoint, targets, planning, margin)

    if endpoint.get("endpoint_pass") is not True:
        return result("arrival_20mm_endpoint_rejected")
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=root, calibration=calibration, targets=targets,
        start_q=arrival.joint_sample.full_q,
        held_hand_q=arrival.joint_sample.full_q[7:],
        held_hand_source="measured", limits=trial.limits,
        axial_waypoint_step_m=trial.axial_waypoint_step_m,
        start_at_preinsert=True)
    if not planning.sampled_planning_pass:
        status = (planning.status if planning.status.startswith("arrival_")
                  else "arrival_" + planning.status)
        return result(status, planning)
    if (planning.transfer_trajectory is None or
            not np.array_equal(planning.transfer_trajectory[0],
                               planning.transfer_trajectory[1]) or
            not np.array_equal(planning.axial_trajectory[0],
                               arrival.joint_sample.full_q)):
        raise ValueError("arrival replan unexpectedly contains a transfer")
    margin = audit_sampled_uncertainty_margins(
        mode=mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    return result(
        "sampled_arrival_20mm_axial_preflight_pass" if
        margin["sampled_margin_pass"] else
        "arrival_uncertainty_margin_rejected", planning, margin)


def write_postshift_arrival_replan(
    result: PostShiftArrivalReplan, output_dir: Path,
) -> Path:
    """Persist only fresh axial planning evidence, not a motor command."""
    if result.status == "sampled_arrival_20mm_axial_preflight_pass" and (
            result.endpoint.get("endpoint_pass") is not True or
            result.planning is None or
            not result.planning.sampled_planning_pass or
            result.uncertainty_margin is None or
            result.uncertainty_margin.get("sampled_margin_pass") is not True):
        raise ValueError("passing arrival replan lacks endpoint/path/margin")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.planning is not None and result.planning.axial_trajectory is not None:
        archive = target / "planned_axial.npz"
        np.savez_compressed(archive, axial=result.planning.axial_trajectory)
        record["planned_axial"] = archive.name
        record["planned_axial_sha256"] = _sha(archive)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def verify_postshift_arrival_replan(
    report_path: Path, *, expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint, checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Recheck sources and exact saved axial bytes; never approve contact."""
    path = Path(report_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    extra = {"planned_axial", "planned_axial_sha256"}
    if (not isinstance(expected, PostShiftArrivalReplan) or
            not isinstance(saved, dict) or
            {key: value for key, value in saved.items() if key not in extra} !=
            expected.to_record() or
            saved.get("robot_ready") is not False or
            saved.get("old_axial_path_reusable") is not False or
            saved.get("axial_contact_authorized") is not False or
            expected.attempt_id != arrival.attempt_id or
            expected.candidate_id != arrival.candidate_id):
        raise ValueError("saved arrival axial report changed")
    root = Path(shared_root).expanduser().resolve()
    old = expected.previous_preflight_report_path.resolve()
    source = expected.arrival_report_path.resolve()
    if (not old.is_file() or _sha(old) !=
            expected.previous_preflight_report_sha256 or
            not source.is_file() or _sha(source) !=
            expected.arrival_report_sha256):
        raise ValueError("arrival axial source report changed")
    verify_postshift_insertion_preflight(
        old, expected=previous, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=root,
        calibration=calibration)
    verify_postshift_arrival_checkpoint(
        arrival, source, preflight=previous, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=root,
        calibration=calibration)
    if (old != Path(json.loads(arrival.handoff_report_path.read_text(
            encoding="utf-8"))["postshift_preflight_report_path"]).resolve() or
            not np.allclose(expected.hypothesis.T_key_hand,
                            expected.targets.T_key_hand, atol=1e-8, rtol=0)):
        raise ValueError("arrival replan uses a different held source")
    expected.candidate_dir.resolve().relative_to(
        AssetPaths(root, mode).candidate_dir.resolve())
    paths = AssetPaths(root, mode)
    files = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": expected.candidate_dir / "wrist_se3.npy",
        "pregrasp_pose": expected.candidate_dir / "pregrasp_pose.npy",
        "grasp_pose": expected.candidate_dir / "grasp_pose.npy",
    }
    hashes = expected.endpoint.get("input_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != set(files) or
            any(not source_file.is_file() or
                _sha(source_file) != hashes[name]
                for name, source_file in files.items()) or
            not np.allclose(expected.endpoint.get("T_key_hand"),
                            expected.hypothesis.T_key_hand,
                            atol=1e-8, rtol=0) or
            not math.isclose(float(expected.endpoint.get(
                "cylinder_yaw_gauge_socket_rad", float("nan"))),
                expected.yaw_gauge_socket_rad, abs_tol=1e-12)):
        raise ValueError("arrival endpoint CAD or held hypothesis changed")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    if not np.allclose(
            validate_se3(expected.endpoint.get("T_socket_key_tested"),
                         name="saved arrival endpoint"),
            np.linalg.inv(socket) @ expected.targets.T_robot_key_verification,
            atol=1e-8, rtol=0):
        raise ValueError("arrival endpoint differs from axial target")
    planning = expected.planning
    passing = saved.get("status") == "sampled_arrival_20mm_axial_preflight_pass"
    if passing and (expected.endpoint.get("endpoint_pass") is not True or
                    planning is None or not planning.sampled_planning_pass or
                    expected.uncertainty_margin is None or
                    expected.uncertainty_margin.get(
                        "sampled_margin_pass") is not True):
        raise ValueError("passing arrival plan lacks endpoint/path/margin")
    if passing:
        repeated = audit_sampled_uncertainty_margins(
            mode=mode, endpoint=expected.endpoint,
            targets=expected.targets, planning=planning,
            bounds=expected.bounds)
        if repeated != expected.uncertainty_margin or not repeated[
                "sampled_margin_pass"]:
            raise ValueError("arrival sampled margins changed")
    name = saved.get("planned_axial")
    if name is None:
        if planning is not None and planning.axial_trajectory is not None:
            raise ValueError("arrival axial trajectory artifact is missing")
        return saved
    if name != "planned_axial.npz" or planning is None:
        raise ValueError("invalid arrival axial archive")
    archive = (path.parent / name).resolve()
    if (not archive.is_relative_to(path.parent) or not archive.is_file() or
            _sha(archive) != saved.get("planned_axial_sha256")):
        raise ValueError("arrival axial trajectory bytes changed")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"axial"}:
            raise ValueError("arrival archive must contain only axial path")
        axial = _path(data["axial"], "saved arrival axial",
                      arrival.joint_sample.full_q,
                      arrival.joint_sample.full_q[7:])
    if (not np.array_equal(axial, planning.axial_trajectory) or
            planning.transfer_trajectory is None or
            not np.array_equal(planning.transfer_trajectory[0],
                               planning.transfer_trajectory[1]) or
            not np.array_equal(planning.transfer_trajectory[0], axial[0])):
        raise ValueError("arrival plan reuses a transfer or stale axial path")
    return saved
