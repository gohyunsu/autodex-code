"""Recheck the *observed* lifted grasp before transfer to the socket.

The initial v8 plan assumes the BODex key/hand relation survives squeeze.
After a physical lift, this module replaces that assumption with a fresh
multiview key pose and measured Franka/Inspire state. It reuses the exact
endpoint screen and existing held-transfer planner; it sends no commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .geometry import validate_se3
from .held_relation import HeldRelation, resolve_postlift_held_relation
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .preflight import InsertionPreflight, plan_held_transfer_and_axial
from .records import AttemptRecord
from .targets import InsertionTargets, build_rigid_insertion_targets
from .trial_preflight import TrialPreflight, _canonical_sha256
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


@dataclass(frozen=True)
class PostLiftPreflight:
    status: str
    attempt_id: str
    candidate_key: tuple[str, str, str]
    session_calibration_sha256: str
    catalog_sha256: str
    trial_key_observation_id: str
    key_observation_id: str
    key_capture_timestamp_s: float
    joint_timestamp_s: float
    live_start_q: np.ndarray
    joint_feedback: LiveRobotState
    T_world_key_observed: np.ndarray
    T_robot_wrist_measured: np.ndarray
    relation: HeldRelation
    endpoint_screen: dict | None
    targets: InsertionTargets | None
    planning: InsertionPreflight | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_postlift_preflight_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_key": list(self.candidate_key),
            "session_calibration_sha256": self.session_calibration_sha256,
            "catalog_sha256": self.catalog_sha256,
            "trial_key_observation_id": self.trial_key_observation_id,
            "key_observation_id": self.key_observation_id,
            "key_capture_timestamp_s": self.key_capture_timestamp_s,
            "joint_timestamp_s": self.joint_timestamp_s,
            "live_start_q": self.live_start_q.tolist(),
            "joint_feedback": self.joint_feedback.to_record(),
            "T_world_key_observed": self.T_world_key_observed.tolist(),
            "T_robot_wrist_measured": self.T_robot_wrist_measured.tolist(),
            "observed_held_relation": self.relation.to_record(),
            "endpoint_screen": self.endpoint_screen,
            "targets": None if self.targets is None else self.targets.to_record(),
            "planning": None if self.planning is None else self.planning.to_record(),
            "scope": "observed_state_replan_not_robot_execution_or_contact_success",
            "robot_ready": False,
        }


def plan_postlift_observed_transfer(
    *, planner, trial: TrialPreflight, attempt: AttemptRecord,
    calibration, catalog: dict, mode: TaskMode, shared_root: Path,
    key_pose_world: np.ndarray, key_observation_id: str,
    key_capture_timestamp_s: float, key_pose_source: str,
    joint_sample: LiveRobotState, max_state_skew_s: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    max_grasp_translation_drift_m: float,
    max_grasp_rotation_drift_deg: float,
    limits: PathAuditLimits, axial_waypoint_step_m: float,
    screen: Callable = screen_grasp_endpoint,
) -> PostLiftPreflight:
    """Replan centered 20 mm transfer from a synchronized post-lift state.

    A visual grasp-success label alone is insufficient: the independently
    observed key and measured wrist must yield a stable held relation. The
    initial trial plan is only provenance for the selected v8 grasp and fixed
    world. Any uncommissioned source, stale sample or changed catalogue fails
    before planning.
    """
    root = Path(shared_root).expanduser().resolve()
    if (not isinstance(trial, TrialPreflight) or
            trial.status != "sampled_planning_pass" or
            trial.selected_candidate_key is None or
            trial.insertion_plan is None or
            not trial.insertion_plan.sampled_planning_pass or
            trial.pickup_plan is None):
        raise ValueError("post-lift transfer needs a passing initial trial preflight")
    if not isinstance(attempt, AttemptRecord) or attempt.mode != mode:
        raise ValueError("post-lift attempt has a different key/socket mode")
    key = trial.selected_candidate_key
    if (attempt.candidate_id != "/".join(key) or
            attempt.tabletop_pose_stem != trial.pose_class["stem"] or
            attempt.labels["grasp_success"] is not True or
            attempt.labels["preinsert_reached"] is not None or
            attempt.labels["insertion_success"] is not None or
            attempt.failure_code is not None):
        raise ValueError("post-lift attempt lacks this grasp's observed lift success")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if (Path(catalog["shared_root"]).expanduser().resolve() != root or
            trial.catalog_sha256 != _canonical_sha256(catalog) or
            trial.session_calibration_sha256 !=
            _canonical_sha256(calibration.record) or
            attempt.session_calibration_sha256 !=
            trial.session_calibration_sha256):
        raise ValueError("post-lift catalogue or frozen socket session changed")
    validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    build_held_scene_from_trial(
        trial_scene=trial.trial_scene, calibration=calibration)
    selected = select_pose_candidates(
        catalog, expected_mode=mode,
        tabletop_pose_stem=attempt.tabletop_pose_stem)
    if selected["status"] != "candidates_available":
        raise ValueError(f"post-lift catalogue unavailable: {selected['status']}")
    matches = [row for row in selected["candidates"]
               if tuple(row["key"]) == key]
    if len(matches) != 1:
        raise ValueError("picked grasp is no longer endpoint-eligible")
    if (key_pose_source != "multiview_foundpose" or
            not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback" or
            not isinstance(key_observation_id, str) or
            not key_observation_id.strip()):
        raise ValueError("post-lift key and wrist need independent live sources")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    timestamp = float(key_capture_timestamp_s)
    joint_time = float(joint_sample.sample_timestamp_s)
    skew = float(max_state_skew_s)
    last_grasp_time = max(event["timestamp_s"] for event in attempt.events
                          if event["stage"] == "grasp_success")
    if (not all(math.isfinite(value) for value in
                (timestamp, joint_time, skew)) or skew <= 0 or
            timestamp <= max(last_grasp_time,
                             trial.key_capture_timestamp_s) or
            joint_time <= last_grasp_time or
            abs(joint_time - timestamp) > skew):
        raise ValueError("post-lift key and joint samples are stale or asynchronous")
    start = np.asarray(joint_sample.full_q, dtype=np.float64)
    if start.shape != (13,) or not np.all(np.isfinite(start)):
        raise ValueError("post-lift FR3/Inspire state must be 13 finite joints")
    limits.validate()
    pose_world = validate_se3(key_pose_world, name="observed post-lift key pose")
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    key_robot = validate_se3(np.linalg.inv(c2r) @ pose_world,
                             name="post-lift T_robot_key")
    wrist_robot = validate_se3(planner.fk_wrist(start),
                               name="measured post-lift wrist FK")
    candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
    nominal_relation = validate_se3(np.load(
        candidate_dir / "wrist_se3.npy", allow_pickle=False),
        name="selected v8 candidate T_key_hand")
    relation = resolve_postlift_held_relation(
        mode=mode, shared_root=root, T_robot_key_observed=key_robot,
        T_robot_hand_measured=wrist_robot,
        candidate_T_key_hand=nominal_relation,
        max_translation_drift_m=max_grasp_translation_drift_m,
        max_rotation_drift_deg=max_grasp_rotation_drift_deg)

    def result(status: str, *, endpoint=None, targets=None,
               planning=None) -> PostLiftPreflight:
        return PostLiftPreflight(
            status, attempt.attempt_id, key,
            trial.session_calibration_sha256, trial.catalog_sha256,
            trial.key_observation_id, key_observation_id,
            timestamp, joint_time, start.copy(), joint_sample, pose_world,
            wrist_robot, relation, endpoint, targets, planning)

    if (relation.translation_drift_m > max_grasp_translation_drift_m or
            relation.rotation_drift_deg > max_grasp_rotation_drift_deg):
        return result("observed_grasp_relation_drift_exceeded")
    endpoint = screen(
        shared_root=root, mode=mode, candidate_dir=candidate_dir,
        minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
        T_key_hand_override=relation.T_key_hand,
        hand_poses_override={"measured_post_lift": start[7:].copy()},
        override_source="multiview_key_pose_plus_live_wrist")
    if (not np.allclose(validate_se3(endpoint.get("T_key_hand"),
                                   name="post-lift endpoint T_key_hand"),
                        relation.T_key_hand, atol=1e-8) or
            not np.allclose(endpoint.get("xy_offset_socket_m"),
                            [0.0, 0.0], atol=1e-12) or
            not np.allclose(endpoint.get("hold_pose_screens", {}).get(
                "measured_post_lift", {}).get("hand_q"),
                start[7:], atol=1e-8)):
        raise ValueError("post-lift endpoint used a different held relation")
    if endpoint.get("endpoint_pass") is not True:
        return result("observed_20mm_endpoint_rejected", endpoint=endpoint)
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=relation.T_key_hand)
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=root, calibration=calibration, targets=targets,
        start_q=start, held_hand_q=start[7:], held_hand_source="measured",
        limits=limits, axial_waypoint_step_m=axial_waypoint_step_m)
    return result(
        "sampled_postlift_preflight_pass" if planning.sampled_planning_pass
        else planning.status, endpoint=endpoint, targets=targets,
        planning=planning)


def write_postlift_preflight(
    result: PostLiftPreflight, output_dir: Path,
) -> Path:
    """Write one immutable observed-state planning bundle, not a replay permit."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.planning is not None and result.planning.transfer_trajectory is not None:
        path = target / "planned_trajectories.npz"
        arrays = {"transfer": result.planning.transfer_trajectory}
        if result.planning.axial_trajectory is not None:
            arrays["axial"] = result.planning.axial_trajectory
        np.savez_compressed(path, **arrays)
        record["planned_trajectories"] = path.name
        record["planned_trajectories_sha256"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return target
