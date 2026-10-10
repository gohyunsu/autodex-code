"""Replan a VLM-proposed 1 mm target from the withdrawn *live* held state.

Reuses the same held transfer/axial preflight as a first trial. It is still a
read-only plan: no guarded insertion, force response, measured finger state,
or physical task outcome is implied by a passing sampled path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .geometry import pose_angle_deg, validate_se3
from .path_audit import PathAuditLimits
from .preflight import InsertionPreflight, plan_held_transfer_and_axial
from .targets import InsertionTargets, build_rigid_insertion_targets
from .xy_retry import XYRetryAssessment


@dataclass(frozen=True)
class XYRetryPreflight:
    status: str
    candidate_key: tuple[str, str, str]
    choice_id: str
    targets: InsertionTargets
    planning: InsertionPreflight
    endpoint_screen: dict
    live_start_q: np.ndarray
    observed_T_key_hand: np.ndarray
    observed_relation_timestamp_s: float
    start_q_timestamp_s: float

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_xy_retry_preflight_v1",
            "status": self.status,
            "candidate_key": list(self.candidate_key),
            "choice_id": self.choice_id,
            "targets": self.targets.to_record(),
            "planning": self.planning.to_record(),
            "endpoint_screen": self.endpoint_screen,
            "live_start_q": self.live_start_q.tolist(),
            "observed_T_key_hand": self.observed_T_key_hand.tolist(),
            "observed_relation_timestamp_s": self.observed_relation_timestamp_s,
            "start_q_timestamp_s": self.start_q_timestamp_s,
            "scope": "read_only_live_state_replan_not_guarded_execution",
            "robot_ready": False,
        }


def plan_xy_retry_from_withdrawn_hold(
    *, planner, assessment: XYRetryAssessment, calibration,
    catalog: dict, mode: TaskMode, shared_root: Path,
    candidate_key: tuple[str, str, str], tabletop_pose_stem: str,
    observed_T_key_hand: np.ndarray, observed_relation_timestamp_s: float,
    live_start_q: np.ndarray, start_q_timestamp_s: float,
    max_state_skew_s: float, held_hand_source: str,
    trial_scene: dict, limits: PathAuditLimits,
    axial_waypoint_step_m: float,
) -> XYRetryPreflight:
    """Freshly screen and replan; never replay the previous failed stroke."""
    root = Path(shared_root).expanduser().resolve()
    if (not isinstance(assessment, XYRetryAssessment) or
            assessment.status != "proposal_requires_live_preflight" or
            assessment.decision is None or
            assessment.decision.status != "propose" or
            assessment.decision.offset_socket_m is None or
            len(assessment.decision.supporting_cameras) < 2 or
            assessment.endpoint_screen is None):
        raise ValueError("retry needs a two-view VLM proposal after withdrawal")
    if Path(catalog.get("shared_root", "")).expanduser().resolve() != root:
        raise ValueError("retry catalogue uses another shared root")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=tabletop_pose_stem)
    if selected["status"] != "candidates_available":
        raise ValueError(f"retry catalogue unavailable: {selected['status']}")
    matches = [row for row in selected["candidates"]
               if tuple(row["key"]) == tuple(candidate_key)]
    if len(matches) != 1:
        raise ValueError("retry grasp is no longer eligible for this tabletop")
    candidate = Path(matches[0]["candidate_dir"])
    if (Path(assessment.endpoint_screen.get("candidate_dir", "")).resolve()
            != candidate.resolve()):
        raise ValueError("VLM proposal belongs to a different grasp")
    choice_id = assessment.decision.choice_id
    rows = [row for row in assessment.endpoint_screen["rows"]
            if row["choice_id"] == choice_id and row["endpoint_pass"]]
    if len(rows) != 1 or not np.allclose(
            rows[0]["xy_offset_socket_m"],
            assessment.decision.offset_socket_m, atol=1e-12):
        raise ValueError("VLM-selected XY target did not pass the prior endpoint gate")
    relation = validate_se3(observed_T_key_hand, name="observed T_key_hand")
    if (assessment.endpoint_screen.get("observed_key_hand_source") !=
            "multiview_key_pose_plus_live_wrist" or
            not np.allclose(
                relation, rows[0]["endpoint_report"]["T_key_hand"],
                atol=1e-8)):
        raise ValueError("observed key/hand relation differs from VLM screening")
    observed_time = float(observed_relation_timestamp_s)
    start_time = float(start_q_timestamp_s)
    skew = float(max_state_skew_s)
    if (not all(math.isfinite(value) for value in
                (observed_time, start_time, skew)) or skew <= 0 or
            abs(observed_time - start_time) > skew):
        raise ValueError("live start joints and observed key relation are stale")
    start = np.asarray(live_start_q, dtype=np.float64)
    if start.shape != (13,) or not np.all(np.isfinite(start)):
        raise ValueError("live withdrawn state must be 13 finite joints")
    limits.validate()
    current_offset = tuple(assessment.endpoint_screen["current_offset_socket_m"])
    old_targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=relation, xy_offset_socket_m=current_offset)
    wrist = validate_se3(planner.fk_wrist(start), name="live withdrawn wrist FK")
    if (np.linalg.norm(wrist[:3, 3] -
                       old_targets.T_robot_hand_preinsert[:3, 3]) >
            limits.goal_position_tolerance_m or
            pose_angle_deg(wrist, old_targets.T_robot_hand_preinsert) >
            limits.goal_rotation_tolerance_deg):
        raise ValueError("live wrist is not at the verified withdrawn hold")
    target_offset = assessment.decision.offset_socket_m
    fresh_screen = screen_grasp_endpoint(
        shared_root=root, mode=mode, candidate_dir=candidate,
        minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
        xy_offset_socket_m=target_offset,
        T_key_hand_override=relation)
    if fresh_screen["endpoint_pass"] is not True:
        raise ValueError("fresh exact 20 mm endpoint screen rejected retry")
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=relation, xy_offset_socket_m=target_offset)
    plan = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial_scene, shared_root=root,
        calibration=calibration, targets=targets, start_q=start,
        held_hand_q=start[7:], held_hand_source=held_hand_source,
        limits=limits, axial_waypoint_step_m=axial_waypoint_step_m)
    return XYRetryPreflight(
        "sampled_retry_preflight_pass" if plan.sampled_planning_pass
        else plan.status,
        tuple(candidate_key), choice_id, targets, plan, fresh_screen,
        start.copy(), relation.copy(), observed_time, start_time)


def write_xy_retry_preflight(result: XYRetryPreflight, output_dir: Path) -> Path:
    """Persist immutable retry evidence and any planned trajectories."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.planning.transfer_trajectory is not None:
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
