"""Replan the complete held-key chain from the measured post-squeeze state.

This does not command a robot. It uses the unchanged v8 lift planner, the
demo's existing transfer/axial planner and sampled whole-key/hand audit, plus
commissioned surface-error bounds. The key/hand relation remains a nominal
grasp hypothesis until independently observed or physically bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .candidates import select_pose_candidates
from .curobo_compat import install_curobo_planner_compat
from .endpoint import screen_grasp_endpoint
from .geometry import validate_se3
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .planner_mode import cartesian_mode_from_planner
from .preflight import (
    InsertionPreflight, _goal_met, _path, plan_held_transfer_and_axial,
)
from .session_runner import SessionRunner
from .targets import InsertionTargets, build_rigid_insertion_targets
from .uncertainty_margin import (
    SurfaceDeviationBounds, audit_sampled_uncertainty_margins,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


@dataclass(frozen=True)
class MeasuredLiftChain:
    status: str
    attempt_id: str
    candidate_id: str
    pickup_execution_log: Path
    pickup_execution_sha256: str
    session_calibration_sha256: str
    trial_preflight_sha256: str
    measured_start: LiveRobotState
    T_key_hand_assumed: np.ndarray
    relation_source: str
    endpoint: dict | None
    targets: InsertionTargets | None
    planning: InsertionPreflight | None
    uncertainty_margin: dict | None
    cartesian_planner_mode: str = "default"

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_measured_lift_chain_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "pickup_execution_log": str(self.pickup_execution_log),
            "pickup_execution_sha256": self.pickup_execution_sha256,
            "session_calibration_sha256": self.session_calibration_sha256,
            "trial_preflight_sha256": self.trial_preflight_sha256,
            "measured_start": self.measured_start.to_record(),
            "T_key_hand_assumed": self.T_key_hand_assumed.tolist(),
            "relation_source": self.relation_source,
            "endpoint": self.endpoint,
            "targets": None if self.targets is None else self.targets.to_record(),
            "planning": None if self.planning is None else self.planning.to_record(),
            "uncertainty_margin": self.uncertainty_margin,
            "cartesian_planner_mode": self.cartesian_planner_mode,
            "scope": "measured_start_sampled_chain_not_motion_or_grasp_success",
            "robot_ready": False,
        }


def plan_measured_lift_chain(
    *, runner: SessionRunner, planner, pickup_execution_log: Path,
    joint_sample: LiveRobotState,
    bounds: SurfaceDeviationBounds,
    limits: PathAuditLimits,
    max_state_age_s: float,
    max_post_squeeze_arm_drift_rad: float,
    max_post_squeeze_hand_drift_raw: float,
    max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    axial_waypoint_step_m: float,
) -> MeasuredLiftChain:
    """Replan lift/transfer/20 mm path without treating squeeze as success.

    Only a measured 13-DOF state after this attempt's completed squeeze is
    accepted. The complete path is audited using the same frozen socket and
    exact CAD inputs as the initial v8 trial. Surface-error bounds must be
    commissioned externally; nominal BODex/MuJoCo relation alone cannot make
    this path executable.
    """
    thresholds = (
        max_state_age_s, max_post_squeeze_arm_drift_rad,
        max_post_squeeze_hand_drift_raw, max_arm_hand_skew_s,
        max_hand_command_error_raw, max_arm_velocity_rad_s,
        axial_waypoint_step_m)
    if (not all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in thresholds) or axial_waypoint_step_m > .005):
        raise ValueError("measured lift needs positive commissioned limits")
    limits.validate()
    bounds.validate()
    if not isinstance(runner, SessionRunner):
        raise TypeError("measured lift needs a frozen SessionRunner")
    attempt = runner.active_attempt
    trial = runner._preflight
    if (attempt is None or trial is None or
            runner.current_decision().action != "await_lift_observation" or
            attempt.events or attempt.failure_code is not None or
            trial.status != "sampled_planning_pass" or
            trial.selected_candidate_key is None or
            attempt.candidate_id != "/".join(trial.selected_candidate_key)):
        raise ValueError("measured lift is not bound to a fresh pickup attempt")
    verified = runner.verify_current_preflight_evidence()
    log_file = Path(pickup_execution_log).expanduser().resolve()
    if (runner._attempt_dir is None or
            log_file != runner._attempt_dir / "pickup_execution.json"):
        raise ValueError("pickup execution log belongs to another attempt")
    log = json.loads(log_file.read_text(encoding="utf-8"))
    if (log.get("schema") != "precision_insertion_pickup_execution_v1" or
            log.get("status") != "squeeze_command_and_feedback_complete" or
            log.get("attempt_id") != attempt.attempt_id or
            log.get("candidate_id") != attempt.candidate_id or
            log.get("preflight_report_sha256") !=
            verified["report_sha256"] or
            log.get("command") !=
            "stock_franka_execute_skip_lift_start_from_current"):
        raise ValueError("pickup completion is not bound to this v8 attempt")
    start_file = runner._attempt_dir / "pickup_started.json"
    if (not start_file.is_file() or
            log.get("pickup_started_sha256") != _sha(start_file)):
        raise ValueError("pickup command-start marker changed or is missing")
    started = json.loads(start_file.read_text(encoding="utf-8"))
    if (started.get("status") != "command_requested" or
            started.get("attempt_id") != attempt.attempt_id or
            started.get("candidate_id") != attempt.candidate_id):
        raise ValueError("pickup command-start marker has another identity")
    report_file = runner._preflight_report_path
    if report_file is None:
        raise ValueError("selected trial report is missing")
    saved_plan = json.loads(report_file.read_text(encoding="utf-8"))
    planner_mode = cartesian_mode_from_planner(planner)
    if (planner_mode != getattr(trial, "cartesian_planner_mode", "default") or
            saved_plan.get("cartesian_planner_mode", "default") != planner_mode):
        raise ValueError(
            "post-squeeze planner mode differs from the bound initial preflight")
    if (log.get("planned_trajectories_sha256") !=
            saved_plan.get("artifacts", {}).get(
                "planned_trajectories_sha256")):
        raise ValueError("pickup used a different saved v8 trajectory")
    if not isinstance(joint_sample, LiveRobotState):
        raise TypeError("measured lift requires live robot feedback")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    completed = float(log.get("completed_at_s", float("nan")))
    saved_post = log.get("post_state")
    post_q = np.asarray(
        saved_post.get("full_q") if isinstance(saved_post, dict) else None,
        dtype=float)
    post_hand = np.asarray(
        saved_post.get("hand_raw_measured")
        if isinstance(saved_post, dict) else None, dtype=float)
    squeeze_raw = np.asarray(log.get("squeeze_action_raw"), dtype=float)
    clock_now = _wall_time()
    if (not isinstance(saved_post, dict) or
            saved_post.get("source") != "robot_joint_feedback" or
            post_q.shape != (13,) or post_hand.shape != (6,) or
            squeeze_raw.shape != (6,) or
            not all(np.all(np.isfinite(row)) for row in
                    (post_q, post_hand, squeeze_raw)) or
            saved_post.get("sample_timestamp_s") != completed or
            not math.isfinite(completed) or
            completed <= attempt.started_at_s or
            joint_sample.sample_timestamp_s < completed or
            joint_sample.sample_timestamp_s > clock_now or
            clock_now - joint_sample.sample_timestamp_s > max_state_age_s or
            float(np.max(np.abs(
                joint_sample.full_q[:7] - post_q[:7]))) >
            max_post_squeeze_arm_drift_rad or
            float(np.max(np.abs(
                joint_sample.hand_raw_measured - post_hand))) >
            max_post_squeeze_hand_drift_raw or
            not np.allclose(joint_sample.hand_raw_commanded,
                            squeeze_raw,
                            atol=max_hand_command_error_raw, rtol=0)):
        raise ValueError("measured held start drifted after the recorded squeeze")
    selection = select_pose_candidates(
        runner.catalog, expected_mode=runner.mode,
        tabletop_pose_stem=attempt.tabletop_pose_stem)
    matches = [row for row in selection["candidates"]
               if tuple(row["key"]) == trial.selected_candidate_key]
    if selection["status"] != "candidates_available" or len(matches) != 1:
        raise ValueError("selected v8 candidate is stale or no longer eligible")
    candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
    relation = validate_se3(np.load(
        candidate_dir / "wrist_se3.npy", allow_pickle=False),
        name="nominal BODex T_key_hand")
    endpoint = screen_grasp_endpoint(
        shared_root=runner.shared_root, mode=runner.mode,
        candidate_dir=candidate_dir,
        minimum_hand_clearance_m=runner.catalog["minimum_hand_clearance_m"],
        T_key_hand_override=relation,
        hand_poses_override={"measured_squeeze": joint_sample.full_q[7:]},
        override_source="nominal_BODex_relation_measured_Inspire_joints")
    if (not np.allclose(validate_se3(
            endpoint.get("T_key_hand"), name="screened key/hand relation"),
            relation, atol=1e-8, rtol=0) or
            not np.allclose(endpoint.get("xy_offset_socket_m"),
                            [0.0, 0.0], atol=1e-12, rtol=0) or
            not np.allclose(endpoint.get("hold_pose_screens", {}).get(
                "measured_squeeze", {}).get("hand_q"),
                joint_sample.full_q[7:], atol=1e-8, rtol=0)):
        raise ValueError("20 mm screen used a different held relation or hand")

    def result(status: str, *, targets=None, planning=None,
               margin=None) -> MeasuredLiftChain:
        return MeasuredLiftChain(
            status, attempt.attempt_id, attempt.candidate_id,
            log_file, _sha(log_file), runner.session_sha256,
            verified["report_sha256"], joint_sample, relation,
            "nominal_BODex_relation_with_commissioned_surface_error_bound",
            endpoint, targets, planning, margin, planner_mode)

    if endpoint.get("endpoint_pass") is not True:
        return result("measured_squeeze_endpoint_rejected")
    # This can run in a new process after pickup, not just in the process
    # where the initial candidate loop installed its cuRobo compatibility.
    install_curobo_planner_compat()
    targets = build_rigid_insertion_targets(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration, T_key_hand=relation)
    lift_pf = planner.plan_lift_preflight(
        np.asarray(joint_sample.full_q, dtype=np.float32),
        trial.trial_scene, lift_h=.10,
        timing_phase="precision_insertion_measured_lift")
    if lift_pf is None:
        return result("measured_lift_unreachable", targets=targets)
    lift = _path(lift_pf.traj, "measured held lift",
                 joint_sample.full_q, joint_sample.full_q[7:])
    start_wrist = validate_se3(planner.fk_wrist(lift[0]),
                               name="measured lift start FK")
    goal_wrist = start_wrist.copy()
    goal_wrist[2, 3] += .10
    if (not _goal_met(validate_se3(planner.fk_wrist(lift[-1]),
                                   name="measured lift endpoint FK"),
                      goal_wrist, limits) or
            not np.allclose(lift_pf.start_full_qpos,
                            joint_sample.full_q, atol=1e-4, rtol=0)):
        raise ValueError("v8 measured lift start or 10 cm goal is inconsistent")
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=runner.shared_root, calibration=runner.calibration,
        targets=targets, start_q=lift[-1],
        held_hand_q=joint_sample.full_q[7:],
        held_hand_source="measured", limits=limits,
        axial_waypoint_step_m=axial_waypoint_step_m,
        lift_trajectory=lift,
        prior_query_records=({"stage": "measured_lift", "success": True,
                              "planner_api": "plan_lift_preflight"},))
    if not planning.sampled_planning_pass:
        return result(planning.status, targets=targets, planning=planning)
    margin = audit_sampled_uncertainty_margins(
        mode=runner.mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    return result(
        "sampled_measured_chain_pass" if margin["sampled_margin_pass"] else
        "sampled_uncertainty_margin_rejected",
        targets=targets, planning=planning, margin=margin)


def write_measured_lift_chain(
    result: MeasuredLiftChain, output_dir: Path,
) -> Path:
    """Save a new replan bundle; never overwrite or grant motor permission."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.planning is not None and result.planning.lift_trajectory is not None:
        name = "planned_trajectories.npz"
        path = target / name
        arrays = {"lift": result.planning.lift_trajectory}
        if result.planning.transfer_trajectory is not None:
            arrays["transfer"] = result.planning.transfer_trajectory
        if result.planning.axial_trajectory is not None:
            arrays["axial"] = result.planning.axial_trajectory
        np.savez_compressed(path, **arrays)
        record["planned_trajectories"] = name
        record["planned_trajectories_sha256"] = _sha(path)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    verify_measured_lift_chain(target / "report.json", expected=result)
    return target


def verify_measured_lift_chain(
    report_path: Path, *, expected: MeasuredLiftChain | None = None,
) -> dict:
    """Recheck saved bytes and path continuity; still not a motion permit."""
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != "precision_insertion_measured_lift_chain_v1" or
            report.get("robot_ready") is not False or
            not isinstance(report.get("attempt_id"), str) or
            not isinstance(report.get("candidate_id"), str)):
        raise ValueError("saved measured lift report has an invalid contract")
    pickup = Path(report.get("pickup_execution_log", "")).resolve()
    if (not pickup.is_file() or
            _sha(pickup) != report.get("pickup_execution_sha256")):
        raise ValueError("saved pickup evidence changed after replan")
    artifact_fields = {"planned_trajectories",
                       "planned_trajectories_sha256"}
    if expected is not None and {
            key: value for key, value in report.items()
            if key not in artifact_fields} != expected.to_record():
        raise ValueError("saved measured lift differs from selected replan")
    if report.get("status") == "sampled_measured_chain_pass":
        planning = report.get("planning")
        margin = report.get("uncertainty_margin")
        if (not isinstance(planning, dict) or
                planning.get("sampled_planning_pass") is not True or
                not isinstance(margin, dict) or
                margin.get("sampled_margin_pass") is not True):
            raise ValueError("saved measured lift pass lacks chain/margin proof")
    filename = report.get("planned_trajectories")
    if filename is not None:
        if not isinstance(filename, str) or not filename:
            raise ValueError("saved measured lift path name is invalid")
        artifact = (path.parent / filename).resolve()
        if (not artifact.is_relative_to(path.parent) or
                not artifact.is_file() or
                _sha(artifact) != report.get("planned_trajectories_sha256")):
            raise ValueError("saved measured lift path bytes changed")
        with np.load(artifact, allow_pickle=False) as archive:
            if "lift" not in archive.files:
                raise ValueError("saved measured lift has no lift trajectory")
            if (report.get("status") == "sampled_measured_chain_pass" and
                    not {"lift", "transfer", "axial"} <=
                    set(archive.files)):
                raise ValueError("passing measured lift lacks full-chain paths")
            start = np.asarray(report["measured_start"]["full_q"], dtype=float)
            held = start[7:]
            lift = _path(archive["lift"], "saved measured lift", start, held)
            tail = lift[-1]
            for name in ("transfer", "axial"):
                if name in archive.files:
                    segment = _path(archive[name], f"saved {name}", tail, held)
                    tail = segment[-1]
    elif report.get("status") == "sampled_measured_chain_pass":
        raise ValueError("passing measured lift has no saved trajectory")
    return report
