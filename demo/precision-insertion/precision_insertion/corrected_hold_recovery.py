"""Return a key held at a failed corrected pre-insertion plan.

Undo the *executed* lateral shifts, transfer and lift in reverse order. This
uses the same FR3 follower and the existing held-object collision audit. The
key is opened only after the held route reaches its planned release pose; a
fresh tabletop observation then feeds the existing repose/reset code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .franka_completion_checked import is_demo_completion_checked_executor
from .geometry import validate_se3
from .held_reverse_route import plan_held_reverse_route
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .timing_profile import timed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_archive(report_path: Path, name: str, attempt_id: str
                    ) -> tuple[np.ndarray, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    archive_name = report.get("planned_trajectories")
    if (report.get("attempt_id") != attempt_id or
            archive_name != "planned_trajectories.npz"):
        raise ValueError(f"{name} source differs from the active attempt")
    archive = report_path.parent / archive_name
    if _sha(archive) != report.get("planned_trajectories_sha256"):
        raise ValueError(f"{name} planned trajectory changed")
    with np.load(archive, allow_pickle=False) as arrays:
        path = np.asarray(arrays[name], dtype=np.float64).copy()
    return path, {"report": str(report_path), "report_sha256": _sha(report_path),
                  "trajectory": str(archive), "trajectory_sha256": _sha(archive)}


def _executed_xy(folder: Path, attempt_id: str) -> tuple[list[np.ndarray], list[dict]]:
    root = folder / "same_trial_xy" / "shifts"
    paths = sorted(root.glob("*/report.json")) if root.is_dir() else []
    shifts, sources = [], []
    for index, path in enumerate(paths):
        if path.parent.name != f"{index:03d}":
            raise ValueError("executed XY shifts are not sequential")
        report = json.loads(path.read_text(encoding="utf-8"))
        trajectory = Path(report.get("trajectory_path", "")).resolve()
        if (report.get("attempt_id") != attempt_id or
                report.get("schema") not in {
                    "precision_insertion_same_trial_xy_shift_v1",
                    "precision_insertion_same_trial_xy_shift_v2"} or
                trajectory != path.parent / "trajectory.npy" or
                _sha(trajectory) != report.get("trajectory_sha256")):
            raise ValueError("executed XY source differs from this attempt")
        shifts.append(np.load(trajectory, allow_pickle=False))
        sources.append({"report": str(path), "report_sha256": _sha(path),
                        "trajectory": str(trajectory),
                        "trajectory_sha256": _sha(trajectory)})
    return shifts, sources


def propagate_held_key_pose(*, planner, failed_key_pose: np.ndarray,
                            failed_q: np.ndarray,
                            current_q: np.ndarray) -> np.ndarray:
    """Carry the failed plan's rigid key-in-hand hypothesis to fresh joints."""
    key = validate_se3(failed_key_pose, name="failed held key pose")
    wrist_at_failure = validate_se3(
        planner.fk_wrist(failed_q), name="failed wrist FK")
    wrist_now = validate_se3(
        planner.fk_wrist(current_q), name="current wrist FK")
    key_hand = np.linalg.inv(key) @ wrist_at_failure
    return validate_se3(wrist_now @ np.linalg.inv(key_hand),
                        name="current held key pose")


class StockCorrectedHoldRecovery:
    """Plan, execute and record a failed corrected-hold return once."""

    def __init__(self, *, executor, planner, read_state, limits: PathAuditLimits,
                 motion_interlock):
        if not is_demo_completion_checked_executor(executor):
            raise TypeError("corrected hold needs the session FR3 executor")
        self.executor = executor
        self.planner = planner
        self.read_state = read_state
        self.limits = limits
        self.motion_interlock = motion_interlock

    def __call__(self, live, runner, decision) -> Path:
        attempt = runner.active_attempt
        if (attempt is None or runner.current_decision() != decision or
                decision.action != "recover_corrected_hold_then_reobserve" or
                attempt.failure_code != "preinsert_path_unavailable"):
            raise ValueError("corrected hold return belongs to another attempt")
        folder = Path(runner._attempt_dir)
        refs = attempt.events[-1]["evidence_refs"]
        axial_failure = "corrected_axial_plan_failure" in refs
        failure_ref = refs[("corrected_axial_plan_failure" if axial_failure
                            else "corrected_xy_plan_failure")]
        failure_path = Path(failure_ref).resolve()
        if axial_failure:
            in_attempt = failure_path.parent.parent == (
                folder / "guarded_axial_handoffs")
            schema = "precision_insertion_same_trial_axial_plan_failure_v1"
        else:
            in_attempt = failure_path.parent == (
                folder / "same_trial_xy" / "plan_failures")
            schema = "precision_insertion_same_trial_xy_plan_failure_v1"
        if not in_attempt:
            raise ValueError("corrected plan failure is not in this attempt")
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if (failure.get("schema") != schema or
                failure.get("attempt_id") != attempt.attempt_id):
            raise ValueError("corrected plan failure differs from this attempt")
        current = self.read_state()
        if not isinstance(current, LiveRobotState):
            raise TypeError("corrected hold needs measured robot joints")
        xy, xy_sources = _executed_xy(folder, attempt.attempt_id)
        lift_report = Path(runner._measured_lift_report_path).resolve()
        transfer_report = Path(runner._postlift_report_path).resolve()
        lift, lift_source = _source_archive(
            lift_report, "lift", attempt.attempt_id)
        transfer, transfer_source = _source_archive(
            transfer_report, "transfer", attempt.attempt_id)
        c2r = np.asarray(runner.calibration.record["c2r"], dtype=np.float64)
        original = np.linalg.inv(c2r) @ runner._preflight.key_pose_world
        current_key = propagate_held_key_pose(
            planner=self.planner,
            failed_key_pose=np.asarray(
                failure["targets"]["T_robot_key_preinsert"]),
            failed_q=np.asarray(failure["measured_start"]["full_q"]),
            current_q=current.full_q)
        with timed("recovery.corrected_hold_plan", phase="recovery",
                   kind="planning", attempt_id=attempt.attempt_id):
            route = plan_held_reverse_route(
                planner=self.planner, trial_scene=runner._preflight.trial_scene,
                shared_root=runner.shared_root, mode=runner.mode,
                calibration=runner.calibration, current_q=current.full_q,
                current_key_pose_robot=current_key,
                original_key_pose_robot=original,
                executed_xy_trajectories=xy,
                executed_transfer_trajectory=transfer,
                executed_lift_trajectory=lift, limits=self.limits,
                arm_velocity_rad_s=self.executor.joint_vmax,
                traj_dt_s=self.executor.traj_dt)
        plan_dir = folder / "corrected_hold_return"
        plan_dir.mkdir(parents=True, exist_ok=False)
        paths = plan_dir / "trajectories.npz"
        np.savez_compressed(paths, held=route.held_trajectory,
                            clear_view=route.clear_view_trajectory)
        plan = plan_dir / "plan.json"
        _write(plan, {
            "schema": "precision_insertion_corrected_hold_return_plan_v1",
            "attempt_id": attempt.attempt_id,
            "candidate_id": attempt.candidate_id,
            "failure": {"path": str(failure_path), "sha256": _sha(failure_path)},
            "sources": {"xy": xy_sources, "transfer": transfer_source,
                        "lift": lift_source},
            "measured_start": current.to_record(),
            "release_key_pose_robot": route.release_key_pose_robot.tolist(),
            "sampled_geometry": route.sampled_geometry,
            "trajectories": {"path": str(paths), "sha256": _sha(paths)},
            "scope": "sampled_plan_not_physical_key_landing",
        })
        started = plan_dir / "started.json"
        _write(started, {"attempt_id": attempt.attempt_id,
                         "plan": str(plan), "requested_at_s": time.time()})
        phase = "reverse_held_motion"
        try:
            if self.motion_interlock() is not True:
                raise PermissionError("corrected return needs live motion opt-in")
            with timed("recovery.corrected_held_motion", phase="recovery",
                       kind="motion", attempt_id=attempt.attempt_id):
                live.clear_view._follow_autodex(route.held_trajectory.copy())
                held_end = live.clear_view._verify_held_endpoint(
                    route.held_trajectory[-1], phase="corrected_return_release")
            phase = "open_hand"
            with timed("recovery.corrected_open_hand", phase="recovery",
                       kind="motion", attempt_id=attempt.attempt_id):
                self.executor.release(None, open_to_init=True)
                opened = live.clear_view.wait_for_open_hand(
                    route.held_trajectory[-1], phase="corrected_return_open")
            phase = "clear_view"
            with timed("recovery.corrected_clear_view", phase="recovery",
                       kind="motion", attempt_id=attempt.attempt_id):
                live.clear_view._follow_autodex(
                    route.clear_view_trajectory.copy())
                final = live.clear_view.verify_state(
                    self.read_state(), phase="corrected_return_clear_view")
            stopped = self.executor.stop_and_acknowledge()
            if stopped.get("acknowledged") is not True:
                raise RuntimeError("corrected return stop was not acknowledged")
            output = plan_dir / "execution.json"
            _write(output, {
                "schema": "precision_insertion_corrected_hold_recovery_v1",
                "attempt_id": attempt.attempt_id,
                "candidate_id": attempt.candidate_id,
                "plan": {"path": str(plan), "sha256": _sha(plan)},
                "held_end": held_end.to_record(),
                "hand_open_at_s": opened.sample_timestamp_s,
                "completed_at_s": final.sample_timestamp_s,
                "measured_state": final.to_record(),
                "hand_open": True, "clear_view_reached": True,
                "stop_acknowledgement": stopped,
                "scope": "measured_return_not_observed_key_landing",
            })
            return output
        except Exception as exc:
            try:
                stopped = self.executor.stop_and_acknowledge()
            except Exception as stop_exc:
                stopped = {"acknowledged": False, "error": str(stop_exc)}
            _write(plan_dir / "failure.json", {
                "attempt_id": attempt.attempt_id, "phase": phase,
                "error_type": type(exc).__name__, "error": str(exc),
                "stop_acknowledgement": stopped})
            raise
