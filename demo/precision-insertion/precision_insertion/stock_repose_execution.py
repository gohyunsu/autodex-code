"""Execute an unchanged directed-v8 reset plan through stock AutoDex FR3.

Both failed-key restoration and directed repose use the same preflighted
pickup, held motion and release paths. Robot feedback is saved at every phase
endpoint; only a later key image can establish the actual landing pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .franka_completion_checked import is_demo_completion_checked_executor
from .live_robot_state import LiveRobotState
from .repose_execution_evidence import (
    PHASES, SCHEMA, STOCK_SOURCE, verify_repose_execution_evidence,
)
from .repose_transition import ReposeTransitionPreflight


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class StockReposeExecutor:
    """One-use-per-plan callback using the session's FR3/Inspire connection."""

    def __init__(self, *, executor, planner, clear_view, read_state,
                 motion_interlock, enable_robot_motion: bool):
        if (not is_demo_completion_checked_executor(executor) or
                not callable(read_state) or
                not callable(motion_interlock) or
                clear_view.executor is not executor):
            raise TypeError("stock repose needs the session FR3, feedback and interlock")
        self.executor = executor
        self.planner = planner
        self.clear_view = clear_view
        self.read_state = read_state
        self.motion_interlock = motion_interlock
        self.enable_robot_motion = enable_robot_motion

    def execute_restoration(self, runner, decision,
                            plan: ReposeTransitionPreflight,
                            preflight_report_path: Path) -> Path:
        """Return a misplaced released key to its original tabletop pose."""
        report_path = Path(preflight_report_path).expanduser().resolve()
        if (runner._attempt_dir is None or
                not report_path.is_relative_to(
                    Path(runner._attempt_dir).resolve() / "reset_restoration") or
                report_path.name != "report.json"):
            raise ValueError("restoration report belongs to another attempt")
        return self._execute(
            runner=runner, decision=decision, plan=plan,
            report_path=report_path, output_dir=report_path.parent.parent,
            expected_action=None)

    def execute_repose(self, runner, decision,
                       plan: ReposeTransitionPreflight) -> Path:
        """Execute this attempt's directed v8 reset preflight."""
        report_path = Path(runner._repose_report_path).expanduser().resolve()
        return self._execute(
            runner=runner, decision=decision, plan=plan,
            report_path=report_path,
            output_dir=Path(runner._attempt_dir).expanduser().resolve(),
            expected_action="repose_execution_gate_required")

    def _execute(self, *, runner, decision, plan, report_path: Path,
                 output_dir: Path, expected_action: str | None) -> Path:
        current_action = runner.current_decision().action
        # begin_repose_attempt opens the episode before calling this callback.
        # That changes the session decision from the gate to observation.
        decision_matches = (
            decision.action == expected_action and
            current_action == "await_repose_observation"
            if expected_action is not None else
            runner.current_decision() == decision)
        if (self.enable_robot_motion is not True or
                not decision_matches or
                runner.active_attempt is None or
                not isinstance(plan, ReposeTransitionPreflight) or
                plan.status != "nominal_reset_preflight_pass_drop_unobserved" or
                plan.pickup_plan is None or
                not plan.pickup_plan.success or
                plan.held_plan is None or
                plan.release_plan is None or
                plan.held_plan.status !=
                "sampled_held_path_pass_release_unplanned" or
                plan.held_plan.sampled_held_path_audit is None or
                plan.held_plan.sampled_held_path_audit.get(
                    "sampled_clear") is not True or
                plan.release_plan.status !=
                "nominal_release_exit_path_pass_drop_unobserved" or
                plan.release_plan.release_geometry_audit is None or
                plan.release_plan.release_geometry_audit.get(
                    "sampled_clear") is not True):
            raise ValueError("stock repose needs this passing live v8 reset plan")
        if self.motion_interlock() is not True:
            raise PermissionError("stock repose motion interlock is closed")
        report_path = Path(report_path).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifacts = report["artifacts"]
        if (report.get("schema") !=
                "precision_insertion_v8_repose_transition_preflight_v1" or
                report.get("status") != plan.status or
                report.get("selected_seed") != plan.selected_seed or
                report.get("robot_ready") is not False):
            raise ValueError("stock repose report differs from selected plan")
        inputs = artifacts.get("input_files")
        if (not isinstance(inputs, dict) or set(inputs) != {
                "session", "catalog", "key_pose_world", "live_start_q",
                "limits"}):
            raise ValueError("stock repose report lacks source files")
        for ref in inputs.values():
            if _sha(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("stock repose input changed before motion")
        scene_path = report_path.parent / artifacts["trial_scene"]
        archive_path = report_path.parent / artifacts["planned_trajectories"]
        if (scene_path.name != artifacts["trial_scene"] or
                archive_path.name != artifacts["planned_trajectories"] or
                _sha(scene_path) != artifacts["trial_scene_sha256"] or
                _sha(archive_path) !=
                artifacts["planned_trajectories_sha256"]):
            raise ValueError("stock repose scene or trajectories changed")
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        if (not isinstance(scene.get("mesh"), dict) or
                not {"target", "fixture_socket"} <= set(scene["mesh"])):
            raise ValueError("stock repose scene lost the key or socket")
        with np.load(archive_path, allow_pickle=False) as archive:
            paths = {name: np.asarray(archive[name], dtype=np.float64)
                     for name in archive.files}
        expected_paths = {
            "pickup_approach": plan.pickup_plan.traj,
            "held_lift": plan.held_plan.lift_trajectory,
            "held_transfer": plan.held_plan.transfer_trajectory,
            "held_descent": plan.held_plan.descent_trajectory,
            "post_release_lift": plan.release_plan.post_release_lift_trajectory,
            "post_release_retract": plan.release_plan.retract_trajectory,
        }
        if (not expected_paths.keys() <= paths.keys() or
                any(value is None or
                    not np.array_equal(paths[name], np.asarray(value))
                    for name, value in expected_paths.items())):
            raise ValueError("stock repose saved paths differ from live plan")
        for first_name, second_name in (
                ("held_lift", "held_transfer"),
                ("held_transfer", "held_descent"),
                ("post_release_lift", "post_release_retract")):
            if not np.allclose(paths[first_name][-1], paths[second_name][0],
                               atol=1e-4, rtol=0):
                raise ValueError("stock repose paths do not join")
        if not np.allclose(
                paths["held_descent"][-1, :7],
                paths["post_release_lift"][0, :7], atol=1e-4, rtol=0):
            raise ValueError("stock repose release does not join held descent")
        start = self.read_state()
        if not isinstance(start, LiveRobotState):
            raise TypeError("stock repose needs measured starting joints")
        self.clear_view._validate_feedback(start, phase="stock_repose_start")
        first = paths["pickup_approach"][0]
        if (np.max(np.abs(start.full_q[:7] - first[:7])) >
                self.clear_view.max_arm_error_rad or
                np.max(np.abs(start.full_q[7:] - first[7:])) >
                self.clear_view.max_hand_error_rad):
            raise ValueError("stock repose pickup start differs from measured robot")
        folder = Path(output_dir).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError("stock repose output parent is missing")
        started = folder / "stock_repose_started.json"
        failure = folder / "stock_repose_failure.json"
        feedback_path = folder / "stock_repose_feedback.json"
        output = folder / "stock_repose_execution.json"
        if any(path.exists() for path in (started, failure, feedback_path, output)):
            raise FileExistsError("this v8 reset already requested motion")
        if self.motion_interlock() is not True:
            raise PermissionError("stock repose motion interlock closed before command")
        report_sha = _sha(report_path)
        _write(started, {
            "schema": "precision_insertion_stock_repose_started_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "preflight_report": {"path": str(report_path),
                                 "sha256": report_sha},
            "pre_state": start.to_record(),
            "requested_at_s": time.time(),
        })
        checkpoints = []
        try:
            # AutoDex's selected v8 approach, pregrasp, grasp and squeeze.
            # The held lift is separately checked against measured squeeze.
            self.executor.execute(
                plan.pickup_plan, planner=self.planner, scene_cfg=scene,
                skip_lift=True, start_from_current=True)
            squeezed = self.clear_view._verify_return_endpoint(
                paths["held_lift"][0], phase="stock_repose_pickup_squeeze")
            checkpoints.append({"name": "pickup_squeeze",
                                "state": squeezed.to_record()})
            for phase, name in (("held_lift", "held_lift"),
                                ("held_transfer", "held_transfer"),
                                ("held_descent", "held_descent")):
                self.clear_view._follow_autodex(paths[name].copy())
                measured = self.clear_view._verify_return_endpoint(
                    paths[name][-1], phase=f"stock_repose_{phase}")
                checkpoints.append({"name": phase,
                                    "state": measured.to_record()})
            final = self.clear_view.finish_planned_release_exit_with_autodex(
                release_plan=plan.release_plan, evidence_sink=checkpoints)
            if (not isinstance(final, LiveRobotState) or
                    [item.get("name") for item in checkpoints] !=
                    list(PHASES) or
                    not np.array_equal(final.full_q,
                                       checkpoints[-1]["state"]["full_q"])):
                raise RuntimeError("stock repose lacks measured phase completion")
            stopped = self.executor.stop_and_acknowledge()
            if stopped.get("acknowledged") is not True:
                raise RuntimeError("stock repose FR3 stop was not acknowledged")
            _write(feedback_path, {
                "schema": "precision_insertion_stock_repose_feedback_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "preflight_report_sha256": report_sha,
                "checkpoints": checkpoints,
                "stop_acknowledgement": stopped,
                "scope": "measured_robot_waypoints_not_physical_key_landing",
            })
            record = {
                "schema": SCHEMA, "source": STOCK_SOURCE,
                "attempt_id": runner.active_attempt.attempt_id,
                "preflight_report_sha256": report_sha,
                "planned_trajectories_sha256": _sha(archive_path),
                "session_calibration_sha256": runner.session_sha256,
                "selected_seed": plan.selected_seed,
                "safety_abort": False,
                "grip_loss_before_release": False,
                "grip_loss_scope": "measured_hand_hold_not_physical_key_presence",
                "stock_feedback": {"path": str(feedback_path),
                                   "sha256": _sha(feedback_path)},
                "scope": "stock_v8_reset_execution_before_fresh_landing_image",
            }
            _write(output, record)
            verify_repose_execution_evidence(
                path=output, attempt_id=runner.active_attempt.attempt_id,
                attempt_started_at_s=runner.active_attempt.started_at_s,
                preflight_report_path=report_path,
                preflight_report_sha256=report_sha,
                session_calibration_sha256=runner.session_sha256,
                selected_seed=plan.selected_seed)
            return output
        except Exception as exc:
            try:
                stopped = self.executor.stop_and_acknowledge()
            except Exception as stop_exc:
                stopped = {"acknowledged": False, "error": str(stop_exc)}
            _write(failure, {
                "schema": "precision_insertion_stock_repose_failure_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "preflight_report": str(report_path),
                "error_type": type(exc).__name__, "error": str(exc),
                "stop_acknowledgement": stopped,
                "robot_state_after_failure": "unknown_requires_recovery",
            })
            raise
