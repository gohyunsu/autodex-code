"""Execute a preplanned socket-axis release/clear-view path with stock FR3.

The plan is prepared before the insertion command. This callback opens the
Inspire hand only after a completed, non-aborted stroke, then ascends the
saved socket axis and follows a stock cuRobo joint retract to clear view.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .franka_completion_checked import is_demo_completion_checked_executor
from .insertion_checkpoint import _metric_record
from .post_insertion_release import (
    SCHEMA as RELEASE_SCHEMA, verify_post_insertion_release,
)
from .socket_release import plan_socket_release_exit, verify_socket_release_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StockSocketRelease:
    """One-use callback for the same FR3 executor and frozen camera session."""

    def __init__(self, *, executor, planner, read_state, path_limits):
        if not is_demo_completion_checked_executor(executor):
            raise TypeError("socket release needs the session's stock FR3 executor")
        if not callable(read_state):
            raise TypeError("socket release needs measured robot feedback")
        self.executor = executor
        self.planner = planner
        self.read_state = read_state
        self.path_limits = path_limits
        self._prepared: tuple[str, Path, str] | None = None

    def prepare(self, live, runner, decision, handoff_report_path: Path) -> Path:
        """Preflight opening and exit before the robot enters the socket."""
        if (runner is not live.opened.runner or
                runner.current_decision() != decision or
                decision.action != "await_guarded_insertion_and_observation" or
                runner.active_attempt is None or
                self._prepared is not None and self._prepared[0] ==
                runner.active_attempt.attempt_id):
            raise ValueError("release preflight belongs to another live attempt")
        report = plan_socket_release_exit(
            runner=runner, planner=self.planner,
            handoff_report_path=handoff_report_path,
            limits=self.path_limits,
            output_dir=Path(runner._attempt_dir) / "socket_release_plan")
        verify_socket_release_plan(report)
        self._prepared = (runner.active_attempt.attempt_id, report, _sha(report))
        return report

    def __call__(self, live, runner, decision, execution: dict) -> Path:
        if (runner is not live.opened.runner or
                runner.current_decision() != decision or
                decision.action != "await_guarded_insertion_and_observation" or
                self._prepared is None or
                self._prepared[0] != runner.active_attempt.attempt_id or
                execution.get("safety_abort") is not False or
                execution.get("grasp_held") is not True):
            raise ValueError("socket release needs this completed held stroke")
        report_path = self._prepared[1]
        if _sha(report_path) != self._prepared[2]:
            raise ValueError("socket release plan changed after preflight")
        report, paths = verify_socket_release_plan(report_path)
        scene_sha = hashlib.sha256(json.dumps(
            runner.calibration.collision_scene, sort_keys=True,
            allow_nan=False).encode()).hexdigest()
        if scene_sha != report["fixed_scene_sha256"]:
            raise ValueError("frozen socket world changed after release preflight")
        if execution.get("axial_handoff") != report["handoff"]:
            raise ValueError("release plan belongs to another axial stroke")
        metric_ref = execution.get("metric_record")
        if not isinstance(metric_ref, dict):
            raise ValueError("release needs a bound stock force metric")
        metric_path = Path(metric_ref["path"]).expanduser().resolve()
        if (_sha(metric_path) != metric_ref.get("sha256") or
                metric_path.parent != Path(runner._attempt_dir).resolve()):
            raise ValueError("release metric changed after the stroke")
        metric, _, _ = _metric_record(
            metric_path, attempt_id=runner.active_attempt.attempt_id,
            candidate_id=runner.active_attempt.candidate_id,
            session_calibration_sha256=runner.session_sha256)
        if (metric["schema"] !=
                "precision_insertion_stock_force_execution_v1" or
                metric["measurement"]["safety_abort"] is not False):
            raise ValueError("socket release cannot follow a force stop")
        folder = Path(runner._attempt_dir)
        started_path = folder / "socket_release_started.json"
        failure_path = folder / "socket_release_failure.json"
        output = folder / "socket_release_execution.json"
        if any(path.exists() for path in (started_path, failure_path, output)):
            raise FileExistsError("this attempt already requested socket release")
        started_at = time.time()
        with started_path.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "precision_insertion_socket_release_started_v1",
                       "attempt_id": runner.active_attempt.attempt_id,
                       "release_plan": {"path": str(report_path),
                                        "sha256": _sha(report_path)},
                       "metric_record": metric_ref,
                       "started_at_s": started_at}, stream,
                      indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        try:
            # Stock release performs its usual controlled Inspire ramp.
            self.executor.release(None, open_to_init=True)
            live.clear_view.wait_for_open_hand(
                paths["axial_exit"][0], phase="socket_after_open")
            release_completed = time.time()
            live.clear_view._follow_autodex(paths["axial_exit"].copy())
            live.clear_view._verify_release_waypoint(
                paths["axial_exit"][-1], phase="socket_axis_exit",
                require_open=True)
            live.clear_view._follow_autodex(paths["retract"].copy())
            final = live.clear_view.verify_state(
                self.read_state(), phase="socket_release_clear_view")
            clear_completed = time.time()
            if not started_at < release_completed <= clear_completed:
                raise RuntimeError("release/clear-view feedback is not ordered")
            record = {
                "schema": RELEASE_SCHEMA,
                "attempt_id": runner.active_attempt.attempt_id,
                "candidate_id": runner.active_attempt.candidate_id,
                "session_calibration_sha256": runner.session_sha256,
                "status": "released_and_clear_view",
                "guarded_execution": {"path": str(metric_path),
                                      "sha256": _sha(metric_path)},
                "release_plan": {"path": str(report_path),
                                 "sha256": _sha(report_path)},
                "started_at_s": started_at,
                "release_completed_at_s": release_completed,
                "clear_view_completed_at_s": clear_completed,
                "measurement": {"hand_open": True,
                                "clear_view_reached": True},
                "final_state": final.to_record(),
                "scope": "measured_open_clear_view_not_physical_insertion_depth",
                "robot_ready": False,
            }
            with output.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True,
                          allow_nan=False)
                stream.write("\n")
            verify_post_insertion_release(
                output, attempt_id=runner.active_attempt.attempt_id,
                candidate_id=runner.active_attempt.candidate_id,
                session_calibration_sha256=runner.session_sha256,
                guarded_execution_path=metric_path)
            self._prepared = None
            return output
        except Exception as exc:
            try:
                stop = self.executor.stop_and_acknowledge()
            except Exception as stop_exc:
                stop = {"acknowledged": False, "error": str(stop_exc)}
            with failure_path.open("x", encoding="utf-8") as stream:
                json.dump({
                    "schema": "precision_insertion_socket_release_failure_v1",
                    "attempt_id": runner.active_attempt.attempt_id,
                    "release_plan": str(report_path),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "stop_acknowledgement": stop,
                }, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            raise
