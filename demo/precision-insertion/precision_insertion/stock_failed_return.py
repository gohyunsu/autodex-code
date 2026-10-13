"""Record the stock AutoDex held-key return after a force-stop ascent.

The motion itself remains in ``ClearViewForPerception``: held transfer,
descent, full Inspire opening, checked lift and stock cuRobo retract. This
recorder saves the measured waypoint feedback and binds it to the plan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .failed_return_evidence import (
    STOCK_CHECKPOINTS, STOCK_SOURCE, verify_failed_return_execution,
)
from .franka_completion_checked import is_demo_completion_checked_executor
from .live_robot_state import LiveRobotState


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class StockFailedReturnRecorder:
    """One stock executor, one saved return plan, one measured execution."""

    def __init__(self, executor):
        if not is_demo_completion_checked_executor(executor):
            raise TypeError("stock failed return needs the session FR3 executor")
        self.executor = executor

    def __call__(self, run_planned_return, report_path, runner, decision) -> Path:
        if (not callable(run_planned_return) or
                runner.current_decision() != decision or
                decision.action != "recover_held_key_then_reobserve" or
                runner.active_attempt is None or runner._attempt_dir is None):
            raise ValueError("stock return belongs to another held-key recovery")
        report_path = Path(report_path).expanduser().resolve()
        folder = Path(runner._attempt_dir).expanduser().resolve()
        if (report_path != folder / "failed_return_preflight" / "report.json" or
                not report_path.is_file()):
            raise ValueError("stock return needs this attempt's saved plan")
        plan = json.loads(report_path.read_text(encoding="utf-8"))
        if (plan.get("schema") != "precision_insertion_failed_return_plan_v1" or
                plan.get("status") !=
                "nominal_failed_return_plan_pass_drop_unobserved" or
                plan.get("return_origin") != "insertion" or
                plan.get("attempt_id") != runner.active_attempt.attempt_id or
                plan.get("candidate_id") != runner.active_attempt.candidate_id or
                plan.get("session_calibration_sha256") != runner.session_sha256 or
                plan.get("catalog_sha256") != runner.catalog_sha256 or
                plan.get("robot_ready") is not False or
                plan.get("held_plan", {}).get("sampled_held_path_pass") is not True or
                plan.get("release_plan", {}).get("nominal_path_pass") is not True):
            raise ValueError("stock return plan differs from live attempt")
        sources = plan.get("sources")
        if (not isinstance(sources, dict) or
                set(sources) != {"withdrawal", "postlift_report"} or
                any(_sha(Path(ref["path"])) != ref["sha256"]
                    for ref in sources.values())):
            raise ValueError("stock return plan sources changed")
        archive_name = plan.get("planned_trajectories")
        if (not isinstance(archive_name, str) or
                Path(archive_name).name != archive_name):
            raise ValueError("stock return trajectory name is invalid")
        archive = report_path.parent / archive_name
        if _sha(archive) != plan["planned_trajectories_sha256"]:
            raise ValueError("stock return planned trajectory changed")
        started_path = folder / "failed_return_stock_started.json"
        failure_path = folder / "failed_return_stock_failure.json"
        feedback_path = folder / "failed_return_stock_feedback.json"
        output = folder / "failed_return_stock_execution.json"
        if any(path.exists() for path in (
                started_path, failure_path, feedback_path, output)):
            raise FileExistsError("stock return was already requested")
        _write(started_path, {
            "schema": "precision_insertion_stock_return_started_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "plan_report": {"path": str(report_path), "sha256": _sha(report_path)},
            "requested_at_s": time.time(),
        })
        checkpoints = []
        try:
            final = run_planned_return(evidence_sink=checkpoints)
            if (not isinstance(final, LiveRobotState) or
                    [row.get("name") for row in checkpoints] !=
                    list(STOCK_CHECKPOINTS) or
                    not np.array_equal(final.full_q,
                                       checkpoints[-1]["state"]["full_q"])):
                raise RuntimeError("stock return lacks measured phase feedback")
            stopped = self.executor.stop_and_acknowledge()
            if stopped.get("acknowledged") is not True:
                raise RuntimeError("FR3 return stop was not acknowledged")
            _write(feedback_path, {
                "schema": "precision_insertion_stock_return_feedback_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "plan_report_sha256": _sha(report_path),
                "checkpoints": checkpoints,
                "stop_acknowledgement": stopped,
                "scope": "measured_waypoints_not_key_landing_or_pose_accuracy",
            })
            record = {
                "schema": "precision_insertion_failed_return_execution_v1",
                "return_origin": "insertion", "source": STOCK_SOURCE,
                "attempt_id": runner.active_attempt.attempt_id,
                "candidate_id": runner.active_attempt.candidate_id,
                "session_calibration_sha256": runner.session_sha256,
                "catalog_sha256": runner.catalog_sha256,
                "safety_abort": False,
                "grip_loss_before_release": False,
                "grip_loss_scope": "measured_hand_hold_not_physical_key_presence",
                "plan_report": {"path": str(report_path),
                                "sha256": _sha(report_path)},
                "stock_feedback": {"path": str(feedback_path),
                                   "sha256": _sha(feedback_path)},
                "hand_open_final": True,
                "arm_clear_view_final": True,
                "scope": "stock_measured_return_before_fresh_camera_landing",
            }
            _write(output, record)
            verify_failed_return_execution(
                path=output, attempt_id=runner.active_attempt.attempt_id,
                candidate_id=runner.active_attempt.candidate_id,
                attempt_started_at_s=runner.active_attempt.started_at_s,
                session_calibration_sha256=runner.session_sha256,
                catalog_sha256=runner.catalog_sha256)
            return output
        except Exception as exc:
            try:
                stopped = self.executor.stop_and_acknowledge()
            except Exception as stop_exc:
                stopped = {"acknowledged": False, "error": str(stop_exc)}
            _write(failure_path, {
                "schema": "precision_insertion_stock_return_failure_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "plan_report": str(report_path),
                "error_type": type(exc).__name__, "error": str(exc),
                "stop_acknowledgement": stopped,
                "robot_state_after_failure": "unknown_requires_recovery",
            })
            raise
