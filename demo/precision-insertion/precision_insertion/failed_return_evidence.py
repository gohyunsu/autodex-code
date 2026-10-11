"""Bind a failed insertion's held-key return plan to external execution.

The plan is read-only until a commissioned controller independently executes
it. The controller log is evidence for ordering and provenance, not proof that
the key landed where expected; a fresh AutoDex key capture supplies that.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT

from .live_robot_state import LiveRobotState
from .repose_preflight import ReposeHeldPreflight
from .repose_release import ReposeReleasePreflight


PHASES = (
    "axial_withdrawal", "held_transfer", "held_descent", "release_open",
    "post_release_lift", "post_release_retract",
)
SOURCES = frozenset({
    "trajectory_feedback", "hand_feedback", "guarded_controller", "safety",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict:
    raw = Path(path).expanduser()
    resolved = raw.resolve()
    if not raw.is_absolute() or not resolved.is_file():
        raise FileNotFoundError(f"failed-return source is missing: {raw}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _finite(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"invalid failed-return {label}")
    return float(value)


def write_failed_return_plan(
    *, output_dir: Path, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, catalog_sha256: str,
    withdrawal_evidence_path: Path, postlift_report_path: Path,
    withdrawn_state: LiveRobotState,
    held: ReposeHeldPreflight, release: ReposeReleasePreflight,
) -> Path:
    """Save exact planned motions and sources without claiming execution."""
    if (not attempt_id or not candidate_id or
            len(session_calibration_sha256) != 64 or
            len(catalog_sha256) != 64 or
            held.status != "sampled_held_path_pass_release_unplanned" or
            release.status !=
            "nominal_release_exit_path_pass_drop_unobserved" or
            not isinstance(held.sampled_held_path_audit, dict) or
            held.sampled_held_path_audit.get("sampled_clear") is not True or
            not isinstance(release.release_geometry_audit, dict) or
            release.release_geometry_audit.get("sampled_clear") is not True):
        raise ValueError("failed return needs bound passing held/release plans")
    arrays = {
        "held_transfer": held.transfer_trajectory,
        "held_descent": held.descent_trajectory,
        "post_release_lift": release.post_release_lift_trajectory,
        "post_release_retract": release.retract_trajectory,
    }
    if any(value is None for value in arrays.values()):
        raise ValueError("failed return lacks a planned motion segment")
    validated = {}
    for name, value in arrays.items():
        path = np.asarray(value, dtype=np.float64)
        if (path.ndim != 2 or path.shape[1] != 13 or len(path) < 2 or
                not np.all(np.isfinite(path))):
            raise ValueError(f"failed return has malformed {name} trajectory")
        validated[name] = path.copy()
    if (not np.allclose(validated["held_transfer"][-1],
                        validated["held_descent"][0], atol=1e-4, rtol=0) or
            not np.allclose(validated["held_descent"][-1, :7],
                            validated["post_release_lift"][0, :7],
                            atol=1e-4, rtol=0) or
            not np.allclose(validated["post_release_lift"][-1],
                            validated["post_release_retract"][0],
                            atol=1e-4, rtol=0) or
            not np.allclose(release.release_hand_q, INSPIRE_INIT,
                            atol=1e-8, rtol=0) or
            not np.allclose(release.retreat_goal_arm_q, FR3_CLEAR_VIEW,
                            atol=1e-8, rtol=0) or
            not np.allclose(validated["post_release_retract"][-1, :7],
                            FR3_CLEAR_VIEW, atol=1e-3, rtol=0)):
        raise ValueError("failed return stages do not join continuously")
    if (not isinstance(withdrawn_state, LiveRobotState) or
            withdrawn_state.source != "robot_joint_feedback" or
            not np.allclose(validated["held_transfer"][0],
                            withdrawn_state.full_q, atol=1e-8, rtol=0)):
        raise ValueError("failed return start differs from measured withdrawal state")
    sources = {
        "withdrawal": _source(withdrawal_evidence_path),
        "postlift_report": _source(postlift_report_path),
    }
    withdrawal = json.loads(Path(sources["withdrawal"]["path"]).read_text(
        encoding="utf-8"))
    if (withdrawal.get("schema") !=
            "precision_insertion_guarded_withdrawal_v1" or
            withdrawal.get("attempt_id") != attempt_id or
            withdrawal.get("candidate_id") != candidate_id or
            withdrawal.get("status") != "withdrawn_to_preinsert_hold" or
            withdrawal.get("key_still_held") is not True or
            withdrawal.get("safety_abort") is not False or
            withdrawal.get("source") != "commissioned_guarded_controller"):
        raise ValueError("failed return lacks this attempt's guarded withdrawal")
    _finite(withdrawal.get("completed_at_s"), "withdrawal time")
    if withdrawn_state.sample_timestamp_s <= withdrawal["completed_at_s"]:
        raise ValueError("failed return state predates guarded withdrawal")
    report = json.loads(Path(sources["postlift_report"]["path"]).read_text(
        encoding="utf-8"))
    if (report.get("attempt_id") != attempt_id or
            report.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            report.get("catalog_sha256") != catalog_sha256 or
            report.get("status") != "sampled_postlift_preflight_pass"):
        raise ValueError("failed return has another attempt's postlift relation")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    archive = target / "planned_trajectories.npz"
    np.savez_compressed(archive, **validated)
    record = {
        "schema": "precision_insertion_failed_return_plan_v1",
        "status": "nominal_failed_return_plan_pass_drop_unobserved",
        "attempt_id": attempt_id, "candidate_id": candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "catalog_sha256": catalog_sha256,
        "sources": sources,
        "withdrawal_completed_at_s": withdrawal["completed_at_s"],
        "withdrawn_joint_state": withdrawn_state.to_record(),
        "held_plan": held.to_record(), "release_plan": release.to_record(),
        "planned_trajectories": archive.name,
        "planned_trajectories_sha256": _sha(archive),
        "robot_ready": False,
        "scope": "planned_return_not_execution_or_landing",
    }
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def verify_failed_return_execution(
    *, path: Path, attempt_id: str, candidate_id: str,
    attempt_started_at_s: float, session_calibration_sha256: str,
    catalog_sha256: str,
) -> dict:
    """Verify ordered external feedback and unchanged planned source bytes."""
    source = Path(path).expanduser().resolve()
    row = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(row, dict) or row.get("schema") !=
            "precision_insertion_failed_return_execution_v1" or
            row.get("source") != "commissioned_external_controller" or
            row.get("attempt_id") != attempt_id or
            row.get("candidate_id") != candidate_id or
            row.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            row.get("catalog_sha256") != catalog_sha256 or
            row.get("safety_abort") is not False or
            row.get("grip_loss_before_release") is not False):
        raise ValueError("failed return execution is not bound to this attempt")
    report_source = row.get("plan_report")
    if not isinstance(report_source, dict) or set(report_source) != {
            "path", "sha256"}:
        raise ValueError("failed return execution lacks a bound plan")
    report_ref = _source(Path(report_source["path"]))
    if report_ref != report_source:
        raise ValueError("failed return plan report changed")
    report = json.loads(Path(report_ref["path"]).read_text(encoding="utf-8"))
    if (report.get("schema") != "precision_insertion_failed_return_plan_v1" or
            report.get("status") !=
            "nominal_failed_return_plan_pass_drop_unobserved" or
            report.get("attempt_id") != attempt_id or
            report.get("candidate_id") != candidate_id or
            report.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            report.get("catalog_sha256") != catalog_sha256 or
            report.get("robot_ready") is not False or
            report.get("held_plan", {}).get("sampled_held_path_pass") is not True or
            report.get("release_plan", {}).get("nominal_path_pass") is not True):
        raise ValueError("failed return plan is not an unchanged passing plan")
    source_files = report.get("sources")
    if not isinstance(source_files, dict) or set(source_files) != {
            "withdrawal", "postlift_report"}:
        raise ValueError("failed return plan lacks its source evidence")
    for evidence in source_files.values():
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
            raise ValueError("failed return plan has malformed source evidence")
        if _source(Path(evidence["path"])) != evidence:
            raise ValueError("failed return plan source changed")
    archive_name = report.get("planned_trajectories")
    if (not isinstance(archive_name, str) or
            Path(archive_name).name != archive_name):
        raise ValueError("failed return has no trajectory archive")
    archive = (Path(report_ref["path"]).parent / archive_name).resolve()
    if (not archive.is_relative_to(Path(report_ref["path"]).parent) or
            not archive.is_file() or _sha(archive) !=
            report.get("planned_trajectories_sha256")):
        raise ValueError("failed return trajectory archive changed")
    measured = report.get("withdrawn_joint_state")
    if (not isinstance(measured, dict) or
            measured.get("schema") != "precision_insertion_live_robot_state_v1" or
            measured.get("source") != "robot_joint_feedback" or
            _finite(measured.get("sample_timestamp_s"), "joint time") <=
            _finite(report.get("withdrawal_completed_at_s"), "withdrawal time")):
        raise ValueError("failed return lacks a later measured start state")
    with np.load(archive, allow_pickle=False) as paths:
        if (set(paths.files) != {
                "held_transfer", "held_descent", "post_release_lift",
                "post_release_retract"} or
                not np.allclose(paths["held_transfer"][0],
                                np.asarray(measured.get("full_q"), dtype=float),
                                atol=1e-8, rtol=0)):
            raise ValueError("failed return planned start differs from feedback")
    phases = row.get("phases")
    if (not isinstance(phases, list) or len(phases) != len(PHASES) or
            [item.get("name") for item in phases if isinstance(item, dict)] !=
            list(PHASES)):
        raise ValueError("failed return needs every ordered execution phase")
    previous = _finite(attempt_started_at_s, "attempt start")
    release_at = None
    for name, item in zip(PHASES, phases):
        started = _finite(item.get("started_at_s"), f"{name} start")
        completed = _finite(item.get("completed_at_s"), f"{name} completion")
        if (started < previous or completed <= started or
                item.get("complete") is not True or
                item.get("safety_abort") is not False or
                (name in PHASES[:3] and item.get("grasp_held") is not True) or
                (name == "release_open" and
                 item.get("hand_open_feedback") is not True)):
            raise ValueError(f"failed return {name} was not completed in order")
        if name == "release_open":
            release_at = completed
        previous = completed
    if (phases[0]["completed_at_s"] !=
            report.get("withdrawal_completed_at_s")):
        raise ValueError("failed return executed another guarded withdrawal")
    evidence = row.get("source_records")
    if not isinstance(evidence, dict) or set(evidence) != SOURCES:
        raise ValueError("failed return lacks independent controller sources")
    seen = {source, Path(report_ref["path"]), archive}
    for named in source_files.values():
        seen.add(Path(named["path"]))
    for named in evidence.values():
        if not isinstance(named, dict) or set(named) != {"path", "sha256"}:
            raise ValueError("failed return controller source is malformed")
        actual = _source(Path(named["path"]))
        if actual != named or Path(actual["path"]) in seen:
            raise ValueError("failed return controller source changed or duplicated")
        seen.add(Path(actual["path"]))
    if (row.get("hand_open_final") is not True or
            row.get("arm_clear_view_final") is not True):
        raise ValueError("failed return did not confirm final clear view")
    return {
        "path": str(source), "sha256": _sha(source),
        "release_completed_at_s": release_at,
        "exit_completed_at_s": previous,
        "plan_report": report_ref,
        "scope": "external_feedback_assertions_plus_later_camera_landing",
    }
