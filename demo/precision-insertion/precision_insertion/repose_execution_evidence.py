"""Validate externally produced repose execution evidence before landing.

This checks provenance and chronology, not whether a controller truly moved
the robot. A commissioned controller must produce the independent feedback
files; a JSON assertion cannot by itself establish physical execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "precision_insertion_repose_execution_evidence_v1"
PHASES = (
    "pickup_squeeze", "held_lift", "held_transfer", "held_descent",
    "release_open", "post_release_lift", "post_release_retract",
)
SOURCE_NAMES = frozenset({
    "trajectory_feedback", "safety", "grasp_state", "hand_feedback",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"invalid repose {name}")
    return float(value)


def verify_repose_execution_evidence(
    *, path: Path, attempt_id: str, attempt_started_at_s: float,
    preflight_report_path: Path, preflight_report_sha256: str,
    session_calibration_sha256: str, selected_seed: dict,
) -> dict:
    """Bind a complete release/exit log to this exact directed reset plan.

    The returned completion times may order a *later* camera observation.
    This function never authorizes motion or sets a success label.
    """
    source = Path(path).expanduser().resolve()
    report_path = Path(preflight_report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("repose preflight report must be a JSON object")
    artifacts = report.get("artifacts")
    if (not isinstance(artifacts, dict) or
            report.get("status") !=
            "nominal_reset_preflight_pass_drop_unobserved" or
            report.get("selected_seed") != selected_seed or
            report.get("robot_ready") is not False or
            _sha(report_path) != preflight_report_sha256):
        raise ValueError("repose execution has no unchanged passing preflight")
    scene_name = artifacts.get("trial_scene")
    inputs = artifacts.get("input_files")
    if (not isinstance(scene_name, str) or
            Path(scene_name).name != scene_name or
            not isinstance(inputs, dict) or set(inputs) != {
                "session", "catalog", "key_pose_world", "live_start_q",
                "limits"}):
        raise ValueError("repose preflight lacks exact source artifacts")
    scene = (report_path.parent / scene_name).resolve()
    if (not scene.is_relative_to(report_path.parent) or
            not scene.is_file() or
            artifacts.get("trial_scene_sha256") != _sha(scene)):
        raise ValueError("repose frozen scene changed")
    for name, evidence in inputs.items():
        if (not isinstance(evidence, dict) or
                set(evidence) != {"path", "sha256"} or
                not isinstance(evidence["path"], str)):
            raise ValueError(f"invalid repose {name} preflight source")
        raw_path = Path(evidence["path"]).expanduser()
        resolved = raw_path.resolve()
        if (not raw_path.is_absolute() or not resolved.is_file() or
                evidence["sha256"] != _sha(resolved)):
            raise ValueError(f"repose {name} preflight source changed")
    relative = artifacts.get("planned_trajectories")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("repose execution has no saved trajectory archive")
    trajectory = (report_path.parent / relative).resolve()
    if (not trajectory.is_relative_to(report_path.parent) or
            not trajectory.is_file() or
            artifacts.get("planned_trajectories_sha256") != _sha(trajectory)):
        raise ValueError("repose planned trajectories changed")
    record = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA or
            record.get("source") != "commissioned_external_controller" or
            record.get("attempt_id") != attempt_id or
            record.get("preflight_report_sha256") != preflight_report_sha256 or
            record.get("planned_trajectories_sha256") != _sha(trajectory) or
            record.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            record.get("selected_seed") != selected_seed or
            record.get("safety_abort") is not False or
            record.get("grip_loss_before_release") is not False):
        raise ValueError("repose execution is not bound to this safe directed plan")
    phases = record.get("phases")
    if (not isinstance(phases, list) or len(phases) != len(PHASES) or
            [row.get("name") for row in phases if isinstance(row, dict)] !=
            list(PHASES)):
        raise ValueError("repose execution needs every ordered phase")
    previous = _finite(attempt_started_at_s, "attempt start")
    release_completed = None
    for name, row in zip(PHASES, phases):
        started = _finite(row.get("started_at_s"), f"{name} start")
        completed = _finite(row.get("completed_at_s"), f"{name} completion")
        if (row.get("complete") is not True or
                row.get("safety_abort") is not False or
                started < previous or completed <= started):
            raise ValueError(f"repose {name} was not safely completed in order")
        if name in {"held_lift", "held_transfer", "held_descent"} and (
                row.get("grasp_held") is not True):
            raise ValueError(f"repose {name} lost the held key")
        if name == "release_open":
            if row.get("hand_open_feedback") is not True:
                raise ValueError("repose release has no open-hand feedback")
            release_completed = completed
        previous = completed
    sources = record.get("source_records")
    if not isinstance(sources, dict) or set(sources) != SOURCE_NAMES:
        raise ValueError("repose lacks independent controller source records")
    source_paths = set()
    forbidden = {source, report_path, trajectory, scene}
    forbidden.update(Path(item["path"]).expanduser().resolve()
                     for item in inputs.values())
    for name, evidence in sources.items():
        if (not isinstance(evidence, dict) or
                set(evidence) != {"path", "sha256"} or
                not isinstance(evidence["path"], str)):
            raise ValueError(f"invalid repose {name} source")
        raw_path = Path(evidence["path"]).expanduser()
        resolved = raw_path.resolve()
        if (not raw_path.is_absolute() or not resolved.is_file() or
                resolved in source_paths or resolved in forbidden or
                evidence["sha256"] != _sha(resolved)):
            raise ValueError(f"repose {name} source is missing or changed")
        source_paths.add(resolved)
    if release_completed is None:
        raise ValueError("repose release phase has no completion")
    return {
        "path": str(source), "sha256": _sha(source),
        "release_completed_at_s": release_completed,
        "exit_completed_at_s": previous,
        "source_records": sources,
        "scope": "external_controller_assertions_not_physical_certification",
    }
