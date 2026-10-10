"""Replay external guarded-contact evidence for one grounded XY retry.

This verifies source consistency and exact path binding, not the accuracy of
an external key-depth producer. The record must never by itself set the task
success label or be interpreted as permission to start another stroke.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import TaskMode
from .grounded_lateral import GroundedLateralPreflight
from .guarded_trace import verify_guarded_contact_trace
from .insertion_checkpoint import _bound_external_metric_claim
from .outcome import KEY_DEPTH_SOURCES
from .postshift_arrival_checkpoint import PostShiftArrivalCheckpoint
from .postshift_arrival_replan import PostShiftArrivalReplan
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .retry_axial_handoff import verify_retry_axial_handoff


_SCHEMA = "precision_insertion_guarded_retry_execution_v1"
_MEASUREMENT_FIELDS = {
    "key_depth_interval_m", "key_depth_source",
    "alignment_within_limits", "safety_abort", "grasp_held",
}
_SOURCE_NAMES = {"key_depth", "alignment", "force_trace", "grasp_state"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_retry_guarded_metric(
    metric_path: Path, *, handoff_report_path: Path,
    expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint,
    checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> tuple[dict, float, float]:
    """Recheck one retry stroke against fresh arrival, path and trace bytes."""
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    handoff = verify_retry_axial_handoff(
        handoff_path, expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    path = Path(metric_path).expanduser().resolve()
    metric = json.loads(path.read_text(encoding="utf-8"))
    handoff_ref = {"path": str(handoff_path), "sha256": _sha(handoff_path)}
    if (not isinstance(metric, dict) or metric.get("schema") != _SCHEMA or
            metric.get("attempt_id") != handoff["attempt_id"] or
            metric.get("candidate_id") != handoff["candidate_id"] or
            metric.get("session_calibration_sha256") !=
                handoff["session_calibration_sha256"] or
            metric.get("retry_axial_handoff") != handoff_ref or
            metric.get("trajectory_archive_sha256") !=
                handoff["trajectory_archive_sha256"] or
            metric.get("robot_ready") is not False or
            metric.get("scope") !=
                "external_retry_samples_not_physical_key_depth_certification"):
        raise ValueError("retry metric differs from fresh axial handoff")
    try:
        started = float(metric["started_at_s"])
        completed = float(metric["completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retry metric lacks stroke times") from exc
    if (not math.isfinite(started) or not math.isfinite(completed) or
            not handoff["decision_timestamp_s"] < started < completed):
        raise ValueError("retry metric stroke does not follow handoff")
    measurement = metric.get("measurement")
    if (not isinstance(measurement, dict) or
            set(measurement) != _MEASUREMENT_FIELDS or
            any(measurement[name] is not None and
                type(measurement[name]) is not bool
                for name in ("alignment_within_limits", "safety_abort",
                             "grasp_held"))):
        raise ValueError("retry metric measurement is incomplete")
    interval = measurement["key_depth_interval_m"]
    if interval is not None:
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(type(value) not in (int, float) or
                    not math.isfinite(value) for value in interval) or
                interval[0] > interval[1]):
            raise ValueError("retry metric key-depth interval is invalid")
    source = measurement["key_depth_source"]
    if ((interval is None and source is not None) or
            (interval is not None and source not in KEY_DEPTH_SOURCES)):
        raise ValueError("retry metric key-depth source is invalid")
    sources = metric.get("source_records")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_NAMES:
        raise ValueError("retry metric needs four named source records")
    source_paths = []
    for name, reference in sources.items():
        if (not isinstance(reference, dict) or
                set(reference) != {"path", "sha256"} or
                not isinstance(reference["path"], str) or
                not isinstance(reference["sha256"], str)):
            raise ValueError(f"retry metric source is malformed: {name}")
        raw_path = Path(reference["path"])
        source_path = raw_path.expanduser().resolve()
        if (not raw_path.is_absolute() or not source_path.is_file() or
                source_path == path or
                _sha(source_path) != reference["sha256"]):
            raise ValueError(f"retry metric source changed: {name}")
        source_paths.append(source_path)
        if name != "force_trace":
            _bound_external_metric_claim(
                name=name, path=source_path, metric=metric,
                started=started, completed=completed)
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("retry metric repeats a source record")
    trace = verify_guarded_contact_trace(
        Path(sources["force_trace"]["path"]))
    if (trace["schema"] !=
            "precision_insertion_guarded_contact_trace_v2" or
            trace["family"] != mode.family or
            trace.get("path_binding") != {
                "axial_handoff_sha256": handoff_ref["sha256"],
                "trajectory_archive_sha256":
                    handoff["trajectory_archive_sha256"],
            } or
            metric.get("contact_limits") != trace["limits"] or
            trace["attempt_id"] != handoff["attempt_id"] or
            trace["candidate_id"] != handoff["candidate_id"] or
            trace["session_calibration_sha256"] !=
                handoff["session_calibration_sha256"] or
            trace["started_at_s"] != started or
            trace["terminal_decision_time_s"] > completed or
            trace["safety_abort"] is not measurement["safety_abort"]):
        raise ValueError("retry metric trace used another path or stroke")
    return metric, started, completed
