"""Replayable, non-actuating evidence for one guarded insertion stroke.

The producer is responsible for calibrated acquisition-time samples and a
robot-side watchdog. Replaying these samples checks internal consistency; it
does *not* authenticate a sensor or measure physical key penetration.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from .guarded_contact import (
    GuardedContactLimits, GuardedContactMonitor, GuardedContactSample,
)


_SCHEMA_V1 = "precision_insertion_guarded_contact_trace_v1"
_SCHEMA_V2 = "precision_insertion_guarded_contact_trace_v2"


def _digest(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"invalid {name} digest")
    return value


def replay_guarded_contact_trace(
    *, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, family: str,
    limits: GuardedContactLimits, started_at_s: float,
    events: list[tuple[GuardedContactSample | None, float]],
    axial_handoff_sha256: str | None = None,
    trajectory_archive_sha256: str | None = None,
) -> dict:
    """Evaluate an ordered sample stream; require a terminal hold or abort.

    ``None`` is a missed-sample deadline check, not a fabricated measurement.
    Decision times must use the same clock as sample acquisition and start.
    """
    if (not isinstance(attempt_id, str) or not attempt_id.strip() or
            not isinstance(candidate_id, str) or not candidate_id.strip() or
            not isinstance(session_calibration_sha256, str) or
            len(session_calibration_sha256) != 64 or
            any(c not in "0123456789abcdef"
                for c in session_calibration_sha256) or
            not isinstance(events, list) or not events):
        raise ValueError("guarded trace needs IDs, session and events")
    handoff_digest = _digest(axial_handoff_sha256, "axial handoff")
    archive_digest = _digest(trajectory_archive_sha256, "trajectory archive")
    if (handoff_digest is None) != (archive_digest is None):
        raise ValueError("guarded trace path binding needs both digests")
    monitor = GuardedContactMonitor(
        family=family, limits=limits, started_at_s=started_at_s)
    rows = []
    previous_time = float(started_at_s)
    for index, (sample, decision_time_s) in enumerate(events):
        if (type(decision_time_s) not in (float, int) or
                not math.isfinite(decision_time_s) or
                decision_time_s <= previous_time or
                (sample is not None and
                 not isinstance(sample, GuardedContactSample))):
            raise ValueError("guarded trace event order or sample is invalid")
        previous_time = float(decision_time_s)
        decision = (monitor.check_sample_deadline(now_s=decision_time_s)
                    if sample is None else monitor.observe(
                        sample, decision_time_s=decision_time_s))
        rows.append({
            "decision_time_s": float(decision_time_s),
            "sample": None if sample is None else asdict(sample),
            "decision": decision.to_record(),
        })
        if decision.action != "continue_preplanned_stroke":
            if index != len(events) - 1:
                raise ValueError("guarded trace has events after a latched stop")
            break
    terminal = rows[-1]["decision"]
    if terminal["action"] == "continue_preplanned_stroke":
        raise ValueError("guarded trace ended without a hold or abort")
    record = {
        "schema": _SCHEMA_V2 if handoff_digest is not None else _SCHEMA_V1,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "family": family,
        "started_at_s": float(started_at_s),
        "terminal_decision_time_s": rows[-1]["decision_time_s"],
        "limits": asdict(limits),
        "events": rows,
        "terminal_action": terminal["action"],
        "safety_abort": terminal["action"] ==
                        "abort_hold_for_supervised_recovery",
        "scope": "replayed_external_samples_not_sensor_or_robot_certification",
        "robot_ready": False,
    }
    if handoff_digest is not None:
        record["path_binding"] = {
            "axial_handoff_sha256": handoff_digest,
            "trajectory_archive_sha256": archive_digest,
        }
    return record


def write_guarded_contact_trace(record: dict, path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    verify_guarded_contact_trace_record(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def verify_guarded_contact_trace_record(record: dict) -> dict:
    if (not isinstance(record, dict) or record.get("schema") not in {
                _SCHEMA_V1, _SCHEMA_V2} or
            record.get("robot_ready") is not False or
            record.get("scope") !=
            "replayed_external_samples_not_sensor_or_robot_certification" or
            not isinstance(record.get("events"), list)):
        raise ValueError("invalid guarded contact trace")
    binding = record.get("path_binding")
    if record["schema"] == _SCHEMA_V2:
        if (not isinstance(binding, dict) or set(binding) != {
                "axial_handoff_sha256", "trajectory_archive_sha256"}):
            raise ValueError("guarded v2 trace needs exact path binding")
    elif binding is not None:
        raise ValueError("guarded v1 trace cannot claim path binding")
    try:
        events = [
            (None if row["sample"] is None else
             GuardedContactSample(**row["sample"]),
             row["decision_time_s"])
            for row in record["events"]
        ]
        expected = replay_guarded_contact_trace(
            attempt_id=record["attempt_id"],
            candidate_id=record["candidate_id"],
            session_calibration_sha256=record["session_calibration_sha256"],
            family=record["family"],
            limits=GuardedContactLimits(**record["limits"]),
            started_at_s=record["started_at_s"], events=events,
            axial_handoff_sha256=(
                binding["axial_handoff_sha256"] if binding else None),
            trajectory_archive_sha256=(
                binding["trajectory_archive_sha256"] if binding else None))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("guarded contact trace cannot be replayed") from exc
    if record != expected:
        raise ValueError("guarded contact trace differs from replay")
    return record


def verify_guarded_contact_trace(path: Path) -> dict:
    return verify_guarded_contact_trace_record(
        json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8")))
