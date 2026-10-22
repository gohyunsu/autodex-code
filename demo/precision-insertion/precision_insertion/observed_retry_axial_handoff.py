"""Hand a visible-key XY retry's saved 20 mm stroke to contact control.

The transfer has already finished and a *new* preinsert hold has been
observed. This packet contains only the axial path; it is not a motor command
or evidence that the key physically entered the socket.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode, select_mode
from .guarded_axial_handoff import _state_from_record
from .live_robot_state import LiveRobotState
from .observed_retry_arrival_checkpoint import verify_observed_retry_arrival
from .records import AttemptRecord
from .retry_preflight import XYRetryPreflight


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_observed_retry_axial_handoff(
    *, attempt: AttemptRecord, preflight: XYRetryPreflight,
    preflight_report_path: Path, arrival_report_path: Path,
    measured_start: LiveRobotState, mode: TaskMode,
    session_calibration_sha256: str, decision_timestamp_s: float,
    output_dir: Path,
) -> Path:
    """Select the already planned axial path after observed corrected arrival."""
    if (not isinstance(attempt, AttemptRecord) or
            not isinstance(preflight, XYRetryPreflight) or
            not isinstance(measured_start, LiveRobotState) or
            not isinstance(mode, TaskMode) or attempt.mode != mode or
            not attempt._pending_retry or not attempt.events or
            attempt.events[-1].get("stage") != "xy_retry" or
            attempt.events[-1].get("value") != preflight.choice_id or
            attempt.labels["insertion_success"] is not False or
            attempt.candidate_id != "/".join(preflight.candidate_key) or
            attempt.session_calibration_sha256 != session_calibration_sha256 or
            preflight.status != "sampled_retry_preflight_pass" or
            preflight.planning.sampled_planning_pass is not True or
            not math.isclose(mode.target_depth_m, .020, abs_tol=1e-9)):
        raise ValueError("axial handoff needs this pending 20 mm XY retry")
    plan_path = Path(preflight_report_path).expanduser().resolve()
    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    if (any(saved.get(key) != value for key, value in
            preflight.to_record().items()) or
            saved.get("planned_trajectories") != "planned_trajectories.npz"):
        raise ValueError("axial handoff differs from the XY retry preflight")
    expected_mode = {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
        "target_depth_m": mode.target_depth_m,
    }
    if preflight.targets.to_record().get("mode") != expected_mode:
        raise ValueError("XY retry target belongs to another insertion mode")
    archive = plan_path.parent / "planned_trajectories.npz"
    if _sha(archive) != saved.get("planned_trajectories_sha256"):
        raise ValueError("axial handoff trajectory archive changed")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"transfer", "axial"}:
            raise ValueError("XY retry preflight lacks both planned paths")
        transfer = np.asarray(data["transfer"], dtype=np.float64)
        axial = np.asarray(data["axial"], dtype=np.float64)
    if (preflight.planning.transfer_trajectory is None or
            preflight.planning.axial_trajectory is None or
            not np.array_equal(transfer,
                               preflight.planning.transfer_trajectory) or
            not np.array_equal(axial, preflight.planning.axial_trajectory) or
            axial.ndim != 2 or axial.shape[1] != 13 or len(axial) < 2 or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(axial[0], transfer[-1]) or
            not np.allclose(axial[:, 7:], axial[0, 7:], atol=1e-8, rtol=0)):
        raise ValueError("XY retry axial path is discontinuous or releases the key")
    arrival_path = Path(arrival_report_path).expanduser().resolve()
    arrival = verify_observed_retry_arrival(arrival_path)
    if (arrival["retry_arrived"] is not True or
            arrival["attempt_id"] != attempt.attempt_id or
            arrival["candidate_id"] != attempt.candidate_id or
            arrival["session_calibration_sha256"] !=
                session_calibration_sha256 or
            arrival["xy_preflight_report_path"] != str(plan_path) or
            arrival["xy_preflight_report_sha256"] != _sha(plan_path)):
        raise ValueError("axial handoff needs this retry's observed hold")
    decision = float(decision_timestamp_s)
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = {
        "schema": "precision_insertion_observed_xy_retry_axial_handoff_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "mode": preflight.targets.to_record()["mode"],
        "xy_preflight_report_path": str(plan_path),
        "xy_preflight_report_sha256": _sha(plan_path),
        "arrival_report_path": str(arrival_path),
        "arrival_report_sha256": _sha(arrival_path),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": _sha(archive),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "axial_sample_count": len(axial),
        "target_depth_m": mode.target_depth_m,
        "measured_start": measured_start.to_record(),
        "decision_timestamp_s": decision,
        "scope": "saved_retry_axial_path_not_contact_or_insertion_success",
        "robot_ready": False,
    }
    report = target / "report.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def verify_observed_retry_axial_handoff(report_path: Path) -> dict:
    """Check the persisted retry contact packet before it is consumed."""
    path = Path(report_path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") !=
            "precision_insertion_observed_xy_retry_axial_handoff_v1" or
            record.get("scope") !=
            "saved_retry_axial_path_not_contact_or_insertion_success" or
            record.get("robot_ready") is not False):
        raise ValueError("invalid observed XY retry axial handoff")
    mode_row = record["mode"]
    mode = select_mode(mode_row["family"], mode_row["gap_mm"])
    if mode_row != {
            "family": mode.family, "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object,
            "target_depth_m": mode.target_depth_m}:
        raise ValueError("retry axial handoff mode changed")
    plan = Path(record["xy_preflight_report_path"])
    arrival_path = Path(record["arrival_report_path"])
    archive = Path(record["trajectory_archive_path"])
    if (not all(source.is_absolute() and source.is_file()
                for source in (plan, arrival_path, archive)) or
            _sha(plan) != record["xy_preflight_report_sha256"] or
            _sha(arrival_path) != record["arrival_report_sha256"] or
            _sha(archive) != record["trajectory_archive_sha256"]):
        raise ValueError("retry axial handoff source changed")
    saved = json.loads(plan.read_text(encoding="utf-8"))
    arrival = verify_observed_retry_arrival(arrival_path)
    if (saved.get("schema") != "precision_insertion_xy_retry_preflight_v1" or
            saved.get("status") != "sampled_retry_preflight_pass" or
            saved.get("candidate_key") != record["candidate_id"].split("/") or
            saved.get("planned_trajectories") != archive.name or
            saved.get("planned_trajectories_sha256") != _sha(archive) or
            saved.get("targets", {}).get("mode") != mode_row or
            arrival.get("retry_arrived") is not True or
            arrival.get("attempt_id") != record["attempt_id"] or
            arrival.get("candidate_id") != record["candidate_id"] or
            arrival.get("session_calibration_sha256") !=
                record["session_calibration_sha256"] or
            arrival.get("xy_preflight_report_path") != str(plan)):
        raise ValueError("retry axial handoff differs from observed arrival")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"transfer", "axial"}:
            raise ValueError("retry axial archive is incomplete")
        transfer = np.asarray(data["transfer"], dtype=np.float64)
        axial = np.asarray(data["axial"], dtype=np.float64)
    if (axial.shape != (record["axial_sample_count"], 13) or
            transfer.ndim != 2 or transfer.shape[1] != 13 or
            len(transfer) < 2 or len(axial) < 2 or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(axial[0], transfer[-1]) or
            not np.array_equal(axial[0], record["axial_start_q"]) or
            not np.array_equal(axial[-1], record["axial_end_q"]) or
            not np.allclose(axial[:, 7:], axial[0, 7:], atol=1e-8, rtol=0)):
        raise ValueError("retry axial handoff path changed")
    _state_from_record(record["measured_start"])
    if (record["target_depth_m"] != mode.target_depth_m or
            not math.isfinite(float(record["decision_timestamp_s"]))):
        raise ValueError("retry axial handoff target or timestamp changed")
    return record
