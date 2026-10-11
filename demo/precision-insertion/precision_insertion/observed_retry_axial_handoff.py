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

from .config import TaskMode
from .guarded_axial_handoff import _positive
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
    max_state_age_s: float, max_start_joint_error_rad: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float, output_dir: Path,
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
    limits = {
        "max_state_age_s": _positive(max_state_age_s, "state age"),
        "max_start_joint_error_rad": _positive(
            max_start_joint_error_rad, "start joint error"),
        "max_arm_hand_skew_s": _positive(max_arm_hand_skew_s, "arm/hand skew"),
        "max_hand_command_error_raw": _positive(
            max_hand_command_error_raw, "hand tracking error"),
        "max_arm_velocity_rad_s": _positive(
            max_arm_velocity_rad_s, "hold velocity"),
    }
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
    measured_start.validate(
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    decision = float(decision_timestamp_s)
    sample = float(measured_start.sample_timestamp_s)
    if (not math.isfinite(decision) or
            not arrival["observation_completed_at_s"] < sample <= decision or
            decision - sample > limits["max_state_age_s"] or
            np.max(np.abs(measured_start.full_q - axial[0])) >
                limits["max_start_joint_error_rad"]):
        raise ValueError("retry axial start is stale or differs from arrival")
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
        "limits": limits,
        "scope": "saved_retry_axial_path_not_contact_or_insertion_success",
        "robot_ready": False,
    }
    report = target / "report.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report
