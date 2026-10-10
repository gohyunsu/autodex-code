"""Read actual FR3/Inspire joints without using commanded hand pose as feedback.

The existing FrankaExecutor.get_hand_qpos() explicitly reports its last
*commanded* hand vector. ParaDex's Inspire controller instead exposes raw
measured motor positions, which AutoDex already knows how to convert to
planner joint order. This adapter performs only reads; it never creates a
controller, moves a motor, or turns a single wrench sample into contact safety.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from autodex.utils.sync import convert_inspire_raw


@dataclass(frozen=True)
class LiveRobotState:
    full_q: np.ndarray
    arm_qvel: np.ndarray
    sample_timestamp_s: float
    arm_timestamp_s: float
    arm_robot_uptime_s: float
    hand_timestamp_s: float
    hand_raw_measured: np.ndarray
    hand_raw_commanded: np.ndarray
    max_hand_command_error_raw: float
    wrench: np.ndarray
    source: str = "robot_joint_feedback"

    def validate(self, *, max_arm_hand_skew_s: float,
                 max_hand_command_error_raw: float,
                 max_arm_velocity_rad_s: float) -> None:
        """Recheck a passed sample; a frozen dataclass does not freeze arrays."""
        limits = (max_arm_hand_skew_s, max_hand_command_error_raw,
                  max_arm_velocity_rad_s)
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in limits):
            raise ValueError("feedback validation limits must be positive")
        if self.source != "robot_joint_feedback":
            raise ValueError("hand pose is not measured robot feedback")
        q = _finite_vector(self.full_q, 13, "FR3/Inspire full qpos")
        velocity = _finite_vector(self.arm_qvel, 7, "FR3 qvel")
        if np.max(np.abs(velocity)) > max_arm_velocity_rad_s:
            raise ValueError("FR3 has not reached a stationary post-lift hold")
        measured = _finite_vector(self.hand_raw_measured, 6,
                                  "measured Inspire raw qpos")
        commanded = _finite_vector(self.hand_raw_commanded, 6,
                                   "commanded Inspire raw action")
        _finite_vector(self.wrench, 6, "FR3 wrench")
        if (np.any(measured < 0) or np.any(measured > 1000) or
                np.any(commanded < 0) or np.any(commanded > 1000)):
            raise ValueError("Inspire feedback/action lies outside 0..1000")
        if not np.allclose(q[7:], convert_inspire_raw(measured[None, :])[0],
                           atol=1e-8, rtol=0):
            raise ValueError("planner hand joints disagree with measured raw motors")
        error = float(np.max(np.abs(measured - commanded)))
        if (not math.isclose(error, float(self.max_hand_command_error_raw),
                             abs_tol=1e-8) or
                error > max_hand_command_error_raw):
            raise ValueError("Inspire measured joints do not track hold command")
        times = (self.sample_timestamp_s, self.arm_timestamp_s,
                 self.hand_timestamp_s, self.arm_robot_uptime_s)
        if (not all(math.isfinite(t) and t > 0 for t in times) or
                abs(self.arm_timestamp_s - self.hand_timestamp_s) >
                max_arm_hand_skew_s or
                not math.isclose(self.sample_timestamp_s,
                                 (self.arm_timestamp_s +
                                  self.hand_timestamp_s) / 2.0,
                                 abs_tol=1e-8)):
            raise ValueError("FR3/Inspire feedback timestamp contract failed")

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_live_robot_state_v1",
            "full_q": self.full_q.tolist(),
            "arm_qvel": self.arm_qvel.tolist(),
            "sample_timestamp_s": self.sample_timestamp_s,
            "arm_timestamp_s": self.arm_timestamp_s,
            "arm_robot_uptime_s": self.arm_robot_uptime_s,
            "hand_timestamp_s": self.hand_timestamp_s,
            "hand_raw_measured": self.hand_raw_measured.tolist(),
            "hand_raw_commanded": self.hand_raw_commanded.tolist(),
            "max_hand_command_error_raw": self.max_hand_command_error_raw,
            "wrench": self.wrench.tolist(),
            "source": self.source,
            "scope": "one_feedback_snapshot_not_motion_or_contact_authorization",
        }


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have {size} finite values")
    return array.copy()


def read_live_franka_inspire_state(
    *, arm, hand, max_arm_hand_skew_s: float,
    max_sample_age_s: float, max_hand_command_error_raw: float,
    max_arm_update_wait_s: float, max_arm_velocity_rad_s: float,
    now_s: float | None = None,
) -> LiveRobotState:
    """Read two controller snapshots on an externally verified common clock.

    Read these during a stationary hold. ParaDex Franka ``get_data()['time']``
    is controller uptime, **not** Unix time. Wait for a *new* streamed state
    and stamp its receipt on the robot PC wall clock. ParaDex hand
    ``get_data()['time']`` is a software read time, not a motor-latched time.
    Their small skew is necessary but not sufficient proof of simultaneity.
    The camera clock still needs separate commissioning.
    """
    thresholds = (max_arm_hand_skew_s, max_sample_age_s,
                  max_hand_command_error_raw, max_arm_update_wait_s,
                  max_arm_velocity_rad_s)
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in thresholds):
        raise ValueError("joint age, skew and hand-error limits must be positive")
    baseline = arm.get_data()
    if not isinstance(baseline, dict):
        raise ValueError("FR3 state stream is unavailable")
    baseline_uptime = float(baseline.get("time", float("nan")))
    if not math.isfinite(baseline_uptime):
        raise ValueError("FR3 controller uptime is invalid")
    deadline = time.monotonic() + max_arm_update_wait_s
    arm_data = None
    arm_receipt_time = None
    while time.monotonic() < deadline:
        sample = arm.get_data()
        if isinstance(sample, dict):
            uptime = float(sample.get("time", float("nan")))
            if math.isfinite(uptime) and uptime > baseline_uptime:
                arm_data = sample
                arm_receipt_time = time.time()
                break
        time.sleep(0.002)
    if arm_data is None:
        raise TimeoutError("FR3 state stream did not advance after sampling began")
    hand_data = hand.get_data()
    now = time.time() if now_s is None else float(now_s)
    if not isinstance(hand_data, dict):
        raise ValueError("FR3 and Inspire feedback must both be available")
    arm_time = float(arm_receipt_time)
    arm_uptime = float(arm_data["time"])
    hand_time = float(hand_data.get("time", float("nan")))
    if (not all(math.isfinite(t) and t > 0 for t in
                (now, arm_time, hand_time)) or
            abs(arm_time - hand_time) > max_arm_hand_skew_s or
            any(t > now or now - t > max_sample_age_s
                for t in (arm_time, hand_time))):
        raise ValueError("FR3/Inspire feedback is stale or not time-aligned")
    arm_q = _finite_vector(arm_data.get("qpos"), 7, "FR3 qpos")
    arm_qvel = _finite_vector(arm_data.get("qvel"), 7, "FR3 qvel")
    measured_raw = _finite_vector(hand_data.get("qpos"), 6,
                                  "Inspire measured raw qpos")
    commanded_raw = _finite_vector(hand_data.get("action"), 6,
                                   "Inspire commanded raw action")
    if (np.any(measured_raw < 0) or np.any(measured_raw > 1000) or
            np.any(commanded_raw < 0) or np.any(commanded_raw > 1000)):
        raise ValueError("Inspire feedback/action lies outside 0..1000")
    error = float(np.max(np.abs(measured_raw - commanded_raw)))
    if error > max_hand_command_error_raw:
        raise ValueError("Inspire measured joints do not track the hold command")
    hand_q = _finite_vector(convert_inspire_raw(measured_raw[None, :])[0],
                            6, "converted Inspire planner qpos")
    wrench = _finite_vector(arm_data.get("wrench"), 6, "FR3 wrench")
    result = LiveRobotState(
        np.concatenate([arm_q, hand_q]), arm_qvel,
        (arm_time + hand_time) / 2.0, arm_time, arm_uptime, hand_time,
        measured_raw, commanded_raw, error, wrench)
    result.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    return result
