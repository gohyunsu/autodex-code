"""Fail-closed sample policy for a future guarded 20 mm insertion actuator.

The caller supplies independently commissioned, time-aligned samples in the
*socket* frame.  In particular, ``nominal_depth_m`` is wrist/FK progress of a
bounded key-in-hand hypothesis, not a measured physical key penetration.
This policy does not send Franka commands, stop a daemon, label task success,
or replace an independent robot-side velocity dead-man/watchdog.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


_ACTION = Literal[
    "continue_preplanned_stroke", "hold_for_key_depth_and_vlm",
    "abort_hold_for_supervised_recovery",
]


@dataclass(frozen=True)
class GuardedContactLimits:
    target_depth_m: float
    max_axial_force_n: float
    max_lateral_force_n: float
    max_torque_nm: float
    max_lateral_error_m: float
    max_axis_tilt_deg: float
    max_yaw_error_deg: float | None
    max_sample_age_s: float
    max_sample_gap_s: float
    max_duration_s: float
    max_depth_step_m: float
    max_depth_regression_m: float
    max_depth_overshoot_m: float

    def validate(self, *, family: str) -> None:
        if family not in {"square", "cylinder"}:
            raise ValueError("guarded contact requires a known key/socket family")
        positive = (
            "target_depth_m", "max_axial_force_n", "max_lateral_force_n",
            "max_torque_nm", "max_lateral_error_m", "max_axis_tilt_deg",
            "max_sample_age_s", "max_sample_gap_s", "max_duration_s",
            "max_depth_step_m",
        )
        nonnegative = ("max_depth_regression_m", "max_depth_overshoot_m")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if (type(value) not in (float, int) or not math.isfinite(value) or
                    value < (0 if name in nonnegative else 1e-15)):
                raise ValueError(f"{name} needs a commissioned finite limit")
        if not math.isclose(self.target_depth_m, .020, rel_tol=0, abs_tol=1e-9):
            raise ValueError("guarded insertion target must be the task's 20 mm")
        if (self.max_yaw_error_deg is None and family == "square"):
            raise ValueError("square insertion needs a yaw-error limit")
        if self.max_yaw_error_deg is not None and (
                type(self.max_yaw_error_deg) not in (float, int) or
                not math.isfinite(self.max_yaw_error_deg) or
                self.max_yaw_error_deg <= 0):
            raise ValueError("yaw-error limit must be finite and positive")


@dataclass(frozen=True)
class GuardedContactSample:
    timestamp_s: float
    nominal_depth_m: float
    force_socket_n: tuple[float, float, float]
    moment_socket_nm: tuple[float, float, float]
    lateral_error_m: float
    axis_tilt_deg: float
    yaw_error_deg: float | None
    hand_command_tracked: bool

    def validate(self, *, family: str) -> None:
        scalars = (self.timestamp_s, self.nominal_depth_m,
                   self.lateral_error_m, self.axis_tilt_deg)
        if (any(type(v) not in (float, int) or not math.isfinite(v)
                for v in scalars) or
                self.lateral_error_m < 0 or self.axis_tilt_deg < 0 or
                type(self.hand_command_tracked) is not bool):
            raise ValueError("guarded contact sample has invalid pose/grip data")
        for name, vector in (("force", self.force_socket_n),
                             ("moment", self.moment_socket_nm)):
            if (not isinstance(vector, (tuple, list)) or len(vector) != 3 or
                    any(type(v) not in (float, int) or not math.isfinite(v)
                        for v in vector)):
                raise ValueError(f"guarded {name} must be a finite socket-frame vector")
        if family == "square" and self.yaw_error_deg is None:
            raise ValueError("square contact sample lacks measured yaw residual")
        if self.yaw_error_deg is not None and (
                type(self.yaw_error_deg) not in (float, int) or
                not math.isfinite(self.yaw_error_deg) or
                self.yaw_error_deg < 0):
            raise ValueError("guarded yaw residual must be finite and nonnegative")


@dataclass(frozen=True)
class GuardedContactDecision:
    action: _ACTION
    reason: str
    sample_count: int
    last_sample_timestamp_s: float | None
    last_nominal_depth_m: float | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_guarded_contact_decision_v1",
            **asdict(self),
            "scope": "sample_policy_not_robot_command_or_key_depth_label",
            "robot_ready": False,
        }


class GuardedContactMonitor:
    """Latched, single-stroke policy; no automatic restart after a stop.

    A real actuator must call ``observe`` before each planned command and
    have its *own independent* watchdog that stops streaming if Python dies.
    Returning ``continue_preplanned_stroke`` is never by itself permission to
    start a new or unplanned motion.
    """

    def __init__(self, *, family: str, limits: GuardedContactLimits,
                 started_at_s: float):
        limits.validate(family=family)
        if (type(started_at_s) not in (float, int) or
                not math.isfinite(started_at_s)):
            raise ValueError("guarded contact needs a finite start timestamp")
        self.family = family
        self.limits = limits
        self.started_at_s = float(started_at_s)
        self._last_timestamp_s: float | None = None
        self._last_depth_m: float | None = None
        self._sample_count = 0
        self._terminal: GuardedContactDecision | None = None

    def _decision(self, action: _ACTION, reason: str) -> GuardedContactDecision:
        result = GuardedContactDecision(
            action, reason, self._sample_count,
            self._last_timestamp_s, self._last_depth_m)
        if action != "continue_preplanned_stroke":
            self._terminal = result
        return result

    def observe(self, sample: GuardedContactSample, *, decision_time_s: float,
                ) -> GuardedContactDecision:
        if self._terminal is not None:
            return self._terminal
        try:
            if not isinstance(sample, GuardedContactSample):
                raise ValueError("missing sensor sample")
            sample.validate(family=self.family)
            if (type(decision_time_s) not in (float, int) or
                    not math.isfinite(decision_time_s)):
                raise ValueError("invalid decision timestamp")
        except (TypeError, ValueError):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "invalid_or_missing_sensor_sample")
        now = float(decision_time_s)
        stamp = float(sample.timestamp_s)
        if (stamp < self.started_at_s or stamp > now or
                now - stamp > self.limits.max_sample_age_s or
                now - self.started_at_s > self.limits.max_duration_s or
                (self._last_timestamp_s is None and
                 stamp - self.started_at_s >
                 self.limits.max_sample_gap_s) or
                (self._last_timestamp_s is not None and
                 (stamp <= self._last_timestamp_s or
                  stamp - self._last_timestamp_s >
                  self.limits.max_sample_gap_s))):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "stale_or_discontinuous_sensor_time")
        self._sample_count += 1
        previous_depth = self._last_depth_m
        self._last_timestamp_s = stamp
        self._last_depth_m = float(sample.nominal_depth_m)
        if not sample.hand_command_tracked:
            return self._decision("abort_hold_for_supervised_recovery",
                                  "inspire_command_tracking_lost")
        fx, fy, fz = sample.force_socket_n
        mx, my, mz = sample.moment_socket_nm
        if (abs(fz) > self.limits.max_axial_force_n or
                math.hypot(fx, fy) > self.limits.max_lateral_force_n or
                math.sqrt(mx * mx + my * my + mz * mz) >
                self.limits.max_torque_nm):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "wrench_limit_exceeded")
        if (sample.lateral_error_m > self.limits.max_lateral_error_m or
                sample.axis_tilt_deg > self.limits.max_axis_tilt_deg or
                (self.family == "square" and sample.yaw_error_deg is not None
                 and sample.yaw_error_deg > self.limits.max_yaw_error_deg)):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "pose_alignment_limit_exceeded")
        if previous_depth is None and sample.nominal_depth_m > 0:
            return self._decision("abort_hold_for_supervised_recovery",
                                  "missing_preinsert_start_sample")
        if (previous_depth is not None and
                sample.nominal_depth_m < previous_depth -
                self.limits.max_depth_regression_m):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "unexpected_axial_regression")
        if (previous_depth is not None and
                sample.nominal_depth_m - previous_depth >
                self.limits.max_depth_step_m):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "implausible_nominal_depth_jump")
        if sample.nominal_depth_m > (self.limits.target_depth_m +
                                     self.limits.max_depth_overshoot_m):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "nominal_stroke_overshoot")
        if sample.nominal_depth_m >= self.limits.target_depth_m:
            return self._decision("hold_for_key_depth_and_vlm",
                                  "nominal_stroke_complete_key_depth_unverified")
        return self._decision("continue_preplanned_stroke",
                              "within_commissioned_sample_limits")

    def check_sample_deadline(self, *, now_s: float) -> GuardedContactDecision:
        """Detect an in-process missing sample; not a robot-side watchdog."""
        if self._terminal is not None:
            return self._terminal
        if (type(now_s) not in (float, int) or not math.isfinite(now_s) or
                now_s < self.started_at_s or
                (self._last_timestamp_s is not None and
                 now_s < self._last_timestamp_s) or
                now_s - self.started_at_s > self.limits.max_duration_s or
                (now_s - (self._last_timestamp_s if
                          self._last_timestamp_s is not None else
                          self.started_at_s) > self.limits.max_sample_gap_s)):
            return self._decision("abort_hold_for_supervised_recovery",
                                  "sensor_deadline_missed")
        return self._decision("continue_preplanned_stroke",
                              "awaiting_next_preplanned_sample")
