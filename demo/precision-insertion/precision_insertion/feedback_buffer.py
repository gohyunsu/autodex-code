"""Read-only Franka/Inspire feedback bracketing camera exposure.

This does not infer a hardware-latched robot timestamp. The measured arm
receipt and hand read times must be independently commissioned against camera
UTC time. Bracketing two stationary feedback samples is stronger than using
one post-inference read, but is not a contact-control safety certificate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import threading
from typing import Callable

import numpy as np

from .key_perception import KeyPoseObservation
from .live_robot_state import LiveRobotState


def _copy_state(state: LiveRobotState) -> LiveRobotState:
    return replace(
        state, full_q=np.asarray(state.full_q).copy(),
        arm_qvel=np.asarray(state.arm_qvel).copy(),
        hand_raw_measured=np.asarray(state.hand_raw_measured).copy(),
        hand_raw_commanded=np.asarray(state.hand_raw_commanded).copy(),
        wrench=np.asarray(state.wrench).copy())


class ExposureStateBuffer:
    """Retain validated measured states through slow SAM/FoundPose inference.

    Both ends of the accepted multi-camera acquisition interval must be
    bracketed by robot feedback. Every stored state between those samples
    must remain within arm and hand hold-drift limits. The nearest measured
    state is returned; no interpolation is presented as measured feedback.
    """

    def __init__(
        self, *, max_samples: int, max_bracket_span_s: float,
        max_key_state_skew_s: float, max_arm_hold_drift_rad: float,
        max_hand_hold_drift_raw: float, max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
    ) -> None:
        if type(max_samples) is not int or max_samples < 2:
            raise ValueError("feedback buffer needs at least two samples")
        for name, value in (
                ("max_bracket_span_s", max_bracket_span_s),
                ("max_key_state_skew_s", max_key_state_skew_s),
                ("max_arm_hold_drift_rad", max_arm_hold_drift_rad),
                ("max_hand_hold_drift_raw", max_hand_hold_drift_raw),
                ("max_arm_hand_skew_s", max_arm_hand_skew_s),
                ("max_hand_command_error_raw", max_hand_command_error_raw),
                ("max_arm_velocity_rad_s", max_arm_velocity_rad_s)):
            if (type(value) not in (int, float) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.max_bracket_span_s = float(max_bracket_span_s)
        self.max_key_state_skew_s = float(max_key_state_skew_s)
        self.max_arm_hold_drift_rad = float(max_arm_hold_drift_rad)
        self.max_hand_hold_drift_raw = float(max_hand_hold_drift_raw)
        self.max_arm_hand_skew_s = float(max_arm_hand_skew_s)
        self.max_hand_command_error_raw = float(max_hand_command_error_raw)
        self.max_arm_velocity_rad_s = float(max_arm_velocity_rad_s)
        self._samples: deque[LiveRobotState] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def append(self, state: LiveRobotState) -> None:
        if not isinstance(state, LiveRobotState):
            raise TypeError("feedback buffer accepts measured LiveRobotState only")
        state.validate(
            max_arm_hand_skew_s=self.max_arm_hand_skew_s,
            max_hand_command_error_raw=self.max_hand_command_error_raw,
            max_arm_velocity_rad_s=self.max_arm_velocity_rad_s)
        frozen = _copy_state(state)
        with self._lock:
            if (self._samples and frozen.sample_timestamp_s <=
                    self._samples[-1].sample_timestamp_s):
                raise ValueError("robot feedback timestamps did not advance")
            self._samples.append(frozen)

    def state_for_observation(
        self, observation: KeyPoseObservation,
    ) -> LiveRobotState:
        if not isinstance(observation, KeyPoseObservation):
            raise TypeError("feedback lookup needs an admitted key observation")
        lower, upper = observation.acquisition_interval_s
        if (not math.isfinite(lower) or not math.isfinite(upper) or
                lower <= 0 or upper < lower):
            raise ValueError("key acquisition interval is invalid")
        with self._lock:
            samples = tuple(self._samples)
        before = [index for index, row in enumerate(samples)
                  if row.sample_timestamp_s <= lower]
        after = [index for index, row in enumerate(samples)
                 if row.sample_timestamp_s >= upper]
        if not before or not after:
            raise ValueError("robot feedback does not bracket key exposure")
        first, last = before[-1], after[0]
        segment = samples[first:last + 1]
        span = segment[-1].sample_timestamp_s - segment[0].sample_timestamp_s
        if span > self.max_bracket_span_s:
            raise ValueError("robot feedback bracket is too wide")
        reference = segment[0]
        for row in segment:
            if (np.max(np.abs(row.full_q[:7] - reference.full_q[:7])) >
                    self.max_arm_hold_drift_rad or
                    np.max(np.abs(row.hand_raw_measured -
                                  reference.hand_raw_measured)) >
                    self.max_hand_hold_drift_raw or
                    np.max(np.abs(row.hand_raw_commanded -
                                  reference.hand_raw_commanded)) >
                    self.max_hand_hold_drift_raw):
                raise ValueError("Franka/Inspire moved during key exposure")
        midpoint = (lower + upper) / 2
        selected = min(segment, key=lambda row: abs(
            row.sample_timestamp_s - midpoint))
        observation.require_state_alignment(
            state_timestamp_s=selected.sample_timestamp_s,
            maximum_skew_s=self.max_key_state_skew_s)
        return _copy_state(selected)


class RobotFeedbackSampler:
    """Background read-only sampling into an ``ExposureStateBuffer``.

    Start it *before* triggering any camera capture. ``sample_state`` may
    call ``read_live_franka_inspire_state(arm=..., hand=..., ...)``; the
    sampler never creates controllers or sends arm/hand commands. A read or
    validation error stops sampling and makes later lookups fail closed.
    """

    def __init__(
        self, buffer: ExposureStateBuffer,
        sample_state: Callable[[], LiveRobotState], *, period_s: float,
    ) -> None:
        if not isinstance(buffer, ExposureStateBuffer) or not callable(
                sample_state):
            raise TypeError("sampler needs a buffer and measured-state reader")
        if (type(period_s) not in (int, float) or
                not math.isfinite(period_s) or period_s <= 0):
            raise ValueError("feedback sampling period must be positive")
        self.buffer = buffer
        self.sample_state = sample_state
        self.period_s = float(period_s)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("feedback sampler already started")
        self._thread = threading.Thread(
            target=self._run, name="precision_feedback", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.buffer.append(self.sample_state())
                self._ready.set()
            except Exception as exc:
                self._failure = exc
                self._stop.set()
                self._ready.set()
                return
            self._stop.wait(self.period_s)

    def wait_until_ready(self, *, timeout_s: float) -> None:
        """Do not trigger a key capture until feedback is actually flowing."""
        if self._thread is None:
            raise RuntimeError("feedback sampler was not started")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("feedback warm-up timeout must be positive")
        if not self._ready.wait(timeout_s):
            raise TimeoutError("no measured feedback arrived before key capture")
        if self._failure is not None:
            raise RuntimeError("feedback sampling failed") from self._failure
        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError("feedback sampler is not running")

    def state_for_observation(
        self, observation: KeyPoseObservation,
    ) -> LiveRobotState:
        if self._thread is None:
            raise RuntimeError("feedback sampler was not started")
        if self._failure is not None:
            raise RuntimeError("feedback sampling failed") from self._failure
        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError("feedback sampler is not running")
        state = self.buffer.state_for_observation(observation)
        if self._failure is not None:
            raise RuntimeError("feedback sampling failed") from self._failure
        return state

    def close(self, *, timeout_s: float = 5.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("sampler shutdown timeout must be positive")
        self._stop.set()
        self._ready.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("feedback reader did not stop")

    def __enter__(self) -> RobotFeedbackSampler:
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
