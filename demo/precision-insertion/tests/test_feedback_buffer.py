"""Exposure-time robot feedback must bracket a stationary key capture."""

from pathlib import Path
import sys
import threading

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.feedback_buffer import (  # noqa: E402
    ExposureStateBuffer, RobotFeedbackSampler,
)
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402


def _state(timestamp, *, arm_angle=0., hand_raw=500.):
    raw = np.full(6, hand_raw, dtype=float)
    arm_q = np.zeros(7)
    arm_q[0] = arm_angle
    q = np.concatenate((arm_q, convert_inspire_raw(raw[None, :])[0]))
    return LiveRobotState(
        q, np.zeros(7), timestamp, timestamp, 500., timestamp,
        raw.copy(), raw.copy(), 0., np.zeros(6))


def _observation(lower=100.021, upper=100.029):
    return KeyPoseObservation(
        "key_001", 25, "key", "square", np.eye(4), "cam_a",
        (lower + upper) / 2, (lower, upper), {}, {}, {}, {}, Path("/tmp/key"))


def _buffer(**changes):
    kwargs = {
        "max_samples": 20,
        "max_bracket_span_s": .05,
        "max_key_state_skew_s": .05,
        "max_arm_hold_drift_rad": .002,
        "max_hand_hold_drift_raw": 2.,
        "max_arm_hand_skew_s": .01,
        "max_hand_command_error_raw": 20.,
        "max_arm_velocity_rad_s": .1,
    }
    kwargs.update(changes)
    return ExposureStateBuffer(**kwargs)


def test_stationary_feedback_brackets_all_key_exposure_views():
    buffer = _buffer()
    for timestamp in (100., 100.02, 100.04):
        buffer.append(_state(timestamp))
    selected = buffer.state_for_observation(_observation())
    assert selected.sample_timestamp_s == 100.02
    selected.full_q[0] = 9.
    assert buffer.state_for_observation(_observation()).full_q[0] == 0.


def test_feedback_motion_between_brackets_rejects_preflight():
    buffer = _buffer()
    buffer.append(_state(100.))
    buffer.append(_state(100.02, arm_angle=.01))
    buffer.append(_state(100.04))
    with pytest.raises(ValueError, match="moved during key exposure"):
        buffer.state_for_observation(_observation(100.001, 100.039))


def test_missing_or_wide_bracket_rejects_and_input_arrays_are_copied():
    buffer = _buffer()
    original = _state(100.)
    buffer.append(original)
    original.full_q[0] = 10.
    with pytest.raises(ValueError, match="does not bracket"):
        buffer.state_for_observation(_observation())
    buffer.append(_state(100.10))
    with pytest.raises(ValueError, match="bracket is too wide"):
        buffer.state_for_observation(_observation())


def test_nonmonotone_feedback_and_reader_failure_reject():
    buffer = _buffer()
    buffer.append(_state(100.))
    with pytest.raises(ValueError, match="did not advance"):
        buffer.append(_state(100.))

    attempted = threading.Event()

    def failed_reader():
        attempted.set()
        raise OSError("controller stream disconnected")

    sampler = RobotFeedbackSampler(buffer, failed_reader, period_s=.001)
    sampler.start()
    assert attempted.wait(timeout=1.)
    sampler._thread.join(timeout=1.)
    with pytest.raises(RuntimeError, match="sampling failed"):
        sampler.wait_until_ready(timeout_s=1.)
    with pytest.raises(RuntimeError, match="sampling failed"):
        sampler.state_for_observation(_observation())
    sampler.close()


def test_sampler_warms_up_before_camera_and_rejects_reuse_after_close():
    buffer = _buffer()
    with RobotFeedbackSampler(
            buffer, lambda: _state(100.), period_s=10.) as sampler:
        sampler.wait_until_ready(timeout_s=1.)
    with pytest.raises(RuntimeError, match="not running"):
        sampler.state_for_observation(_observation())
