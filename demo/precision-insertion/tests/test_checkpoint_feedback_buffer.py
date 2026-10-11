"""Existing feedback sampler also serves raw VLM camera checkpoints."""

from pathlib import Path
import sys
import time

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.feedback_buffer import ExposureStateBuffer  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402


def _state(stamp, *, arm_angle=0.):
    raw = np.full(6, 500., dtype=float)
    arm = np.zeros(7)
    arm[0] = arm_angle
    return LiveRobotState(
        np.concatenate([arm, convert_inspire_raw(raw[None, :])[0]]),
        np.zeros(7), stamp, stamp, 500., stamp,
        raw.copy(), raw.copy(), 0., np.zeros(6))


def _buffer():
    return ExposureStateBuffer(
        max_samples=20, max_bracket_span_s=.05,
        max_key_state_skew_s=.05, max_arm_hold_drift_rad=.002,
        max_hand_hold_drift_raw=2., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=20., max_arm_velocity_rad_s=.1)


def _capture(center):
    names = ("cam_a", "cam_b")
    return RawCameraCapture(
        "held_1", 7,
        {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in names},
        {name: index for index, name in enumerate(names, 1)},
        {"request_id": 7, "frames": {
            "cam_a": {"timestamp_s": center - .002, "max_error_s": .001},
            "cam_b": {"timestamp_s": center + .002, "max_error_s": .001}}})


def test_raw_checkpoint_is_bracketed_by_existing_feedback():
    now = time.time()
    buffer = _buffer()
    for stamp in (now - .02, now, now + .02):
        buffer.append(_state(stamp))
    result = buffer.state_for_capture(_capture(now))
    assert abs(result.sample_timestamp_s - now) < 1e-5
    result.full_q[0] = 9.
    assert buffer.state_for_capture(_capture(now)).full_q[0] == 0.
    assert buffer.latest(max_age_s=.1).sample_timestamp_s == now + .02


def test_raw_checkpoint_rejects_missing_bracket_and_motion():
    now = time.time()
    buffer = _buffer()
    buffer.append(_state(now - .02))
    with pytest.raises(ValueError, match="does not bracket"):
        buffer.state_for_capture(_capture(now))
    buffer.append(_state(now, arm_angle=.01))
    buffer.append(_state(now + .02))
    with pytest.raises(ValueError, match="moved during raw exposure"):
        buffer.state_for_capture(_capture(now))
