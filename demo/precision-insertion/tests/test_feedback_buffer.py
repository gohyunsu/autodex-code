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
from precision_insertion.live_robot_state import (  # noqa: E402
    FeedbackTimeAlignmentUnavailable, FrankaStateUnavailable,
    LiveRobotState,
)
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402


def _state(timestamp, *, arm_angle=0., hand_raw=500., arm_velocity=0.,
           hand_command_error=0.):
    raw = np.full(6, hand_raw, dtype=float)
    arm_q = np.zeros(7)
    arm_q[0] = arm_angle
    q = np.concatenate((arm_q, convert_inspire_raw(raw[None, :])[0]))
    qvel = np.zeros(7)
    qvel[0] = arm_velocity
    commanded = raw.copy() + hand_command_error
    return LiveRobotState(
        q, qvel, timestamp, timestamp, 500., timestamp,
        raw.copy(), commanded, abs(hand_command_error), np.zeros(6))


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


def test_session_long_buffer_records_motion_but_admits_only_later_hold():
    buffer = _buffer(
        max_sampling_arm_velocity_rad_s=2.,
        max_sampling_hand_command_error_raw=200.)
    buffer.append(_state(100.))
    buffer.append(_state(
        100.02, arm_angle=.01, arm_velocity=.5,
        hand_command_error=80.))
    buffer.append(_state(100.04, arm_angle=.02))
    with pytest.raises(ValueError, match="moved during key exposure"):
        buffer.state_for_observation(_observation(100.001, 100.039))
    buffer.append(_state(100.06, arm_angle=.02))
    buffer.append(_state(100.08, arm_angle=.02))
    selected = buffer.state_for_observation(_observation(100.061, 100.079))
    assert selected.sample_timestamp_s == 100.06
    assert selected.full_q[0] == .02


def test_raw_vlm_capture_rejects_motion_even_if_arm_returns_to_same_pose():
    buffer = _buffer(max_sampling_arm_velocity_rad_s=2.)
    buffer.append(_state(100.))
    buffer.append(_state(100.02, arm_velocity=.5))
    buffer.append(_state(100.04))
    capture = RawCameraCapture(
        "raw_1", 9,
        {camera: np.zeros((2, 2, 3), dtype=np.uint8)
         for camera in ("cam_a", "cam_b")},
        {"cam_a": 1, "cam_b": 1},
        {"request_id": 9, "frames": {
            "cam_a": {"timestamp_s": 100.021, "max_error_s": 0.},
            "cam_b": {"timestamp_s": 100.029, "max_error_s": 0.}}})
    with pytest.raises(ValueError, match="moved during raw exposure"):
        buffer.state_for_capture(capture)


def test_missing_bracket_rejects_but_long_stationary_request_is_allowed():
    buffer = _buffer()
    original = _state(100.)
    buffer.append(original)
    original.full_q[0] = 10.
    with pytest.raises(ValueError, match="does not bracket"):
        buffer.state_for_observation(_observation())
    buffer.append(_state(100.10))
    assert buffer.state_for_observation(_observation()).full_q[0] == 0.


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


def test_sampler_recovers_after_a_transient_fr3_update_timeout():
    buffer = _buffer()
    resumed = threading.Event()
    append = buffer.append

    def record_append(state):
        append(state)
        if state.sample_timestamp_s >= 100.04:
            resumed.set()

    buffer.append = record_append
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FrankaStateUnavailable("FR3 state stream is stale")
        return _state(100. + (calls - (calls > 2)) * .02)

    with RobotFeedbackSampler(buffer, reader, period_s=.001) as sampler:
        sampler.wait_until_ready(timeout_s=1.)
        assert resumed.wait(timeout=1.)
        selected = sampler.state_for_observation(_observation())
        assert selected.sample_timestamp_s == 100.02


def test_sampler_recovers_after_one_unaligned_arm_hand_pair():
    buffer = _buffer()
    resumed = threading.Event()
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FeedbackTimeAlignmentUnavailable(
                "FR3/Inspire feedback is stale or not time-aligned")
        if calls == 4:
            resumed.set()
        return _state(100. + (calls - (calls > 2)) * .02)

    with RobotFeedbackSampler(buffer, reader, period_s=.001) as sampler:
        sampler.wait_until_ready(timeout_s=1.)
        assert resumed.wait(timeout=1.)
        assert sampler.state_for_observation(_observation()).sample_timestamp_s == 100.02


def test_held_object_command_gap_does_not_reject_stationary_exposure():
    buffer = _buffer()
    for timestamp in (100., 100.02, 100.04):
        buffer.append(_state(timestamp, hand_command_error=80.))
    selected = buffer.state_for_observation(_observation())
    assert selected.max_hand_command_error_raw == 80.
    assert selected.sample_timestamp_s == 100.02


def test_sampler_never_admits_capture_without_a_live_fr3_update():
    def stalled_reader():
        raise FrankaStateUnavailable("FR3 state stream is stale")

    with RobotFeedbackSampler(_buffer(), stalled_reader, period_s=.001) as sampler:
        with pytest.raises(TimeoutError, match="no measured feedback"):
            sampler.wait_until_ready(timeout_s=.02)
        with pytest.raises(ValueError, match="does not bracket"):
            sampler.state_for_observation(_observation())


def test_sampler_never_admits_capture_with_only_unaligned_feedback():
    def unaligned_reader():
        raise FeedbackTimeAlignmentUnavailable(
            "FR3/Inspire feedback is stale or not time-aligned")

    with RobotFeedbackSampler(_buffer(), unaligned_reader,
                            period_s=.001) as sampler:
        with pytest.raises(TimeoutError, match="no measured feedback"):
            sampler.wait_until_ready(timeout_s=.02)
        with pytest.raises(ValueError, match="does not bracket"):
            sampler.state_for_observation(_observation())


def test_sampler_warms_up_before_camera_and_rejects_reuse_after_close():
    buffer = _buffer()
    with RobotFeedbackSampler(
            buffer, lambda: _state(100.), period_s=10.) as sampler:
        sampler.wait_until_ready(timeout_s=1.)
    with pytest.raises(RuntimeError, match="not running"):
        sampler.state_for_observation(_observation())
