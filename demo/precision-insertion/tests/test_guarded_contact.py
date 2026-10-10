"""A planned 20 mm stroke is monitored, never labeled successful here."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.guarded_contact import (  # noqa: E402
    GuardedContactLimits, GuardedContactMonitor, GuardedContactSample,
)


def _limits():
    return GuardedContactLimits(
        target_depth_m=.020, max_axial_force_n=10.,
        max_lateral_force_n=5., max_torque_nm=.5,
        max_lateral_error_m=.002, max_axis_tilt_deg=4.,
        max_yaw_error_deg=5., max_sample_age_s=.05,
        max_sample_gap_s=.1, max_duration_s=1.,
        max_depth_step_m=.005,
        max_depth_regression_m=.0001,
        max_depth_overshoot_m=.0005,
    )


def _sample(t=1.01, depth=0.):
    return GuardedContactSample(
        timestamp_s=t, nominal_depth_m=depth,
        force_socket_n=(0., 0., 1.),
        moment_socket_nm=(0., 0., .01), lateral_error_m=.0005,
        axis_tilt_deg=1., yaw_error_deg=1.,
        hand_command_tracked=True,
    )


def test_guarded_stroke_reaches_hold_without_task_success_label():
    monitor = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    first = monitor.observe(_sample(), decision_time_s=1.02)
    assert first.action == "continue_preplanned_stroke"
    for index, depth in enumerate((.004, .008, .012, .016), start=1):
        assert monitor.observe(
            _sample(1.01 + index * .02, depth),
            decision_time_s=1.02 + index * .02,
        ).action == "continue_preplanned_stroke"
    second = monitor.observe(_sample(1.11, .020), decision_time_s=1.12)
    assert second.action == "hold_for_key_depth_and_vlm"
    assert second.reason == "nominal_stroke_complete_key_depth_unverified"
    assert "insertion_success" not in second.to_record()
    assert second.to_record()["robot_ready"] is False
    assert monitor.observe(_sample(1.05, .021), decision_time_s=1.06) == second


@pytest.mark.parametrize("changed,reason", [
    ({"force_socket_n": (0., 0., 11.)}, "wrench_limit_exceeded"),
    ({"force_socket_n": (4., 4., 0.)}, "wrench_limit_exceeded"),
    ({"moment_socket_nm": (0., 0., .6)}, "wrench_limit_exceeded"),
    ({"lateral_error_m": .003}, "pose_alignment_limit_exceeded"),
    ({"axis_tilt_deg": 5.}, "pose_alignment_limit_exceeded"),
    ({"yaw_error_deg": 6.}, "pose_alignment_limit_exceeded"),
    ({"hand_command_tracked": False}, "inspire_command_tracking_lost"),
])
def test_guarded_threshold_abort_is_latched(changed, reason):
    monitor = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    aborted = monitor.observe(
        replace(_sample(), **changed), decision_time_s=1.02)
    assert aborted.action == "abort_hold_for_supervised_recovery"
    assert aborted.reason == reason
    assert monitor.observe(_sample(1.03, .02), decision_time_s=1.04) == aborted


def test_guarded_time_and_depth_history_fail_closed():
    monitor = GuardedContactMonitor(
        family="cylinder", limits=replace(_limits(), max_yaw_error_deg=None),
        started_at_s=1.)
    first = monitor.observe(
        replace(_sample(depth=0.), yaw_error_deg=None),
        decision_time_s=1.02)
    assert first.action == "continue_preplanned_stroke"
    monitor.observe(replace(_sample(1.03, .001), yaw_error_deg=None),
                    decision_time_s=1.04)
    reversed_depth = monitor.observe(
        replace(_sample(1.05, 0.), yaw_error_deg=None),
        decision_time_s=1.06)
    assert reversed_depth.reason == "unexpected_axial_regression"

    jumped = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    jumped.observe(_sample(), decision_time_s=1.02)
    assert jumped.observe(_sample(1.03, .01), decision_time_s=1.04).reason == (
        "implausible_nominal_depth_jump")

    stale = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    assert stale.observe(_sample(), decision_time_s=1.2).reason == (
        "stale_or_discontinuous_sensor_time")
    missed = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    assert missed.check_sample_deadline(now_s=1.11).reason == (
        "sensor_deadline_missed")
    assert missed.observe(_sample(), decision_time_s=1.02).reason == (
        "sensor_deadline_missed")


def test_guarded_rejects_nominal_stroke_overshoot_after_valid_start():
    limits = replace(_limits(), max_depth_step_m=.03)
    monitor = GuardedContactMonitor(
        family="square", limits=limits, started_at_s=1.)
    monitor.observe(_sample(), decision_time_s=1.02)
    assert monitor.observe(_sample(1.03, .021), decision_time_s=1.04).reason == (
        "nominal_stroke_overshoot")


def test_guarded_rejects_missing_square_yaw_or_invalid_force():
    with pytest.raises(ValueError, match="yaw-error limit"):
        GuardedContactMonitor(
            family="square", limits=replace(_limits(), max_yaw_error_deg=None),
            started_at_s=1.)
    monitor = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    answer = monitor.observe(
        replace(_sample(), yaw_error_deg=None), decision_time_s=1.02)
    assert answer.reason == "invalid_or_missing_sensor_sample"
    assert answer.action == "abort_hold_for_supervised_recovery"
    late_start = GuardedContactMonitor(
        family="square", limits=_limits(), started_at_s=1.)
    assert late_start.observe(
        _sample(depth=.019), decision_time_s=1.02).reason == (
            "missing_preinsert_start_sample")
