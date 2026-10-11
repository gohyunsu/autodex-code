"""Measured clear-view gate and the ordered post-release return."""

from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.clear_view import ClearViewForPerception  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.repose_release import ReposeReleasePreflight  # noqa: E402


def _fixture():
    actual = {"arm": np.asarray(FR3_CLEAR_VIEW).copy() + .2,
              "raw": np.full(6, 500.), "velocity": np.zeros(7)}
    events = []

    class Executor:
        def home(self, *, clear_view):
            assert clear_view is True
            events.append("initial_home")
            actual["arm"] = np.asarray(FR3_CLEAR_VIEW).copy()
            actual["raw"] = np.full(6, 1000.)

    def read_state():
        now = time.time()
        raw = actual["raw"].copy()
        return LiveRobotState(
            np.r_[actual["arm"], convert_inspire_raw(raw[None])[0]],
            actual["velocity"].copy(), now, now, 10., now,
            raw, raw.copy(), 0., np.zeros(6))

    guard = ClearViewForPerception(Executor(), read_state, .1, 20., .05, 1.)
    return guard, actual, events


def test_initial_home_and_each_key_capture_require_measured_open_clear_view():
    guard, actual, events = _fixture()
    with pytest.raises(ValueError, match="not at clear view"):
        guard.before_key_capture()
    guard.prepare_initial()
    assert events == ["initial_home"]
    assert np.allclose(guard.before_key_capture().full_q[7:], INSPIRE_INIT)
    actual["raw"][:] = 500.
    with pytest.raises(ValueError, match="not open"):
        guard.before_key_capture()
    actual["raw"][:] = 1000.
    actual["arm"][0] += .1
    with pytest.raises(ValueError, match="not at clear view"):
        guard.before_key_capture()
    assert events == ["initial_home"]  # no unsafe free-space home on retry


def test_failed_trial_release_opens_ascends_then_retracts_before_recapture():
    guard, actual, events = _fixture()
    low = np.r_[actual["arm"], INSPIRE_INIT]
    high = low.copy()
    high[0] += .04
    parked = np.r_[FR3_CLEAR_VIEW, INSPIRE_INIT]
    plan = ReposeReleasePreflight(
        "nominal_release_exit_path_pass_drop_unobserved",
        np.vstack([low, high]), np.vstack([high, parked]),
        {"sampled_clear": True}, (), INSPIRE_INIT.copy(),
        np.asarray(FR3_CLEAR_VIEW).copy())

    def open_hand(q):
        assert np.array_equal(q, INSPIRE_INIT)
        events.append("open")
        actual["raw"][:] = 1000.

    def follow(path):
        events.append("ascend" if len(events) == 1 else "retract")
        actual["arm"] = path[-1, :7].copy()

    guard.finish_planned_release_exit(
        release_plan=plan, open_hand=open_hand,
        follow_joint_trajectory=follow)
    assert events == ["open", "ascend", "retract"]
    guard.before_key_capture()


def test_release_exit_rejects_wrong_start_and_unchecked_plan_without_motion():
    guard, actual, events = _fixture()
    actual["arm"] += .2
    low = np.r_[FR3_CLEAR_VIEW + .2, INSPIRE_INIT]
    high = low.copy()
    high[0] += .04
    parked = np.r_[FR3_CLEAR_VIEW, INSPIRE_INIT]
    good = ReposeReleasePreflight(
        "nominal_release_exit_path_pass_drop_unobserved",
        np.vstack([low, high]), np.vstack([high, parked]),
        {"sampled_clear": True}, (), INSPIRE_INIT.copy(),
        np.asarray(FR3_CLEAR_VIEW).copy())
    with pytest.raises(ValueError, match="missed the release-exit waypoint"):
        guard.finish_planned_release_exit(
            release_plan=good, open_hand=lambda _: events.append("open"),
            follow_joint_trajectory=lambda _: events.append("move"))
    assert events == []
    unchecked = ReposeReleasePreflight(
        good.status, good.post_release_lift_trajectory,
        good.retract_trajectory, {"sampled_clear": False}, (),
        good.release_hand_q, good.retreat_goal_arm_q)
    with pytest.raises(ValueError, match="geometry audit"):
        guard.finish_planned_release_exit(
            release_plan=unchecked, open_hand=lambda _: events.append("open"),
            follow_joint_trajectory=lambda _: events.append("move"))
    assert events == []
