"""Measured clear-view gate and the ordered post-release return."""

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.clear_view import ClearViewForPerception  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.repose_preflight import ReposeHeldPreflight  # noqa: E402
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


def test_missed_grasp_uses_stock_release_reset_with_original_key_obstacle():
    guard, actual, events = _fixture()
    plan = SimpleNamespace(success=True)
    planner = object()
    scene = {"mesh": {
        "target": {"file_path": "/key.obj", "pose": [0, 0, 0]},
        "fixture_socket": {"file_path": "/socket.obj"},
    }}

    def release(received, *, open_to_init):
        assert received is plan and open_to_init is True
        events.append("AutoDex.release")
        actual["raw"][:] = 1000.

    def reset(received, received_planner, world):
        assert received is plan and received_planner is planner
        assert world["mesh"]["fixture_socket"] == scene["mesh"]["fixture_socket"]
        assert world["mesh"]["failed_grasp_original_key"] == scene["mesh"]["target"]
        world["mesh"]["target"]["pose"][0] = 9
        events.append("AutoDex.reset")
        actual["arm"] = np.asarray(FR3_CLEAR_VIEW).copy()

    guard.executor.release = release
    guard.executor.reset = reset
    guard.prepare_failed_grasp_for_capture(
        pickup_plan=plan, planner=planner, trial_scene=scene)
    assert events == ["AutoDex.release", "AutoDex.reset"]
    assert scene["mesh"]["target"]["pose"] == [0, 0, 0]
    guard.before_key_capture()


def test_missed_grasp_reset_failure_cannot_start_key_capture():
    guard, _, events = _fixture()
    guard.executor.release = lambda *_args, **_kwargs: events.append("release")

    def reset(*_args):
        raise RuntimeError("AutoDex reset preflight rejected")

    guard.executor.reset = reset
    with pytest.raises(RuntimeError, match="preflight rejected"):
        guard.prepare_failed_grasp_for_capture(
            pickup_plan=SimpleNamespace(success=True), planner=object(),
            trial_scene={"mesh": {"target": {}, "fixture_socket": {}}})
    assert events == ["release"]
    with pytest.raises(ValueError, match="not at clear view"):
        guard.before_key_capture()


def test_planned_release_uses_stock_executor_hand_trajectory():
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

    def release(received, *, open_to_init):
        assert received is None and open_to_init is True
        events.append("AutoDex.release")
        actual["raw"][:] = 1000.

    def follow(path, *, hand_traj):
        assert path.shape == (2, 13)
        assert hand_traj.shape == (2, 6)
        assert np.all(hand_traj == 1000.)
        events.append("AutoDex.follow")
        actual["arm"] = path[-1, :7].copy()

    guard.executor.release = release
    guard.executor._convert = lambda _q: np.full(6, 1000.)
    guard.executor.follow_joint_trajectory = follow
    guard.finish_planned_release_exit_with_autodex(release_plan=plan)
    assert events == ["AutoDex.release", "AutoDex.follow", "AutoDex.follow"]
    guard.before_key_capture()


def test_failed_insertion_return_uses_stock_held_motion_then_release():
    guard, actual, events = _fixture()
    start = guard.read_state().full_q.copy()
    mid = start.copy()
    mid[0] += .04
    low = mid.copy()
    low[0] += .04
    open_low = low.copy()
    open_low[7:] = INSPIRE_INIT
    high = open_low.copy()
    high[0] += .04
    parked = np.r_[FR3_CLEAR_VIEW, INSPIRE_INIT]
    held = ReposeHeldPreflight(
        "sampled_held_path_pass_release_unplanned",
        np.vstack([start, start]), np.vstack([start, mid]),
        np.vstack([mid, low]), {"sampled_clear": True}, (), np.eye(4),
        .04, start[7:].copy(), "measured")
    release_plan = ReposeReleasePreflight(
        "nominal_release_exit_path_pass_drop_unobserved",
        np.vstack([open_low, high]), np.vstack([high, parked]),
        {"sampled_clear": True}, (), INSPIRE_INIT.copy(),
        np.asarray(FR3_CLEAR_VIEW).copy())

    def follow(path, *, hand_traj):
        events.append("held" if hand_traj[-1, 0] == 500 else "open_exit")
        actual["arm"] = path[-1, :7].copy()
        actual["raw"] = hand_traj[-1].copy()

    def release(_plan, *, open_to_init):
        assert open_to_init
        events.append("release")
        actual["raw"][:] = 1000.

    guard.executor._convert = lambda q: (
        np.full(6, 1000.) if np.allclose(q, INSPIRE_INIT)
        else np.full(6, 500.))
    guard.executor.follow_joint_trajectory = follow
    guard.executor.release = release
    guard.complete_failed_insertion_return_with_autodex(
        held_plan=held, release_plan=release_plan)
    assert events == ["held", "held", "release", "open_exit", "open_exit"]
    guard.before_key_capture()
