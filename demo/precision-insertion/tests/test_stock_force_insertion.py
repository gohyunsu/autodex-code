"""Offline checks for stock force stop and reverse-axis recovery."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import stock_force_insertion as stock  # noqa: E402


def test_reverse_uses_only_visited_axis_segment():
    path = np.zeros((6, 7))
    path[:, 0] = np.linspace(0., .1, 6)
    path[:, 1] = np.linspace(0., .2, 6)
    measured = path[2] * .7 + path[3] * .3
    reverse = stock.reverse_executed_path(path, measured)
    np.testing.assert_allclose(reverse[0], measured)
    np.testing.assert_allclose(reverse[-1], path[0])
    assert not np.any(np.isclose(reverse[:, 0], path[-1, 0]))
    with pytest.raises(RuntimeError, match="off the planned"):
        stock.reverse_executed_path(path, measured + np.array(
            [0., 0., .1, 0., 0., 0., 0.]))


def test_stock_force_stop_ascends_without_speed_override(monkeypatch):
    calls = []
    arm_state = {"qpos": np.zeros(7), "wrench": np.array([0., 0., 13., 0., 0., 0.])}
    executor = SimpleNamespace(arm=SimpleNamespace(get_data=lambda: arm_state),
                               _last_stop_on_contact=False)

    def follow(path, **kwargs):
        calls.append((np.asarray(path).copy(), kwargs))
        if len(calls) == 1:
            executor._last_stop_on_contact = True
            arm_state["qpos"] = np.asarray(path)[2].copy()
        else:
            executor._last_stop_on_contact = False
            arm_state["qpos"] = np.asarray(path)[-1].copy()

    executor._follow = follow
    executor.stop_and_acknowledge = lambda: pytest.fail(
        "completion-checked _follow already acknowledged the stop")
    monkeypatch.setattr(stock, "is_demo_completion_checked_executor", lambda _: True)
    adapter = stock.FrankaForceStopInsertionAdapter(executor)
    axial = np.zeros((5, 13))
    axial[:, 0] = np.linspace(0., .08, 5)
    result = adapter.follow_stock_force_insertion(axial)
    assert result["safety_abort"] is True
    assert result["retreat_completed"] is True
    assert len(calls) == 2
    assert calls[0][1]["stop_wrench_z"] == 12.
    assert all("speed" not in kwargs for _, kwargs in calls)
    np.testing.assert_allclose(calls[1][0][0], axial[2, :7])
    np.testing.assert_allclose(calls[1][0][-1], axial[0, :7])


def test_stock_normal_completion_does_not_withdraw(monkeypatch):
    calls = []
    executor = SimpleNamespace(_last_stop_on_contact=False)
    executor._follow = lambda path, **kwargs: calls.append((path, kwargs))
    executor.stop_and_acknowledge = lambda: pytest.fail(
        "completion-checked _follow already acknowledged the stop")
    monkeypatch.setattr(stock, "is_demo_completion_checked_executor", lambda _: True)
    axial = np.zeros((3, 13))
    axial[:, 0] = [0., .01, .02]
    result = stock.FrankaForceStopInsertionAdapter(
        executor).follow_stock_force_insertion(axial)
    assert result["safety_abort"] is False
    assert result["retreat_completed"] is False
    assert len(calls) == 1
