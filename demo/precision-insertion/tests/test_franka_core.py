"""FR3 runtime construction reuses AutoDex objects and releases them."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import franka_core  # noqa: E402


def _settings():
    return franka_core.FrankaFeedbackSettings(.1, 1., 20., .2, .05)


def _executor(events):
    return SimpleNamespace(
        arm_dof=7, hand_name="inspire", arm=object(), hand=object(),
        home=lambda **_kwargs: events.append("home"),
        shutdown=lambda: events.append("shutdown"))


def test_franka_core_uses_stock_fr3_objects_without_homing(monkeypatch):
    events = []
    executor = _executor(events)
    planner = SimpleNamespace(_n_arm=7, _hand="fr3_inspire")
    sample = object()
    monkeypatch.setattr(franka_core, "read_live_franka_inspire_state",
                        lambda **kwargs: (events.append(kwargs) or sample))
    core = franka_core.open_franka_core(
        _settings(), executor_factory=lambda: executor,
        planner_factory=lambda: planner)
    assert events == []
    assert core.executor is executor and core.planner is planner
    assert core.clear_view.executor is executor
    assert core.read_state() is sample
    assert events[0]["arm"] is executor.arm
    core.close()
    core.close()
    assert events[-1] == "shutdown"
    assert events.count("shutdown") == 1
    assert "home" not in events


def test_franka_core_rejects_xarm_and_releases_connection():
    events = []
    executor = _executor(events)
    executor.arm_dof = 6
    with pytest.raises(TypeError, match="Franka/Inspire"):
        franka_core.open_franka_core(
            _settings(), executor_factory=lambda: executor,
            planner_factory=lambda: SimpleNamespace(
                _n_arm=7, _hand="fr3_inspire"))
    assert events == ["shutdown"]


def test_franka_core_checks_cuda_planner_before_opening_franka():
    events = []

    def broken_planner():
        raise RuntimeError("cuRobo unavailable")

    with pytest.raises(RuntimeError, match="cuRobo unavailable"):
        franka_core.open_franka_core(
            _settings(), executor_factory=lambda: (
                events.append("connect") or _executor(events)),
            planner_factory=broken_planner)
    assert events == []
