"""FR3 runtime construction reuses AutoDex objects and releases them."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import franka_core  # noqa: E402


def _settings():
    return franka_core.FrankaFeedbackSettings(.1, 1., 20., .05)


def _executor(events):
    return SimpleNamespace(
        arm_dof=7, hand_name="inspire", arm=object(), hand=object(),
        home=lambda **_kwargs: events.append("home"),
        set_speed_profile_planner=lambda planner: events.append(
            ("speed_profile_planner", planner)),
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
    assert events == [("speed_profile_planner", planner)]
    assert core.executor is executor and core.planner is planner
    assert core.clear_view.executor is executor
    assert core.read_state() is sample
    assert events[1]["arm"] is executor.arm
    core.close()
    core.close()
    assert events[-1] == "shutdown"
    assert events.count("shutdown") == 1
    assert "home" not in events


def test_franka_core_passes_one_selected_mount_to_planner_and_executor(monkeypatch):
    from autodex import planner as stock_planner
    from precision_insertion import franka_completion_checked

    urdf = (Path(__file__).resolve().parents[3] / "autodex" / "planner" /
            "src" / "curobo" / "content" / "assets" / "robot" /
            "fr3_inspire_description" / "fr3_inspire_oct12_mount.urdf")
    monkeypatch.setenv("PRECISION_FR3_INSPIRE_URDF", str(urdf))
    seen = {}

    def planner_factory(**kwargs):
        seen["planner"] = kwargs
        return SimpleNamespace(_n_arm=7, _hand="fr3_inspire")

    def executor_factory(**kwargs):
        seen["executor"] = kwargs
        return _executor([])

    monkeypatch.setitem(stock_planner.__dict__, "GraspPlanner", planner_factory)
    monkeypatch.setattr(franka_completion_checked,
                        "completion_checked_franka_class",
                        lambda: executor_factory)
    core = franka_core.open_franka_core(_settings())
    assert seen["planner"] == {
        "hand": "fr3_inspire", "robot_urdf_path": str(urdf)}
    assert seen["executor"]["hand_name"] == "inspire"
    assert seen["executor"]["object_safety_speed_scale"] == .5
    from precision_insertion.mount_geometry import link7_to_hand
    import numpy as np

    np.testing.assert_allclose(seen["executor"]["link7_to_hand"],
                               link7_to_hand(urdf))
    core.close()


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


def test_franka_core_owns_motion_tolerant_exposure_sampler(monkeypatch):
    events = []
    executor = _executor(events)
    planner = SimpleNamespace(_n_arm=7, _hand="fr3_inspire")
    buffer_settings = {}

    def buffer_factory(**kwargs):
        buffer_settings.update(kwargs)
        return object()

    class Sampler:
        def __init__(self, buffer, reader, *, period_s):
            assert period_s == .01
            self.reader = reader

        def start(self):
            events.append("sampler_start")

        def wait_until_ready(self, *, timeout_s):
            assert timeout_s == 2.
            events.append("sampler_ready")

        def close(self):
            events.append("sampler_close")

    monkeypatch.setattr(franka_core, "ExposureStateBuffer", buffer_factory)
    monkeypatch.setattr(franka_core, "RobotFeedbackSampler", Sampler)
    monkeypatch.setattr(franka_core, "read_live_franka_inspire_state",
                        lambda **kwargs: kwargs)
    core = franka_core.open_franka_core(
        _settings(), executor_factory=lambda: executor,
        planner_factory=lambda: planner)
    settings = franka_core.FrankaSamplingSettings(
        100, .1, .05, .002, 2., 200., 2., .01, 2.)
    sampler = core.start_feedback_sampler(settings)
    sample = sampler.reader()
    assert sample["max_arm_velocity_rad_s"] == 2.
    assert sample["max_hand_command_error_raw"] == 200.
    assert buffer_settings["max_arm_velocity_rad_s"] == .05
    assert buffer_settings["max_sampling_arm_velocity_rad_s"] == 2.
    core.close()
    assert events == [("speed_profile_planner", planner),
                      "sampler_start", "sampler_ready",
                      "sampler_close", "shutdown"]
