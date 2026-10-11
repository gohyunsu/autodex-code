"""The built-in robot-PC factory owns FR3 resources and reuses stage wiring."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import franka_integration  # noqa: E402
from precision_insertion.live_runtime import RunLiveOptions  # noqa: E402


def _options(tmp_path, config):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", enable_robot_motion=True,
        robot_config_path=path)


def _config():
    return {
        "schema": "precision_insertion_franka_integration_v1",
        "feedback": {
            "max_arm_hand_skew_s": .1, "max_sample_age_s": 1.,
            "max_hand_command_error_raw": 20.,
            "max_arm_update_wait_s": .2,
            "max_arm_velocity_rad_s": .05,
        },
        "stages_factory": "robot_pc_stages:configure",
    }


def test_franka_integration_reuses_core_and_existing_stage_assembly(
        monkeypatch, tmp_path):
    events = []
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: events.append("close"))
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda feedback: (events.append(feedback) or core))
    module = ModuleType("robot_pc_stages")

    def configure(received_core, live, options):
        assert received_core is core and live is session
        assert options.robot_config_path.is_file()
        return {"shared": {"backend": backend}, "stages": {"task": {}},
                "held_relation_route": "observed"}

    module.configure = configure
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = object()
    session = object()

    def assemble(live, **kwargs):
        assert live is session
        assert kwargs["shared"] == {
            "backend": backend, "planner": core.planner,
            "read_state": core.read_state}
        assert set(kwargs["stages"]) == {"task"}
        assert set(kwargs["stages"]["task"]) == {
            "capture_preinsert", "capture_final"}
        assert kwargs["held_relation_route"] == "observed"
        assert kwargs["enable_robot_motion"] is True
        return ("assembled",)

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    integration = franka_integration.factory(_options(tmp_path, _config()))
    assert integration.clear_view is core.clear_view
    assert integration.build_components(session) == ("assembled",)
    integration.close()
    assert events[-1] == "close"


@pytest.mark.parametrize("override", [False, True])
def test_franka_integration_reuses_checked_core_for_pickup_unless_overridden(
        monkeypatch, tmp_path, override):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    external = object()
    module.configure = lambda *_: {
        "shared": {"backend": object()},
        "stages": {"pickup": {"executor": external} if override else {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def assemble(_live, **kwargs):
        assert kwargs["stages"]["pickup"]["executor"] is (
            external if override else core.executor)
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    integration = franka_integration.factory(_options(tmp_path, _config()))
    integration.build_components(object())


def test_franka_integration_builds_measured_lift_adapter_when_configured(
        monkeypatch, tmp_path):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {
        "shared": {"backend": object()}, "stages": {"lift": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    received = []
    adapter = object()

    def build_adapter(**kwargs):
        received.append(kwargs)
        return adapter

    monkeypatch.setattr(franka_integration, "FrankaMeasuredLiftAdapter",
                        build_adapter)

    def assemble(_live, **kwargs):
        assert kwargs["stages"]["lift"]["adapter"] is adapter
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    binary = tmp_path / "franka_daemon"
    config["lift_monitor"] = {
        "daemon_binary_path": str(binary), "max_force_norm_n": 20.,
        "max_torque_norm_nm": 3., "max_hand_tracking_error_raw": 25.,
        "max_state_gap_s": .05, "minimum_samples": 3}
    options = _options(tmp_path, config)
    franka_integration.factory(options).build_components(object())
    assert len(received) == 1
    assert received[0]["executor"] is core.executor
    assert received[0]["daemon_binary_path"] == binary
    assert received[0]["trace_root"] == options.run_output_dir / "controller_traces"
    assert received[0]["limits"].minimum_samples == 3


@pytest.mark.parametrize("family", ["square", "cylinder"])
def test_franka_integration_reuses_held_motion_adapter_across_stages(
        monkeypatch, tmp_path, family):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    custom = object()
    module.configure = lambda *_: {
        "shared": {"backend": object()},
        "stages": {"lift": {}, "task": {}, "retry": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    created = []

    def build_adapter(**kwargs):
        created.append(kwargs)
        return held_adapter

    held_adapter = object()
    monkeypatch.setattr(franka_integration, "FrankaMeasuredLiftAdapter",
                        build_adapter)

    def assemble(_live, **kwargs):
        stages = kwargs["stages"]
        assert stages["lift"]["adapter"] is held_adapter
        assert stages["task"]["transfer_adapter"] is held_adapter
        assert "transfer_adapter" not in stages["retry"]
        assert "lateral_adapter" not in stages["retry"]
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    config["held_motion_monitor"] = {
        "daemon_binary_path": str(tmp_path / "franka_daemon"),
        "max_force_norm_n": 20., "max_torque_norm_nm": 3.,
        "max_hand_tracking_error_raw": 25., "max_state_gap_s": .05,
        "minimum_samples": 3}
    live = SimpleNamespace(opened=SimpleNamespace(
        runner=SimpleNamespace(mode=SimpleNamespace(family=family))))
    franka_integration.factory(_options(tmp_path, config)).build_components(live)
    assert len(created) == 1


def test_duplicate_motion_monitor_config_fails_before_robot_connection(
        monkeypatch, tmp_path):
    connected = []
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: connected.append(True))
    monitor = {
        "daemon_binary_path": str(tmp_path / "franka_daemon"),
        "max_force_norm_n": 20., "max_torque_norm_nm": 3.,
        "max_hand_tracking_error_raw": 25., "max_state_gap_s": .05,
        "minimum_samples": 3}
    config = _config()
    config["lift_monitor"] = monitor
    config["held_motion_monitor"] = monitor
    with pytest.raises(ValueError, match="not both"):
        franka_integration.factory(_options(tmp_path, config))
    assert connected == []


def test_invalid_lift_monitor_fails_before_robot_connection(monkeypatch, tmp_path):
    connected = []
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: connected.append(True))
    config = _config()
    config["lift_monitor"] = {
        "daemon_binary_path": "relative/franka_daemon",
        "max_force_norm_n": 20., "max_torque_norm_nm": 3.,
        "max_hand_tracking_error_raw": 25., "max_state_gap_s": .05,
        "minimum_samples": 3}
    with pytest.raises(ValueError, match="absolute daemon path"):
        franka_integration.factory(_options(tmp_path, config))
    assert connected == []


def test_invalid_stage_factory_fails_before_robot_connection(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: events.append("connect"))
    config = _config()
    config["stages_factory"] = "not a module"
    with pytest.raises(ValueError, match="stage factory"):
        franka_integration.factory(_options(tmp_path, config))
    assert events == []


def test_invalid_feedback_fails_before_robot_connection(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: events.append("connect"))
    config = _config()
    config["feedback"]["max_arm_velocity_rad_s"] = -1
    with pytest.raises(ValueError, match="finite and positive"):
        franka_integration.factory(_options(tmp_path, config))
    assert events == []


def test_stage_factory_cannot_replace_fr3_planner_or_feedback(
        monkeypatch, tmp_path):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {
        "shared": {"planner": object()}, "stages": {}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    integration = franka_integration.factory(_options(tmp_path, _config()))
    with pytest.raises(ValueError, match="belong to core"):
        integration.build_components(object())


def test_franka_integration_starts_and_injects_exposure_sampler(
        monkeypatch, tmp_path):
    events = []
    sampler = SimpleNamespace(
        state_for_observation=lambda _: None,
        state_for_capture=lambda _: None)
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: events.append("close"))

    def start(settings):
        assert settings.max_sampling_arm_velocity_rad_s == 2.
        core.sampler = sampler
        events.append("sampler_start")

    core.start_feedback_sampler = start
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    backend = object()
    monkeypatch.setattr(franka_integration, "load_vlm_backend",
                        lambda **kwargs: backend)
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {"shared": {}, "stages": {}}
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def assemble(_live, **kwargs):
        assert kwargs["shared"]["state_for_observation"] is (
            sampler.state_for_observation)
        assert kwargs["shared"]["state_for_capture"] is (
            sampler.state_for_capture)
        assert kwargs["shared"]["backend"] is backend
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    config["sampling"] = {
        "max_samples": 100, "max_bracket_span_s": .1,
        "max_key_state_skew_s": .05, "max_arm_hold_drift_rad": .002,
        "max_hand_hold_drift_raw": 2.,
        "max_sampling_hand_command_error_raw": 200.,
        "max_sampling_arm_velocity_rad_s": 2.,
        "period_s": .01, "warmup_timeout_s": 2.,
    }
    integration = franka_integration.factory(_options(tmp_path, config))
    integration.build_components(SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(image_hw=(1080, 1920)))))
    integration.close()
    assert events == ["sampler_start", "close"]


def test_first_party_qwen_and_fresh_checkpoint_callbacks(monkeypatch, tmp_path):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    custom_capture = lambda *_: None
    module.configure = lambda *_: {
        "shared": {},
        "stages": {"lift": {}, "task": {"capture_final": custom_capture},
                   "retry": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    loaded = []
    backend = object()

    def load(**kwargs):
        loaded.append(kwargs)
        return backend

    monkeypatch.setattr(franka_integration, "load_vlm_backend", load)
    captures = []

    def make(*, phase, timeout_s):
        captures.append((phase, timeout_s))
        return lambda *_: phase

    monkeypatch.setattr(franka_integration, "make_checkpoint_capture", make)

    def assemble(_live, **kwargs):
        assert kwargs["shared"]["backend"] is backend
        stages = kwargs["stages"]
        assert stages["lift"]["capture_after_lift"]() == "after_lift"
        assert stages["task"]["capture_preinsert"]() == "preinsert"
        assert stages["task"]["capture_final"] is custom_capture
        assert stages["retry"]["capture_withdrawn"]() == "preinsert"
        assert "capture_postshift" not in stages["retry"]
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    config["vlm"] = {"capture_timeout_s": 6.0}
    integration = franka_integration.factory(_options(tmp_path, config))
    integration.build_components(SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(image_hw=(1080, 1920)))))
    assert loaded == [{
        "mode": "local", "model_id": None, "max_input_size": (1920, 1088),
        "max_new_tokens": 512, "require_native_pixels": True}]
    assert len(captures) == 3
    assert all(timeout == 6.0 for _phase, timeout in captures)
