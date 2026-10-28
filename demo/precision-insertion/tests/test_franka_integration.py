"""The built-in robot-PC factory owns FR3 resources and reuses stage wiring."""

import json
from dataclasses import replace
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import franka_integration  # noqa: E402
from precision_insertion.live_runtime import RunLiveOptions  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.tip_frame_window import TipFrameWindow  # noqa: E402


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


def test_obsolete_update_wait_is_ignored_in_existing_robot_config(tmp_path):
    old = _config()
    path = _options(tmp_path, old).robot_config_path
    with_old = franka_integration.load_franka_integration_config(
        path, import_stages=False)["feedback"]
    old["feedback"].pop("max_arm_update_wait_s")
    path.write_text(json.dumps(old), encoding="utf-8")
    without_old = franka_integration.load_franka_integration_config(
        path, import_stages=False)["feedback"]
    assert with_old == without_old


def test_hand_force_override_is_explicit_and_validated(tmp_path):
    config = _config()
    path = _options(tmp_path, config).robot_config_path
    assert franka_integration.load_franka_integration_config(
        path, import_stages=False)["hand_force_set_raw"] is None
    config["hand_force_set_raw"] = [400, 400, 400, 400, 600, 400]
    path.write_text(json.dumps(config), encoding="utf-8")
    assert franka_integration.load_franka_integration_config(
        path, import_stages=False)["hand_force_set_raw"] == (
            400, 400, 400, 400, 600, 400)
    config["hand_force_set_raw"] = [400, 400, 400]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="six Inspire values"):
        franka_integration.load_franka_integration_config(
            path, import_stages=False)


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


def test_zerodex_local_qwen_load_overlaps_startup_and_is_reused(
        monkeypatch, tmp_path):
    monkeypatch.setenv("PRECISION_PRELOAD_VLM", "1")
    monkeypatch.delenv("PRECISION_VISIBLE_GRIP_CAMERA_IDS", raising=False)
    events = []
    loaded = Event()
    release = Event()
    backend = object()
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: events.append("core_closed"))
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {"shared": {}, "stages": {}}
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def load_backend(**kwargs):
        events.append("qwen_load_started")
        assert kwargs["max_input_size"] == (2048, 1536)
        loaded.set()
        assert release.wait(2)
        events.append("qwen_load_finished")
        return backend

    monkeypatch.setattr(franka_integration, "load_vlm_backend", load_backend)
    monkeypatch.setattr(franka_integration, "assemble_live_components",
                        lambda _live, **kwargs: (
                            events.append("assembled") or
                            (kwargs["shared"]["backend"],)))
    launch = SimpleNamespace(
        image_hw=(1536, 2048), shared_root=tmp_path,
        camera_ids={"camera_a"})
    options = replace(_options(tmp_path, _config()), launch_snapshot=launch)
    integration = franka_integration.factory(options)
    assert loaded.wait(2)
    events.append("clear_view_could_run")
    release.set()
    live = SimpleNamespace(cameras=SimpleNamespace(launch=launch))
    assert integration.build_components(live) == (backend,)
    integration.close()
    assert events == ["qwen_load_started", "clear_view_could_run",
                      "qwen_load_finished", "assembled", "core_closed"]


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


def test_cylinder_factory_injects_stock_force_and_release_with_existing_limits(
        monkeypatch, tmp_path):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    limits = object()
    interlock = lambda: True
    module.configure = lambda *_: {
        "shared": {"backend": object()},
        "stages": {"task": {}, "key_planning": {}, "held_return": {
                       "limits": limits, "return_parameters": {
                           "max_withdrawal_state_gap_s": .5}},
                   "transfer_recovery": {},
                   "pickup": {"motion_interlock": interlock},
                   "repose_execution": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    force = object()
    release = object()
    return_recorder = object()
    monkeypatch.setattr(franka_integration, "FrankaForceStopInsertionAdapter",
                        lambda executor: force if executor is core.executor else None)

    def make_release(**kwargs):
        assert kwargs == {"executor": core.executor, "planner": core.planner,
                          "read_state": core.read_state, "path_limits": limits}
        return release

    monkeypatch.setattr(franka_integration, "StockSocketRelease", make_release)
    monkeypatch.setattr(franka_integration, "StockFailedReturnRecorder",
                        lambda executor: return_recorder
                        if executor is core.executor else None)
    restoration = lambda *_args: None
    repose = lambda *_args: None

    def make_reset(**kwargs):
        assert kwargs["executor"] is core.executor
        assert kwargs["planner"] is core.planner
        assert kwargs["read_state"] is core.read_state
        assert kwargs["motion_interlock"] is interlock
        assert kwargs["enable_robot_motion"] is True
        return SimpleNamespace(execute_restoration=restoration,
                               execute_repose=repose)

    monkeypatch.setattr(franka_integration, "StockReposeExecutor", make_reset)

    def assemble(_live, **kwargs):
        task = kwargs["stages"]["task"]
        assert task["insertion_adapter"] is force
        assert task["release_and_clear"] is release
        assert kwargs["stages"]["held_return"]["record_execution"] is (
            return_recorder)
        assert kwargs["stages"]["transfer_recovery"]["record_execution"] is (
            return_recorder)
        assert kwargs["stages"]["transfer_recovery"]["limits"] is limits
        assert kwargs["stages"]["transfer_recovery"]["return_parameters"] == (
            {"max_transfer_state_gap_s": .5})
        assert kwargs["stages"]["held_return"]["return_parameters"] == (
            {"max_withdrawal_state_gap_s": .5})
        for stage in ("key_planning", "held_return", "transfer_recovery",
                      "repose_execution"):
            assert kwargs["stages"][stage]["key_prompt"] == (
                "white plastic cylindrical key")
        assert kwargs["stages"]["held_return"]["execute_restoration"] is (
            restoration)
        assert kwargs["stages"]["repose_execution"]["execute_repose"] is (
            repose)
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    live = SimpleNamespace(opened=SimpleNamespace(
        runner=SimpleNamespace(mode=SimpleNamespace(family="cylinder"))))
    franka_integration.factory(_options(tmp_path, _config())).build_components(live)


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
        "daemon_binary_path": str(binary),
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
    insertion_adapter = object()
    monkeypatch.setattr(franka_integration, "FrankaForceStopInsertionAdapter",
                        lambda executor: insertion_adapter)

    def assemble(_live, **kwargs):
        stages = kwargs["stages"]
        assert stages["lift"]["adapter"] is held_adapter
        assert stages["task"]["transfer_adapter"] is held_adapter
        if family == "cylinder":
            assert stages["task"]["insertion_adapter"] is insertion_adapter
        assert "transfer_adapter" not in stages["retry"]
        assert "lateral_adapter" not in stages["retry"]
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    config["held_motion_monitor"] = {
        "daemon_binary_path": str(tmp_path / "franka_daemon"),
        "max_state_gap_s": .05,
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
        "max_state_gap_s": .05,
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
        "max_state_gap_s": .05,
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
        cameras=SimpleNamespace(launch=SimpleNamespace(
            image_hw=(1080, 1920), shared_root=tmp_path))))
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
        assert "capture_withdrawn" not in stages["retry"]
        assert "capture_postshift" not in stages["retry"]
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    config = _config()
    config["vlm"] = {"capture_timeout_s": 6.0}
    integration = franka_integration.factory(_options(tmp_path, config))
    integration.build_components(SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(
            image_hw=(1080, 1920), shared_root=tmp_path))))
    assert loaded == [{
        "mode": "local", "model_id": str(
            tmp_path / "AutoDex/models/Qwen3-VL-2B-Instruct"),
        "max_input_size": (1920, 1088),
        "max_new_tokens": 128, "require_native_pixels": True,
        "parallel_views": 1,
        "classification_image_size": (384, 288)}]
    assert len(captures) == 2
    assert all(timeout == 6.0 for _phase, timeout in captures)


def test_zerodex_local_lift_uses_configured_visible_grip_views(monkeypatch, tmp_path):
    selected = ("26053248", "25322639", "25305462")
    monkeypatch.setenv("PRECISION_VISIBLE_GRIP_CAMERA_IDS", ",".join(selected))
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {"shared": {}, "stages": {}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = SimpleNamespace()
    monkeypatch.setattr(franka_integration, "load_vlm_backend",
                        lambda **_kwargs: backend)

    def assemble(_live, **kwargs):
        assert kwargs["shared"]["backend"] is backend
        assert backend.visible_grip_camera_ids == selected
        assert backend.supports_visible_grip_lift is True
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    live = SimpleNamespace(cameras=SimpleNamespace(launch=SimpleNamespace(
        camera_ids=(*selected, "25322642"), image_hw=(1536, 2048),
        shared_root=tmp_path)))
    integration = franka_integration.factory(_options(tmp_path, _config()))
    integration.build_components(live)
    integration.close()


def test_parallel_vlm_views_require_independent_api_backend():
    assert franka_integration._vlm_settings({
        "mode": "gemini", "model_id": "test-model", "parallel_views": 3,
    })["parallel_views"] == 3
    with pytest.raises(ValueError, match="invalid VLM"):
        franka_integration._vlm_settings({"parallel_views": 2})


def test_tracked_mode_attaches_exact_preinsert_camera_window(monkeypatch, tmp_path):
    capture = RawCameraCapture("pre_1", 1, {}, {"25322651": 42}, {})
    window = TipFrameWindow("service_unavailable", "25322651", 42, ())
    calls = []
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    monkeypatch.setattr(franka_integration,
                        "FrankaForceStopInsertionAdapter", lambda _: object())
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {
        "shared": {"backend": object()},
        "stages": {"task": {"capture_preinsert": lambda *_: capture}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def fetch(**kwargs):
        calls.append(kwargs)
        return window

    monkeypatch.setattr(franka_integration, "fetch_tip_frame_window", fetch)

    def assemble(_live, **kwargs):
        result = kwargs["stages"]["task"]["capture_preinsert"](
            object(), object(), object())
        assert result.temporal_window is window
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    options = replace(_options(tmp_path, _config()), xy_estimator="tracked_tip")
    live = SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(
            camera_ids=("25322651",), image_hw=(1080, 1920),
            shared_root=tmp_path)),
        opened=SimpleNamespace(runner=SimpleNamespace(
            mode=SimpleNamespace(family="cylinder"))))
    integration = franka_integration.factory(options)
    integration.build_components(live)
    assert calls == [{"host": "capture2", "port": 16951,
                      "camera_id": "25322651", "target_frame_id": 42}]


def test_sam_roi_worker_warms_before_retry_and_closes_with_session(
    monkeypatch, tmp_path,
):
    events = []
    loading, release = Event(), Event()
    core = SimpleNamespace(
        clear_view=object(), planner=object(), executor=object(),
        read_state=lambda: None, sampler=None, close=lambda: events.append("core_close"))
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: (events.append("core_open") or core))
    monkeypatch.setattr(franka_integration,
                        "FrankaForceStopInsertionAdapter", lambda _: object())
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {
        "shared": {"backend": SimpleNamespace(
            close=lambda: events.append("vlm_close"))},
        "stages": {"task": {}, "retry": {}, "key_planning": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    python = tmp_path / "sam-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("PRECISION_SAM_PYTHON", str(python))
    monkeypatch.setenv("PRECISION_SAM_CHECKPOINT", str(checkpoint))

    class MaskWorker:
        def __init__(self, **kwargs):
            events.append(("warm", kwargs))
            loading.set()
            assert release.wait(3.)
            kwargs["stderr_path"].write_text("SAM worker log\n", encoding="utf-8")

        def close(self):
            events.append("sam_close")

    monkeypatch.setattr(franka_integration, "SamRoiBackend", MaskWorker)

    def assemble(_live, **kwargs):
        retry = kwargs["stages"]["retry"]
        assert retry["xy_estimator"] == "sam_half_ellipse"
        assert isinstance(retry["sam_segmentor"].ready(), MaskWorker)
        kwargs["stages"]["key_planning"]["before_preflight"]()
        events.append("assembled")
        return ()

    monkeypatch.setattr(franka_integration, "assemble_live_components", assemble)
    options = replace(_options(tmp_path, _config()),
                      xy_estimator="sam_half_ellipse")
    live = SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(
            shared_root=tmp_path, image_hw=(1080, 1920))),
        opened=SimpleNamespace(runner=SimpleNamespace(
            mode=SimpleNamespace(family="cylinder"))))
    integration = franka_integration.factory(options)
    assert loading.wait(1.)
    assert "core_open" in events
    release.set()
    integration.build_components(live)
    options.run_output_dir.mkdir()
    integration.close()
    warm = next(value for value in events if isinstance(value, tuple))
    assert warm[1]["python_path"] == python
    assert warm[1]["checkpoint"] == checkpoint
    assert events.index("assembled") < events.index("sam_close")
    assert events[-3:] == ["vlm_close", "sam_close", "core_close"]
    assert (options.run_output_dir / "sam_roi_worker.stderr.log").read_text() == (
        "SAM worker log\n")


def test_sam_model_load_failure_surfaces_before_key_preflight(monkeypatch, tmp_path):
    python = tmp_path / "sam-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("PRECISION_SAM_PYTHON", str(python))
    monkeypatch.setenv("PRECISION_SAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(franka_integration, "SamRoiBackend",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("SAM CUDA unavailable")))
    events = []
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: events.append("core_close"))
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: (events.append("core_open") or core))
    module = ModuleType("robot_pc_stages")
    module.configure = lambda *_: {
        "shared": {"backend": object()}, "stages": {"key_planning": {}}}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        franka_integration, "assemble_live_components",
        lambda _live, **kwargs: kwargs["stages"]["key_planning"][
            "before_preflight"]())
    options = replace(_options(tmp_path, _config()),
                      xy_estimator="sam_half_ellipse")
    integration = franka_integration.factory(options)
    live = SimpleNamespace(
        cameras=SimpleNamespace(launch=SimpleNamespace(
            shared_root=tmp_path, image_hw=(1080, 1920))),
        opened=SimpleNamespace(runner=SimpleNamespace(
            mode=SimpleNamespace(family="cylinder"))))
    with pytest.raises(RuntimeError, match="SAM CUDA unavailable"):
        integration.build_components(live)
    integration.close()
    assert events == ["core_open", "core_close"]


def test_plan_only_does_not_load_unused_xy_sam_worker(monkeypatch, tmp_path):
    core = SimpleNamespace(
        clear_view=object(), planner=object(), read_state=lambda: None,
        sampler=None, close=lambda: None)
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    monkeypatch.setattr(
        franka_integration, "SamRoiBackend",
        lambda **_kwargs: pytest.fail("plan-only never reaches XY segmentation"))
    options = replace(_options(tmp_path, _config()),
                      xy_estimator="sam_half_ellipse", stop_after_planning=True)
    integration = franka_integration.factory(options)
    integration.close()


def test_sam_worker_closes_if_fr3_core_fails(monkeypatch, tmp_path):
    python = tmp_path / "sam-python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("PRECISION_SAM_PYTHON", str(python))
    monkeypatch.setenv("PRECISION_SAM_CHECKPOINT", str(checkpoint))
    events = []

    class MaskWorker:
        def __init__(self, **_kwargs):
            events.append("sam_open")

        def close(self):
            events.append("sam_close")

    monkeypatch.setattr(franka_integration, "SamRoiBackend", MaskWorker)
    monkeypatch.setattr(franka_integration, "open_franka_core",
                        lambda _: (_ for _ in ()).throw(RuntimeError("FR3 offline")))
    options = replace(_options(tmp_path, _config()),
                      xy_estimator="sam_half_ellipse")
    with pytest.raises(RuntimeError, match="FR3 offline"):
        franka_integration.factory(options)
    assert events == ["sam_open", "sam_close"]


@pytest.mark.parametrize("missing", ["python", "checkpoint"])
def test_sam_configuration_fails_before_fr3_core_opens(
    monkeypatch, tmp_path, missing,
):
    python = tmp_path / "sam-python"
    checkpoint = tmp_path / "sam.pt"
    if missing != "python":
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
    if missing != "checkpoint":
        checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("PRECISION_SAM_PYTHON", str(python))
    monkeypatch.setenv("PRECISION_SAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(
        franka_integration, "open_franka_core",
        lambda _: pytest.fail("FR3 core opened before SAM inputs were checked"))
    options = replace(_options(tmp_path, _config()),
                      xy_estimator="sam_half_ellipse")
    label = "Python" if missing == "python" else "checkpoint"
    with pytest.raises(FileNotFoundError, match=f"SAM3 {label}"):
        franka_integration.factory(options)
