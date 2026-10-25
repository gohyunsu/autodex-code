"""The board-9 FR3 binding assembles every stock AutoDex live action."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_pc_stages import configure  # noqa: E402
from precision_insertion.franka_integration import (  # noqa: E402
    _with_checkpoint_captures, factory as franka_factory,
)
from precision_insertion.live_runtime import RunLiveOptions  # noqa: E402
from precision_insertion import franka_integration  # noqa: E402
from precision_insertion.live_orchestration import (  # noqa: E402
    compose_live_handlers,
)
from precision_insertion.runtime_assembly import (  # noqa: E402
    assemble_live_components,
)


def _fixture(*, board="9", gap=20, sampler=True, scene_mode="charuco_board"):
    launch = SimpleNamespace(
        mode=SimpleNamespace(family="cylinder", gap_mm=gap),
        board_id=board, scene_mode=scene_mode)
    clear_view = SimpleNamespace(
        before_key_capture=lambda: None,
        verify_state=lambda state, **_kwargs: state)
    live = SimpleNamespace(
        cameras=SimpleNamespace(launch=launch), clear_view=clear_view,
        opened=SimpleNamespace(runner=SimpleNamespace(mode=launch.mode)))
    calls = []

    def read_state():
        calls.append("feedback")
        return object()

    core = SimpleNamespace(
        sampler=object() if sampler else None, read_state=read_state)
    options = SimpleNamespace(enable_robot_motion=True)
    return core, live, options, calls


def test_board9_binding_assembles_all_current_actions_without_hardware():
    core, live, options, calls = _fixture()
    parameters = configure(core, live, options)
    assert calls == []  # constructing the binding never opens a controller
    assert parameters["held_relation_route"] == "nominal"
    assert parameters["stages"]["pickup"]["lift_bounds"] is None
    assert parameters["stages"]["retry"]["retry_limits"].max_total_offset_m is None
    assert parameters["stages"]["task"]["insertion_limits"].max_execution_duration_s == 90
    assert parameters["stages"]["task"]["handoff_parameters"] == {}
    assert parameters["shared"]["motion_interlock"]()
    assert calls == []

    stages = _with_checkpoint_captures(parameters["stages"], 10.)
    stages["pickup"]["executor"] = object()
    stages["lift"]["adapter"] = object()
    stages["task"].update(
        transfer_adapter=object(), insertion_adapter=object(),
        release_and_clear=lambda *_args: None)
    stages["held_return"].update(
        execute_restoration=lambda *_args: None,
        record_execution=lambda *_args: None)
    stages["repose_execution"]["execute_repose"] = lambda *_args: None
    parameters["shared"].update(
        planner=object(), read_state=lambda: None,
        state_for_observation=lambda _observation: None,
        state_for_capture=lambda _capture: None,
        backend=SimpleNamespace(infer=lambda *_args: None))
    components = assemble_live_components(
        live, shared=parameters["shared"], stages=stages,
        enable_robot_motion=True, held_relation_route="nominal")
    assert len(components) == 10
    assert len(compose_live_handlers(live, *components)) == 18
    assert all("white" in stages[name]["key_prompt"] for name in (
        "key_planning", "held_return", "transfer_recovery",
        "repose_execution"))


@pytest.mark.parametrize("gap", [1, 3, 5, 10, 15, 20])
def test_board9_binding_accepts_configured_cylinder_gaps_without_hardware(gap):
    core, live, options, calls = _fixture(gap=gap)
    assert configure(core, live, options)["held_relation_route"] == "nominal"
    assert calls == []


def test_board_free_binding_assembles_same_stages_without_hardware():
    core, live, options, calls = _fixture(board=None, scene_mode="board_free")
    parameters = configure(core, live, options)
    assert set(parameters["stages"]) == {
        "key_planning", "pickup", "lift", "held_relation", "task", "retry",
        "held_return", "transfer_recovery", "repose_planning",
        "repose_execution"}
    assert calls == []


@pytest.mark.parametrize("change", [{"board": "11"}, {"gap": 2},
                                     {"sampler": False}])
def test_board9_binding_rejects_another_fixture_or_missing_sampler(change):
    core, live, options, _calls = _fixture(**change)
    with pytest.raises(ValueError):
        configure(core, live, options)


def test_franka_integration_injects_stock_callbacks_and_assembles_board9(
        tmp_path, monkeypatch):
    core, live, _options, _calls = _fixture()
    live.cameras.launch.image_hw = (1536, 2048)
    live.cameras.launch.shared_root = tmp_path
    live.clear_view.executor = object()
    core.planner = object()
    core.executor = live.clear_view.executor
    core.clear_view = live.clear_view
    core.close = lambda: None
    core.sampler = SimpleNamespace(
        state_for_observation=lambda _observation: None,
        state_for_capture=lambda _capture: None)
    core.start_feedback_sampler = lambda _settings: core.sampler
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda _: core)
    monkeypatch.setattr(franka_integration, "load_vlm_backend",
                        lambda **_kwargs: SimpleNamespace(infer=lambda *_a: None))
    monkeypatch.setattr(franka_integration, "FrankaMeasuredLiftAdapter",
                        lambda **_kwargs: object())
    monkeypatch.setattr(franka_integration, "FrankaForceStopInsertionAdapter",
                        lambda _executor: object())
    monkeypatch.setattr(franka_integration, "StockSocketRelease",
                        lambda **_kwargs: lambda *_a: None)
    monkeypatch.setattr(franka_integration, "StockFailedReturnRecorder",
                        lambda _executor: lambda *_a: None)
    monkeypatch.setattr(franka_integration, "StockReposeExecutor",
                        lambda **_kwargs: SimpleNamespace(
                            execute_restoration=lambda *_a: None,
                            execute_repose=lambda *_a: None))
    config = {
        "schema": "precision_insertion_franka_integration_v1",
        "feedback": {
            "max_arm_hand_skew_s": .1, "max_sample_age_s": 1.,
            "max_hand_command_error_raw": 30.,
            "max_arm_update_wait_s": .2,
            "max_arm_velocity_rad_s": .05,
        },
        "sampling": {
            "max_samples": 16384, "max_bracket_span_s": 10.,
            "max_key_state_skew_s": 5., "max_arm_hold_drift_rad": .01,
            "max_hand_hold_drift_raw": 30.,
            "max_sampling_hand_command_error_raw": 200.,
            "max_sampling_arm_velocity_rad_s": 2.,
            "period_s": .01, "warmup_timeout_s": 2.,
        },
        "held_motion_monitor": {
            "daemon_binary_path": str(tmp_path / "franka_daemon"),
            "max_state_gap_s": .5, "minimum_samples": 3,
        },
        "stages_factory": "robot_pc_stages:configure",
    }
    config_path = tmp_path / "franka.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    options = RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", enable_robot_motion=True,
        robot_config_path=config_path)
    components = franka_factory(options).build_components(live)
    assert len(components) == 10
    assert len(compose_live_handlers(live, *components)) == 18
    transfer_recovery = components[7]
    assert ("max_transfer_state_gap_s" in
            transfer_recovery.return_parameters)
    assert ("max_withdrawal_state_gap_s" not in
            transfer_recovery.return_parameters)
