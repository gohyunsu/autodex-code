"""The runnable entry point composes one complete camera-session pipeline."""

import os
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_runtime  # noqa: E402
from precision_insertion.live_orchestration import _LIVE_ACTIONS  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
import run_live  # noqa: E402


def test_live_cli_imports_stock_and_demo_packages_from_another_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run_live.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--enable-robot-motion" in result.stdout
    assert "--exp-name" in result.stdout


def test_default_exp_path_matches_stock_autodex_hierarchy(monkeypatch, tmp_path):
    mode = select_mode("cylinder", 20)
    monkeypatch.setattr(run_live, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=mode))
    output = run_live.default_run_output(tmp_path / "launch.json",
                                         "precision-insertion")
    assert output.parent.parent == (Path.home() / "shared_data" / "AutoDex" /
                                    "experiment" / "precision-insertion" /
                                    "inspire" / mode.key_object)
    assert output.parent.name == "_pipeline_runs"
    with pytest.raises(ValueError, match="path-safe"):
        run_live.default_run_output(tmp_path / "launch.json", "../other")


def test_static_preflight_reads_v8_files_without_opening_robot(
        monkeypatch, tmp_path):
    from precision_insertion import franka_integration

    launch_path = tmp_path / "launch.json"
    catalog_path = tmp_path / "catalog.json"
    launch_path.write_text("{}\n", encoding="utf-8")
    catalog_path.write_text("{}\n", encoding="utf-8")
    mode = select_mode("cylinder", 20)
    monkeypatch.setattr(live_runtime, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=mode, shared_root=tmp_path,
                                        readiness=lambda: {"board_id": "9"}))
    monkeypatch.setattr(live_runtime, "preflight_endpoint_catalog_file",
                        lambda **_kwargs: {"candidates": [
                            {"eligible": True}, {"eligible": False}]})
    monkeypatch.setattr(franka_integration, "open_franka_core", lambda *_args:
                        pytest.fail("static preflight opened the robot"))
    monkeypatch.setattr(franka_integration, "import_module", lambda *_args:
                        pytest.fail("static preflight imported stage code"))
    robot_path = tmp_path / "robot.json"
    robot_path.write_text(json.dumps({
        "schema": "precision_insertion_franka_integration_v1",
        "feedback": {
            "max_arm_hand_skew_s": .1, "max_sample_age_s": 1.,
            "max_hand_command_error_raw": 20.,
            "max_arm_update_wait_s": .2,
            "max_arm_velocity_rad_s": .05},
        "stages_factory": "static_robot_pc_stages:configure",
    }), encoding="utf-8")
    report = live_runtime.inspect_live_inputs(
        launch_config_path=launch_path, endpoint_catalog_path=catalog_path,
        robot_config_path=robot_path)
    assert report["eligible_candidate_count"] == 1
    assert report["launch"]["board_id"] == "9"
    assert report["robot_config"]["stage_factory_syntax_valid"] is True
    assert report["robot_config"]["stage_factory_module_on_path"] is False
    assert report["robot_config"]["stage_factory_imported"] is False
    assert report["robot_ready"] is False


def test_stage_module_path_check_does_not_execute_package(monkeypatch, tmp_path):
    package = tmp_path / "robot_pc_package"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package must not run in static preflight')\n",
        encoding="utf-8")
    (package / "stages.py").write_text(
        "raise RuntimeError('stage module must not run in static preflight')\n",
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert live_runtime._stage_module_on_path(
        "robot_pc_package.stages:configure") is True
    assert "robot_pc_package" not in sys.modules
    assert "robot_pc_package.stages" not in sys.modules


def test_integration_is_complete_and_closed_after_session(monkeypatch, tmp_path):
    monkeypatch.setattr(live_runtime, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=SimpleNamespace(family="cylinder")))
    class ClearView:
        pass

    monkeypatch.setattr(live_runtime, "ClearViewForPerception", ClearView)
    closed = []
    module = ModuleType("test_robot_integration")

    def factory(options):
        assert options.enable_robot_motion is True

        def components(live):
            return [SimpleNamespace(
                live=live,
                handlers=lambda: {name: lambda *_: None for name in _LIVE_ACTIONS})]

        return live_runtime.LiveRuntimeIntegration(
            ClearView(), components, lambda: closed.append(True))

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    live = SimpleNamespace()
    result = SimpleNamespace(status="success")

    def run(**kwargs):
        assert set(kwargs["build_handlers"](live)) == _LIVE_ACTIONS
        return result

    monkeypatch.setattr(live_runtime, "run_live_precision_session", run)
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", enable_robot_motion=True)
    assert live_runtime.run_integrated_live_session(
        options, integration_spec="test_robot_integration:factory") is result
    assert closed == [True]


def test_integration_refuses_motion_without_opt_in_or_missing_handlers(
        monkeypatch, tmp_path):
    monkeypatch.setattr(live_runtime, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=SimpleNamespace(family="cylinder")))
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run")
    with pytest.raises(PermissionError, match="opt-in"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="missing:factory")
    with pytest.raises(ValueError, match="nonnegative"):
        live_runtime.run_integrated_live_session(
            live_runtime.RunLiveOptions(
                options.launch_config_path, options.endpoint_catalog_path,
                options.run_output_dir, max_xy_retries=-1,
                enable_robot_motion=True),
            integration_spec="missing:factory")

    class ClearView:
        pass

    monkeypatch.setattr(live_runtime, "ClearViewForPerception", ClearView)
    closed = []
    module = ModuleType("incomplete_robot_integration")
    module.factory = lambda _: live_runtime.LiveRuntimeIntegration(
        ClearView(),
        lambda live: [SimpleNamespace(
            live=live,
            handlers=lambda: {"capture_fresh_key": lambda *_: None})],
        lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def run(**kwargs):
        kwargs["build_handlers"](SimpleNamespace())

    monkeypatch.setattr(live_runtime, "run_live_precision_session", run)
    with pytest.raises(ValueError, match="unconfigured actions"):
        live_runtime.run_integrated_live_session(
            live_runtime.RunLiveOptions(
                options.launch_config_path, options.endpoint_catalog_path,
                options.run_output_dir, enable_robot_motion=True),
            integration_spec="incomplete_robot_integration:factory")
    assert closed == [True]


def test_square_live_mode_stops_before_constructing_robot_integration(
        monkeypatch, tmp_path):
    monkeypatch.setattr(live_runtime, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=SimpleNamespace(family="square")))
    monkeypatch.setattr(live_runtime, "load_runtime_integration", lambda *_a:
                        pytest.fail("square must not construct robot adapter"))
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", enable_robot_motion=True)
    with pytest.raises(ValueError, match="cylinder mode only"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="ignored:factory")


def test_static_preflight_launch_stops_before_robot_integration(
        monkeypatch, tmp_path):
    monkeypatch.setattr(live_runtime, "load_session_launch", lambda _path:
                        SimpleNamespace(mode=SimpleNamespace(family="cylinder"),
                                        execution_scope="static_preflight_only"))
    monkeypatch.setattr(live_runtime, "load_runtime_integration", lambda *_a:
                        pytest.fail("static launch opened robot integration"))
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", enable_robot_motion=True)
    with pytest.raises(PermissionError, match="static preflight"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="ignored:factory")


def test_cylinder_live_rejects_legacy_label_route_before_robot_integration(
        monkeypatch, tmp_path):
    monkeypatch.setattr(live_runtime, "load_runtime_integration", lambda *_a:
                        pytest.fail("legacy label must not construct robot adapter"))
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", label_mode="evidence_fused",
        enable_robot_motion=True)
    with pytest.raises(ValueError, match="vlm_only or manual"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="ignored:factory")


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_xy_weight_is_rejected_before_robot_integration(
        monkeypatch, tmp_path, weight):
    monkeypatch.setattr(live_runtime, "load_runtime_integration", lambda *_a:
                        pytest.fail("bad gain must not construct robot adapter"))
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run", xy_update_weight=weight, enable_robot_motion=True)
    with pytest.raises(ValueError, match="XY update weight"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="ignored:factory")
