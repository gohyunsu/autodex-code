"""The runnable entry point composes one complete camera-session pipeline."""

import os
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
