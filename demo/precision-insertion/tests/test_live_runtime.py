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


def test_live_cli_imports_stock_and_demo_packages_from_another_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run_live.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--enable-robot-motion" in result.stdout


def test_integration_is_complete_and_closed_after_session(monkeypatch, tmp_path):
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
    options = live_runtime.RunLiveOptions(
        tmp_path / "launch.json", tmp_path / "catalog.json",
        tmp_path / "run")
    with pytest.raises(PermissionError, match="opt-in"):
        live_runtime.run_integrated_live_session(
            options, integration_spec="missing:factory")
    with pytest.raises(ValueError, match="zero or one"):
        live_runtime.run_integrated_live_session(
            live_runtime.RunLiveOptions(
                options.launch_config_path, options.endpoint_catalog_path,
                options.run_output_dir, max_xy_retries=2,
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
