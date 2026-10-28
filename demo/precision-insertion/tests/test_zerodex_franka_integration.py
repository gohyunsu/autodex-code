"""ZeroDex arm routing tests; no robot or camera connections are opened."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autodex.utils.robot_config import FR3_CLEAR_VIEW
from precision_insertion import zerodex_franka_integration as route
from precision_insertion.live_runtime import LiveRuntimeIntegration, RunLiveOptions


def test_curobo_collision_debug_print_is_process_local():
    world_mesh = SimpleNamespace()

    sentinel = object()
    previous = world_mesh.__dict__.get("print", sentinel)
    restore = route._quiet_curobo_world_mesh_debug(world_mesh)
    try:
        assert world_mesh.print("tensor value") is None
    finally:
        restore()
    assert world_mesh.__dict__.get("print", sentinel) is previous


def test_local_pymodbus_client_import_matches_paradex():
    route.install_zerodex_modbus_compat()
    from pymodbus.client import ModbusTcpClient

    assert callable(ModbusTcpClient)
    route.install_zerodex_modbus_compat()


def _process(root: Path, pid: int, *, fci_ip: str, command_port: int = 5555):
    process = root / str(pid)
    process.mkdir()
    (process / "comm").write_text("franka_daemon\n", encoding="utf-8")
    argv = ("/workspace/paradex/cpp/franka_daemon/build/franka_daemon",
            fci_ip, "--command_port", str(command_port),
            "--state_port", "5556")
    (process / "cmdline").write_bytes(b"\0".join(
        part.encode("utf-8") for part in argv) + b"\0")


def test_inspect_zerodex_daemon_requires_exact_robot_and_ports(tmp_path):
    _process(tmp_path, 10, fci_ip="172.16.0.2")
    with pytest.raises(RuntimeError, match="found 0"):
        route.inspect_zerodex_daemon(proc_root=tmp_path)
    _process(tmp_path, 11, fci_ip="172.16.1.11", command_port=5557)
    with pytest.raises(RuntimeError, match="found 0"):
        route.inspect_zerodex_daemon(proc_root=tmp_path)
    _process(tmp_path, 12, fci_ip="172.16.1.11")
    assert route.inspect_zerodex_daemon(proc_root=tmp_path) == {
        "daemon_pid": 12, "fr3_fci_ip": "172.16.1.11",
        "client_host": "127.0.0.1"}
    _process(tmp_path, 13, fci_ip="172.16.1.11")
    with pytest.raises(RuntimeError, match="found 2"):
        route.inspect_zerodex_daemon(proc_root=tmp_path)


def _options(tmp_path: Path) -> RunLiveOptions:
    (tmp_path / "clear_view_arm_q.json").write_text(json.dumps({
        "schema": "precision_insertion_zerodex_clear_view_v1",
        "robot_fci_ip": "172.16.1.11",
        "arm_q": [0.1, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7],
    }), encoding="utf-8")
    return RunLiveOptions(
        launch_config_path=tmp_path / "launch.json",
        endpoint_catalog_path=tmp_path / "catalog.json",
        run_output_dir=tmp_path / "run",
        robot_config_path=tmp_path / "robot.json",
        enable_robot_motion=True)


def test_factory_routes_process_only_and_restores_after_close(tmp_path, monkeypatch):
    from paradex.utils.system import network_info

    binary = tmp_path / "franka_daemon"
    binary.write_bytes(b"test executable")
    monkeypatch.setattr(route, "load_franka_integration_config",
                        lambda *args, **kwargs: {
                            "held_motion_monitor": (binary, object()),
                            "lift_monitor": None})
    monkeypatch.setattr(route, "inspect_zerodex_daemon", lambda: {})
    monkeypatch.setattr(route, "_quiet_curobo_world_mesh_debug", lambda: lambda: None)
    original = network_info["franka"]
    original_clear_view = FR3_CLEAR_VIEW.copy()
    observed = []
    monkeypatch.setattr(route, "stock_factory", lambda options: (
        observed.append((network_info["franka"], FR3_CLEAR_VIEW.copy())) or
        LiveRuntimeIntegration(
            clear_view=object(), build_components=lambda live: (),
            close=lambda: observed.append(network_info["franka"]))))
    integration = route.factory(_options(tmp_path))
    assert network_info["franka"] == "127.0.0.1"
    np.testing.assert_allclose(FR3_CLEAR_VIEW, [0.1, -0.7, 0, -2.3, 0, 1.5, 0.7])
    integration.close()
    assert observed[0][0] == "127.0.0.1"
    np.testing.assert_allclose(observed[0][1],
                               [0.1, -0.7, 0, -2.3, 0, 1.5, 0.7])
    assert observed[1] == "127.0.0.1"
    assert network_info["franka"] == original
    np.testing.assert_allclose(FR3_CLEAR_VIEW, original_clear_view)


def test_factory_restores_process_route_if_stock_factory_fails(tmp_path, monkeypatch):
    from paradex.utils.system import network_info

    monkeypatch.setattr(route, "load_franka_integration_config",
                        lambda *args, **kwargs: {
                            "held_motion_monitor": None, "lift_monitor": None})
    monkeypatch.setattr(route, "inspect_zerodex_daemon", lambda: {})
    monkeypatch.setattr(route, "_quiet_curobo_world_mesh_debug", lambda: lambda: None)
    original = network_info["franka"]
    original_clear_view = FR3_CLEAR_VIEW.copy()

    def fail(options):
        assert network_info["franka"] == "127.0.0.1"
        np.testing.assert_allclose(FR3_CLEAR_VIEW,
                                   [0.1, -0.7, 0, -2.3, 0, 1.5, 0.7])
        raise RuntimeError("startup failed")

    monkeypatch.setattr(route, "stock_factory", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        route.factory(_options(tmp_path))
    assert network_info["franka"] == original
    np.testing.assert_allclose(FR3_CLEAR_VIEW, original_clear_view)


def test_factory_does_not_touch_robot_when_daemon_binary_missing(tmp_path, monkeypatch):
    from paradex.utils.system import network_info

    monkeypatch.setattr(route, "load_franka_integration_config",
                        lambda *args, **kwargs: {
                            "held_motion_monitor": (tmp_path / "missing", object()),
                            "lift_monitor": None})
    monkeypatch.setattr(route, "stock_factory", lambda options: pytest.fail(
        "must not open the robot"))
    original = network_info["franka"]
    with pytest.raises(FileNotFoundError, match="daemon binary is missing"):
        route.factory(_options(tmp_path))
    assert network_info["franka"] == original


def test_clear_view_loader_rejects_wrong_robot_or_nonfinite_q(tmp_path):
    options = _options(tmp_path)
    np.testing.assert_allclose(
        route.load_zerodex_clear_view(options.robot_config_path),
        [0.1, -0.7, 0, -2.3, 0, 1.5, 0.7])
    source = tmp_path / "clear_view_arm_q.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["robot_fci_ip"] = "172.16.0.2"
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong ZeroDex FR3"):
        route.load_zerodex_clear_view(options.robot_config_path)
    record["robot_fci_ip"] = "172.16.1.11"
    record["arm_q"][0] = float("nan")
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid seven-joint"):
        route.load_zerodex_clear_view(options.robot_config_path)
