"""Use the existing AutoDex FR3 executor with ZeroDex's local ParaDex daemon.

This profile changes only process-local ParaDex arm routing and the clear-view
joint target. The daemon must already be running for the ZeroDex FR3; this
module never launches it or claims ownership of the robot's FCI connection.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from autodex.utils.robot_config import FR3_CLEAR_VIEW

from .franka_integration import factory as stock_factory
from .franka_integration import load_franka_integration_config
from .live_runtime import LiveRuntimeIntegration, RunLiveOptions
from .stock_daemon_identity import _ports_and_timeout


ZERODEX_FR3_FCI_IP = os.environ.get(
    "PRECISION_ZERODEX_FR3_FCI_IP", "172.16.1.11")
LOCAL_DAEMON_HOST = os.environ.get(
    "PRECISION_ZERODEX_DAEMON_HOST", "127.0.0.1")
_CLEAR_VIEW_SCHEMA = "precision_insertion_zerodex_clear_view_v1"


def _quiet_curobo_world_mesh_debug(world_mesh=None) -> Callable[[], None]:
    """Keep stock tensor debug prints out of cuRobo CUDA graph capture.

    The vendored world_mesh module has three unconditional tensor prints.
    Formatting those tensors launches CUDA work during graph capture, so this
    process-local override replaces only that module's ``print`` lookup.
    """
    if world_mesh is None:
        from curobo.geom.sdf import world_mesh

    missing = object()
    previous = world_mesh.__dict__.get("print", missing)
    world_mesh.print = lambda *_args, **_kwargs: None

    def restore() -> None:
        if previous is missing:
            world_mesh.__dict__.pop("print", None)
        else:
            world_mesh.print = previous

    return restore


def install_zerodex_modbus_compat() -> None:
    """Expose the installed pymodbus 2.x client to ParaDex's 3.x import.

    The local ParaDex Inspire driver imports ``pymodbus.client.ModbusTcpClient``
    although this host has pymodbus 2.5.3, where that class lives in
    ``pymodbus.client.sync``. Keep the alias inside this process instead of
    editing the shared ParaDex checkout or capture4 environment.
    """
    import pymodbus.client as client

    if not hasattr(client, "ModbusTcpClient"):
        from pymodbus.client.sync import ModbusTcpClient

        client.ModbusTcpClient = ModbusTcpClient


def load_zerodex_clear_view(config_path: Path) -> np.ndarray:
    """Read the measured clear-view joints beside the ZeroDex robot profile."""
    source = Path(config_path).expanduser().resolve().parent / "clear_view_arm_q.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("schema") != _CLEAR_VIEW_SCHEMA or
            record.get("robot_fci_ip") != ZERODEX_FR3_FCI_IP):
        raise ValueError(f"wrong ZeroDex FR3 clear-view record: {source}")
    q = np.asarray(record.get("arm_q"), dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError(f"invalid seven-joint ZeroDex clear view: {source}")
    return q


def inspect_zerodex_daemon(
    *, proc_root: Path = Path("/proc"),
    expected_fci_ip: str = ZERODEX_FR3_FCI_IP,
) -> dict:
    """Identify the one local ParaDex daemon targeting this rig's FR3.

    Reading process metadata does not open a robot connection. The command
    and state ports are those used by the unchanged ParaDex FrankaController.
    """
    matches = []
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if (process / "comm").read_text(encoding="utf-8").strip() != "franka_daemon":
                continue
            raw = (process / "cmdline").read_bytes().rstrip(b"\0")
            argv = tuple(part.decode("utf-8") for part in raw.split(b"\0"))
            command_port, state_port, _ = _ports_and_timeout(argv)
            if (len(argv) < 2 or argv[1] != expected_fci_ip or
                    command_port != 5555 or state_port != 5556):
                continue
            matches.append(int(process.name))
        except (OSError, UnicodeError, ValueError):
            continue
    if len(matches) != 1:
        raise RuntimeError(
            "ZeroDex needs one local franka_daemon targeting "
            f"{expected_fci_ip} on ports 5555/5556; found {len(matches)}"
        )
    return {"daemon_pid": matches[0], "fr3_fci_ip": expected_fci_ip,
            "client_host": LOCAL_DAEMON_HOST}


def factory(options: RunLiveOptions) -> LiveRuntimeIntegration:
    """Route only this precision-insertion process to ZeroDex's FR3 daemon."""
    if not isinstance(options, RunLiveOptions) or options.robot_config_path is None:
        raise ValueError("ZeroDex FR3 integration needs --robot-config")
    if options.enable_robot_motion is not True:
        raise PermissionError("ZeroDex FR3 integration needs explicit robot-motion opt-in")
    parsed = load_franka_integration_config(
        options.robot_config_path, import_stages=False)
    monitor = parsed["held_motion_monitor"] or parsed["lift_monitor"]
    if monitor is not None and not monitor[0].is_file():
        raise FileNotFoundError(f"ZeroDex FR3 daemon binary is missing: {monitor[0]}")
    clear_view_q = load_zerodex_clear_view(options.robot_config_path)
    inspect_zerodex_daemon()

    # Imported FR3_CLEAR_VIEW references in the executor, clear-view gate and
    # release/recovery planners all point to this one ndarray. Updating it in
    # place before stock_factory constructs the core keeps their target
    # identical without changing the AutoDex source or global calibration.
    # ParaDex's get_arm() reads this module-level dictionary. Mutating it in
    # this single-session process avoids editing the shared network.json.
    from paradex.utils.system import network_info

    install_zerodex_modbus_compat()
    restore_curobo_print = _quiet_curobo_world_mesh_debug()
    previous_clear_view = FR3_CLEAR_VIEW.copy()
    FR3_CLEAR_VIEW[:] = clear_view_q
    previous_host = network_info["franka"]
    network_info["franka"] = LOCAL_DAEMON_HOST
    try:
        integration = stock_factory(options)
    except BaseException:
        restore_curobo_print()
        network_info["franka"] = previous_host
        FR3_CLEAR_VIEW[:] = previous_clear_view
        raise

    def close() -> None:
        try:
            integration.close()
        finally:
            restore_curobo_print()
            network_info["franka"] = previous_host
            FR3_CLEAR_VIEW[:] = previous_clear_view

    return replace(integration, close=close)
