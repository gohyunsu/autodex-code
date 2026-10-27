"""Use the existing AutoDex FR3 executor with ZeroDex's local ParaDex daemon.

This profile changes only the process-local ParaDex arm address. The daemon
must already be running for the ZeroDex FR3; this module never launches it or
claims ownership of the robot's FCI connection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .franka_integration import factory as stock_factory
from .franka_integration import load_franka_integration_config
from .live_runtime import LiveRuntimeIntegration, RunLiveOptions
from .stock_daemon_identity import _ports_and_timeout


ZERODEX_FR3_FCI_IP = "172.16.1.11"
LOCAL_DAEMON_HOST = "127.0.0.1"


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
    inspect_zerodex_daemon()

    # ParaDex's get_arm() reads this module-level dictionary. Mutating it in
    # this single-session process avoids editing the shared network.json.
    from paradex.utils.system import network_info

    previous_host = network_info["franka"]
    network_info["franka"] = LOCAL_DAEMON_HOST
    try:
        integration = stock_factory(options)
    except BaseException:
        network_info["franka"] = previous_host
        raise

    def close() -> None:
        try:
            integration.close()
        finally:
            network_info["franka"] = previous_host

    return replace(integration, close=close)
