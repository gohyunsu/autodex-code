"""Identify the running ParaDex daemon before noncontact FR3 motion.

Record the installed binary hash for traceability without pinning one build.
The runtime process, local ports and configured stream expiry are checked;
this is not a physical timeout or E-stop test. Other controllers still use
the separately reviewed commissioning-record route.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
from pathlib import Path

from .franka_completion_checked import is_demo_completion_checked_executor
from .franka_measured_lift import FrankaMeasuredLiftAdapter


STOCK_STREAM_EXPIRY_S = 0.2


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ports_and_timeout(argv: tuple[str, ...]) -> tuple[int, int, float]:
    if len(argv) < 2 or Path(argv[0]).name != "franka_daemon":
        raise ValueError("running process is not the ParaDex daemon")
    fields = argv[2:]
    if len(fields) % 2:
        raise ValueError("daemon launch flags are incomplete")
    values = {"--command_port": 5555, "--state_port": 5556,
              "--stream_timeout_ms": 200}
    seen: set[str] = set()
    for name, raw in zip(fields[::2], fields[1::2]):
        if name not in values or name in seen:
            raise ValueError("daemon launch contains unknown or repeated flags")
        seen.add(name)
        try:
            values[name] = int(raw)
        except ValueError as exc:
            raise ValueError("daemon launch contains a noninteger flag") from exc
    command = values["--command_port"]
    state = values["--state_port"]
    timeout_ms = values["--stream_timeout_ms"]
    if not (0 < command < 65536 and 0 < state < 65536 and
            10 <= timeout_ms <= 2000):
        raise ValueError("daemon ports or stream timeout are invalid")
    return command, state, timeout_ms / 1000.0


def verify_stock_daemon_identity(
    *, adapter: FrankaMeasuredLiftAdapter, max_command_age_s: float,
    proc_root: Path = Path("/proc"),
) -> dict:
    """Bind a stock follower to this host's exact running watchdog binary.

    This route is limited to the first-party measured lift/transfer adapter
    and a local ParaDex process configured for the controller's FR3 ports.
    Docker's PID namespace can deny access to /proc/<pid>/exe even while the
    daemon's local ports are usable, so the installed binary hash is recorded
    and the process is matched by its command. A rebuild does not invalidate
    otherwise identical runtime behavior.
    It never reports physical watchdog commissioning as passed.
    """
    if (type(adapter) is not FrankaMeasuredLiftAdapter or
            not is_demo_completion_checked_executor(adapter.executor) or
            type(max_command_age_s) not in (int, float) or
            not math.isfinite(max_command_age_s) or
            max_command_age_s < STOCK_STREAM_EXPIRY_S):
        raise ValueError("stock noncontact route needs its checked executor and 200 ms expiry")
    arm = adapter.executor.arm
    try:
        local = ipaddress.ip_address(arm.ip).is_loopback
        command_port = int(arm.command_port)
        state_port = int(arm.state_port)
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError("stock FR3 controller has no local daemon endpoint") from exc
    if not local:
        raise ValueError("stock daemon identity can only verify a local FR3 endpoint")
    binary = Path(adapter.daemon_binary_path).expanduser().resolve()
    if not binary.is_file():
        raise ValueError(f"FR3 daemon binary is missing: {binary}")
    binary_sha256 = _sha(binary)
    matches = []
    for process in Path(proc_root).iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if (process / "comm").read_text(encoding="utf-8").strip() != "franka_daemon":
                continue
            raw = (process / "cmdline").read_bytes().rstrip(b"\0")
            argv = tuple(part.decode("utf-8") for part in raw.split(b"\0"))
            port, state, expiry = _ports_and_timeout(argv)
            if (port != command_port or state != state_port or
                    expiry > max_command_age_s):
                continue
            matches.append((int(process.name), argv, expiry))
        except (OSError, UnicodeError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError("no unique matching running stock FR3 watchdog daemon")
    pid, argv, expiry = matches[0]
    return {
        "schema": "precision_insertion_stock_daemon_identity_v1",
        "daemon_binary_path": str(binary),
        "daemon_binary_sha256": binary_sha256,
        "daemon_pid": pid,
        "daemon_cmdline_sha256": hashlib.sha256(
            "\0".join(argv).encode("utf-8")).hexdigest(),
        "command_expiry_s": expiry,
        "physical_watchdog_tested": False,
        "source": "installed_stock_paradex_daemon_runtime_identity",
    }
