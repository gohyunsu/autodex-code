"""Identify the running stock ParaDex daemon before noncontact FR3 motion.

The checked-in hash is for the installed robot-PC binary whose source has a
200 ms joint-velocity stream expiry. This is a runtime identity check, not a
claim that a physical timeout or E-stop test was performed. Other controllers
still use the separately reviewed commissioning-record route.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
from pathlib import Path

from .franka_completion_checked import is_demo_completion_checked_executor
from .franka_measured_lift import FrankaMeasuredLiftAdapter


AUDITED_STOCK_DAEMON_SHA256 = (
    "363e4247cb3f9d7f3b25e9c5ae1e1dc8766775ce104407e33c48541e62b60bab"
)
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

    This route is limited to the first-party measured lift/transfer adapter,
    the audited installed binary and a local ParaDex process configured for
    the controller's FR3 ports. Docker's PID namespace can deny access to
    /proc/<pid>/exe even while the daemon's local ports are usable, so the
    installed binary is hashed once and the process is matched by its command.
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
    if not binary.is_file() or _sha(binary) != AUDITED_STOCK_DAEMON_SHA256:
        raise ValueError("installed FR3 daemon differs from audited stock binary")
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
        "daemon_binary_sha256": AUDITED_STOCK_DAEMON_SHA256,
        "daemon_pid": pid,
        "daemon_cmdline_sha256": hashlib.sha256(
            "\0".join(argv).encode("utf-8")).hexdigest(),
        "command_expiry_s": expiry,
        "physical_watchdog_tested": False,
        "source": "installed_stock_paradex_daemon_runtime_identity",
    }
