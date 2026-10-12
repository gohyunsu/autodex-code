#!/usr/bin/env python3
"""Read current FR3/Inspire feedback without constructing motion controllers.

The FR3 path subscribes to the daemon's state PUB socket. The Inspire path
uses Modbus function 03 (read holding registers) only. This probe never opens
the FR3 command socket, writes a hand register, or constructs FrankaExecutor.
It does not certify camera-time alignment, a stationary hold, or motion safety.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import msgpack
import zmq


def _finite_vector(value: object, size: int, name: str) -> list[float]:
    if (not isinstance(value, (tuple, list)) or len(value) != size or
            any(type(v) not in (int, float) or not math.isfinite(v)
                for v in value)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return [float(v) for v in value]


def _arm_state(payload: bytes, receipt_s: float) -> dict | None:
    state = msgpack.unpackb(payload, raw=False)
    if not isinstance(state, dict) or state.get("type") != "state_update":
        return None
    uptime = state.get("timestamp")
    if (type(uptime) not in (int, float) or
            not math.isfinite(uptime) or uptime <= 0):
        raise ValueError("FR3 state uptime is invalid")
    return {
        "receipt_utc_s": receipt_s,
        "robot_uptime_s": float(uptime),
        "qpos": _finite_vector(state.get("qpos"), 7, "FR3 qpos"),
        "qvel": _finite_vector(state.get("qvel"), 7, "FR3 qvel"),
        "wrench": _finite_vector(state.get("wrench"), 6, "FR3 wrench"),
    }


def read_arm_feedback(
    *, host: str, port: int = 5556, timeout_s: float = 5.0,
) -> dict:
    """Subscribe to two advancing FR3 state packets; never open port 5555."""
    if (not host or not 0 < port < 65536 or port == 5555 or
            not 0 < timeout_s <= 30):
        raise ValueError("invalid FR3 feedback endpoint or timeout")
    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    first = None
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            remaining_ms = max(1, min(200, int(
                (deadline - time.monotonic()) * 1000)))
            if not sock.poll(remaining_ms):
                continue
            state = _arm_state(sock.recv(), time.time())
            if state is None:
                continue
            if first is None:
                first = state
            elif state["robot_uptime_s"] > first["robot_uptime_s"]:
                return {
                    "first_robot_uptime_s": first["robot_uptime_s"],
                    "last": state,
                    "max_abs_qvel_rad_s": max(abs(v) for v in state["qvel"]),
                    "source": "franka_state_pub_read_only",
                }
        raise TimeoutError("FR3 state PUB did not produce two advancing packets")
    finally:
        sock.close()
        context.term()


def _read_six(client: object, address: int, name: str) -> list[int]:
    reply = client.read_holding_registers(address=address, count=6)
    if reply is None or reply.isError():
        raise RuntimeError(f"Inspire {name} register read failed")
    values = getattr(reply, "registers", None)
    if (not isinstance(values, list) or len(values) != 6 or
            any(type(value) is not int or not 0 <= value <= 1000
                for value in values)):
        raise ValueError(f"Inspire {name} needs six raw 0..1000 registers")
    return values


def read_hand_feedback(
    *, host: str, port: int, timeout_s: float = 3.0,
    client_factory=None,
) -> dict:
    """Read actual and commanded Inspire angles with Modbus function 03."""
    if not host or not 0 < port < 65536 or not 0 < timeout_s <= 30:
        raise ValueError("invalid Inspire feedback endpoint or timeout")
    if client_factory is None:
        from pymodbus.client import ModbusTcpClient
        client_factory = ModbusTcpClient
    client = client_factory(host=host, port=port, timeout=timeout_s)
    try:
        if not client.connect():
            raise ConnectionError("Inspire Modbus connection failed")
        measured = _read_six(client, 1546, "angleAct")
        commanded = _read_six(client, 1486, "angleSet")
        return {
            "receipt_utc_s": time.time(),
            "raw_measured": measured,
            "raw_command_register": commanded,
            "max_command_error_raw": max(abs(a - b) for a, b in
                                         zip(measured, commanded)),
            "source": "inspire_modbus_function_03_read_only",
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--franka-host", required=True)
    parser.add_argument("--franka-state-port", type=int, default=5556)
    parser.add_argument("--inspire-host", required=True)
    parser.add_argument("--inspire-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    report = {
        "schema": "precision_insertion_feedback_only_probe_v1",
        "scope": "live_feedback_read_only_no_motion_command",
        "robot_motion_command_sent": False,
        "camera_time_aligned": False,
    }
    try:
        report["franka"] = read_arm_feedback(
            host=args.franka_host, port=args.franka_state_port,
            timeout_s=args.timeout_s)
        report["inspire"] = read_hand_feedback(
            host=args.inspire_host, port=args.inspire_port,
            timeout_s=args.timeout_s)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output), "error": report.get("error"),
        "franka_feedback": "franka" in report,
        "inspire_feedback": "inspire" in report,
    }, sort_keys=True))
    if "error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
