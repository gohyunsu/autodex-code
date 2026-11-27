"""Robotiq 2F-85 client for the Capture4-owned RS-485 server.

The camera PC owns the serial adapter.  This module deliberately contains
only the network client used by the precision runtime, so installing or
changing the shared Capture4 service is not part of a robot episode.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from .end_effector import robotiq_position_from_q, robotiq_q_from_position


class RobotiqRemote:
    """ParaDex-like controller facade backed by Capture4 ZMQ."""

    def __init__(self, host: str | None = None, *, command_port: int | None = None,
                 state_port: int | None = None, timeout_ms: int = 3000) -> None:
        self.host = host or os.environ.get(
            "PRECISION_ROBOTIQ_HOST", "147.46.219.136")
        self.command_port = int(command_port or os.environ.get(
            "PRECISION_ROBOTIQ_COMMAND_PORT", "5565"))
        self.state_port = int(state_port or os.environ.get(
            "PRECISION_ROBOTIQ_STATE_PORT", "5566"))
        self.timeout_ms = int(timeout_ms)
        self.speed = int(os.environ.get("PRECISION_ROBOTIQ_SPEED", "255"))
        self.force = int(os.environ.get("PRECISION_ROBOTIQ_FORCE", "255"))
        if not 0 <= self.speed <= 255 or not 0 <= self.force <= 255:
            raise ValueError("Robotiq speed and force must be in [0, 255]")
        self._context = zmq.Context()
        self._command = self._new_command_socket()
        self._state = self._context.socket(zmq.SUB)
        self._state.setsockopt(zmq.SUBSCRIBE, b"")
        self._state.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._state.setsockopt(zmq.LINGER, 0)
        self._state.connect(f"tcp://{self.host}:{self.state_port}")
        self._last_command_position = 0
        self.save_path: str | None = None

    def _new_command_socket(self):
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{self.host}:{self.command_port}")
        return socket

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._command.send(msgpack.packb(payload, use_bin_type=True))
            packed = self._command.recv()
        except zmq.Again as exc:
            self._command.close(linger=0)
            self._command = self._new_command_socket()
            raise TimeoutError(
                f"Capture4 Robotiq request timed out: {payload['type']}") from exc
        response = msgpack.unpackb(packed, raw=False)
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error", "Robotiq request failed"))
        return response

    @staticmethod
    def _state_body(response: dict[str, Any]) -> dict[str, Any]:
        state = response.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Capture4 Robotiq response has no state")
        return state

    def ping(self) -> dict[str, Any]:
        return self._request({"type": "ping"})

    def read_state(self) -> dict[str, Any]:
        response = self._request({"type": "read_state"})
        state = self._state_body(response)
        self._last_command_position = int(state.get(
            "requested_position", self._last_command_position))
        return {**state, "time_ns": int(response.get("time_ns", time.time_ns()))}

    def get_data(self) -> dict[str, Any]:
        state = self.read_state()
        state["qpos"] = robotiq_q_from_position(state["position"])
        state["action"] = robotiq_q_from_position(
            state.get("requested_position", self._last_command_position))
        state["time"] = state["time_ns"] / 1e9
        return state

    def _set_position(self, position: int, *, speed: int, force: int) -> dict:
        response = self._request({
            "type": "set_position", "position": int(position),
            "speed": int(speed), "force": int(force),
        })
        self._last_command_position = int(position)
        return self._state_body(response)

    def move(self, q: np.ndarray, *, speed: int | None = None,
             force: int | None = None,
             wait: bool = True, timeout_s: float = 5.0) -> dict[str, Any]:
        """Command one planner-space joint and optionally wait for completion."""
        target = robotiq_position_from_q(q)
        state = self._set_position(
            target, speed=self.speed if speed is None else speed,
            force=self.force if force is None else force)
        if not wait:
            return state
        deadline = time.monotonic() + float(timeout_s)
        while True:
            state = self.read_state()
            # object_state 1/2 means contact while opening/closing; 3 means the
            # requested position was reached.  Either is a completed Robotiq
            # motion, whereas object_state 0 with go_to=true is still moving.
            if not bool(state.get("go_to", False)) or int(
                    state.get("object_state", 0)) in (1, 2, 3):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError("Robotiq motion did not finish within timeout")
            time.sleep(0.02)

    def close_until_contact(self, *, speed: int | None = None,
                            force: int | None = None,
                            timeout_s: float = 5.0) -> np.ndarray:
        state = self._set_position(
            255, speed=self.speed if speed is None else speed,
            force=self.force if force is None else force)
        deadline = time.monotonic() + float(timeout_s)
        while True:
            state = self.read_state()
            if int(state.get("object_state", 0)) in (1, 2, 3) or not bool(
                    state.get("go_to", False)):
                return robotiq_q_from_position(state["position"])
            if time.monotonic() >= deadline:
                raise TimeoutError("Robotiq close did not finish within timeout")
            time.sleep(0.02)

    def start(self, save_path: str) -> None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.save_path = str(save_path)

    def stop(self) -> None:
        self.save_path = None

    def end(self) -> None:
        self._command.close(linger=0)
        self._state.close(linger=0)
        self._context.term()
