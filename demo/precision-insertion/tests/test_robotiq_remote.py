"""Robotiq network adapter preserves the planner/rPR command contract."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time

import msgpack
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.end_effector import robotiq_q_from_position  # noqa: E402
from precision_insertion.robotiq_remote import RobotiqRemote  # noqa: E402


def _client(states):
    client = object.__new__(RobotiqRemote)
    client.speed = 255
    client.force = 255
    client._last_command_position = 0
    client.sent = []
    client._set_position = lambda position, *, speed, force: (
        client.sent.append((position, speed, force)) or states[0])
    iterator = iter(states)
    client.read_state = lambda: next(iterator)
    return client


def test_open_joint_maps_to_rpr_zero_and_uses_profile_registers():
    final = {"position": 0, "requested_position": 0,
             "object_state": 3, "go_to": True}
    client = _client([final])

    state = client.move(np.asarray([0.0]))

    assert client.sent == [(0, 255, 255)]
    assert state == final


def test_close_until_contact_returns_measured_not_requested_joint():
    moving = {"position": 100, "requested_position": 255,
              "object_state": 0, "go_to": True}
    contact = {"position": 166, "requested_position": 255,
               "object_state": 2, "go_to": True}
    client = _client([moving, contact])

    measured = client.close_until_contact()

    assert client.sent == [(255, 255, 255)]
    np.testing.assert_allclose(measured, robotiq_q_from_position(166))


def test_feedback_and_motion_requests_share_one_req_socket_serially():
    class StrictReqSocket:
        def __init__(self):
            self.in_flight = False
            self.types = []

        def send(self, packed):
            if self.in_flight:
                raise AssertionError("REQ send interleaved before its recv")
            self.in_flight = True
            self.types.append(msgpack.unpackb(packed, raw=False)["type"])
            # Release the GIL so the other worker would interleave here if
            # RobotiqRemote did not own a request lock.
            time.sleep(.02)

        def recv(self):
            assert self.in_flight
            self.in_flight = False
            return msgpack.packb({"ok": True}, use_bin_type=True)

    client = object.__new__(RobotiqRemote)
    client._request_lock = threading.RLock()
    client._command = StrictReqSocket()
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(
            client._request, ({"type": "read_state"}, {"type": "ping"})))

    assert answers == [{"ok": True}, {"ok": True}]
    assert sorted(client._command.types) == ["ping", "read_state"]
