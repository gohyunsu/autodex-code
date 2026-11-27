"""Robotiq network adapter preserves the planner/rPR command contract."""

from pathlib import Path
import sys

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
