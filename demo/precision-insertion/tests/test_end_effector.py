"""End-effector selection keeps Inspire and Franka-gripper contracts distinct."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.end_effector import (  # noqa: E402
    end_effector_spec,
    full_dof_end_effector_spec,
    gripper_width_from_hand_q,
    hand_q_from_gripper_width,
    normalize_end_effector,
    planner_end_effector_spec,
    robotiq_position_from_q,
    robotiq_q_from_position,
)


@pytest.mark.parametrize(
    ("requested", "name", "hand", "hand_dof", "full_dof"),
    [
        ("inspire", "inspire", "fr3_inspire", 6, 13),
        ("franka_gripper", "franka_gripper", "fr3_gripper", 2, 9),
        ("gripper", "franka_gripper", "fr3_gripper", 2, 9),
        ("robotiq_2f85", "robotiq_2f85", "fr3_robotiq_2f85", 1, 8),
        ("robotiq", "robotiq_2f85", "fr3_robotiq_2f85", 1, 8),
    ],
)
def test_end_effector_contracts_are_separate(
        requested, name, hand, hand_dof, full_dof):
    spec = end_effector_spec(requested)
    assert spec.name == name
    assert spec.planner_hand == hand
    assert spec.hand_dof == hand_dof
    assert spec.full_dof == full_dof
    assert full_dof_end_effector_spec(full_dof).name == name


def test_environment_defaults_to_historical_inspire(monkeypatch):
    monkeypatch.delenv("PRECISION_END_EFFECTOR", raising=False)
    assert end_effector_spec().name == "inspire"
    assert normalize_end_effector(None) == "inspire"


@pytest.mark.parametrize(
    ("hand", "ee_link", "name"),
    [
        ("fr3_inspire", "base_link", "inspire"),
        ("fr3_gripper", "fr3_hand", "franka_gripper"),
        ("fr3_robotiq_2f85", "robotiq_85_base_link", "robotiq_2f85"),
    ],
)
def test_planner_contract_selects_matching_end_effector(hand, ee_link, name):
    planner = SimpleNamespace(
        _hand=hand, _n_arm=7,
        _robot_cfg={"kinematics": {"ee_link": ee_link}},
    )
    assert planner_end_effector_spec(planner).name == name


def test_planner_contract_rejects_cross_wired_end_effector():
    planner = SimpleNamespace(
        _hand="fr3_gripper", _n_arm=7,
        _robot_cfg={"kinematics": {"ee_link": "base_link"}},
    )
    with pytest.raises(ValueError, match="fr3_hand planner contract"):
        planner_end_effector_spec(planner)


def test_gripper_width_round_trip_uses_two_half_width_joints():
    q = hand_q_from_gripper_width(0.06)
    np.testing.assert_allclose(q, [0.03, 0.03])
    assert gripper_width_from_hand_q(q) == pytest.approx(0.06)


def test_robotiq_status_round_trip_uses_one_mimic_driver_joint():
    for position in (0, 17, 128, 255):
        q = robotiq_q_from_position(position)
        assert q.shape == (1,)
        assert robotiq_position_from_q(q) == position
