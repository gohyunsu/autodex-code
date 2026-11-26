from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DEMO = Path(__file__).resolve().parents[1]
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from precision_insertion.analytic_gripper import (  # noqa: E402
    analytic_franka_gripper_grasp, classify_cylinder_tabletop)


def _pose(rotation: np.ndarray, translation=(0.5, 0.1, 0.0)) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def test_standing_key_uses_horizontal_side_grasp():
    key = _pose(np.eye(3))
    result = analytic_franka_gripper_grasp(
        key, table_normal_robot=np.asarray([0.0, 0.0, 1.0]))
    assert result.tabletop_class == "end_down"
    assert abs(result.T_robot_hand[2, 2]) < 1e-9
    assert np.allclose(result.T_robot_hand[:3, 0], [0.0, 0.0, 1.0])
    assert np.allclose(result.pregrasp_q, [0.04, 0.04])
    assert np.allclose(result.grasp_q, [0.015, 0.015])
    assert np.allclose(key @ result.T_key_hand, result.T_robot_hand)


def test_lying_key_uses_top_down_grasp_and_measured_yaw():
    # local key z points along robot +x
    rotation = np.asarray([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    key = _pose(rotation)
    result = analytic_franka_gripper_grasp(
        key, table_normal_robot=np.asarray([0.0, 0.0, 1.0]))
    assert result.tabletop_class == "side_down"
    assert np.allclose(result.T_robot_hand[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(np.abs(result.T_robot_hand[:3, 0]), [1.0, 0.0, 0.0])
    assert result.T_robot_pregrasp[2, 3] > result.T_robot_hand[2, 3]
    assert np.allclose(key @ result.T_key_hand, result.T_robot_hand)


def test_end_exchange_does_not_change_tabletop_class_or_physical_grasp():
    key_a = _pose(np.eye(3))
    flip = np.diag([1.0, -1.0, -1.0])
    key_b = _pose(flip)
    # End exchange is a rotation about the cylinder center, not its local
    # z=0 end-frame origin.
    key_b[:3, 3] = key_a[:3, 3] + np.asarray([0.0, 0.0, 0.080])
    normal = np.asarray([0.0, 0.0, 1.0])
    a = analytic_franka_gripper_grasp(key_a, table_normal_robot=normal)
    b = analytic_franka_gripper_grasp(key_b, table_normal_robot=normal)
    assert classify_cylinder_tabletop(
        key_b, table_normal_robot=normal) == "end_down"
    assert a.tabletop_class == b.tabletop_class
    # The D-infinity center and selected hand frame are identical even though
    # the local object origin/end label changed.
    assert np.allclose(a.T_robot_hand, b.T_robot_hand)


def test_nearest_class_assignment_has_no_unobservable_rejection_band():
    angle = np.deg2rad(50.0)
    rotation_y = np.asarray([[np.cos(angle), 0.0, np.sin(angle)],
                             [0.0, 1.0, 0.0],
                             [-np.sin(angle), 0.0, np.cos(angle)]])
    assert classify_cylinder_tabletop(
        _pose(rotation_y), table_normal_robot=[0.0, 0.0, 1.0]) == "side_down"
