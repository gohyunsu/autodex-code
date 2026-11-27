from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


DEMO = Path(__file__).resolve().parents[1]
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from precision_insertion.analytic_gripper import (  # noqa: E402
    analytic_franka_gripper_grasp,
    analytic_franka_gripper_grasp_for_candidate,
    classify_cylinder_tabletop,
)


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
    assert result.T_robot_hand[2, 3] == pytest.approx(0.055)
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


def test_upright_candidate_ids_sample_eight_symmetry_equivalent_sides():
    key = _pose(np.eye(3))
    grasps = [analytic_franka_gripper_grasp_for_candidate(
        key, table_normal_robot=np.asarray([0.0, 0.0, 1.0]),
        identifier=("table", "0", f"{index:03d}"))
        for index in range(8)]
    approaches = np.asarray([grasp.T_robot_hand[:3, 2] for grasp in grasps])
    assert np.allclose(approaches[:, 2], 0.0, atol=1e-9)
    assert len({tuple(np.round(row, 8)) for row in approaches}) == 8
    assert all(np.allclose(key @ grasp.T_key_hand, grasp.T_robot_hand)
               for grasp in grasps)


def test_side_down_candidate_ids_are_top_down_wrist_flips():
    rotation = np.asarray([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    key = _pose(rotation)
    a, b = [analytic_franka_gripper_grasp_for_candidate(
        key, table_normal_robot=np.asarray([0.0, 0.0, 1.0]),
        identifier=("table", "1", f"{index:03d}"))
        for index in range(2)]
    assert np.allclose(a.T_robot_hand[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(a.T_robot_hand[:3, 2], b.T_robot_hand[:3, 2])
    assert np.allclose(a.T_robot_hand[:3, 0], -b.T_robot_hand[:3, 0])
    assert np.allclose(a.T_robot_hand[:3, 1], -b.T_robot_hand[:3, 1])


def test_candidate_identifier_must_match_measured_tabletop_class():
    with pytest.raises(ValueError, match="tabletop class"):
        analytic_franka_gripper_grasp_for_candidate(
            _pose(np.eye(3)), table_normal_robot=[0.0, 0.0, 1.0],
            identifier=("table", "1", "000"))


def test_robotiq_grasp_uses_physical_closing_axis_and_one_joint():
    rotation = np.asarray([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    result = analytic_franka_gripper_grasp(
        _pose(rotation), table_normal_robot=[0.0, 0.0, 1.0],
        end_effector="robotiq_2f85")
    # Robotiq closes along local x; cylinder axis is local -y; approach is z.
    np.testing.assert_allclose(
        np.cross(result.T_robot_hand[:3, 0], result.T_robot_hand[:3, 1]),
        result.T_robot_hand[:3, 2], atol=1e-9)
    np.testing.assert_allclose(
        np.abs(result.T_robot_hand[:3, 1]), [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.pregrasp_q, [0.0])
    np.testing.assert_allclose(result.grasp_q, [0.53])
