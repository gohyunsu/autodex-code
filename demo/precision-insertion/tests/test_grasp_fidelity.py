import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import precision_insertion.grasp_fidelity as fidelity
from precision_insertion.grasp_fidelity import (
    cylinder_pose_change, pose7_to_se3, rigid_pose_change,
    simulated_visual_penetration_audit, trajectory_closure_audit,
    trajectory_rigid_closure_audit,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def test_pose7_mujoco_quaternion_order():
    T = pose7_to_se3([0, 0, 0, math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert np.allclose(T[:3, 0], [0, 1, 0])
    with pytest.raises(ValueError):
        pose7_to_se3([0, 0, 0, 0, 0, 0, 0])


def test_symmetry_quotient_uses_physical_center_not_endcap_origin():
    original = np.eye(4)
    # The same 80-mm symmetric cylinder, turned end-for-end about its center.
    flipped = np.eye(4)
    flipped[:3, :3] = np.diag([1.0, -1.0, -1.0])
    flipped[:3, 3] = [0.0, 0.0, 0.08]
    result = cylinder_pose_change(initial_key=original, final_key=flipped,
                                  initial_hand=original, final_hand=original,
                                  key_height_m=0.08)
    assert result["center_world_displacement_m"] == pytest.approx(0.0)
    assert result["center_in_hand_displacement_m"] == pytest.approx(0.0)
    assert result["symmetry_reduced_axis_tilt_deg"] == pytest.approx(0.0)


def test_trajectory_audits_squeeze_separately_from_gravity():
    trajectory = {
        "phase": ["pregrasp", "squeeze", "force_gravity", "force_gravity"],
        "object_pose": [_pose(), _pose(x=0.02), _pose(x=0.025), _pose(x=0.025)],
        "robot_qpos": [_pose() + [0] * 12] * 4,
    }
    report = trajectory_closure_audit(trajectory, key_height_m=0.08)
    assert report["end_squeeze"]["center_in_hand_displacement_m"] == pytest.approx(0.02)
    assert report["first_gravity_step"]["center_in_hand_displacement_m"] == pytest.approx(0.025)
    assert report["end_gravity"]["symmetry_reduced_axis_tilt_deg"] == pytest.approx(0.0)
    trajectory["phase"][1] = "grasp"
    with pytest.raises(ValueError, match="lacks squeeze"):
        trajectory_closure_audit(trajectory, key_height_m=0.08)


def test_square_closure_keeps_full_relative_rotation_and_center():
    original = np.eye(4)
    rotated = np.eye(4)
    rotated[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = rigid_pose_change(
        initial_key=original, final_key=rotated,
        initial_hand=original, final_hand=original,
        key_center_local_m=np.array([.01, 0, 0]),
    )
    assert result["center_in_hand_displacement_m"] == pytest.approx(
        math.sqrt(2) * .01)
    assert result["full_relative_rotation_deg"] == pytest.approx(90)
    trajectory = {
        "phase": ["pregrasp", "squeeze", "force_gravity"],
        "object_pose": [_pose(), _pose(x=.003), _pose(x=.004)],
        "robot_qpos": [_pose() + [0] * 12] * 3,
    }
    closure = trajectory_rigid_closure_audit(
        trajectory, key_center_local_m=np.zeros(3))
    assert closure["end_squeeze"]["center_in_hand_displacement_m"] == (
        pytest.approx(.003))
    assert closure["end_gravity"]["center_in_hand_displacement_m"] == (
        pytest.approx(.004))


def test_achieved_visual_audit_uses_mujoco_joint_order_and_dynamic_key(monkeypatch):
    monkeypatch.setattr(fidelity, "_verify_cylinder_mesh", lambda _: None)
    observed = []

    def fake_visual(**kwargs):
        observed.append(kwargs)
        return {"sample_points_over_threshold": 0}

    monkeypatch.setattr(fidelity, "_visual_penetration_at_pose", fake_visual)
    robot = _pose() + list(range(12))
    trajectory = {
        "phase": ["pregrasp", "squeeze", "force_gravity"],
        "robot_qpos": [robot] * 3,
        "object_pose": [_pose(), _pose(x=0.01), _pose(x=0.02)],
    }
    report = simulated_visual_penetration_audit(
        trajectory=trajectory, key_mesh_path="unused", robot_urdf="unused")
    assert list(report) == ["end_squeeze", "end_gravity"]
    assert np.array_equal(observed[0]["hand_q"], [0, 1, 4, 6, 8, 10])
    assert observed[0]["T_key_hand"][0, 3] == pytest.approx(-0.01)
    assert observed[1]["T_key_hand"][0, 3] == pytest.approx(-0.02)


def test_analytic_cylinder_inside_depth():
    points = np.array([[0, 0, 0.04], [0.01, 0, 0.04],
                       [0.02, 0, 0.04], [0, 0, -0.01]])
    assert np.allclose(fidelity._cylinder_inside_depth(points),
                       [0.015, 0.005, 0.0, 0.0])
