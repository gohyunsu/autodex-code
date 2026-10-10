"""Cylinder tip/axis fitting keeps the unobservable yaw on its prior gauge."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.postshift_pose import (  # noqa: E402
    reconstruct_axisymmetric_held_hypothesis,
    tip_axis_visual_surface_bound,
)


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _case():
    socket = np.eye(4)
    socket[:3, :3] = _rz(.3)
    socket[:3, 3] = [.4, -.2, .1]
    prior_key_socket = np.eye(4)
    prior_key_socket[:3, :3] = _rz(.7) @ np.diag([1., -1., -1.])
    prior_key_socket[:3, 3] = [.001, -.002, .16]
    hand = socket @ prior_key_socket
    grasp = np.eye(4)
    tip = np.array([0., 0., .08])
    predicted_tip = prior_key_socket[:3, :3] @ tip + prior_key_socket[:3, 3]
    axis = prior_key_socket[:3, :3] @ np.array([0., 0., 1.])
    return socket, hand, grasp, tip, predicted_tip, axis, prior_key_socket


def test_reconstructs_tip_axis_and_preserves_prior_yaw_gauge():
    socket, hand, grasp, tip, old_tip, old_axis, old_pose = _case()
    angle = math.radians(2.)
    # Rotate the old insertion axis about socket Y, then shift the tip.
    ry = np.array([[math.cos(angle), 0., math.sin(angle)],
                   [0., 1., 0.],
                   [-math.sin(angle), 0., math.cos(angle)]])
    new_axis = ry @ old_axis
    new_tip = old_tip + np.array([.0003, -.0002, .0001])
    result = reconstruct_axisymmetric_held_hypothesis(
        T_robot_socket=socket, T_robot_hand_measured=hand,
        T_key_hand_prior=grasp, tip_key_m=tip,
        insertion_axis_key=[0., 0., 1.], tip_socket_m=new_tip,
        insertion_axis_socket=new_axis,
        max_tip_prior_residual_m=.001,
        max_axis_prior_residual_deg=3.)
    assert np.allclose(result.T_socket_key[:3, :3] @ tip +
                       result.T_socket_key[:3, 3], new_tip)
    assert np.allclose(result.T_socket_key[:3, :3] @ [0., 0., 1.], new_axis)
    assert np.allclose(result.T_socket_key[:3, :3] @ [1., 0., 0.],
                       ry @ old_pose[:3, :3] @ [1., 0., 0.])
    assert np.allclose(result.T_robot_key @ result.T_key_hand, hand)
    assert math.isclose(result.axis_prior_residual_deg, 2., abs_tol=1e-7)
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["unobservable_dof"] == (
        "rotation_about_cylinder_axis")


def test_rejects_wrong_end_large_axis_change_and_bad_limits():
    socket, hand, grasp, tip, old_tip, old_axis, _ = _case()
    base = dict(T_robot_socket=socket, T_robot_hand_measured=hand,
                T_key_hand_prior=grasp, tip_key_m=tip,
                insertion_axis_key=[0., 0., 1.], tip_socket_m=old_tip,
                insertion_axis_socket=old_axis,
                max_tip_prior_residual_m=.001,
                max_axis_prior_residual_deg=3.)
    with pytest.raises(ValueError, match="disagrees"):
        reconstruct_axisymmetric_held_hypothesis(
            **{**base, "tip_socket_m": old_tip + [.005, 0., 0.]})
    with pytest.raises(ValueError, match="disagrees"):
        reconstruct_axisymmetric_held_hypothesis(
            **{**base, "insertion_axis_socket": [0., .5, -.8660254]})
    with pytest.raises(ValueError, match="commissioned"):
        reconstruct_axisymmetric_held_hypothesis(
            **{**base, "max_axis_prior_residual_deg": 100.})
    with pytest.raises(ValueError, match="nonzero"):
        reconstruct_axisymmetric_held_hypothesis(
            **{**base, "insertion_axis_socket": [0., 0., 0.]})


def test_tip_axis_surface_bound_covers_angular_lever_arm():
    vertices = np.array([[0., 0., 0.], [0., 0., .08]])
    got = tip_axis_visual_surface_bound(
        key_vertices_m=vertices, tip_key_m=[0., 0., .08],
        max_tip_error_m=.0002, max_axis_error_deg=1.)
    assert math.isclose(got, .0002 + .16 * math.sin(math.radians(.5)))
    with pytest.raises(ValueError, match="metric CAD"):
        tip_axis_visual_surface_bound(
            key_vertices_m=vertices, tip_key_m=[0., 0., .08],
            max_tip_error_m=-.001, max_axis_error_deg=1.)
