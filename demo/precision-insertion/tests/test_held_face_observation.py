"""Camera-facing observation geometry and inverse tilted-tip translation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.held_face_observation import (  # noqa: E402
    rank_face_view_tilts, tilted_observation_goals,
    xy_correction_from_tilted_tip,
)
from precision_insertion.held_key_tip import triangulate_tip_proxy  # noqa: E402


def _poses():
    socket = np.eye(4)
    key = np.eye(4)
    key[:3, :3] = np.diag([1., -1., -1.])
    key[:3, 3] = [0., 0., .25]
    wrist = np.eye(4)
    wrist[:3, 3] = [.06, -.02, .34]
    return key, wrist, socket


def test_tilted_wrist_goals_keep_rigid_key_relation_and_return_pose():
    key, wrist, socket = _poses()
    goals = tilted_observation_goals(
        T_robot_key_preinsert=key, T_robot_wrist_current=wrist,
        T_robot_socket=socket, raise_m=.04, tilt_deg=90.,
        axis_azimuth_deg=15.)
    relation = np.linalg.inv(key) @ wrist
    assert np.allclose(goals["T_robot_key_raised"][:3, 3],
                       key[:3, 3] + [0., 0., .04])
    assert np.allclose(goals["T_robot_wrist_raised"],
                       goals["T_robot_key_raised"] @ relation)
    assert np.allclose(goals["T_robot_wrist_observation"],
                       goals["T_robot_key_observation"] @ relation)
    assert np.allclose(goals["T_robot_wrist_return"], wrist)


def test_tilted_face_rank_prefers_separated_cameras_and_vertical_has_none():
    key, _wrist, socket = _poses()
    cameras = {
        "a": [.5, .5, .7],
        "b": [-.5, .5, .7],
        "near_a": [.505, .5, .7],
    }
    assert rank_face_view_tilts(
        T_robot_key_preinsert=key, T_robot_socket=socket,
        camera_centers_robot=cameras, tip_z_m=.08, raise_m=.04,
        tilt_degrees=(0.,), azimuth_degrees=(0.,)) == []
    ranked = rank_face_view_tilts(
        T_robot_key_preinsert=key, T_robot_socket=socket,
        camera_centers_robot=cameras, tip_z_m=.08, raise_m=.04,
        tilt_degrees=(90.,), azimuth_degrees=(0.,))
    assert len(ranked) == 1
    assert "b" in ranked[0]["best_pair"]
    assert ranked[0]["best_pair_parallax_sin"] > .4
    assert all(value > 0 for value in ranked[0]["face_dot_by_camera"].values())


def test_tilted_tip_inverse_recovers_socket_xy_translation():
    key, wrist, socket = _poses()
    goals = tilted_observation_goals(
        T_robot_key_preinsert=key, T_robot_wrist_current=wrist,
        T_robot_socket=socket, raise_m=.04, tilt_deg=75.,
        axis_azimuth_deg=60.)
    tilted = goals["T_robot_key_observation"]
    nominal_tip = (tilted @ np.array([0., 0., .08, 1.]))[:3]
    actual_unrotated_error = np.array([.003, -.002, 0.])
    observed_tip = (nominal_tip +
                    goals["R_robot_tilt"] @ actual_unrotated_error)
    K = np.array([[900., 0., 1024.], [0., 900., 768.], [0., 0., 1.]])
    image_points = []
    for name, center_x in (("left", -.12), ("right", .12)):
        T_camera_robot = np.eye(4)
        T_camera_robot[:3, 3] = [-center_x, 0., .8]
        projection = K @ T_camera_robot[:3]
        homogeneous = projection @ np.r_[observed_tip, 1.]
        image_points.append((name, projection,
                             homogeneous[:2] / homogeneous[2]))
    triangulated = triangulate_tip_proxy(image_points)
    assert np.allclose(triangulated["point_socket_m"], observed_tip,
                       atol=1e-8)
    result = xy_correction_from_tilted_tip(
        observed_tip_robot_m=triangulated["point_socket_m"],
        T_robot_key_observation=tilted,
        R_robot_tilt=goals["R_robot_tilt"],
        T_robot_socket=socket, tip_z_m=.08)
    assert np.allclose(result["key_translation_socket_m"],
                       actual_unrotated_error, atol=1e-12)
    assert np.allclose(result["xy_correction_socket_m"],
                       [-.003, .002], atol=1e-12)
