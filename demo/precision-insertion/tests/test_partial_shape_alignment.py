import math

import numpy as np
from scipy.spatial.transform import Rotation

from precision_insertion.partial_shape_alignment import (
    PartialCylinderView, estimate_partial_cylinder_alignment,
    partial_shape_measurement_weights,
)


def _camera(position, target=(0., 0., .1)):
    position = np.asarray(position, float)
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack((right, down, forward))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ position
    return T


def _project(K, T, point):
    camera = T[:3, :3] @ point + T[:3, 3]
    homogeneous = K @ camera
    return homogeneous[:2] / homogeneous[2]


def _observations(tip, axis, *, outlier_camera=None):
    K = np.array([[1200., 0., 1024.],
                  [0., 1200., 768.],
                  [0., 0., 1.]])
    cameras = {
        "left": _camera((-.45, -.30, .32)),
        "right": _camera((.45, -.25, .30)),
        "front": _camera((.02, .50, .25)),
        "side": _camera((-.50, .18, .22)),
    }
    rear = tip - .08 * axis
    rows = []
    for name, T in cameras.items():
        uv_tip = _project(K, T, tip)
        uv_rear = _project(K, T, rear)
        if name == outlier_camera:
            uv_tip = uv_tip + np.array([75., -55.])
        rows.append(PartialCylinderView(
            name, K, T, tuple(uv_tip), tuple(uv_tip - uv_rear)))
    return rows


def test_partial_shape_fit_recovers_xy_and_tilt():
    tip = np.array([.008, -.005, .105])
    axis = Rotation.from_rotvec([.055, -.035, 0.]).apply([0., 0., -1.])
    result = estimate_partial_cylinder_alignment(
        _observations(tip, axis)[:3], nominal_tip_z_socket_m=tip[2],
        key_length_m=.08)
    assert result["status"] == "diagnostic_partial_shape_pose"
    assert np.allclose(result["tip_socket_m"][:2], tip[:2], atol=1e-6)
    recovered = np.asarray(result["insertion_axis_socket"])
    angle = math.acos(float(np.clip(recovered @ axis, -1., 1.)))
    assert angle < 1e-5
    assert result["jacobian_rank"] == 4


def test_partial_shape_fit_robust_loss_limits_one_bad_view():
    tip = np.array([.006, -.004, .105])
    axis = Rotation.from_rotvec([.04, -.02, 0.]).apply([0., 0., -1.])
    result = estimate_partial_cylinder_alignment(
        _observations(tip, axis, outlier_camera="side"),
        nominal_tip_z_socket_m=tip[2], key_length_m=.08,
        robust_pixel_scale=8.)
    assert result["status"] == "diagnostic_partial_shape_pose"
    # The outlier remains in the transparent per-view report, while the
    # inlier cameras keep the fitted tip near the actual metric position.
    assert np.linalg.norm(np.asarray(result["tip_socket_m"][:2]) - tip[:2]) < .003
    residuals = {row["camera_id"]: row["center_residual_norm_px"]
                 for row in result["per_view"]}
    assert residuals["side"] > max(residuals[camera]
                                   for camera in ("left", "right", "front"))


def test_partial_shape_fit_does_not_promote_one_view_to_metric_pose():
    tip = np.array([.003, .002, .105])
    axis = np.array([0., 0., -1.])
    result = estimate_partial_cylinder_alignment(
        _observations(tip, axis)[:1], nominal_tip_z_socket_m=tip[2],
        key_length_m=.08)
    assert result["status"] == "insufficient_multiview_shape_geometry"
    assert result["camera_ids"] == ["left"]


def test_visible_geometry_gives_complementary_cap_and_shaft_weights():
    side = partial_shape_measurement_weights(
        {"major_radius_px": 15., "minor_radius_px": 3.},
        {"visible_axial_span_px": 100., "visible_lateral_span_px": 30.,
         "solidity": .9})
    end_on = partial_shape_measurement_weights(
        {"major_radius_px": 25., "minor_radius_px": 24.},
        {"visible_axial_span_px": 52., "visible_lateral_span_px": 50.,
         "solidity": .95})
    assert side["shaft_axis_weight"] > end_on["shaft_axis_weight"]
    assert end_on["distal_center_weight"] > side["distal_center_weight"]
