"""Measured wrist motion distinguishes the moving key from a static rim."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.same_trial_visual_feedback import (  # noqa: E402
    update_held_key_xy,
)


def _observation(u):
    return {"alignment": {
        "nominal_tip_z_socket_m": .1,
        "sam_half_ellipse_views": [{
            "camera_id": "camera", "admitted": True,
            "sam_half_ellipse": {"center_full_px": [u, 500.],
                                 "major_radius_px": 14.},
        }],
    }}


def _calibration():
    return {"c2r": np.eye(4).tolist(), "camera_calibration": {
        "intrinsics_full": {"camera": {"K_undist": [
            [1000., 0., 500.], [0., 1000., 500.], [0., 0., 1.]]}},
        "extrinsics_full": {"camera": np.eye(4).tolist()},
    }}


def test_temporal_feedback_tracks_face_that_follows_measured_motion():
    result = update_held_key_xy(
        previous=_observation(500.), current=_observation(530.),
        measured_xy_socket_m=[.003, 0.], measured_axis_socket_m=0.,
        calibration=_calibration(), socket_robot=np.eye(4),
        previous_camera_id="camera", previous_xy_socket_m=[0., 0.],
        key_radius_m=.015)
    assert result["status"] == "tracked"
    np.testing.assert_allclose(result["fused_xy_socket_m"], [.003, 0.])
    assert result["camera_rows"][0]["innovation_px"] < 1e-8


def test_temporal_feedback_does_not_follow_stationary_socket_rim():
    result = update_held_key_xy(
        previous=_observation(500.), current=_observation(500.),
        measured_xy_socket_m=[.003, 0.], measured_axis_socket_m=0.,
        calibration=_calibration(), socket_robot=np.eye(4),
        previous_camera_id="camera", previous_xy_socket_m=[0., 0.],
        key_radius_m=.015)
    assert result["status"] == "lost"
    assert result["camera_rows"][0]["innovation_px"] == 30.


def test_cumulative_feedback_finds_static_rim_when_one_step_is_subpixel():
    # With a distant camera, 3 mm moves only 3 px. One-frame association is
    # ambiguous, but five moves should have displaced the held key by 15 px.
    calibration = _calibration()
    calibration["camera_calibration"]["intrinsics_full"]["camera"][
        "K_undist"][0][0] = 100.
    calibration["camera_calibration"]["intrinsics_full"]["camera"][
        "K_undist"][1][1] = 100.
    result = update_held_key_xy(
        previous=_observation(500.), current=_observation(500.),
        measured_xy_socket_m=[.003, 0.], measured_axis_socket_m=0.,
        calibration=calibration, socket_robot=np.eye(4),
        previous_camera_id="camera", previous_xy_socket_m=[0., 0.],
        key_radius_m=.015, anchor=_observation(500.),
        cumulative_xy_socket_m=[.015, 0.])
    assert result["camera_rows"][0]["innovation_px"] == 3.
    assert result["camera_rows"][0]["anchor_response"]["resolvable"]
    assert result["status"] == "lost"
