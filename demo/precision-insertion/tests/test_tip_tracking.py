"""Saved-image endpoint tracking and metric XY without a robot or VLM."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.grounded_alignment import GroundedLineView
from precision_insertion.tip_tracking import (
    KeyAppearance, detect_visible_tip, track_tip_frames,
    tracked_tip_xy_alignment,
)
from precision_insertion.xy_overlay import CalibratedXYFrame


def test_visible_tip_follows_white_key_and_records_source_frames():
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    reference[80:300, 290:350] = 245
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:300, 290:350] = 255
    frame = np.zeros_like(reference)
    frame[80:305, 315:375] = 245
    appearance = KeyAppearance.from_reference(reference, mask)
    tip = detect_visible_tip(
        frame, appearance,
        projected_rear_px=(320, 80), projected_tip_px=(320, 300))
    assert tip is not None
    assert abs(tip.tip_px[0] - 345) < 2
    assert abs(tip.tip_px[1] - 299) < 3
    result = track_tip_frames(
        [(100, 1000, frame), (101, 1033, frame)], appearance,
        target_frame_id=101,
        projected_rear_px=(320, 80), projected_tip_px=(320, 300))
    assert result["status"] == "observed"
    assert [row["source_frame_id"] for row in result["observations"]] == [100, 101]
    assert [row["read_unix_ns"] for row in result["observations"]] == [1000, 1033]


def test_four_calibrated_tip_views_give_bounded_xy_increment():
    key_tip = np.array([.020, .005, .085])
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    views = []
    for index, (x, y) in enumerate(((-.12, 0.), (.12, 0.),
                                   (0., -.12), (0., .12))):
        T = np.eye(4)
        T[:3, 3] = [-x, -y, 1.]
        h = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        uv = tuple(h[:2] / h[2])
        frame = CalibratedXYFrame(
            str(index), float(index), Image.new("RGB", (640, 480)), T, K)
        views.append(GroundedLineView(frame, uv, None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=.085,
        search_index=0, max_reprojection_px=3., min_parallax_deg=5.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] == "diagnostic_metric_xy_correction"
    np.testing.assert_allclose(result["tip_socket_m"], key_tip, atol=1e-6)
    assert len(result["inlier_cameras"]) == 4
    assert np.linalg.norm(result["bounded_xy_increment_socket_m"]) <= .010000001


def test_centered_multiview_tip_has_no_lateral_step():
    key_tip = np.array([0., 0., .085])
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    views = []
    for index, (x, y) in enumerate(((-.12, 0.), (.12, 0.),
                                   (0., -.12), (0., .12))):
        T = np.eye(4)
        T[:3, 3] = [-x, -y, 1.]
        h = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        frame = CalibratedXYFrame(
            str(index), float(index), Image.new("RGB", (640, 480)), T, K)
        views.append(GroundedLineView(frame, tuple(h[:2] / h[2]), None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=.085, search_index=0,
        max_reprojection_px=3., min_parallax_deg=5.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert np.linalg.norm(result["bounded_xy_increment_socket_m"]) < 1e-9
    assert len(result["inlier_cameras"]) == 4


@pytest.mark.parametrize("outlier_px", [100., 500.])
def test_bad_tip_view_does_not_shift_the_multiview_xy_point(outlier_px):
    key_tip = np.array([.010, -.005, 1.])
    K = np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]])
    views = []
    for index, (x, y) in enumerate(((-.20, 0.), (-.10, .10),
                                    (0., -.10), (.10, .05), (.20, -.05))):
        T = np.eye(4)
        T[:3, 3] = [-x, -y, 0.]
        projected = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        uv = projected[:2] / projected[2]
        if index == 4:
            uv += [outlier_px, -outlier_px]
        frame = CalibratedXYFrame(
            str(index), 100., Image.new("RGB", (2048, 1536)), T, K)
        views.append(GroundedLineView(frame, tuple(uv), None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=1., search_index=0,
        max_reprojection_px=4., min_parallax_deg=1.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["inlier_cameras"] == ["0", "1", "2", "3"]
    np.testing.assert_allclose(result["tip_socket_m"], key_tip, atol=1e-6)


def test_pair_seed_does_not_turn_two_good_views_and_one_bad_view_into_metric():
    key_tip = np.array([.010, -.005, 1.])
    K = np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]])
    views = []
    for index, (x, y) in enumerate(((-.20, 0.), (-.10, .10), (.20, -.05))):
        T = np.eye(4)
        T[:3, 3] = [-x, -y, 0.]
        projected = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        uv = projected[:2] / projected[2]
        if index == 2:
            uv += [500., -500.]
        frame = CalibratedXYFrame(
            str(index), 100., Image.new("RGB", (2048, 1536)), T, K)
        views.append(GroundedLineView(frame, tuple(uv), None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=1., search_index=0,
        max_reprojection_px=4., min_parallax_deg=1.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] != "diagnostic_metric_xy_correction"
    assert result["inlier_cameras"] == []


def test_nearly_parallel_tip_views_do_not_claim_metric_depth():
    key_tip = np.array([.010, -.005, 1.])
    K = np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]])
    views = []
    for index, camera_x in enumerate((-.0001, 0., .0001)):
        T = np.eye(4)
        T[0, 3] = -camera_x
        projected = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        uv = tuple(projected[:2] / projected[2])
        frame = CalibratedXYFrame(
            str(index), 100., Image.new("RGB", (2048, 1536)), T, K)
        views.append(GroundedLineView(frame, uv, None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=1., search_index=0,
        max_reprojection_px=3., min_parallax_deg=5.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] != "diagnostic_metric_xy_correction"
    assert result["tracker_reason"] == "insufficient_camera_parallax"


def test_behind_camera_rays_do_not_become_metric_xy_advice():
    key_tip = np.array([.020, .005, .085])
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    views = []
    for index, camera_x in enumerate((-.12, 0., .12)):
        T = np.eye(4)
        T[:3, 3] = [-camera_x, 0., -1.]
        # Algebraic projection gives consistent pixels, but the point has
        # negative camera depth and cannot be visible in any of these views.
        projected = K @ (T[:3, :3] @ key_tip + T[:3, 3])
        frame = CalibratedXYFrame(
            str(index), 100., Image.new("RGB", (640, 480)), T, K)
        views.append(GroundedLineView(
            frame, tuple(projected[:2] / projected[2]), None))
    result = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=.085, search_index=0,
        max_reprojection_px=3., min_parallax_deg=5.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] == "abstain"
    assert "bounded_xy_increment_socket_m" not in result
    assert result["inlier_cameras"] == []
    assert result["tip_socket_m"] is None


def test_missing_tip_views_do_not_create_a_blind_next_trial_step():
    result = tracked_tip_xy_alignment(
        [], nominal_tip_z_socket_m=.085, search_index=2,
        max_reprojection_px=3., min_parallax_deg=5.,
        max_nominal_tip_height_error_m=.01)
    assert result["status"] == "abstain"
    assert result["reason"] == "no_consistent_visible_key_tip"
    assert result["xy_source"] == "none"
    assert result["inlier_cameras"] == []
    assert "bounded_xy_increment_socket_m" not in result


def test_missing_target_frame_does_not_invent_tip():
    reference = np.full((100, 100, 3), 240, dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    appearance = KeyAppearance.from_reference(reference, mask)
    result = track_tip_frames(
        [(1, 1, reference)], appearance, target_frame_id=10,
        projected_rear_px=(20, 20), projected_tip_px=(20, 80))
    assert result["status"] == "target_frame_missing"
    assert result["observations"] == []
