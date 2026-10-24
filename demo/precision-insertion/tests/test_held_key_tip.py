"""Geometry contracts for the read-only VLM-box / SAM3 key-end estimate."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.held_key_tip import (  # noqa: E402
    distal_key_mask_boundary_center, distal_key_mask_boundary_fit,
    distal_key_mask_edge_midpoint, distal_key_mask_half_ellipse,
    padded_key_box_roi,
    parse_normalized_key_box, projected_key_axis_crop,
    triangulate_tip_proxy,
)


def test_normalized_qwen_box_maps_to_original_crop_pixels():
    answer = '```json\n{"bbox":[100,250,750,900],"label":"key"}\n```'
    assert parse_normalized_key_box(
        answer, canvas_size_wh=(512, 448)) == (51.2, 112., 384., 403.2)
    assert padded_key_box_roi(
        (51.2, 112., 384., 403.2), canvas_size_wh=(512, 448)) == (
            26, 87, 409, 428)
    assert parse_normalized_key_box(
        '{"bbox":[100,1100,750,900]}', canvas_size_wh=(512, 448)) is None


def test_key_axis_crop_preserves_camera_pixels_and_planned_direction():
    K = np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, 3] = [0., 0., 1.]
    # Local +Z points toward image +Y at fixed camera depth.
    T[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    roi, rear, tip = projected_key_axis_crop(
        image_size_wh=(2048, 1536), intrinsics=K,
        T_camera_robot=np.eye(4), T_robot_key=T, tip_z_m=.08)
    assert rear == (1024., 768.)
    assert tip == (1024., 848.)
    assert roi == (768, 552, 1280, 1064)


def test_mask_endpoint_uses_planned_axis_only_to_choose_distal_end():
    mask = np.zeros((220, 180), dtype=np.uint8)
    mask[40:200, 60:120] = 255
    point = distal_key_mask_edge_midpoint(
        mask, planned_rear_px=(590., 740.),
        planned_tip_px=(590., 900.), mask_origin_px=(500., 700.))
    assert np.allclose(point, (589.5, 899.))
    reversed_point = distal_key_mask_edge_midpoint(
        mask, planned_rear_px=(590., 900.),
        planned_tip_px=(590., 740.), mask_origin_px=(500., 700.))
    assert np.allclose(reversed_point, (589.5, 740.))
    assert distal_key_mask_edge_midpoint(
        np.zeros((20, 20), dtype=np.uint8), planned_rear_px=(0., 0.),
        planned_tip_px=(0., 10.)) is None


def test_two_real_camera_rays_recover_a_spatial_proxy():
    K = np.array([[1100., 0., 1024.], [0., 1100., 768.], [0., 0., 1.]])
    point = np.array([.02, -.01, .5])
    observations = []
    for name, center in (("left", -.08), ("right", .08)):
        E = np.eye(4)
        E[0, 3] = -center
        P = K @ E[:3]
        h = P @ np.r_[point, 1.]
        observations.append((name, P, h[:2]/h[2]))
    result = triangulate_tip_proxy(observations)
    assert np.allclose(result["point_socket_m"], point, atol=1e-7)
    assert max(result["reprojection_error_px"].values()) < 1e-6


def test_distal_mask_boundary_fit_uses_visible_end_not_whole_mask_box():
    mask = np.zeros((240, 180), dtype=np.uint8)
    for x in range(50, 130):
        # A tilted real mask boundary plus one spur that should not set the
        # center of the end face.
        end = int(190 + .2 * (x-90))
        mask[30:end+1, x] = 255
    mask[200:225, 54] = 255
    fit = distal_key_mask_boundary_fit(
        mask, planned_rear_px=(90., 30.), planned_tip_px=(90., 200.))
    assert fit is not None
    assert len(fit["boundary_samples_full_px"]) >= 5
    assert np.allclose(fit["center_full_px"], (89.5, 190.), atol=3.)
    assert np.allclose(distal_key_mask_boundary_center(
        mask, planned_rear_px=(90., 30.), planned_tip_px=(90., 200.)),
        fit["center_full_px"])


def test_sam3_half_ellipse_recovers_center_of_synthetic_round_end():
    mask = np.zeros((240, 180), dtype=np.uint8)
    for x in range(50, 130):
        lateral = (x-89.5)/40.
        far = int(180 + 12*np.sqrt(max(0., 1-lateral*lateral)))
        mask[25:far+1, x] = 255
    fit = distal_key_mask_half_ellipse(
        mask, planned_rear_px=(90., 25.), planned_tip_px=(90., 192.))
    assert fit is not None
    assert np.allclose(fit["center_full_px"], (89.5, 180.), atol=4.)
    assert fit["major_radius_px"] > fit["minor_radius_px"] > 0


def test_sam3_half_ellipse_ignores_disconnected_white_hand_patch():
    mask = np.zeros((240, 180), dtype=np.uint8)
    for x in range(50, 130):
        lateral = (x-89.5)/40.
        far = int(180 + 12*np.sqrt(max(0., 1-lateral*lateral)))
        mask[45:far+1, x] = 255
    clean = distal_key_mask_half_ellipse(
        mask, planned_rear_px=(90., 45.), planned_tip_px=(90., 192.))
    mask[0:40, 0:40] = 255
    with_patch = distal_key_mask_half_ellipse(
        mask, planned_rear_px=(90., 45.), planned_tip_px=(90., 192.))
    assert clean is not None and with_patch is not None
    assert np.allclose(clean["center_full_px"],
                       with_patch["center_full_px"], atol=1e-6)
