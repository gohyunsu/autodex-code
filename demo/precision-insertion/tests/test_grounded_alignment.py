"""Synthetic calibrated-camera tests; not a physical accuracy claim."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.grounded_alignment import (  # noqa: E402
    AlignmentLimits, GroundedView, GroundedLineView,
    estimate_grounded_alignment, estimate_grounded_line_alignment,
    observe_grounded_key_axis, observe_grounded_cylinder_axis,
)
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402


def _camera(name, center):
    center = np.asarray(center, dtype=float)
    focus = np.array([0, 0, 0.10])
    forward = (focus - center) / np.linalg.norm(focus - center)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    E = np.eye(4)
    E[:3, :3] = np.vstack((right, down, forward))
    E[:3, 3] = -E[:3, :3] @ center
    K = np.array([[1150.0, 0, 640], [0, 1150, 360], [0, 0, 1]])
    return CalibratedXYFrame(name, 12.0, Image.new("RGB", (1280, 720)), E, K)


def _pixel(frame, point):
    camera = frame.T_camera_socket[:3, :3] @ point + frame.T_camera_socket[:3, 3]
    projected = frame.intrinsics @ camera
    return tuple((projected[:2] / projected[2]).tolist())


def _rig(tip, ref, *, outlier=False):
    cameras = [
        _camera("front", (0.22, 0.00, 0.22)),
        _camera("side", (0.00, 0.22, 0.22)),
        _camera("back", (-0.22, 0.00, 0.22)),
    ]
    rows = [GroundedView(frame, _pixel(frame, tip), _pixel(frame, ref))
            for frame in cameras]
    if outlier:
        frame = rows[-1].frame
        rows[-1] = GroundedView(frame, (20.0, 40.0), (70.0, 90.0))
    return rows


def _limits(**overrides):
    values = dict(pixel_sigma_px=0.20, max_reprojection_px=1.0,
                  min_parallax_deg=5.0, max_axis_tilt_deg=4.0,
                  max_20mm_axis_sweep_m=0.0015,
                  max_lateral_uncertainty_95_m=0.0005,
                  systematic_lateral_sigma_m=0.00005,
                  cad_spacing_sigma_m=0.0003, minimum_views=2)
    values.update(overrides)
    return AlignmentLimits(**values)


def _estimate(rows, **kwargs):
    return estimate_grounded_alignment(
        rows, landmark_spacing_m=0.05, socket_rim_z_m=0.055,
        verification_depth_m=0.02, limits=_limits(), **kwargs)


def test_continuous_xy_offset_comes_from_triangulated_tip_and_axis():
    tip = np.array([0.0016, -0.0002, 0.09])
    ref = tip + [0, 0, 0.05]
    result = _estimate(_rig(tip, ref))
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["xy_correction_socket_m"] == pytest.approx(
        [-.0016, .0002], abs=1e-7)
    assert np.linalg.norm(result["bounded_xy_increment_socket_m"]) == (
        pytest.approx(.001))
    assert result["bounded_xy_increment_socket_m"][1] > 0
    assert result["mean_error_xy_m"] == pytest.approx(tip[:2], abs=1e-7)
    assert result["robot_ready"] is False


def test_wrong_view_landmark_is_rejected_not_averaged():
    tip = np.array([0.0017, 0, 0.09])
    result = _estimate(_rig(tip, tip + [0, 0, 0.05], outlier=True))
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["rejected_cameras"] == ["back"]
    assert result["xy_correction_socket_m"] == pytest.approx(
        [-.0017, 0.], abs=1e-7)


def test_one_point_only_cannot_determine_axis():
    tip = np.array([0.0016, 0, 0.09])
    rows = [GroundedView(v.frame, v.tip_uv_px, None)
            for v in _rig(tip, tip + [0, 0, 0.05])]
    assert _estimate(rows)["reason"] == (
        "insufficient_visible_corresponding_axis_landmarks")


def test_tilt_and_square_yaw_cannot_be_solved_by_xy():
    tip = np.array([0.002, 0, 0.09])
    ref = tip + np.array([0.012, 0, 0.05]) * (
        0.05 / np.linalg.norm([0.012, 0, 0.05]))
    assert _estimate(_rig(tip, ref))["reason"] == (
        "axis_tilt_requires_reorientation_not_xy")
    assert _estimate(_rig(tip, tip + [0, 0, 0.05]),
                     yaw_relevant=True)["reason"] == (
        "two_collinear_landmarks_cannot_estimate_square_key_yaw")


def test_tiny_error_does_not_justify_even_a_continuous_increment():
    tip = np.array([0.0002, 0, 0.09])
    assert _estimate(_rig(tip, tip + [0, 0, 0.05]))["reason"] == (
        "continuous_xy_correction_not_confident")


def test_submillimetre_offset_is_not_forced_into_a_cardinal_step():
    # This was incorrectly rejected when forced into a full 1 mm move.
    # Continuous XY uses the observed error; only an actual execution
    # increment exceeding 1 mm would be capped.
    tip = np.array([.00072, 0., .09])
    result = estimate_grounded_alignment(
        _rig(tip, tip + [0, 0, .05]), landmark_spacing_m=.05,
        socket_rim_z_m=.055, verification_depth_m=.02,
        limits=_limits(systematic_lateral_sigma_m=.0001))
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["xy_correction_socket_m"] == pytest.approx(
        [-.00072, 0.], abs=1e-7)
    assert result["bounded_xy_increment_socket_m"] == pytest.approx(
        [-.00072, 0.], abs=1e-7)
    assert result["increment_squared_error_improvement_lower_95_m2"] > 0


def test_no_parallax_and_uncertain_calibration_abstain():
    tip = np.array([0.0016, 0, 0.09])
    ref = tip + [0, 0, 0.05]
    rows = _rig(tip, ref)
    close_frames = [_camera("near_a", (.22, 0, .22)),
                    _camera("near_b", (.221, 0, .22))]
    close_rows = [GroundedView(f, _pixel(f, tip), _pixel(f, ref))
                  for f in close_frames]
    assert estimate_grounded_alignment(
        close_rows, landmark_spacing_m=.05, socket_rim_z_m=.055,
        verification_depth_m=.02,
        limits=_limits(min_parallax_deg=5.0))["reason"] == (
            "insufficient_multiview_parallax")
    assert estimate_grounded_alignment(
        rows, landmark_spacing_m=.05, socket_rim_z_m=.055,
        verification_depth_m=.02,
        limits=_limits(systematic_lateral_sigma_m=.001))["reason"] == (
            "lateral_uncertainty_exceeds_budget")


class _Backend:
    def __init__(self, answer):
        self.answer = answer

    def infer(self, images, prompt):
        assert len(images) == 1
        assert "UNDISTORTED" in prompt
        assert "overlay" in prompt
        return self.answer


def test_vlm_returns_only_validated_raw_image_pixels():
    frame = _rig(np.array([0, 0, .09]), np.array([0, 0, .14]))[0].frame
    response = json.dumps({"tip_px": [600, 300],
                           "axis_ref_px": [610, 290], "evidence": "rim"})
    views, observations = observe_grounded_key_axis(_Backend(response), [frame])
    assert views[0].tip_uv_px == (600, 300)
    assert observations[0].parse_error is None
    invalid, bad = observe_grounded_key_axis(_Backend(response.replace(
        "600", "99999")), [frame])
    assert invalid[0].tip_uv_px is None
    assert bad[0].parse_error is not None


def _line_rig(tip, axis, *, outlier=False):
    frames = [_camera("front", (.22, 0, .22)),
              _camera("side", (0, .22, .22)),
              _camera("back", (-.22, 0, .22)),
              _camera("other", (0, -.22, .22))]
    rows = [GroundedLineView(
        frame, _pixel(frame, tip),
        (_pixel(frame, tip - .012 * axis),
         _pixel(frame, tip + .03 * axis))) for frame in frames]
    if outlier:
        row = rows[-1]
        rows[-1] = GroundedLineView(
            row.frame, (20., 20.), ((100., 90.), (150., 130.)))
    return rows


def _line_estimate(rows, **limit_overrides):
    return estimate_grounded_line_alignment(
        rows, socket_rim_z_m=.055, verification_depth_m=.02,
        limits=_limits(minimum_views=3, **limit_overrides))


def test_cylinder_tip_plus_visible_axis_line_needs_no_second_physical_point():
    tip = np.array([.0018, -.0002, .09])
    result = _line_estimate(_line_rig(tip, np.array([0, 0, -1.])))
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["xy_correction_socket_m"] == pytest.approx(
        [-.0018, .0002], abs=1e-6)
    assert np.linalg.norm(result["bounded_xy_increment_socket_m"]) == (
        pytest.approx(.001))
    assert result["mean_error_xy_m"] == pytest.approx(tip[:2], abs=1e-6)
    assert len(result["inlier_cameras"]) == 4


def test_cylinder_line_proposes_continuous_submillimetre_correction():
    tip = np.array([.00072, 0., .09])
    result = _line_estimate(
        _line_rig(tip, np.array([0, 0, -1.])),
        systematic_lateral_sigma_m=.0001)
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["bounded_xy_increment_socket_m"] == pytest.approx(
        [-.00072, 0.], abs=1e-6)


def test_line_estimator_rejects_bad_view_and_tilt():
    tip = np.array([.0018, 0, .09])
    result = _line_estimate(_line_rig(tip, np.array([0, 0, -1.]),
                                      outlier=True))
    assert result["status"] == "diagnostic_metric_xy_correction"
    assert result["rejected_cameras"] == ["other"]
    axis = np.array([.25, 0., -.96824583655])
    assert _line_estimate(_line_rig(tip, axis))["reason"] == (
        "axis_tilt_requires_reorientation_not_xy")


def test_line_grounding_abstains_if_shaft_or_tip_hidden():
    tip = np.array([.0018, 0, .09])
    rows = _line_rig(tip, np.array([0, 0, -1.]))
    missing = [GroundedLineView(v.frame, v.tip_uv_px, None) for v in rows]
    assert _line_estimate(missing)["reason"] == (
        "insufficient_visible_tip_and_shaft_axis_views")
    response = json.dumps({"tip_px": [600, 300], "axis_line_px": None,
                           "evidence": "shaft hidden by fingers"})
    grounded, records = observe_grounded_cylinder_axis(
        _Backend(response), [rows[0].frame])
    assert grounded[0].axis_line_uv_px is None
    assert records[0].parse_error is None


def test_metric_grounding_rejects_a_resized_local_backend():
    class ResizedBackend:
        native_pixel_coordinates = False

        def infer(self, images, prompt):
            raise AssertionError("grounding must reject before inference")

    frame = _line_rig(np.array([0., 0., .09]),
                      np.array([0., 0., -1.]))[0].frame
    with pytest.raises(ValueError, match="forbids local VLM image resizing"):
        observe_grounded_cylinder_axis(ResizedBackend(), [frame])
