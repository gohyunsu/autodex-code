"""1 mm VLM overlays must resolve in original calibrated pixels."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.xy_overlay import (  # noqa: E402
    CalibratedXYFrame, build_xy_candidate_overlays, camera_socket_transform,
)
from precision_insertion.xy_voting import axis_1mm_proposals  # noqa: E402


def _frame(camera_id: str, focal: float) -> CalibratedXYFrame:
    T = np.eye(4)
    T[2, 3] = 1.0
    K = np.array([[focal, 0, 500], [0, focal, 400], [0, 0, 1]])
    return CalibratedXYFrame(
        camera_id, 100.0, Image.new("RGB", (1000, 800), (70, 80, 90)), T, K)


def test_resolvable_two_view_overlays_preserve_raw_and_order():
    choices = axis_1mm_proposals((0.0, 0.0))
    result = build_xy_candidate_overlays(
        choices=choices, frames=(_frame("a", 4000), _frame("b", 4000)),
        socket_plane_z_m=0.0, minimum_anchor_separation_px=3.0,
        crop_width_px=320)
    assert result.status == "views_ready"
    assert len(result.views) == 2
    assert result.choice_ids == tuple(choice.choice_id for choice in choices)
    assert result.per_camera["a"]["minimum_pairwise_separation_original_px"] == (
        pytest.approx(4.0))
    assert result.views[0].raw.size == (640, 360)
    assert result.views[0].overlay.tobytes() != result.views[0].raw.tobytes()
    assert result.to_record()["robot_ready"] is False


def test_upscaling_cannot_rescue_unresolved_one_mm_motion():
    result = build_xy_candidate_overlays(
        choices=axis_1mm_proposals((0.0, 0.0)),
        frames=(_frame("a", 1000), _frame("b", 1000)),
        socket_plane_z_m=0.0, minimum_anchor_separation_px=2.0,
        crop_width_px=320, display_scale=10)
    assert result.status == "insufficient_pixel_resolvable_views"
    assert not result.views
    assert result.per_camera["a"]["reason"] == (
        "one_mm_candidates_not_pixel_resolvable")


def test_camera_transform_uses_world_to_camera_then_robot_to_world():
    T_camera_world = np.eye(4)
    T_camera_world[:3, 3] = [1, 0, 0]
    T_world_robot = np.eye(4)
    T_world_robot[:3, 3] = [0, 2, 0]
    T_robot_socket = np.eye(4)
    T_robot_socket[:3, 3] = [0, 0, 3]
    result = camera_socket_transform(
        T_camera_world=T_camera_world, T_world_robot=T_world_robot,
        T_robot_socket=T_robot_socket)
    np.testing.assert_allclose(result[:3, 3], [1, 2, 3])


def test_rejects_duplicate_views_and_bad_pixel_budget():
    choices = axis_1mm_proposals((0.0, 0.0))
    with pytest.raises(ValueError, match="unique"):
        build_xy_candidate_overlays(
            choices=choices, frames=(_frame("a", 4000), _frame("a", 4000)),
            socket_plane_z_m=0, minimum_anchor_separation_px=2,
            crop_width_px=320)
    with pytest.raises(ValueError, match="positive pixel budget"):
        build_xy_candidate_overlays(
            choices=choices, frames=(_frame("a", 4000),),
            socket_plane_z_m=0, minimum_anchor_separation_px=0,
            crop_width_px=320)
