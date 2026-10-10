"""Synthetic rear/axis depth geometry; not a physical VLM accuracy test."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.exposed_depth import (  # noqa: E402
    ExposedDepthLimits, estimate_exposed_depth,
    estimate_exposed_depth_for_mode,
)
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.grounded_alignment import (  # noqa: E402
    GroundedLineView, observe_exposed_key_rear_axis,
)
from test_grounded_alignment import _Backend, _line_rig  # noqa: E402


def _limits(**changes):
    values = dict(
        minimum_views=3, max_capture_skew_s=.03,
        min_rear_parallax_deg=5., max_reprojection_px=1.,
        max_axis_line_error_px=1., min_axis_plane_angle_deg=5.,
        max_axis_tilt_deg=4., rear_position_error_bound_m=.0001,
        axis_angle_error_bound_deg=.1,
        rim_height_error_bound_m=.0001,
        cad_tip_projection_error_bound_m=.0001)
    values.update(changes)
    return ExposedDepthLimits(**values)


def _estimate(rear_z, **changes):
    views = _line_rig(np.array([0., 0., rear_z]),
                      np.array([0., 0., -1.]))
    return estimate_exposed_depth(
        views, socket_rim_z_m=.055, key_rear_to_tip_m=.08,
        limits=_limits(**changes))


def test_nominal_20mm_does_not_claim_verified_20mm_with_uncertainty():
    result = _estimate(.115)
    assert result["status"] == "bounded_visual_depth"
    assert result["nominal_depth_m"] == pytest.approx(.020, abs=1e-6)
    assert result["key_depth_interval_m"][0] < .020
    assert result["key_depth_interval_m"][1] > .020
    assert result["robot_ready"] is False


def test_deeper_visible_rear_yields_depth_lower_bound_over_20mm():
    result = _estimate(.114)
    assert result["key_depth_interval_m"][0] > .020
    assert result["insertion_axis_socket"] == pytest.approx([0., 0., -1.],
                                                             abs=1e-6)
    assert len(result["inlier_cameras"]) == 4


def test_hidden_rear_stale_cameras_or_tilt_abstain():
    views = _line_rig(np.array([0., 0., .115]),
                      np.array([0., 0., -1.]))
    hidden = [GroundedLineView(v.frame, None, v.axis_line_uv_px)
              for v in views]
    assert estimate_exposed_depth(
        hidden, socket_rim_z_m=.055, key_rear_to_tip_m=.08,
        limits=_limits())["reason"] == (
            "rear_center_or_axis_not_visible_in_enough_views")
    stale = [replace(view, frame=replace(
        view.frame, timestamp_s=12.2)) if i == 0 else view
             for i, view in enumerate(views)]
    assert estimate_exposed_depth(
        stale, socket_rim_z_m=.055, key_rear_to_tip_m=.08,
        limits=_limits())["reason"] == (
            "multiview_exposures_not_synchronized")
    tilted = _line_rig(np.array([0., 0., .115]),
                       np.array([.25, 0., -.96824583655]))
    assert estimate_exposed_depth(
        tilted, socket_rim_z_m=.055, key_rear_to_tip_m=.08,
        limits=_limits())["reason"] == "key_axis_tilt_exceeds_limit"


def test_exposed_rear_vlm_uses_distinct_pixels_and_abstains_on_bad_json():
    frame = _line_rig(np.array([0., 0., .115]),
                      np.array([0., 0., -1.]))[0].frame
    answer = json.dumps({
        "rear_px": [600, 300], "axis_line_px": [[610, 290], [620, 280]],
        "evidence": "visible rear face"})
    views, records = observe_exposed_key_rear_axis(_Backend(answer), [frame])
    assert views[0].tip_uv_px == (600, 300)
    assert records[0].stage == "exposed_rear_axis_line_grounding"
    assert "rear_px" in records[0].prompt
    assert "opposite the socket" in records[0].prompt
    invalid, bad = observe_exposed_key_rear_axis(_Backend("not json"), [frame])
    assert invalid[0].tip_uv_px is None
    assert bad[0].parse_error is not None


def test_exposed_depth_rejects_uncommissioned_error_bounds():
    with pytest.raises(ValueError, match="commissioned positive bound"):
        _estimate(.115, axis_angle_error_bound_deg=0.)


@pytest.mark.parametrize("family,gap_mm,rim,length", [
    ("square", 1.5, .0585, .0855),
    ("cylinder", 15., .055, .080),
])
def test_depth_geometry_uses_same_v8_task_asset_as_planning(
        tmp_path, family, gap_mm, rim, length):
    mode = select_mode(family, gap_mm)
    path = AssetPaths(tmp_path, mode).task_geometry
    path.parent.mkdir(parents=True)
    rotation = np.diag([1., -1., -1.])
    entry = np.eye(4)
    entry[:3, :3] = rotation
    entry[2, 3] = rim + length
    verification = entry.copy()
    verification[2, 3] -= .020
    rear_name = ("handle_rear_z_m" if family == "square" else
                 "grasp_rear_z_m")
    geometry = {
        "units": "m", "socket_pose_object": mode.socket_object,
        "key_object": mode.key_object,
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_verification": verification.tolist(),
        "verification_insertion_depth_m": .020,
        "insertion_direction_socket": [0., 0., -1.],
        "key_frame": {"insertion_axis": [0., 0., 1.],
                      rear_name: 0., "tip_z_m": length},
        "socket_entry_plane_z_m": rim,
    }
    path.write_text(json.dumps(geometry), encoding="utf-8")
    rear = np.array([0., 0., rim + length - .021])
    views = _line_rig(rear, np.array([0., 0., -1.]))
    result = estimate_exposed_depth_for_mode(
        views, mode=mode, shared_root=tmp_path, limits=_limits())
    assert result["status"] == "bounded_visual_depth"
    assert result["nominal_depth_m"] == pytest.approx(.021, abs=1e-6)
    assert result["mode"]["socket_object"] == mode.socket_object
    geometry["socket_rim_z_m"] = rim + .001
    path.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="rim differs"):
        estimate_exposed_depth_for_mode(
            views, mode=mode, shared_root=tmp_path, limits=_limits())
