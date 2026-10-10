"""Measured-wrist depth remains a conditional key-tip hypothesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.conditional_wrist_depth import (  # noqa: E402
    conditional_key_tip_depth,
)
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.targets import InsertionTargets  # noqa: E402
from precision_insertion.uncertainty_margin import (  # noqa: E402
    SurfaceDeviationBounds,
)


def _setup(tmp_path, *, family="square", yaw=0., xy=(0., 0.)):
    mode = select_mode(family, 1.5 if family == "square" else 20.)
    rotation = np.diag([1., -1., -1.])
    def pose(height):
        value = np.eye(4)
        value[:3, :3] = rotation
        value[2, 3] = height
        return value
    entry, verification, preinsert = pose(.18), pose(.16), pose(.21)
    geometry = {
        "units": "m", "socket_pose_object": mode.socket_object,
        "key_object": mode.key_object,
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_verification": verification.tolist(),
        "T_socket_key_preinsert": preinsert.tolist(),
        "verification_insertion_depth_m": .020,
        "socket_entry_plane_z_m": .100,
        "insertion_direction_socket": [0., 0., -1.],
        "key_frame": {"insertion_axis": [0., 0., 1.], "tip_z_m": .080},
    }
    path = tmp_path / "task_geometry.json"
    path.write_text(json.dumps(geometry), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cosine, sine = np.cos(yaw), np.sin(yaw)
    gauge = np.array([[cosine, -sine, 0.],
                      [sine, cosine, 0.], [0., 0., 1.]])
    for transform in (entry, verification, preinsert):
        transform[:3, :3] = gauge @ transform[:3, :3]
        transform[:2, 3] += xy
    targets = InsertionTargets(
        mode, digest, "0" * 64, np.eye(4), preinsert, entry, verification,
        preinsert.copy(), entry.copy(), verification.copy(),
        np.array([0., 0., -1.]), xy, .03, yaw)
    kwargs = dict(
        mode=mode, targets=targets, task_geometry_path=path,
        T_robot_socket=np.eye(4),
        T_robot_hand_measured=verification,
        surface_bounds=SurfaceDeviationBounds(
            .001, .001, "commissioned_future_trial_surface_bound"))
    return kwargs


def test_exact_nominal_20mm_does_not_prove_physical_key_depth(tmp_path):
    result = conditional_key_tip_depth(**_setup(tmp_path))
    assert result["nominal_key_tip_depth_m"] == pytest.approx(.020)
    assert result["conditional_depth_interval_m"] == pytest.approx(
        [.019, .021])
    assert result["tip_lateral_residual_to_planned_entry_m"] == pytest.approx(0.)
    assert result["tip_offset_from_socket_axis_m"] == pytest.approx(0.)
    assert result["key_depth_source"] is None
    assert result["robot_ready"] is False


def test_conditional_depth_respects_cylinder_yaw_gauge(tmp_path):
    result = conditional_key_tip_depth(**_setup(
        tmp_path, family="cylinder", yaw=.7))
    assert result["nominal_key_tip_depth_m"] == pytest.approx(.020)
    assert result["key_axis_tilt_deg"] == pytest.approx(0.)


def test_planned_retry_offset_is_not_socket_axis_alignment(tmp_path):
    result = conditional_key_tip_depth(**_setup(tmp_path, xy=(.001, 0.)))
    assert result["tip_lateral_residual_to_planned_entry_m"] == pytest.approx(0.)
    assert result["tip_offset_from_socket_axis_m"] == pytest.approx(.001)


def test_wrong_cad_or_changed_target_is_rejected(tmp_path):
    kwargs = _setup(tmp_path)
    kwargs["task_geometry_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="CAD differs"):
        conditional_key_tip_depth(**kwargs)
    kwargs = _setup(tmp_path)
    kwargs["targets"].T_robot_key_entry[0, 3] = .01
    with pytest.raises(ValueError, match="entry target"):
        conditional_key_tip_depth(**kwargs)


def test_uncommissioned_surface_bound_is_rejected(tmp_path):
    kwargs = _setup(tmp_path)
    kwargs["surface_bounds"] = SurfaceDeviationBounds(
        .001, .001, "measured_MuJoCo_scatter")
    with pytest.raises(ValueError, match="future-trial surface"):
        conditional_key_tip_depth(**kwargs)
