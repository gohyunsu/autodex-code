"""Cylinder frame ambiguity must not be mistaken for post-squeeze slip."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.held_relation import (  # noqa: E402
    resolve_postlift_held_relation,
)


def _cylinder_symmetry(tmp_path: Path, mode):
    path = (tmp_path / "object_processing" / mode.key_object /
            "processed_data" / "info" / "symmetry.json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "type": "Dinf", "center": [0, 0, 0.04],
        "axes": [{"axis": [0, 0, 1], "fold": "inf"},
                 {"axis": [1, 0, 0], "fold": 2}],
    }), encoding="utf-8")


def _around_center(rotation, center=(0, 0, 0.04)):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    c = np.asarray(center, dtype=float)
    transform[:3, 3] = c - rotation @ c
    return transform


@pytest.mark.parametrize("flip", [False, True])
def test_cylinder_axial_yaw_and_end_exchange_preserve_held_relation(
    tmp_path, flip,
):
    mode = select_mode("cylinder", 20)
    _cylinder_symmetry(tmp_path, mode)
    yaw = np.deg2rad(123)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    if flip:
        rotation = rotation @ np.diag([1, -1, -1])
    observed = _around_center(rotation)
    nominal = np.eye(4)
    nominal[:3, 3] = [0.025, 0.0, 0.015]
    result = resolve_postlift_held_relation(
        mode=mode, shared_root=tmp_path,
        T_robot_key_observed=observed,
        T_robot_hand_measured=nominal,
        candidate_T_key_hand=nominal,
        max_translation_drift_m=0.003,
        max_rotation_drift_deg=5.0)
    assert result.translation_drift_m < 1e-9
    assert result.rotation_drift_deg < 1e-6
    assert result.symmetry_branch == (
        "end_exchange_and_axial_yaw" if flip else "axial_yaw")
    np.testing.assert_allclose(result.center_in_robot_m, [0, 0, 0.04],
                               atol=1e-9)
    np.testing.assert_allclose(result.T_key_hand, nominal, atol=1e-9)


def test_real_center_shift_is_not_removed_by_cylinder_symmetry(tmp_path):
    mode = select_mode("cylinder", 20)
    _cylinder_symmetry(tmp_path, mode)
    key = np.eye(4)
    key[0, 3] = 0.006
    result = resolve_postlift_held_relation(
        mode=mode, shared_root=tmp_path,
        T_robot_key_observed=key,
        T_robot_hand_measured=np.eye(4),
        candidate_T_key_hand=np.eye(4),
        max_translation_drift_m=0.003,
        max_rotation_drift_deg=5.0)
    assert result.translation_drift_m == pytest.approx(0.006)


def test_square_uses_full_pose_not_cylinder_equivalence(tmp_path):
    mode = select_mode("square", 1.5)
    key = np.eye(4)
    key[0, 3] = 0.004
    result = resolve_postlift_held_relation(
        mode=mode, shared_root=tmp_path,
        T_robot_key_observed=key,
        T_robot_hand_measured=np.eye(4),
        candidate_T_key_hand=np.eye(4),
        max_translation_drift_m=0.003,
        max_rotation_drift_deg=5.0)
    assert result.translation_drift_m == pytest.approx(0.004)
    assert result.symmetry_branch == "identity"
