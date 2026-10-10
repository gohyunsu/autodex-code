"""A nominal 20 mm pose cannot silently cover grasp/sensor uncertainty."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.targets import InsertionTargets  # noqa: E402
from precision_insertion.uncertainty_margin import (  # noqa: E402
    SurfaceDeviationBounds, audit_sampled_uncertainty_margins,
    relation_rotation_surface_bound,
)


def _inputs():
    mode = select_mode("square", 1.5)
    pose = np.eye(4)
    key_hash, socket_hash, geometry_hash = "a" * 64, "b" * 64, "c" * 64
    targets = InsertionTargets(
        mode=mode, task_geometry_sha256=geometry_hash,
        socket_collision_mesh_sha256=socket_hash, T_key_hand=pose,
        T_robot_key_preinsert=pose, T_robot_key_entry=pose,
        T_robot_key_verification=pose, T_robot_hand_preinsert=pose,
        T_robot_hand_entry=pose, T_robot_hand_verification=pose,
        insertion_axis_robot=np.array([0, 0, -1.0]),
        xy_offset_socket_m=(0, 0), preinsert_clearance_m=0.01)
    endpoint = {
        "schema": "precision_insertion_endpoint_screen_v1",
        "endpoint_pass": True,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "T_key_hand": pose.tolist(),
        "key_socket_fit": {"colliding": False,
                           "minimum_surface_distance_m": 0.002},
        "minimum_observed_hand_clearance_m": 0.006,
        "minimum_required_hand_clearance_m": 0.001,
        "input_sha256": {"key_mesh": key_hash, "socket_mesh": socket_hash,
                         "task_geometry": geometry_hash},
    }
    audit = {
        "schema": "precision_insertion_sampled_held_path_audit_v1",
        "sampled_clear": True,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "limits": {"minimum_hand_clearance_m": 0.001},
        "input_sha256": {"key_mesh": key_hash,
                         "mesh/fixture_socket": socket_hash,
                         "task_geometry": geometry_hash},
        "minimum_surface_distances_m": {
            "key->mesh/fixture_socket": 0.0015,
            "key->cuboid/table": 0.004,
            "hand/hand->mesh/fixture_socket": 0.005,
            "hand/hand->cuboid/table": 0.007,
        },
    }
    planning = InsertionPreflight(
        "sampled_planning_pass", None, None, None, audit, 1,
        np.zeros(6), "measured", ())
    bounds = SurfaceDeviationBounds(
        key_surface_m=0.001, hand_surface_m=0.002,
        source="commissioned_future_trial_surface_bound")
    return mode, endpoint, targets, planning, bounds


def test_nominal_path_needs_extra_clearance_for_future_surface_errors():
    mode, endpoint, targets, planning, bounds = _inputs()
    result = audit_sampled_uncertainty_margins(
        mode=mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    assert result["sampled_margin_pass"] is True
    assert result["robot_ready"] is False
    assert result["checks"]["path/key->mesh/fixture_socket"][
        "remaining_margin_m"] == pytest.approx(0.0005)


def test_endpoint_and_path_margins_reject_noncolliding_nominal_geometry():
    mode, endpoint, targets, planning, bounds = _inputs()
    endpoint["key_socket_fit"]["minimum_surface_distance_m"] = 0.0005
    planning.sampled_held_path_audit["minimum_surface_distances_m"][
        "hand/hand->mesh/fixture_socket"] = 0.002
    result = audit_sampled_uncertainty_margins(
        mode=mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    assert result["sampled_margin_pass"] is False
    assert result["checks"]["endpoint/key->socket"]["clear"] is False
    assert result["checks"]["path/hand/hand->mesh/fixture_socket"][
        "clear"] is False


def test_unmatched_sources_or_empirical_scatter_are_not_admitted():
    mode, endpoint, targets, planning, bounds = _inputs()
    with pytest.raises(ValueError, match="future-trial"):
        audit_sampled_uncertainty_margins(
            mode=mode, endpoint=endpoint, targets=targets, planning=planning,
            bounds=SurfaceDeviationBounds(0.001, 0.002, "empirical_grasp_scatter"))
    endpoint["input_sha256"]["socket_mesh"] = "d" * 64
    with pytest.raises(ValueError, match="sources disagree"):
        audit_sampled_uncertainty_margins(
            mode=mode, endpoint=endpoint, targets=targets,
            planning=planning, bounds=bounds)


def test_missing_key_socket_path_distance_fails_closed():
    mode, endpoint, targets, planning, bounds = _inputs()
    del planning.sampled_held_path_audit["minimum_surface_distances_m"][
        "key->mesh/fixture_socket"]
    with pytest.raises(ValueError, match="lacks key/socket"):
        audit_sampled_uncertainty_margins(
            mode=mode, endpoint=endpoint, targets=targets,
            planning=planning, bounds=bounds)


def test_rotation_bound_uses_entire_key_not_just_center():
    pose = np.eye(4)
    pose[0, 3] = 0.03
    vertices = np.array([[0.03, 0, 0], [0.03, 0, 0.08]])
    result = relation_rotation_surface_bound(
        T_key_hand=pose, key_vertices=vertices,
        relation_translation_bound_m=0.001,
        relation_rotation_bound_deg=1.0)
    assert result == pytest.approx(
        0.001 + 2 * 0.08 * np.sin(np.deg2rad(0.5)))
