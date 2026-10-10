"""Conservative *sampled* clearance audit for an uncertain held grasp.

Distances already come from the exact 20 mm endpoint and sampled held-path
audits. This module does not estimate future pickup error from empirical
scatter, certify a sensor, check between-sample swept volumes, or command a
robot. Its surface-deviation bounds must be commissioned independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .config import TaskMode
from .geometry import validate_se3
from .preflight import InsertionPreflight
from .targets import InsertionTargets


@dataclass(frozen=True)
class SurfaceDeviationBounds:
    """Future-trial worst-case input, not a value learned by this module.

    Each bound is the maximum *relative to the frozen fixture* displacement
    of any point of that moving mesh on the whole relevant motion, including
    calibration, grasp repeatability/slip, FK/controller tracking, finger
    geometry (for hand) and socket/CAD errors. A joint-angle or key-origin
    bound alone is insufficient. These inputs require physical commissioning.
    """

    key_surface_m: float
    hand_surface_m: float
    source: str

    def validate(self) -> None:
        if self.source != "commissioned_future_trial_surface_bound":
            raise ValueError("empirical grasp scatter is not a future-trial surface bound")
        for name in ("key_surface_m", "hand_surface_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def relation_rotation_surface_bound(
    *, T_key_hand: np.ndarray, key_vertices: np.ndarray,
    relation_translation_bound_m: float, relation_rotation_bound_deg: float,
) -> float:
    """Bound key-point shift from an uncertain T_key_hand at fixed wrist.

    With ``T_key_hand = [R,t]``, a key point ``v`` is ``R.T @ (v-t)`` in hand
    coordinates. For ||delta t||<=e and rotation angle<=a, its displacement
    is at most ``e + 2 max||v-t|| sin(a/2)``. Wrist/socket/finger errors must
    be added *separately* before comparing with fixture clearance.
    """
    transform = validate_se3(T_key_hand, name="calibrated T_key_hand")
    vertices = np.asarray(key_vertices, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or
            not np.all(np.isfinite(vertices))):
        raise ValueError("key vertices must be finite metric xyz points")
    translation = float(relation_translation_bound_m)
    rotation = float(relation_rotation_bound_deg)
    if (not math.isfinite(translation) or translation < 0 or
            not math.isfinite(rotation) or not 0 <= rotation <= 180):
        raise ValueError("relation error bounds must be finite and nonnegative")
    radius = float(np.max(np.linalg.norm(vertices - transform[:3, 3], axis=1)))
    return translation + 2.0 * radius * math.sin(math.radians(rotation) / 2.0)


def audit_sampled_uncertainty_margins(
    *, mode: TaskMode, endpoint: Mapping[str, Any],
    targets: InsertionTargets, planning: InsertionPreflight,
    bounds: SurfaceDeviationBounds,
) -> dict:
    """Require nominal clearances to exceed explicit future surface errors.

    This is a necessary condition for the *sampled* nominal geometry only.
    A passing result is never sufficient for physical transfer or insertion.
    """
    bounds.validate()
    if not isinstance(targets, InsertionTargets) or targets.mode != mode:
        raise ValueError("uncertainty audit needs matching rigid task targets")
    if (not isinstance(planning, InsertionPreflight) or
            not planning.sampled_planning_pass or
            not isinstance(endpoint, Mapping) or
            endpoint.get("schema") != "precision_insertion_endpoint_screen_v1" or
            endpoint.get("endpoint_pass") is not True):
        raise ValueError("uncertainty audit needs passing exact endpoint and path")
    if endpoint.get("mode") != {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
    }:
        raise ValueError("endpoint was screened for a different task mode")
    if not np.allclose(validate_se3(endpoint.get("T_key_hand"),
                                    name="screened T_key_hand"),
                       targets.T_key_hand, atol=1e-8, rtol=0):
        raise ValueError("endpoint and planned held relation disagree")
    audit = planning.sampled_held_path_audit
    if (not isinstance(audit, dict) or audit.get("sampled_clear") is not True or
            audit.get("schema") != "precision_insertion_sampled_held_path_audit_v1" or
            audit.get("mode") != {"family": mode.family, "gap_mm": mode.gap_mm}):
        raise ValueError("held path lacks a matching passing geometry audit")
    endpoint_hashes = endpoint.get("input_sha256", {})
    path_hashes = audit.get("input_sha256", {})
    if (not isinstance(endpoint_hashes, dict) or
            not isinstance(path_hashes, dict) or
            endpoint_hashes.get("key_mesh") != path_hashes.get("key_mesh") or
            endpoint_hashes.get("task_geometry") !=
            path_hashes.get("task_geometry") or
            endpoint_hashes.get("socket_mesh") !=
            path_hashes.get("mesh/fixture_socket") or
            endpoint_hashes.get("task_geometry") !=
            targets.task_geometry_sha256 or
            endpoint_hashes.get("socket_mesh") !=
            targets.socket_collision_mesh_sha256 or
            any(not isinstance(value, str) or len(value) != 64
                for value in (endpoint_hashes.get("key_mesh"),
                              endpoint_hashes.get("task_geometry"),
                              endpoint_hashes.get("socket_mesh")))):
        raise ValueError("endpoint, path and target CAD sources disagree")

    distances = audit.get("minimum_surface_distances_m")
    if (not isinstance(distances, dict) or not distances or
            "key->mesh/fixture_socket" not in distances or
            not any(name.startswith("hand/") for name in distances)):
        raise ValueError("held path lacks key/socket or hand/world distances")
    key_fit = endpoint.get("key_socket_fit", {})
    endpoint_hand = endpoint.get("minimum_observed_hand_clearance_m")
    endpoint_required_hand = endpoint.get("minimum_required_hand_clearance_m")
    path_required_hand = audit.get("limits", {}).get("minimum_hand_clearance_m")
    if (not isinstance(key_fit, dict) or
            key_fit.get("colliding") is not False):
        raise ValueError("endpoint has no valid key/socket clearance")

    checks: dict[str, dict] = {}

    def add(name: str, distance: object, required: float) -> None:
        try:
            measured = float(distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing sampled distance: {name}") from exc
        if not math.isfinite(measured) or measured < 0:
            raise ValueError(f"invalid sampled distance: {name}")
        checks[name] = {
            "nominal_distance_m": measured,
            "required_with_uncertainty_m": required,
            "remaining_margin_m": measured - required,
            "clear": measured > required,
        }

    for value, name in ((endpoint_required_hand, "endpoint hand clearance"),
                        (path_required_hand, "path hand clearance")):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
    add("endpoint/key->socket", key_fit.get("minimum_surface_distance_m"),
        bounds.key_surface_m)
    add("endpoint/hand->socket", endpoint_hand,
        float(endpoint_required_hand) + bounds.hand_surface_m)
    for pair, distance in sorted(distances.items()):
        if pair.startswith("key->"):
            required = bounds.key_surface_m
        elif pair.startswith("hand/"):
            required = float(path_required_hand) + bounds.hand_surface_m
        else:
            raise ValueError(f"unknown moving geometry in path audit: {pair}")
        add(f"path/{pair}", distance, required)
    passed = all(row["clear"] for row in checks.values())
    return {
        "schema": "precision_insertion_sampled_uncertainty_margin_v1",
        "status": "sampled_margin_pass_not_robot_ready" if passed else
                  "sampled_margin_rejected",
        "sampled_margin_pass": passed,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "bounds": {"key_surface_m": bounds.key_surface_m,
                   "hand_surface_m": bounds.hand_surface_m,
                   "source": bounds.source},
        "checks": checks,
        "not_validated": [
            "physical authenticity or coverage of supplied future-trial bounds",
            "continuous swept geometry between trajectory samples",
            "actual grip slip, controller tracking, guarded contact or task success",
        ],
        "scope": "sampled_clearance_diagnostic_not_robot_motion_authorization",
        "robot_ready": False,
    }
