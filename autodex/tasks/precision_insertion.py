"""Evidence-gated insertion scenario selection and bounded retry decisions.

This module does not command a robot.  Its output is a *proposed* correction in
the frozen socket frame; a commissioned insertion controller must check and
execute any trajectory.  Grasp stability and endpoint geometry are deliberately
not promoted to full-task simulation or hardware success.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


VALIDATION_ORDER = (
    "geometry_only", "sampled_insertion_geometry", "grasp_sim_pass",
    "full_task_sim_pass", "hardware_validated",
)


@dataclass(frozen=True)
class InsertionMode:
    name: str
    key: str
    socket: str
    socket_gap_mm: float
    yaw_relevant: bool
    target_depth_mm: float = 20.0


def mode_config(mode: str, *, gap_mm: float | None = None) -> InsertionMode:
    if mode == "square":
        gap = 1.5 if gap_mm is None else float(gap_mm)
        if gap not in (0.1, 0.3, 0.5, 1.0, 1.5):
            raise ValueError("square gap must be 0.1, 0.3, 0.5, 1.0 or 1.5 mm")
        label = str(gap).replace(".", "p")
        return InsertionMode(mode, f"precision_key_{label}mm",
                             "precision_socket_unified", gap, True)
    if mode == "cylinder":
        gap = 20.0 if gap_mm is None else float(gap_mm)
        if gap not in (1, 3, 5, 10, 15, 20):
            raise ValueError("cylinder radial gap must be 1, 3, 5, 10, 15 or 20 mm")
        return InsertionMode(mode, "precision_key_cylinder_r15_h80",
                             f"precision_socket_cylinder_gap_{int(gap):02d}mm",
                             gap, False)
    raise ValueError("mode must be 'square' or 'cylinder'")


def _scenario(id_: str, mode: InsertionMode, pose: int, path: Path,
              level: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": id_, "mode": mode.name, "key": mode.key, "socket": mode.socket,
        "gap_mm": mode.socket_gap_mm, "tabletop_pose": pose,
        "candidate_dir": str(path), "validation_level": level,
        "evidence": dict(evidence), "full_task_sim_pass": False,
        "hardware_ready": False,
    }


def build_catalog(shared_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Index verified historical grasp-sim passes, never invent insertion passes."""
    square = mode_config("square")
    root = (shared_root / "AutoDex" / "sim_filter_pass" / "inspire" /
            "precision_insertion_unconstrained_20mm_10k_top50" /
            square.key / "table" / "4")
    passes = report["curobo_mujoco_pilot"]["mujoco_gravity_stability_pass_ids"]
    scenarios = []
    missing = []
    for candidate_id in passes:
        candidate = root / str(candidate_id)
        expected = ("grasp_pose.npy", "pregrasp_pose.npy", "wrist_se3.npy")
        if not all((candidate / name).is_file() for name in expected):
            missing.append(str(candidate))
            continue
        scenarios.append(_scenario(
            f"square-pose004-grasp{candidate_id}", square, 4, candidate,
            "grasp_sim_pass", {
                "sampled_endpoint_20mm_geometry": True,
                "curobo_tabletop_scene_clearance": True,
                "mujoco_gravity_stability": True,
                "declared_contact_above_handle_front": str(candidate_id) not in
                report["curobo_mujoco_pilot"]["successes_without_declared_contact_above_handle_front_ids"],
                "whole_hand_continuous_insertion": "not_tested",
                "socket_contact_dynamics": "not_tested",
                "fixed_fixture_franka_transfer": "not_tested",
            }))
    cylinder_scenarios = []
    new_manifest = (shared_root / "AutoDex" / "precision_insertion" /
                    "cylindrical" / "asset_manifest.json")
    old_manifest = shared_root / "AutoDex" / "precision_insertion" / "cylindrical_assets.json"
    manifest_path = new_manifest if new_manifest.is_file() else old_manifest
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if "mode" in manifest and (
                manifest["mode"] != "cylinder" or
                manifest.get("verification_depth_m") != 0.020):
            raise ValueError("unexpected cylindrical asset manifest")
        for socket in manifest["sockets"]:
            gap = float(socket.get("radial_clearance_mm", socket.get("gap_mm")))
            mode = mode_config("cylinder", gap_mm=gap)
            socket_name = socket.get("object", socket.get("name"))
            collision = socket.get("collision_mesh") or str(
                shared_root / "object_processing" / mode.socket / "processed_data" /
                "mesh" / "static_collision.obj")
            if socket_name != mode.socket or not Path(collision).is_file():
                raise ValueError(f"cylinder socket geometry missing/mismatched: {mode.socket}")
            cylinder_scenarios.append({
                "id": f"cylinder-gap{int(gap):02d}-pose000-axial-fit",
                "mode": "cylinder", "key": mode.key, "socket": mode.socket,
                "gap_mm": gap, "tabletop_pose": 0, "candidate_dir": None,
                "validation_level": "geometry_only", "full_task_sim_pass": False,
                "hardware_ready": False,
                "evidence": {"nominal_axial_fit_20mm": True,
                             "grasp_candidate": "not_generated",
                             "franka_transfer": "not_tested",
                             "socket_contact_dynamics": "not_tested"},
            })
    scenarios.extend(cylinder_scenarios)
    return {
        "schema_version": 1,
        "definition": "An inventory of evidence, not a list of insertion successes.",
        "modes": {
            "square": {"default_gap_mm": 1.5, "scenario_count": len(scenarios) - len(cylinder_scenarios)},
            "cylinder": {"default_gap_mm": 20,
                         "scenario_count": len(cylinder_scenarios),
                         "reason": "Only nominal CAD fit; BODex and insertion simulation are pending."},
        },
        "scenarios": scenarios,
        "missing_candidate_directories": missing,
    }


def select_scenario(catalog: Mapping[str, Any], mode: str,
                    *, minimum_level: str = "grasp_sim_pass",
                    scenario_id: str | None = None,
                    key: str | None = None, socket: str | None = None,
                    gap_mm: float | None = None,
                    tabletop_pose: int | None = None) -> dict[str, Any]:
    if minimum_level not in VALIDATION_ORDER:
        raise ValueError(f"unknown validation level: {minimum_level}")
    eligible = [s for s in catalog["scenarios"]
                if s["mode"] == mode and
                VALIDATION_ORDER.index(s["validation_level"]) >=
                VALIDATION_ORDER.index(minimum_level) and
                (scenario_id is None or s["id"] == scenario_id) and
                (key is None or s["key"] == key) and
                (socket is None or s["socket"] == socket) and
                (gap_mm is None or s["gap_mm"] == gap_mm) and
                (tabletop_pose is None or s["tabletop_pose"] == tabletop_pose)]
    if not eligible:
        raise RuntimeError(f"No {mode} scenario reaches {minimum_level}; "
                           "do not execute unvalidated insertion on the robot")
    # Avoid known declared shaft/front contact when a safer pilot alternative
    # exists.  Even those alternatives still require whole-hand path checks.
    eligible.sort(key=lambda s: (
        bool(s["evidence"].get("declared_contact_above_handle_front", True)),
        int(s["id"].rsplit("grasp", 1)[-1]) if "grasp" in s["id"] else 0))
    return eligible[0]


def require_robot_ready(scenario: Mapping[str, Any]) -> None:
    """Fail closed until an independently audited full-task record is present."""
    if scenario.get("validation_level") != "hardware_validated" or not scenario.get("hardware_ready"):
        raise RuntimeError("robot execution blocked: no hardware-validated "
                           "full insertion scenario and commissioned controller")


@dataclass(frozen=True)
class RetryDecision:
    status: str
    reason: str
    offset_xy_m: tuple[float, float]
    attempt: int

    def to_record(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason,
                "offset_xy_m": list(self.offset_xy_m), "attempt": self.attempt}


def decide_retry(*, mode: InsertionMode, previous_offset_xy_m: tuple[float, float],
                 attempt: int, observation: Mapping[str, Any],
                 max_attempts: int = 4, gain: float = 0.5,
                 max_step_m: float = 0.0005,
                 max_total_m: float = 0.002) -> RetryDecision:
    """Use metric pose residual, with VLM only as a read-only failure gate.

    ``pose_error_xy_m`` is observed key-minus-target in the socket frame, not
    a pixel offset or a VLM-generated metric estimate.  A new trajectory must
    still pass planning/collision/force gates before execution.
    """
    old = tuple(float(x) for x in previous_offset_xy_m)
    if len(old) != 2 or not all(math.isfinite(x) for x in old):
        raise ValueError("finite 2D previous offset required")
    depth = observation.get("measured_depth_mm")
    depth_ok = depth is not None and math.isfinite(float(depth)) and float(depth) >= mode.target_depth_mm
    if depth_ok and observation.get("abort_reason") is None and observation.get("grasp_held") is True:
        if (observation.get("force_within_limits") is True and
                observation.get("vlm_class") in ("partial_insertion", "seated") and
                observation.get("vlm_confidence", 0.0) >= 0.7):
            return RetryDecision("success", "measured_depth_force_and_visual_consistent", old, attempt)
        return RetryDecision("inspect", "depth_reached_but_success_evidence_incomplete", old, attempt)
    if observation.get("abort_reason") in ("force_limit", "collision", "emergency_stop"):
        return RetryDecision("stop", "safety_abort", old, attempt)
    if observation.get("grasp_held") is not True or observation.get("vlm_class") == "slip":
        return RetryDecision("stop", "grasp_not_verified", old, attempt)
    if attempt >= max_attempts:
        return RetryDecision("stop", "attempt_limit", old, attempt)
    if observation.get("vlm_class") not in ("misaligned", "rim_jam"):
        return RetryDecision("inspect", "visual_failure_not_localized", old, attempt)
    if observation.get("vlm_confidence", 0.0) < 0.7:
        return RetryDecision("inspect", "vlm_low_confidence", old, attempt)
    if observation.get("pose_uncertainty_mm", float("inf")) > 0.5:
        return RetryDecision("inspect", "pose_uncertainty_too_large", old, attempt)
    error = observation.get("pose_error_xy_m")
    if not isinstance(error, (list, tuple)) or len(error) != 2 or not all(
            isinstance(x, (float, int)) and math.isfinite(x) for x in error):
        return RetryDecision("inspect", "metric_pose_residual_missing", old, attempt)
    if math.hypot(float(error[0]), float(error[1])) < 0.0001:
        return RetryDecision("inspect", "pose_residual_too_small_to_explain_jam", old, attempt)
    proposed = [old[i] - max(-max_step_m, min(max_step_m, gain * float(error[i])))
                for i in range(2)]
    norm = math.hypot(*proposed)
    if norm > max_total_m:
        proposed = [x * max_total_m / norm for x in proposed]
    return RetryDecision("propose_retry", "bounded_socket_frame_xy_correction",
                         (proposed[0], proposed[1]), attempt + 1)


def read_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
