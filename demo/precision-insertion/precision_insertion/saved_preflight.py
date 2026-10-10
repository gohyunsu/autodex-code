"""Independently verify a saved passing v8 precision-insertion plan bundle.

This is an artifact and joint-trajectory contract check.  The original
planning process has already ended; hashes and self-reported collision
results do not reproduce its cuRobo/MuJoCo computation or authorize motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .geometry import validate_se3
from .path_audit import _array_sha256


_TRAJECTORIES = ("pickup_approach", "held_lift", "transfer", "axial")
_NPZ_FIELDS = frozenset((*_TRAJECTORIES, "pickup_pregrasp", "pickup_grasp",
                         "pickup_wrist", "held_hand_q"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_artifact(directory: Path, name: object, digest: object) -> Path:
    if (not isinstance(name, str) or not name or
            Path(name).name != name or name in {".", ".."} or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("saved preflight artifact path or digest is invalid")
    path = directory / name
    if not path.is_file() or path.is_symlink() or _sha(path) != digest:
        raise ValueError(f"saved preflight artifact changed: {name}")
    return path


def _path(value: np.ndarray, name: str) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 13 or len(path) < 2 or
            not np.all(np.isfinite(path))):
        raise ValueError(f"{name} must be a finite 13-DOF trajectory")
    return path


def verify_saved_passing_trial(report_path: Path) -> dict:
    """Recheck source hashes, selected grasp and held-path joint invariants.

    The return value deliberately carries ``robot_ready=False``.  A fresh
    live camera/robot state, physical key–hand relation and guarded executor
    remain separate mandatory gates.
    """
    supplied_report = Path(report_path).expanduser()
    if supplied_report.is_symlink():
        raise ValueError("saved trial preflight report must not be a symlink")
    report_file = supplied_report.resolve()
    if not report_file.is_file():
        raise FileNotFoundError("saved trial preflight report is missing")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or
            report.get("schema") != "precision_insertion_trial_preflight_v2" or
            report.get("status") != "sampled_planning_pass" or
            report.get("robot_ready") is not False):
        raise ValueError("expected a saved passing read-only trial preflight")
    selected = report.get("selected_candidate_key")
    plan = report.get("insertion_plan")
    attempted = report.get("attempted_candidates")
    if (not isinstance(selected, list) or len(selected) != 3 or
            any(not isinstance(part, str) or not part for part in selected) or
            not isinstance(plan, dict) or
            plan.get("schema") != "precision_insertion_planning_preflight_v1" or
            plan.get("status") != "sampled_planning_pass" or
            plan.get("sampled_planning_pass") is not True or
            not isinstance(plan.get("sampled_held_path_audit"), dict) or
            plan["sampled_held_path_audit"].get("sampled_clear") is not True or
            not isinstance(attempted, list) or not attempted or
            not isinstance(attempted[-1], dict) or
            attempted[-1].get("key") != selected or
            attempted[-1].get("pickup_preflight_pass") is not True or
            attempted[-1].get("insertion_preflight_status") !=
            "sampled_planning_pass"):
        raise ValueError("saved preflight selection and planning result disagree")
    artifacts = report.get("artifacts")
    if (not isinstance(artifacts, dict) or set(artifacts) != {
            "trial_scene", "trial_scene_sha256", "planned_trajectories",
            "planned_trajectories_sha256"}):
        raise ValueError("saved preflight needs exactly its two hashed artifacts")
    directory = report_file.parent
    scene_file = _bound_artifact(
        directory, artifacts["trial_scene"], artifacts["trial_scene_sha256"])
    trajectory_file = _bound_artifact(
        directory, artifacts["planned_trajectories"],
        artifacts["planned_trajectories_sha256"])
    scene = json.loads(scene_file.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError("saved trial scene must be a JSON object")
    with np.load(trajectory_file, allow_pickle=False) as archive:
        if set(archive.files) != _NPZ_FIELDS:
            raise ValueError("saved preflight trajectory fields changed")
        trajectories = {name: _path(archive[name], name)
                        for name in _TRAJECTORIES}
        pregrasp = np.asarray(archive["pickup_pregrasp"], dtype=np.float64)
        grasp = np.asarray(archive["pickup_grasp"], dtype=np.float64)
        wrist = np.asarray(archive["pickup_wrist"], dtype=np.float64)
        held = np.asarray(archive["held_hand_q"], dtype=np.float64)
    if (pregrasp.shape != (6,) or grasp.shape != (6,) or
            held.shape != (6,) or
            not all(np.all(np.isfinite(value)) for value in
                    (pregrasp, grasp, held))):
        raise ValueError("saved Inspire grasp or held joints are invalid")
    validate_se3(wrist, name="saved pickup wrist")
    start = np.asarray(report.get("live_start_q"), dtype=np.float64)
    reported_held = np.asarray(plan.get("held_hand_q"), dtype=np.float64)
    if (start.shape != (13,) or not np.all(np.isfinite(start)) or
            reported_held.shape != (6,) or
            not np.allclose(trajectories["pickup_approach"][0], start,
                            rtol=0, atol=1e-6) or
            not np.allclose(reported_held, held, rtol=0, atol=1e-6)):
        raise ValueError("saved start or held hand differs from report")
    limits = report.get("limits")
    step_limit = (limits.get("max_joint_step_rad")
                  if isinstance(limits, dict) else None)
    if (type(step_limit) not in (float, int) or
            not math.isfinite(step_limit) or step_limit <= 0):
        raise ValueError("saved joint-step limit is invalid")
    maximum_step = 0.0
    maximum_hand_drift = 0.0
    for name in ("held_lift", "transfer", "axial"):
        path = trajectories[name]
        maximum_step = max(maximum_step, float(np.max(np.abs(np.diff(
            path, axis=0)))))
        maximum_hand_drift = max(maximum_hand_drift, float(np.max(np.abs(
            path[:, 7:] - held))))
    if maximum_step > step_limit + 1e-8 or maximum_hand_drift > 1e-6:
        raise ValueError("saved held trajectory violates joint-step or hand lock")
    if (not np.allclose(
            trajectories["pickup_approach"][-1, :7],
            trajectories["held_lift"][0, :7], rtol=0, atol=1e-4) or
            not np.allclose(trajectories["held_lift"][-1],
                            trajectories["transfer"][0], rtol=0, atol=1e-4) or
            not np.allclose(trajectories["transfer"][-1],
                            trajectories["axial"][0], rtol=0, atol=1e-4)):
        raise ValueError("saved pickup/lift/transfer/axial boundaries disconnect")
    sample_counts = plan["sampled_held_path_audit"].get("sample_counts")
    if (not isinstance(sample_counts, dict) or
            sample_counts.get("lift") != len(trajectories["held_lift"]) or
            sample_counts.get("transfer") != len(trajectories["transfer"]) or
            sample_counts.get("descent") != len(trajectories["axial"])):
        raise ValueError("saved sampled audit counts differ from trajectory")
    audit_hashes = plan["sampled_held_path_audit"].get("input_sha256")
    expected_hashes = {
        "lift_trajectory": _array_sha256(trajectories["held_lift"]),
        "transfer_trajectory": _array_sha256(trajectories["transfer"]),
        "descent_trajectory": _array_sha256(trajectories["axial"]),
        "held_hand_q": _array_sha256(held),
    }
    if (not isinstance(audit_hashes, dict) or
            any(audit_hashes.get(name) != digest for name, digest in
                expected_hashes.items()) or
            plan.get("sample_counts") != {
                "lift": len(trajectories["held_lift"]),
                "transfer": len(trajectories["transfer"]),
                "axial": len(trajectories["axial"]),
            }):
        raise ValueError("saved held paths differ from sampled audit inputs")
    return {
        "schema": "precision_insertion_saved_passing_preflight_verification_v1",
        "candidate_id": "/".join(selected),
        "report_sha256": _sha(report_file),
        "trial_scene_sha256": artifacts["trial_scene_sha256"],
        "planned_trajectories_sha256": artifacts[
            "planned_trajectories_sha256"],
        "sample_counts": {name: len(path) for name, path in
                          trajectories.items()},
        "maximum_held_joint_step_rad": maximum_step,
        "maximum_held_hand_drift_rad": maximum_hand_drift,
        "scope": "saved_artifact_integrity_not_recomputed_collision_or_motion",
        "robot_ready": False,
    }
