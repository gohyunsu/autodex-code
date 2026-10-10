"""A saved v8 plan is inspectable, but never a robot-motion permit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.path_audit import _array_sha256  # noqa: E402
from precision_insertion.saved_preflight import (  # noqa: E402
    verify_saved_passing_trial,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, dict, dict]:
    start = np.zeros(13)
    held = np.full(6, .2)
    pickup = np.stack([start, start.copy()])
    pickup[-1, 0] = .1
    lift = np.stack([pickup[-1].copy(), pickup[-1].copy()])
    lift[:, 7:] = held
    lift[-1, 0] += .01
    transfer = np.stack([lift[-1].copy(), lift[-1].copy()])
    transfer[-1, 1] += .01
    axial = np.stack([transfer[-1].copy(), transfer[-1].copy()])
    axial[-1, 2] += .01
    arrays = {
        "pickup_approach": pickup,
        "pickup_pregrasp": np.zeros(6),
        "pickup_grasp": held.copy(),
        "pickup_wrist": np.eye(4),
        "held_lift": lift,
        "transfer": transfer,
        "axial": axial,
        "held_hand_q": held,
    }
    np.savez_compressed(tmp_path / "planned_trajectories.npz", **arrays)
    (tmp_path / "trial_scene.json").write_text("{}\n", encoding="utf-8")
    audit = {
        "sampled_clear": True,
        "sample_counts": {"lift": 2, "transfer": 2, "descent": 2},
        "input_sha256": {
            "lift_trajectory": _array_sha256(lift),
            "transfer_trajectory": _array_sha256(transfer),
            "descent_trajectory": _array_sha256(axial),
            "held_hand_q": _array_sha256(held),
        },
    }
    report = {
        "schema": "precision_insertion_trial_preflight_v2",
        "status": "sampled_planning_pass", "robot_ready": False,
        "selected_candidate_key": ["table", "4", "5102"],
        "attempted_candidates": [{
            "key": ["table", "4", "5102"],
            "pickup_preflight_pass": True,
            "insertion_preflight_status": "sampled_planning_pass",
        }],
        "insertion_plan": {
            "schema": "precision_insertion_planning_preflight_v1",
            "status": "sampled_planning_pass", "sampled_planning_pass": True,
            "held_hand_q": held.tolist(),
            "sample_counts": {"lift": 2, "transfer": 2, "axial": 2},
            "sampled_held_path_audit": audit,
        },
        "live_start_q": start.tolist(),
        "limits": {"max_joint_step_rad": .12},
        "artifacts": {
            "planned_trajectories": "planned_trajectories.npz",
            "planned_trajectories_sha256": _sha(
                tmp_path / "planned_trajectories.npz"),
            "trial_scene": "trial_scene.json",
            "trial_scene_sha256": _sha(tmp_path / "trial_scene.json"),
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, report, arrays


def test_saved_passing_preflight_verifies_fixed_hand_and_boundaries(tmp_path):
    report_path, _report, _arrays = _bundle(tmp_path)
    verified = verify_saved_passing_trial(report_path)
    assert verified["candidate_id"] == "table/4/5102"
    assert verified["sample_counts"]["axial"] == 2
    assert verified["maximum_held_hand_drift_rad"] == 0
    assert verified["robot_ready"] is False


def test_saved_preflight_rejects_mutated_npz_and_path_traversal(tmp_path):
    report_path, report, _arrays = _bundle(tmp_path)
    (tmp_path / "planned_trajectories.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_saved_passing_trial(report_path)
    report["artifacts"]["planned_trajectories"] = "../unbound.npz"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="path or digest"):
        verify_saved_passing_trial(report_path)


def test_saved_preflight_rejects_held_hand_motion_even_with_updated_hashes(
        tmp_path):
    report_path, report, arrays = _bundle(tmp_path)
    arrays["transfer"] = arrays["transfer"].copy()
    arrays["transfer"][-1, 7] += .01
    np.savez_compressed(tmp_path / "planned_trajectories.npz", **arrays)
    report["artifacts"]["planned_trajectories_sha256"] = _sha(
        tmp_path / "planned_trajectories.npz")
    report["insertion_plan"]["sampled_held_path_audit"]["input_sha256"][
        "transfer_trajectory"] = _array_sha256(arrays["transfer"])
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hand lock"):
        verify_saved_passing_trial(report_path)


def test_saved_preflight_rejects_audit_array_digest_mismatch(tmp_path):
    report_path, report, _arrays = _bundle(tmp_path)
    report["insertion_plan"]["sampled_held_path_audit"]["input_sha256"][
        "transfer_trajectory"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="sampled audit inputs"):
        verify_saved_passing_trial(report_path)
