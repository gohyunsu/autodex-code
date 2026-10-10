import json
from pathlib import Path

import numpy as np
import pytest

from scripts.precision_insertion.build_pose_task_catalog import build_catalogs


def _pose(root: Path, key: str, index: int) -> None:
    folder = root / "object_processing" / key / "processed_data" / "info" / "tabletop"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{index:03d}.npy", np.eye(4))


def _grasp(root: Path, key: str, pose: int, candidate: int) -> None:
    folder = (root / "AutoDex" / "candidates" / "inspire" / "v8" / key /
              "table" / str(pose) / str(candidate))
    folder.mkdir(parents=True)
    np.save(folder / "wrist_se3.npy", np.eye(4))
    np.save(folder / "pregrasp_pose.npy", np.zeros(6))
    np.save(folder / "grasp_pose.npy", np.zeros(6))


def test_pose_inventory_never_calls_missing_pass_impossible(tmp_path):
    key = "precision_key_1p5mm"
    _pose(tmp_path, key, 0)
    _pose(tmp_path, key, 4)
    _grasp(tmp_path, key, 4, 104)
    scenarios = {"scenarios": [{
        "id": "pose4-grasp104", "mode": "square", "gap_mm": 1.5,
        "key": key, "socket": "precision_socket_unified", "tabletop_pose": 4,
        "candidate_dir": "diagnostic", "validation_level": "grasp_sim_pass",
        "full_task_sim_pass": False, "hardware_ready": False,
    }]}
    poses, seeds = build_catalogs(tmp_path, scenarios, variants=(("square", 1.5),))
    assert len(poses["rows"]) == 2
    assert len(seeds["transitions"]) == 2
    by_pose = {row["tabletop_pose"]: row for row in poses["rows"]}
    assert by_pose[0]["evidence_state"] == "no_candidate_evidence_yet"
    assert by_pose[4]["evidence_state"] == "grasp_candidates_below_full_task_gate"
    assert by_pose[4]["runtime_pool_candidates"][0]["id"] == "104"
    assert all(row["absence_verdict"] == "unknown_not_proven" for row in by_pose.values())
    assert not any(row["may_trigger_reorient_for_no_feasible_scenario"]
                   for row in by_pose.values())
    source_four = next(row for row in seeds["transitions"] if row["cell"] == "4_0")
    assert {seed["source"] for seed in source_four["source_grasp_hypotheses"]} == {
        "v8_source_pose_grasp_not_pair_validated",
        "scenario_grasp_not_pair_validated",
    }


def test_reorient_diagnostic_is_not_runtime_seed(tmp_path):
    key = "precision_key_1p5mm"
    _pose(tmp_path, key, 0)
    _pose(tmp_path, key, 4)
    scene = (tmp_path / "AutoDex" / "scene" / "inspire" / key /
             "reorient_12" / "4_0.json")
    scene.parent.mkdir(parents=True)
    scene.write_text("{}")
    diagnostic = (tmp_path / "AutoDex" / "precision_insertion" / "experiments" /
                  "reorientation_from_stable_grasp" / "4_0" / "104")
    diagnostic.mkdir(parents=True)
    for name, value in (("wrist_se3.npy", np.eye(4)),
                        ("pregrasp_pose.npy", np.zeros(6)),
                        ("grasp_pose.npy", np.zeros(6))):
        np.save(diagnostic / name, value)
    preview = (tmp_path / "AutoDex" / "precision_insertion" /
               "presentation_assets" / "06_reorientation" / "stable_grasp_104" /
               "reorientation_pose_004_to_000.json")
    preview.parent.mkdir(parents=True)
    preview.write_text(json.dumps({
        "status": "curobo_continuous_reorientation_plan_not_physical_validation",
        "selected": {"cell": "4_0", "candidate": "104"},
    }))
    _, seed_catalog = build_catalogs(
        tmp_path, {"scenarios": []}, variants=(("square", 1.5),))
    transition = next(row for row in seed_catalog["transitions"]
                      if row["cell"] == "4_0")
    assert transition["scene_by_release_height_cm"]["12"] == str(scene)
    assert transition["diagnostic_seeds"][0]["id"] == "104"
    assert transition["runtime_reset_candidates"] == []
    assert transition["motion_preview"]["physical_validation"] is False
    assert transition["evidence_state"] == "diagnostic_curobo_motion_preview_only"
    assert transition["ready_for_automatic_reorient"] is False


def test_reject_contradictory_full_task_claim(tmp_path):
    key = "precision_key_1p5mm"
    _pose(tmp_path, key, 0)
    scenario = {"scenarios": [{
        "mode": "square", "gap_mm": 1.5, "key": key,
        "socket": "precision_socket_unified", "tabletop_pose": 0,
        "validation_level": "full_task_sim_pass", "full_task_sim_pass": False,
        "hardware_ready": False,
    }]}
    with pytest.raises(ValueError, match="inconsistent full-task evidence"):
        build_catalogs(tmp_path, scenario, variants=(("square", 1.5),))


def test_raw_bodex_run_is_not_a_runtime_reorient_seed(tmp_path):
    key = "precision_key_cylinder_r15_h80"
    for pose in (0, 1):
        _pose(tmp_path, key, pose)
    raw = (tmp_path / "AutoDex" / "bodex_raw" / "inspire" / "pilot" /
           f"{key}_grip_proxy" / "reorient_12" / "0_1" / "0")
    raw.mkdir(parents=True)
    np.save(raw / "bodex_info.npy", {"success": False})
    _, seeds = build_catalogs(tmp_path, {"scenarios": []},
                              variants=(("cylinder", 20),))
    transition = next(row for row in seeds["transitions"]
                      if row["cell"] == "0_1")
    assert transition["bodex_raw_runs"][0]["saved_optimization_seeds"] == 1
    assert transition["bodex_raw_runs"][0]["run"] == "pilot"
    assert transition["evidence_state"] == "raw_bodex_proposals_not_screened"
    assert transition["runtime_reset_candidates"] == []
    assert transition["ready_for_automatic_reorient"] is False


def test_legacy_square_proxy_raw_run_is_inventoried(tmp_path):
    key = "precision_key_1p5mm"
    for pose in (0, 4):
        _pose(tmp_path, key, pose)
    raw = (tmp_path / "AutoDex" / "bodex_raw" / "inspire" / "pilot" /
           "precision_key_handle_contact_proxy" / "reorient_12" /
           "0_4" / "42")
    raw.mkdir(parents=True)
    np.save(raw / "bodex_info.npy", {"success": False})
    _, seeds = build_catalogs(tmp_path, {"scenarios": []},
                              variants=(("square", 1.5),))
    transition = next(row for row in seeds["transitions"]
                      if row["cell"] == "0_4")
    assert transition["bodex_raw_runs"][0]["proposal_object"] == (
        "precision_key_handle_contact_proxy")
