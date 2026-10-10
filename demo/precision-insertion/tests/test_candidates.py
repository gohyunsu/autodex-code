"""Explicit-root v8 candidate indexing, endpoint gating and loader reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.candidates import (  # noqa: E402
    build_endpoint_catalog, planner_candidate_override, select_pose_candidates,
)
from precision_insertion.config import select_mode  # noqa: E402


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_fixture(tmp_path):
    mode = select_mode("square", 1.5)
    paths = AssetPaths(tmp_path, mode)
    shared_files = [
        paths.key_planning_mesh, paths.raw_mesh(mode.key_object),
        paths.socket_collision_mesh, paths.task_geometry, paths.robot_urdf,
    ]
    for file in shared_files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("mesh test", encoding="utf-8")
    paths.key_tabletop_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("000", "001"):
        np.save(paths.key_tabletop_dir / f"{stem}.npy", np.eye(4))
    candidates = []
    for scene_id, stem, grasp_id in (("0", "000", "1"),
                                     ("0", "000", "2"),
                                     ("1", "001", "3")):
        scene_path = paths.scene_dir / f"{scene_id}.json"
        _write_json(scene_path, {"meta": {"pose_idx": stem}})
        candidate = paths.candidate_dir / "table" / scene_id / grasp_id
        candidate.mkdir(parents=True)
        wrist = np.eye(4)
        wrist[0, 3] = int(grasp_id) / 1000
        np.save(candidate / "wrist_se3.npy", wrist)
        np.save(candidate / "pregrasp_pose.npy", np.zeros(6))
        np.save(candidate / "grasp_pose.npy", np.ones(6) * 0.2)
        _write_json(candidate / "sim_eval.json", {"success": True})
        _write_json(candidate / "simulation_validation.json", {
            "status": "passed",
            "full_object_mesh_sha256": _sha(paths.key_planning_mesh),
        })
        candidates.append(candidate)

    input_paths = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
    }

    def screen(*, candidate_dir, **_kwargs):
        files = {
            **input_paths,
            "wrist_se3": candidate_dir / "wrist_se3.npy",
            "grasp_pose": candidate_dir / "grasp_pose.npy",
        }
        return {
            "endpoint_pass": candidate_dir.name != "2",
            "input_sha256": {key: _sha(path) for key, path in files.items()},
        }

    return mode, paths, candidates, screen


def test_catalog_selects_pose_and_reuses_v8_loader(tmp_path):
    mode, paths, candidates, screen = _candidate_fixture(tmp_path)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    assert catalog["complete_scan"] is True
    assert catalog["eligible_count"] == 2
    assert [row["tabletop_pose_stem"] for row in catalog["candidates"]] == [
        "000", "000", "001"]
    _write_json(tmp_path / "AutoDex" / "experiment" / "v8" / "coverage" /
                f"cov_v8_cand_{mode.key_object}.json", {
                    "grasps": [
                        {"type": "table", "sid": "0", "gid": "1", "covers": [0, 1]},
                        {"type": "table", "sid": "1", "gid": "3", "covers": [2]},
                    ]})
    selected = select_pose_candidates(catalog, tabletop_pose_stem="000")
    assert selected["status"] == "candidates_available"
    assert [row["key"] for row in selected["candidates"]] == [["table", "0", "1"]]
    assert selected["candidates"][0]["uncovered_scene_gain"] == 2

    T_robot_key = np.eye(4)
    T_robot_key[0, 3] = 0.4
    wrist, pregrasp, grasp, info, openposes = planner_candidate_override(
        catalog=catalog, selected=selected["candidates"],
        pose_robot_key=T_robot_key, tabletop_pose_stem="000")
    assert info == [("table", "0", "1")]
    assert wrist.shape == (1, 4, 4)
    assert wrist[0, 0, 3] == pytest.approx(0.401)
    assert pregrasp.shape == (1, 6)
    assert grasp.shape == (1, 6)
    assert openposes == [None]

    excluded = select_pose_candidates(
        catalog, tabletop_pose_stem="000", attempted=[("table", "0", "1")])
    assert excluded["status"] == "no_eligible_in_screened_pool"
    assert "not impossibility proof" in excluded["reason"]


def test_catalog_detects_changed_scene_and_evidence(tmp_path):
    mode, paths, candidates, screen = _candidate_fixture(tmp_path)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    (candidates[0] / "sim_eval.json").write_text('{"success": false}')
    result = select_pose_candidates(catalog, tabletop_pose_stem="000")
    assert result["status"] == "catalog_stale"
    assert "grasp evidence changed" in result["reason"]


def test_pilot_or_bad_scene_cannot_become_complete_catalog(tmp_path, monkeypatch):
    mode, paths, candidates, screen = _candidate_fixture(tmp_path)
    pilot = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, max_candidates=1, screen=screen)
    assert pilot["complete_scan"] is False
    assert select_pose_candidates(pilot, tabletop_pose_stem="000")["status"] == (
        "catalog_incomplete")
    even_full_prefix = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, max_candidates=3, screen=screen)
    assert even_full_prefix["complete_scan"] is False
    with monkeypatch.context() as patcher:
        patcher.setattr(
            "precision_insertion.candidates._candidate_dirs", lambda _root: [])
        empty = build_endpoint_catalog(
            shared_root=tmp_path, mode=mode,
            minimum_hand_clearance_m=0.0002, screen=screen)
    assert empty["complete_scan"] is False
    (paths.scene_dir / "0.json").write_text("{}", encoding="utf-8")
    invalid = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    assert invalid["complete_scan"] is False
    assert len(invalid["errors"]) == 2


def test_changed_candidate_pool_and_unapproved_override_rejected(tmp_path):
    mode, paths, candidates, screen = _candidate_fixture(tmp_path)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    with pytest.raises(ValueError, match="do not match eligible"):
        planner_candidate_override(
            catalog=catalog,
            selected=[{"key": ["table", "0", "2"],
                       "candidate_dir": str(candidates[1])}],
            pose_robot_key=np.eye(4), tabletop_pose_stem="000")
    new_dir = paths.candidate_dir / "table" / "1" / "4"
    new_dir.mkdir()
    changed = select_pose_candidates(catalog, tabletop_pose_stem="000")
    assert changed["status"] == "catalog_stale"
