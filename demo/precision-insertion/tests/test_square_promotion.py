"""Square v8 promotion reuses the stock filter without overwriting a scene."""

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
    build_endpoint_catalog, select_pose_candidates,
)
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.square_promotion import promote_square_scene  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path):
    mode = select_mode("square", 1.5)
    paths = AssetPaths(root, mode)
    for file in (paths.raw_mesh(mode.key_object), paths.key_planning_mesh,
                 paths.socket_collision_mesh, paths.task_geometry,
                 paths.robot_urdf):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("CAD fixture", encoding="utf-8")
    paths.key_tabletop_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.key_tabletop_dir / "004.npy", np.eye(4))
    scene_urdf = root / "object_processing" / mode.key_object / "processed_data" / "urdf" / "coacd.urdf"
    scene_urdf.parent.mkdir(parents=True, exist_ok=True)
    scene_urdf.write_text("robot fixture", encoding="utf-8")
    _write(paths.scene_dir / "4.json", {
        "meta": {"pose_idx": "004"},
        "scene": {"mesh": {"target": {
            "file_path": str(paths.key_planning_mesh),
            "urdf_path": str(scene_urdf),
        }}},
    })
    source = root / "stage" / mode.key_object / "table" / "4"
    passed = root / "pass" / mode.key_object / "table" / "4"
    for folder in (source / "104", passed / "104"):
        folder.mkdir(parents=True)
        np.save(folder / "wrist_se3.npy", np.eye(4))
        np.save(folder / "pregrasp_pose.npy", np.zeros(6))
        np.save(folder / "grasp_pose.npy", np.ones(6))
        np.save(folder / "bodex_info.npy", np.zeros(2))
    np.save(source / "104" / "coll_valid.npy", True)
    _write(source / "104" / "contact_screen.json", {
        "accepted": True, "contact_policy_mode": "report-only",
        "quality": {"grasp_error_max": .12,
                    "contact_distance_mean_abs_m": .005},
    })
    _write(source / "104" / "sim_eval.json", {
        "success": True, "hand": "inspire", "version": "pilot",
    })
    _write(source / "104" / "sim_traj.json", {
        "phase": ["pregrasp", "squeeze", "force_gravity"],
        "object_pose": [[], [], []], "robot_qpos": [[], [], []],
    })
    highres = root / "highres.json"
    _write(highres, {
        "status": "sampled_prefilter_not_trajectory_or_physical_validation",
        "contact_policy_mode": "disabled",
        "sampling": {"samples_per_link": 3000},
        "scene": str(root / "quality" / mode.key_object / "table" / "4"),
        "tabletop_pose": str(paths.key_tabletop_dir / "004.npy"),
        "task_geometry": str(paths.task_geometry),
        "socket_mesh": str(paths.socket_collision_mesh),
        "passed_candidates": ["104"],
    })

    def screen(*, candidate_dir, **_kwargs):
        files = {
            "key_mesh": paths.raw_mesh(mode.key_object),
            "socket_mesh": paths.socket_collision_mesh,
            "task_geometry": paths.task_geometry,
            "robot_urdf": paths.robot_urdf,
            "wrist_se3": candidate_dir / "wrist_se3.npy",
            "pregrasp_pose": candidate_dir / "pregrasp_pose.npy",
            "grasp_pose": candidate_dir / "grasp_pose.npy",
        }
        return {"endpoint_pass": True,
                "input_sha256": {name: _sha(path)
                                 for name, path in files.items()}}

    return mode, paths, source, passed, highres, screen


def test_complete_square_pilot_promotes_once_and_catalog_binds_bytes(tmp_path):
    mode, paths, source, passed, highres, screen = _fixture(tmp_path)
    output = paths.candidate_dir / "table" / "4"
    args = dict(
        shared_root=tmp_path, mode=mode, source_scene=source,
        pass_scene=passed, highres_report=highres, output_scene=output,
        simulation_version="pilot", minimum_hand_clearance_m=.0002,
        screen=screen)
    result = promote_square_scene(**args)
    assert result["candidate_ids"] == ["104"]
    assert result["robot_ready"] is False
    assert (output / "highres_report.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        promote_square_scene(**args)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=.0002, screen=screen)
    assert catalog["complete_scan"] is True
    assert catalog["eligible_count"] == 1
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem="004")
    assert [row["key"] for row in selected["candidates"]] == [
        ["table", "4", "104"]]
    (output / "highres_report.json").write_text("{}", encoding="utf-8")
    assert select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem="004")[
            "status"] == "catalog_stale"
    (output / "104" / "sim_traj.json").write_text("{}", encoding="utf-8")
    stale = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=.0002, screen=screen)
    assert stale["eligible_count"] == 0
    assert stale["candidates"][0]["grasp_stability_reason"] == (
        "stale_square_source_file")


def test_square_promotion_rejects_incomplete_or_changed_pass_pool(tmp_path):
    mode, paths, source, passed, highres, screen = _fixture(tmp_path)
    args = dict(
        shared_root=tmp_path, mode=mode, source_scene=source,
        pass_scene=passed, highres_report=highres,
        output_scene=paths.candidate_dir / "table" / "4",
        simulation_version="pilot", minimum_hand_clearance_m=.0002,
        screen=screen)
    (passed / "104" / "wrist_se3.npy").write_bytes(b"different")
    with pytest.raises(ValueError, match="pass pool differs"):
        promote_square_scene(**args)
    assert not args["output_scene"].exists()


def test_square_promotion_rejects_failed_endpoint(tmp_path):
    mode, paths, source, passed, highres, _screen = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exact 20 mm endpoint failed"):
        promote_square_scene(
            shared_root=tmp_path, mode=mode, source_scene=source,
            pass_scene=passed, highres_report=highres,
            output_scene=paths.candidate_dir / "table" / "4",
            simulation_version="pilot", minimum_hand_clearance_m=.0002,
            screen=lambda **_kw: {"endpoint_pass": False,
                                  "input_sha256": {}})
    assert not (paths.candidate_dir / "table" / "4").exists()
