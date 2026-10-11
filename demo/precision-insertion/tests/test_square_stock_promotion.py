"""New full-key square grasps append to v8 with source-bound endpoint evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.candidates import build_endpoint_catalog  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
import precision_insertion.square_stock_promotion as promotion  # noqa: E402


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_append_stock_square_grasp_without_replacing_old_scene(tmp_path, monkeypatch):
    mode = select_mode("square", 1.0)
    paths = AssetPaths(tmp_path, mode)
    for path in (paths.raw_mesh(mode.key_object), paths.key_planning_mesh,
                 paths.socket_collision_mesh, paths.task_geometry,
                 paths.robot_urdf):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CAD", encoding="utf-8")
    old = paths.candidate_dir / "table/0/78"
    old.mkdir(parents=True)
    scene = paths.scene_dir / "0.json"
    _write(scene, {"meta": {"pose_idx": "000"},
                   "scene": {"mesh": {"target": {
                       "file_path": str(paths.key_planning_mesh)}}}})
    paths.key_tabletop_dir.mkdir(parents=True)
    np.save(paths.key_tabletop_dir / "000.npy", np.eye(4))

    raw_root, pass_root, screen = (tmp_path / name for name in
                                   ("raw", "pass", "screen"))
    raw = raw_root / mode.key_object / "table/0/7"
    passed = pass_root / mode.key_object / "table/0/7"
    for directory in (raw, passed):
        directory.mkdir(parents=True)
        for name, value in (("wrist_se3.npy", np.eye(4)),
                            ("pregrasp_pose.npy", np.zeros(6)),
                            ("grasp_pose.npy", np.ones(6)),
                            ("bodex_info.npy", np.zeros(2))):
            np.save(directory / name, value)
    np.save(raw / "coll_valid.npy", np.array(True))
    _write(raw / "sim_eval.json", {
        "success": True, "hand": "inspire", "version": "v8"})
    _write(raw / "sim_traj.json", {
        "phase": ["squeeze", "force_gravity"],
        "object_pose": [[], []], "robot_qpos": [[], []]})
    inputs = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": passed / "wrist_se3.npy",
        "pregrasp_pose": passed / "pregrasp_pose.npy",
        "grasp_pose": passed / "grasp_pose.npy",
    }
    report = screen / mode.key_object / "table/0/7.json"
    _write(report, {
        "schema": "precision_insertion_endpoint_screen_v1",
        "endpoint_pass": True, "hand_socket_clear_at_20mm": True,
        "key_socket_fit": {"colliding": False},
        "candidate_dir": str(passed),
        "mode": {"family": "square", "gap_mm": 1.0,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "verification_depth_m": 0.02,
        "xy_offset_socket_m": [0.0, 0.0],
        "minimum_required_hand_clearance_m": .0002,
        "input_sha256": {name: _sha(path) for name, path in inputs.items()},
    })
    counts = {"generated": 1000, "mujoco_stable": 1}
    _write(screen / "summary.json", {
        "schema": "precision_insertion_square_tabletop_1000_screen_v1",
        "expected_proposals_per_scene": 1000,
        "raw_root": str(raw_root),
        "original_autodex_sim_candidate_root": str(pass_root),
        "minimum_hand_clearance_m": .0002,
        "modes": {mode.key_object: {
            "gap_mm": 1.0, "socket_object": mode.socket_object,
            "errors": [], "scene_counts": {"0": counts},
            "eligible_offline_grasp_ids": ["table/0/7"]}},
        "robot_ready": False,
    })
    monkeypatch.setattr(promotion, "SCENES", ("0",))
    monkeypatch.setattr(promotion, "audit_tabletop_cell",
                        lambda **_kwargs: (counts, [("7", passed)]))
    manifest = tmp_path / "new_manifest.json"
    args = dict(shared_root=tmp_path, mode=mode, screen_root=screen,
                candidate_id_offset=12011000, output_manifest=manifest)
    preview = promotion.promote_screened_batch(**args)
    assert preview["installed"] is False
    assert not manifest.exists()
    installed = promotion.promote_screened_batch(**args, install=True)
    new = paths.candidate_dir / "table/0/12011007"
    assert installed["installed"] is True
    assert new.is_dir() and old.is_dir() and manifest.is_file()
    with pytest.raises(FileExistsError):
        promotion.promote_screened_batch(**args, install=True)

    def fake_screen(*, candidate_dir, **_kwargs):
        return {"endpoint_pass": True, "input_sha256": {
            name: _sha(path) for name, path in {
                **{key: value for key, value in inputs.items()
                   if key not in ("wrist_se3", "pregrasp_pose", "grasp_pose")},
                "wrist_se3": candidate_dir / "wrist_se3.npy",
                "pregrasp_pose": candidate_dir / "pregrasp_pose.npy",
                "grasp_pose": candidate_dir / "grasp_pose.npy",
            }.items()}}

    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode, minimum_hand_clearance_m=.0002,
        screen=fake_screen)
    assert catalog["eligible_count"] == 1
    (new / "sim_traj.json").write_text("{}", encoding="utf-8")
    stale = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode, minimum_hand_clearance_m=.0002,
        screen=fake_screen)
    assert stale["eligible_count"] == 0
