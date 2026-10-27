"""Full-length proposal covers the key, without replacing exact-CAD gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_full_length_cylinder_proposal import (  # noqa: E402
    HEIGHT_M, KEY, PROPOSAL, SCENES, build,
)
from stage_cylinder_tabletop_proposals import (  # noqa: E402
    check_scene_pair, stage,
)
import screen_cylinder_tabletop_batch as cylinder_screen  # noqa: E402
import audit_cylinder_grasp_fidelity as cylinder_fidelity  # noqa: E402
from stage_cylinder_reorient_proposals import (  # noqa: E402
    _check_scene_pair,
)


def test_full_length_asset_preserves_exact_source_scenes(tmp_path):
    source = tmp_path / "object_processing" / KEY
    info = source / "processed_data" / "info"
    tabletop = info / "tabletop"
    tabletop.mkdir(parents=True)
    (info / "simplified.json").write_text(json.dumps({
        "gravity_center": [0, 0, 0.04], "obb": [0.03, 0.03, 0.08],
        "mass": 1.0,
    }))
    urdf = source / "processed_data" / "urdf" / "coacd.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text(f'<robot name="{KEY}"></robot>')
    exact_mesh = source / "processed_data" / "mesh" / "simplified.obj"
    exact_mesh.parent.mkdir(parents=True)
    exact_mesh.write_text("o test\n")
    for stem in (0, 1):
        np.save(tabletop / f"{stem:03d}.npy", np.eye(4))
    for kind, cells in SCENES.items():
        for cell in cells:
            path = (tmp_path / "AutoDex" / "scene" / "inspire" /
                    KEY / kind / f"{cell}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {"pose_idx": f"{int(cell):03d}"} if kind == "table" else {
                "scene_type": kind, "pose_i": "000", "pose_j": "001",
                "h": .12, "version": "v8"}
            path.write_text(json.dumps({
                "scene": {"mesh": {"target": {
                    "pose": [0, 0, .08, 1, 0, 0, 0],
                    "scale": [1, 1, 1],
                    "file_path": str(exact_mesh),
                    "urdf_path": str(urdf)}},
                    "cuboid": {"table": {}}},
                "meta": meta,
            }))
    report = build(shared_root=tmp_path)
    assert report["proposal_object"] == PROPOSAL
    assert report["proposal_contact_region_restricted"] is False
    assert report["proposal_height_m"] == HEIGHT_M
    mesh = trimesh.load(
        tmp_path / "object_processing" / PROPOSAL /
        "processed_data/mesh/simplified.obj", force="mesh")
    assert mesh.is_watertight
    assert mesh.bounds[:, 2] == pytest.approx([0.0, HEIGHT_M])
    assert check_scene_pair(tmp_path, "0")["tabletop_pose_stem"] == "000"
    assert _check_scene_pair(tmp_path, "0_1")["cell"] == "0_1"
    with pytest.raises(FileExistsError):
        build(shared_root=tmp_path)


def test_side_down_only_stage_and_screen_keep_scene_count(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    staged = tmp_path / "staged"
    key = "precision_key_cylinder_r15_h80"
    proxy = f"{key}_full_length_proposal"
    for scene in ("0", "1"):
        seed = raw / proxy / "table" / scene / "0"
        seed.mkdir(parents=True)
        for name in ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
                     "bodex_info.npy"):
            (seed / name).write_bytes(b"test")
    monkeypatch.setattr(
        "stage_cylinder_tabletop_proposals.check_scene_pair",
        lambda _root, scene: {"scene_id": scene})
    report = stage(raw, staged, tmp_path, 1, ("1",))
    assert report["seed_count_total"] == 1
    assert [row["scene_id"] for row in report["scenes"]] == ["1"]
    assert (staged / key / "table/1/0/wrist_se3.npy").is_file()
    assert not (staged / key / "table/0").exists()

    monkeypatch.setattr(cylinder_screen, "CYLINDER_RADIAL_GAPS_MM", (20.0,))
    monkeypatch.setattr(cylinder_screen, "audit_tabletop_cell",
                        lambda **kwargs: ({"scene": kwargs["scene_id"]}, []))
    monkeypatch.setattr(cylinder_screen, "_screen_gap",
                        lambda *_: ("20mm", {"eligible_offline_grasp_ids": []}))
    screened = cylinder_screen.run(
        tmp_path, staged, tmp_path / "passed", tmp_path / "screen", 1, 0.0)
    assert screened["scene_ids"] == ["1"]
    assert screened["total_raw_proposals"] == 1
    assert screened["scene_counts"] == {"1": {"scene": "1"}}


def test_side_down_only_fidelity_audit_uses_screen_scene_ids(tmp_path, monkeypatch):
    key = "precision_key_cylinder_r15_h80"
    stage = tmp_path / "stage"
    pool = tmp_path / "pool"
    candidate = pool / key / "table/1/699"
    candidate.mkdir(parents=True)
    raw = stage / key / "table/1/699"
    raw.mkdir(parents=True)
    (raw / "sim_eval.json").write_text('{"success": true}')
    (raw / "sim_traj.json").write_text('{}')
    summary = tmp_path / "screen.json"
    summary.write_text(json.dumps({
        "schema": "precision_insertion_cylinder_1000_per_scene_screen_v1",
        "physical_key_object": key,
        "scene_ids": ["1"],
        "scene_counts": {"1": {"mujoco_stable": 1}},
        "stage_root": str(stage),
        "original_autodex_sim_candidate_root": str(pool),
    }))
    mesh = tmp_path / "key.obj"
    urdf = tmp_path / "robot.urdf"
    mesh.write_bytes(b"mesh")
    urdf.write_bytes(b"robot")
    monkeypatch.setattr(cylinder_fidelity, "AssetPaths", lambda *_: SimpleNamespace(
        raw_mesh=lambda _: mesh, robot_urdf=urdf))
    monkeypatch.setattr(cylinder_fidelity, "trajectory_closure_audit",
                        lambda *_args, **_kwargs: {"end_squeeze": {
                            "center_in_hand_displacement_m": 0.0}})
    monkeypatch.setattr(cylinder_fidelity, "nominal_visual_penetration_audit",
                        lambda **_kwargs: {"hold": {"sample_points_over_threshold": 0}})
    monkeypatch.setattr(cylinder_fidelity, "simulated_visual_penetration_audit",
                        lambda **_kwargs: {"hold": {"sample_points_over_threshold": 0}})
    report = cylinder_fidelity.run(
        summary_path=summary, shared_root=tmp_path,
        output_root=tmp_path / "audit")
    assert report["scene_ids"] == ["1"]
    assert report["candidates"] == ["table/1/699"]
