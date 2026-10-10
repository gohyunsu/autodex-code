"""v8 reset scenes are proposal assets, never socket-aware robot plans."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reorient_assets import (  # noqa: E402
    audit_v8_reorient_assets, prepare_v8_reorient_scenes,
)


MODE = select_mode("cylinder", 20)


def _asset_tree(root: Path) -> Path:
    object_dir = root / "object_processing" / MODE.key_object
    tabletop = object_dir / "processed_data/info/tabletop"
    tabletop.mkdir(parents=True)
    for index, angle in ((0, 0.0), (1, 90.0)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        np.save(tabletop / f"{index:03d}.npy", pose)
    mesh = trimesh.creation.cylinder(radius=0.015, height=0.08)
    simplified = object_dir / "processed_data/mesh/simplified.obj"
    simplified.parent.mkdir(parents=True)
    mesh.export(simplified)
    raw = object_dir / "raw_mesh" / f"{MODE.key_object}.obj"
    raw.parent.mkdir()
    mesh.export(raw)
    urdf = object_dir / "processed_data/urdf/coacd.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot name='test'/>")
    return object_dir


def test_generates_and_audits_directed_v8_scenes(tmp_path):
    object_dir = _asset_tree(tmp_path)
    manifest_path = tmp_path / "scene_manifest.json"
    prepared = prepare_v8_reorient_scenes(
        shared_root=tmp_path, mode=MODE, manifest_path=manifest_path,
        heights_cm=(0,))
    assert prepared["new_scene_count"] == 2
    assert prepared["directed_scene_count"] == 2
    assert prepared["new_bodex_scene_files"] == 2
    assert prepared["new_sim_filter_scene_files"] == 2
    assert prepared["robot_ready"] is False
    assert (object_dir / "scene/reorient_0/0_1.json").is_file()
    assert (object_dir / "scene/reorient_0/1_0.json").is_file()
    assert (tmp_path / "AutoDex/scene/inspire" / MODE.key_object /
            "reorient_0/0_1.json").is_file()
    audit = audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
    assert len(audit["directed_pairs"]) == 2
    assert all(row["scene_heights_cm"] == [0] for row in audit["directed_pairs"])
    assert not any(row["has_any_stable_seed"] for row in audit["directed_pairs"])
    assert audit["stock_reset_runner_compatible"] is False
    mirror = (tmp_path / "AutoDex/scene/inspire" / MODE.key_object /
              "reorient_0/0_1.json")
    mirror.unlink()
    incomplete = audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
    assert incomplete["directed_pairs"][0]["scene_heights_cm"] == []
    assert incomplete["directed_pairs"][0][
        "bodex_scene_missing_sim_filter_mirror_heights_cm"] == [0]
    again = prepare_v8_reorient_scenes(
        shared_root=tmp_path, mode=MODE,
        manifest_path=tmp_path / "scene_manifest_2.json", heights_cm=(0,))
    assert again["new_scene_count"] == 1
    assert again["new_bodex_scene_files"] == 0
    assert again["new_sim_filter_scene_files"] == 1
    with pytest.raises(FileExistsError):
        prepare_v8_reorient_scenes(
            shared_root=tmp_path, mode=MODE,
            manifest_path=manifest_path, heights_cm=(0,))


def test_rejects_corrupt_existing_scene_and_counts_only_sim_pass(tmp_path):
    object_dir = _asset_tree(tmp_path)
    prepare_v8_reorient_scenes(
        shared_root=tmp_path, mode=MODE,
        manifest_path=tmp_path / "scene_manifest.json", heights_cm=(12,))
    cell = (tmp_path / "AutoDex/candidates/inspire/reset_12" /
            MODE.key_object / "reorient_12/0_1/5")
    cell.mkdir(parents=True)
    np.save(cell / "wrist_se3.npy", np.eye(4))
    np.save(cell / "pregrasp_pose.npy", np.zeros(6))
    np.save(cell / "grasp_pose.npy", np.zeros(6))
    (cell / "sim_eval.json").write_text('{"success": false}')
    audit = audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
    assert audit["directed_pairs"][0][
        "stable_reset_seed_counts_by_height_cm"]["12"] == 0
    (cell / "sim_eval.json").write_text('{"success": true}')
    audit = audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
    assert audit["directed_pairs"][0][
        "reported_mujoco_pass_counts_by_height_cm"]["12"] == 1
    assert audit["directed_pairs"][0][
        "stable_reset_seed_counts_by_height_cm"]["12"] == 0
    assert audit["directed_pairs"][0][
        "reported_passes_rejected_by_loader_by_height_cm"]["12"] == ["5"]
    sim_file = (tmp_path / "AutoDex/scene/inspire" / MODE.key_object /
                "reorient_12/0_1.json")
    sim_scene = json.loads(sim_file.read_text())
    sim_scene["meta"]["geometry_object"] = MODE.key_object
    sim_scene["meta"]["grasp_target_object"] = MODE.key_object
    sim_file.write_text(json.dumps(sim_scene))
    audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
    scene_file = object_dir / "scene/reorient_12/0_1.json"
    scene = json.loads(scene_file.read_text())
    scene["meta"]["pose_j"] = "099"
    scene_file.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="metadata differs"):
        audit_v8_reorient_assets(shared_root=tmp_path, mode=MODE)
