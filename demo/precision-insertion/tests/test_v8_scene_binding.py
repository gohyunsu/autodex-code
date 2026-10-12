"""The current v8 scene generator must use the pose's object tree for CAD."""

from __future__ import annotations

import json

import numpy as np

from src.grasp_generation import adaptive_orchestrator


def test_adaptive_table_scene_binds_mesh_to_selected_object_root(
    tmp_path, monkeypatch,
):
    object_root = tmp_path / "object_processing"
    scene_root = tmp_path / "scenes"
    monkeypatch.setattr(adaptive_orchestrator, "get_scene_dir",
                        lambda *_args: str(scene_root))
    previous = adaptive_orchestrator.scene_generation.obj_path
    try:
        adaptive_orchestrator.scene_generation.obj_path = str(
            tmp_path / "AutoDex/object/paradex")
        adaptive_orchestrator.write_scene_file(
            "inspire", "precision_key_cylinder_r15_h80", str(object_root),
            "table", {"id": "0", "pose_idx": "000", "pose": np.eye(4)},
            {}, None)
        target = json.loads((scene_root / "0.json").read_text())[
            "scene"]["mesh"]["target"]
        assert target["file_path"] == str(
            object_root / "precision_key_cylinder_r15_h80" /
            "processed_data/mesh/simplified.obj")
        assert target["urdf_path"] == str(
            object_root / "precision_key_cylinder_r15_h80" /
            "processed_data/urdf/coacd.urdf")
        assert adaptive_orchestrator.scene_generation.obj_path == str(
            tmp_path / "AutoDex/object/paradex")
    finally:
        adaptive_orchestrator.scene_generation.obj_path = previous
