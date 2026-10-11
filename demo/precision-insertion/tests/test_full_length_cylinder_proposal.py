"""Full-length proposal covers the key, without replacing exact-CAD gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_full_length_cylinder_proposal import (  # noqa: E402
    HEIGHT_M, KEY, PROPOSAL, SCENES, build,
)
from stage_cylinder_tabletop_proposals import (  # noqa: E402
    check_scene_pair,
)
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
