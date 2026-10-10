"""Proxy staging and full-key reset-filter evidence stay separate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_cylinder_reorient_pilot import audit  # noqa: E402
from stage_cylinder_reorient_proposals import KEY, PAIRS, PROXY, stage  # noqa: E402


def _stage_fixture(root: Path, *, expected: int = 2) -> tuple[Path, Path]:
    raw = root / "raw"
    scene_root = root / "AutoDex/scene/inspire"
    for obj in (PROXY, KEY):
        mesh = root / "object_processing" / obj / "processed_data/mesh/simplified.obj"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("o test\n")
        for cell in PAIRS:
            i, j = map(int, cell.split("_"))
            scene = {
                "meta": {"scene_type": "reorient_12", "pose_i": f"{i:03d}",
                         "pose_j": f"{j:03d}", "h": 0.12, "version": "v8"},
                "scene": {
                    "mesh": {"target": {"file_path": str(mesh),
                                        "pose": [0, 0, 0, 1, 0, 0, 0],
                                        "scale": [1, 1, 1]}},
                    "cuboid": {"table_i": {}, "table_j": {}},
                },
            }
            path = scene_root / obj / "reorient_12" / f"{cell}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scene))
    for cell in PAIRS:
        for seed in range(expected):
            seed_dir = raw / PROXY / "reorient_12" / cell / str(seed)
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "wrist_se3.npy", np.eye(4))
            np.save(seed_dir / "pregrasp_pose.npy", np.zeros(6))
            np.save(seed_dir / "grasp_pose.npy", np.ones(6))
            np.save(seed_dir / "bodex_info.npy", np.zeros(1))
    return raw, root / "staged"


def test_stage_and_audit_stock_filter_results(tmp_path):
    raw, staged = _stage_fixture(tmp_path)
    manifest = stage(raw_root=raw, stage_root=staged,
                     shared_root=tmp_path, expected_per_cell=2)
    assert manifest["total_raw_proposals"] == 4
    assert manifest["robot_ready"] is False
    with pytest.raises(FileExistsError):
        stage(raw_root=raw, stage_root=staged,
              shared_root=tmp_path, expected_per_cell=2)

    outcomes = {
        ("0_1", 0): (True, None, True),
        ("0_1", 1): (False, "scene_collision", False),
        ("1_0", 0): (True, "no_object_contact", False),
        ("1_0", 1): (True, None, False),
    }
    for (cell, seed), (clear, reason, success) in outcomes.items():
        seed_dir = staged / KEY / "reorient_12" / cell / str(seed)
        np.save(seed_dir / "coll_valid.npy", clear)
        result = {"hand": "inspire", "version": "v8", "success": success}
        if reason is not None:
            result["reason"] = reason
        (seed_dir / "sim_eval.json").write_text(json.dumps(result))
    output = tmp_path / "audit.json"
    report = audit(stage_root=staged, output=output)
    assert report["totals"] == {
        "raw": 4, "scene_clear": 3, "squeeze_contact": 2,
        "mujoco_stable": 1, "scene_collision": 1,
        "no_object_contact": 1, "mujoco_unstable": 1,
    }
    assert report["cells"][0]["mujoco_stable_seed_ids"] == [0]
    assert report["robot_ready"] is False
    with pytest.raises(FileExistsError):
        audit(stage_root=staged, output=output)


def test_audit_rejects_incomplete_or_inconsistent_filter(tmp_path):
    raw, staged = _stage_fixture(tmp_path, expected=1)
    stage(raw_root=raw, stage_root=staged,
          shared_root=tmp_path, expected_per_cell=1)
    with pytest.raises(FileNotFoundError):
        audit(stage_root=staged, output=tmp_path / "audit.json")
    for cell in PAIRS:
        seed_dir = staged / KEY / "reorient_12" / cell / "0"
        np.save(seed_dir / "coll_valid.npy", False)
        (seed_dir / "sim_eval.json").write_text(json.dumps({
            "hand": "inspire", "version": "v8", "success": True,
            "reason": "scene_collision"}))
    with pytest.raises(ValueError, match="inconsistent collision"):
        audit(stage_root=staged, output=tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()
