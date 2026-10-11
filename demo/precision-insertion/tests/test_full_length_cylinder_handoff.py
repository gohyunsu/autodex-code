"""Full-length cylinder addendum is hash-bound and installs append-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_full_length_cylinder_handoff import (  # noqa: E402
    KEY, SCHEMA, _scene_semantics, verify,
)
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from rehydrate_full_length_cylinder_handoff import rehydrate  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    target = tmp_path / "recipient"
    paths = AssetPaths(target, select_mode("cylinder", 20))
    for path, content in ((paths.raw_mesh(KEY), b"raw-key"),
                          (paths.key_planning_mesh, b"planning-key")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    scene_semantics = {}
    pose_sha = {}
    for index, stem in (("0", "000"), ("1", "001")):
        scene = (target / "AutoDex/scene/inspire" / KEY / "table" /
                 f"{index}.json")
        scene.parent.mkdir(parents=True, exist_ok=True)
        scene.write_text(json.dumps({
            "meta": {"pose_idx": stem},
            "scene": {"cuboid": {}, "mesh": {"target": {
                "pose": np.eye(4).tolist(), "scale": [1, 1, 1],
                "file_path": str(paths.key_planning_mesh),
                "urdf_path": str(target / "recipient.urdf"),
            }}},
        }), encoding="utf-8")
        scene_semantics[index] = _scene_semantics(scene)
        pose = paths.key_tabletop_dir / f"{stem}.npy"
        pose.parent.mkdir(parents=True, exist_ok=True)
        np.save(pose, np.eye(4))
        pose_sha[stem] = _sha(pose)
    bundle = tmp_path / "bundle"
    candidate = (bundle / "payload/shared_data/AutoDex/candidates/inspire/v8" /
                 KEY / "table/0/20017")
    candidate.mkdir(parents=True)
    (candidate / "wrist_se3.npy").write_bytes(b"candidate-17")
    (bundle / "README.md").write_text("offline only", encoding="utf-8")
    manifest = {
        "schema": SCHEMA, "robot_ready": False,
        "candidate_ids": ["table/0/20017"],
        "physical_key_raw_mesh_sha256": _sha(paths.raw_mesh(KEY)),
        "physical_key_planning_mesh_sha256": _sha(paths.key_planning_mesh),
        "scene_semantics": scene_semantics,
        "tabletop_pose_sha256": pose_sha,
        "file_sha256": {str(path.relative_to(bundle)): _sha(path)
                        for path in bundle.rglob("*") if path.is_file()},
    }
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, target, paths.candidate_dir / "table/0/20017"


def test_rehydrate_append_only_and_reject_conflict(tmp_path):
    bundle, target, candidate = _fixture(tmp_path)
    assert verify(bundle)["candidate_ids"] == ["table/0/20017"]
    dry = rehydrate(bundle_root=bundle, target_shared_root=target)
    assert dry["candidates_to_install"] == 1 and not candidate.exists()
    done = rehydrate(bundle_root=bundle, target_shared_root=target,
                     install=True)
    assert done["candidates_to_install"] == 1
    assert (candidate / "wrist_se3.npy").read_bytes() == b"candidate-17"
    assert rehydrate(bundle_root=bundle, target_shared_root=target)[
        "identical_existing_candidates"] == 1
    (candidate / "wrist_se3.npy").write_bytes(b"different")
    with pytest.raises(ValueError, match="conflicts"):
        rehydrate(bundle_root=bundle, target_shared_root=target, install=True)


def test_handoff_tamper_rejected_before_install(tmp_path):
    bundle, target, candidate = _fixture(tmp_path)
    source = (bundle / "payload/shared_data/AutoDex/candidates/inspire/v8" /
              KEY / "table/0/20017/wrist_se3.npy")
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        rehydrate(bundle_root=bundle, target_shared_root=target, install=True)
    assert not candidate.exists()
