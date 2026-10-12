"""Rebinding source-bound v8 scenes must preserve geometry and provenance."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rehydrate_cylinder_key_scenes import (  # noqa: E402
    KEY_REL, SCENE_REL, rehydrate,
)


def _fixture(tmp_path):
    source = tmp_path / "bundle" / "shared_data"
    origin = tmp_path / "producer" / "shared_data"
    target = tmp_path / "recipient" / "shared_data"
    for subpath in (KEY_REL / "mesh/simplified.obj",
                    KEY_REL / "urdf/coacd.urdf"):
        installed = target / subpath
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_text("source verified geometry")
    for stem in (0, 1):
        scene = source / SCENE_REL / f"{stem}.json"
        scene.parent.mkdir(parents=True, exist_ok=True)
        scene.write_text(json.dumps({
            "meta": {"pose_idx": f"{stem:03d}"},
            "scene": {"mesh": {"target": {
                "pose": [0, 0, .08, 1, 0, 0, 0],
                "file_path": str(origin / KEY_REL / "mesh/simplified.obj"),
                "urdf_path": str(origin / KEY_REL / "urdf/coacd.urdf"),
            }}, "cuboid": {"table": {"dims": [2, 2, .2]}}},
        }))
    return source, origin, target


def test_rehydrate_v8_scenes_only_in_staging_root(tmp_path):
    source, origin, target = _fixture(tmp_path)
    options = dict(source_shared_root=source,
                   source_origin_shared_root=origin,
                   target_shared_root=target)
    preflight = rehydrate(**options)
    assert preflight["files_to_install"] == 2
    assert not (target / SCENE_REL / "0.json").exists()
    installed = rehydrate(**options, install=True)
    assert installed["robot_ready"] is False
    for stem in (0, 1):
        original = (source / SCENE_REL / f"{stem}.json").read_bytes()
        assert (target / "AutoDex/precision_insertion/cylindrical/"
                "key_scene_relocation/original_scenes" /
                f"{stem}.json").read_bytes() == original
        rebound = json.loads((target / SCENE_REL / f"{stem}.json").read_text())
        assert rebound["scene"]["mesh"]["target"]["file_path"] == str(
            target / KEY_REL / "mesh/simplified.obj")


def test_rehydrate_rejects_another_recipient_scene(tmp_path):
    source, origin, target = _fixture(tmp_path)
    conflict = target / SCENE_REL / "0.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text('{"unrelated": true}')
    with pytest.raises(ValueError, match="recipient scene differs"):
        rehydrate(source_shared_root=source,
                  source_origin_shared_root=origin,
                  target_shared_root=target)
