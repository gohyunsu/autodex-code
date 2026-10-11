"""Square full-key relocation binds each v8 scene to its own key mesh."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehydrate_square_fullkey_handoff import (  # noqa: E402
    _relocate_scene, _scene_relative,
)


def test_square_scene_relocation_uses_the_selected_key():
    origin = Path("/old/shared_data")
    target = Path("/new/shared_data")
    key = "precision_key_0p5mm"
    relative = _scene_relative(key, 3)
    mesh = origin / "object_processing" / key
    source = {
        "scene": {"mesh": {"target": {
            "file_path": str(mesh / "processed_data/mesh/simplified.obj"),
            "urdf_path": str(mesh / "processed_data/urdf/coacd.urdf"),
        }}}, "meta": {"pose_idx": "003"},
    }
    relocated = json.loads(_relocate_scene(
        relative, json.dumps(source).encode(), key=key,
        origin=origin, target=target))
    target_mesh = relocated["scene"]["mesh"]["target"]
    assert target_mesh["file_path"] == str(
        target / "object_processing" / key /
        "processed_data/mesh/simplified.obj")
    assert str(origin) not in json.dumps(relocated)
    with pytest.raises(ValueError, match="unexpected source path"):
        _relocate_scene(relative, json.dumps(relocated).encode(), key=key,
                        origin=origin, target=target)
