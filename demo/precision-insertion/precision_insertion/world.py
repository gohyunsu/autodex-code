"""Build the session's fixed-fixture collision world without legacy hooks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

from autodex.utils.conversion import se32cart

from .geometry import validate_se3


def add_fixed_mesh_fixtures(
    scene_cfg: dict,
    fixed_fixtures: Mapping[str, Mapping] | None,
) -> dict:
    """Return a new cuRobo scene with validated socket fixture meshes.

    The original scene is never mutated. The fixture name cannot shadow the
    target object or another mesh. The caller owns the free-key versus
    attached-key scene transition after grasp.
    """
    result = copy.deepcopy(scene_cfg)
    if not fixed_fixtures:
        return result
    meshes = result.setdefault("mesh", {})
    for name, fixture in sorted(fixed_fixtures.items()):
        if not isinstance(name, str) or not name or name == "target":
            raise ValueError(f"invalid fixed fixture name: {name!r}")
        if name in meshes:
            raise ValueError(f"fixed fixture would replace scene mesh {name!r}")
        if not isinstance(fixture, Mapping):
            raise ValueError(f"fixed fixture {name!r} must be a mapping")
        pose_robot = validate_se3(
            fixture.get("pose_robot"), name=f"fixed fixture {name} pose_robot")
        mesh_value = fixture.get("collision_mesh")
        if not isinstance(mesh_value, (str, Path)) or not str(mesh_value):
            raise FileNotFoundError(f"fixed fixture {name!r} collision mesh missing")
        mesh_path = Path(mesh_value).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"fixed fixture {name!r} collision mesh not found: {mesh_path}")
        meshes[name] = {
            "pose": se32cart(pose_robot).tolist(),
            "file_path": str(mesh_path),
        }
    return result
