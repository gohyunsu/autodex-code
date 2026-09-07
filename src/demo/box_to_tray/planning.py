"""Planner-side use of the measured containers.

The demo reuses ``src/demo/inference/run_demo.py`` for every motion.  Only the
world differs: a step may pick an object out of a real container, so every
measured container is collision geometry, and an object picked *out of* one
rests on its floor rather than on the table.  Both corrections are applied
through that runner's ``scene_cfg_hook``, which keeps the proven motion code
unforked.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from src.demo.box_to_tray.fixtures import ContainerFixture, add_container_obstacles


def object_pose_robot(scene_cfg: Mapping[str, object]) -> np.ndarray:
    """Robot-frame 4x4 pose of the planning target in a ``scene_cfg``."""
    from autodex.utils.conversion import cart2se3

    return cart2se3(np.asarray(scene_cfg["mesh"]["target"]["pose"], dtype=np.float64))


def set_object_pose(scene_cfg: dict, pose_robot: np.ndarray) -> dict:
    from autodex.utils.conversion import se32cart

    cfg = dict(scene_cfg)
    mesh = {k: dict(v) for k, v in cfg["mesh"].items()}
    mesh["target"]["pose"] = se32cart(np.asarray(pose_robot, dtype=np.float64)).tolist()
    cfg["mesh"] = mesh
    return cfg


def make_scene_hook(
    fixtures: Sequence[ContainerFixture],
    *,
    source: Optional[ContainerFixture] = None,
    require_in_source: bool = True,
    snap_max_m: float = 0.02,
    model: str = "mesh",
    load_vertices: Optional[Callable[[str], np.ndarray]] = None,
    report: Optional[Callable[[str], None]] = print,
) -> Callable[[dict], dict]:
    """Build the ``scene_cfg_hook`` for one step of the demo.

    ``source`` is the container this step picks *out of*, or ``None`` for an
    object standing on the table.  The returned hook

    1. rejects a pose outside the source container *before the arm moves*, so a
       detection that landed on the table or on the rim cannot start a pick,
    2. raises an implausibly sunken pose onto that container's measured floor —
       the same correction the table demos apply against the table, and
    3. merges *every* measured container into the planning world, since the arm
       has to avoid the ones it is not picking from as well.  ``model`` selects
       the container's own mesh (exact, the default) or the fitted wall cuboids;
       see :func:`add_container_obstacles`.
    """
    if isinstance(fixtures, ContainerFixture):
        fixtures = [fixtures]

    def hook(scene_cfg: dict) -> dict:
        pose = object_pose_robot(scene_cfg)
        cfg = scene_cfg
        if source is not None:
            if require_in_source and not source.contains(pose):
                raise RuntimeError(
                    f"the estimated object centre {np.round(pose[:3, 3], 3).tolist()} is "
                    f"outside the measured {source.name} interior; put the object in it "
                    "(or re-measure if the container moved). No robot motion was sent.")
            loader = load_vertices
            if loader is None:
                from src.demo.continuous_basket.tabletop import load_mesh_vertices
                loader = load_mesh_vertices
            from src.demo.continuous_basket.tabletop import raise_to_table

            floor_z = source.floor_z
            vertices = loader(scene_cfg["mesh"]["target"]["file_path"])
            corrected, applied, bottom = raise_to_table(
                pose, vertices, surface_z=floor_z, max_raise_m=snap_max_m)
            if applied > 0:
                cfg = set_object_pose(cfg, corrected)
                if report is not None:
                    report(f"[{source.name}] raised the planning pose by "
                           f"{applied * 1000:.1f} mm onto the measured floor "
                           f"({floor_z:.3f} m)")
            elif bottom < floor_z and report is not None:
                report(f"[{source.name}] the object mesh bottom sits "
                       f"{(floor_z - bottom) * 1000:.1f} mm below the measured floor; "
                       f"the correction exceeds {snap_max_m * 1000:.0f} mm and was refused")
        return add_container_obstacles(cfg, fixtures, model=model)

    return hook
