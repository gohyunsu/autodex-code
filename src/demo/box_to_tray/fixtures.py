"""Open containers (box, bowl, ...) as measured planning fixtures.

``open_box``, ``smallbowl`` and friends are normal AutoDex objects: each has a
mesh under the v8 asset root and a FoundPose representation, so a fixture's 6-D
pose is estimated by exactly the same distributed init pipeline the picked
object uses.  This module turns that pose plus the mesh into the things the
planner needs:

* **collision walls** — four upright cuboids and the inner floor, derived from
  the mesh's own outer and interior footprints.  Cuboids rather than the mesh
  itself because the planner drops every mesh obstacle from the worlds it
  builds with ``include_obj_obstacle=False`` (lift, carry, retreat); cuboids
  survive those paths, so the box is still an obstacle while the hand is
  climbing out of it.
* **two heights and an interior region** — the floor an object rests on inside
  it (used to correct a sunken pose), the rim (which is also where an object
  being put *into* the container is released), and the interior footprint that
  decides whether a detection is actually inside it.

The interior is found by casting vertical rays down onto the posed mesh: a ray
inside lands on the floor, a ray on the rim lands near the top, and a ray
outside misses.  That is exact for the real asset, unlike a silhouette hull,
which cannot see a concavity at all.

The footprint frame comes from the posed geometry, never from the pose's own
rotation: asset frames are not consistent (the box meshes stand on local z, the
bowl meshes on local y), so reading a yaw off the pose would rotate a bowl's
walls into nonsense.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BOX_OBJ = "open_box"
# Vertical safety band applied to the measured interior footprint before it is
# used to accept a detection or to place the walls.
DEFAULT_INTERIOR_INSET = 0.015
# A ray that lands this far below the rim is inside the box rather than on it.
RIM_TOLERANCE_M = 0.02


def footprint_rect(points_xy: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Minimum-area rectangle ``(centre, size, yaw)`` of a footprint.

    Taken from the posed geometry so it holds for any asset frame, and for a
    round fixture where "the yaw" is arbitrary but the rectangle is not.
    """
    import cv2

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (sx, sy), angle_deg = cv2.minAreaRect(pts)
    yaw = math.radians(float(angle_deg))
    size = (float(sx), float(sy))
    if size[1] > size[0]:                    # keep the long side first
        size = (size[1], size[0])
        yaw += math.pi / 2.0
    yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
    return (float(cx), float(cy)), size, float(yaw)


def yaw_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def yaw_quat(yaw: float) -> list:
    """cuRobo ``[qw, qx, qy, qz]`` for a rotation about robot +z."""
    return [float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))]


def point_in_region(point: Sequence[float], region: Mapping[str, object]) -> bool:
    """Whether a robot-frame point lies inside an oriented interior region."""
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {p.shape}")
    if not (float(region["z_min"]) <= p[2] <= float(region["z_max"])):
        return False
    local = yaw_rot(float(region["yaw"])).T @ (p[:2] - np.asarray(region["center_xy"], float))
    half = np.asarray(region["size_xy"], dtype=np.float64) / 2.0
    return bool(np.all(np.abs(local) <= half))


def _rect_from_local(local_xy: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Axis-aligned centre/size of points already expressed in the box frame."""
    lo = local_xy.min(axis=0)
    hi = local_xy.max(axis=0)
    return ((float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)),
            (float(hi[0] - lo[0]), float(hi[1] - lo[1])))


def load_mesh(mesh_path: str):
    """Load a fixture mesh without letting trimesh weld or reorder anything."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def transform_mesh(mesh, pose_robot: np.ndarray):
    posed = mesh.copy()
    posed.apply_transform(np.asarray(pose_robot, dtype=np.float64))
    return posed


def measure_container(mesh, pose_robot: np.ndarray, *, grid: int = 48,
                      interior_inset: float = DEFAULT_INTERIOR_INSET,
                      top_clearance: float = 0.10,
                      rim_tolerance: float = RIM_TOLERANCE_M,
                      floor_percentile: float = 5.0) -> dict:
    """Measure floor/rim heights and the outer and interior footprints.

    Everything is returned in the robot frame; the footprints are expressed in
    the fitted rectangle's own yawed frame, so a rotated box is described
    tightly rather than by an inflated axis-aligned box.  A round bowl gets a
    square footprint, whose corners are empty air: modelling them as wall is
    conservative, not wrong.
    """
    pose = np.asarray(pose_robot, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {pose.shape}")
    posed = transform_mesh(mesh, pose)
    vertices = np.asarray(posed.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("the box mesh has no vertices")
    rect_centre, outer_size, yaw = footprint_rect(vertices[:, :2])
    R = yaw_rot(yaw)
    centre_xy = np.asarray(rect_centre, dtype=np.float64)
    local = (vertices[:, :2] - centre_xy) @ R          # == R.T @ (xy - centre)
    outer_centre_local, _measured = _rect_from_local(local)
    base_z = float(vertices[:, 2].min())
    rim_z = float(vertices[:, 2].max())

    # Vertical probes over the outer footprint.  A hit well below the rim is
    # the box interior; a hit near the rim is the wall top; a miss is outside.
    us = np.linspace(-outer_size[0] / 2, outer_size[0] / 2, grid) + outer_centre_local[0]
    vs = np.linspace(-outer_size[1] / 2, outer_size[1] / 2, grid) + outer_centre_local[1]
    gu, gv = np.meshgrid(us, vs, indexing="ij")
    probes_local = np.stack([gu.reshape(-1), gv.reshape(-1)], axis=1)
    probes_xy = probes_local @ R.T + centre_xy
    origins = np.concatenate(
        [probes_xy, np.full((probes_xy.shape[0], 1), rim_z + 0.05)], axis=1)
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))
    locations, index_ray, _tri = posed.ray.intersects_location(
        origins, directions, multiple_hits=True)
    interior_local = np.empty((0, 2))
    floor_z = base_z
    if len(index_ray):
        # Keep the highest hit per ray: that is the first surface seen going down.
        top_hit = np.full(origins.shape[0], -np.inf)
        np.maximum.at(top_hit, index_ray, locations[:, 2])
        inside = np.isfinite(top_hit) & (top_hit < rim_z - rim_tolerance)
        if inside.any():
            interior_local = probes_local[inside]
            floor_z = float(np.percentile(top_hit[inside], floor_percentile))
    if len(interior_local) < 4:
        raise RuntimeError(
            "no interior surface was found on the box mesh; the pose may be wrong "
            "or the mesh is not an open container")
    interior_centre_local, interior_size = _rect_from_local(interior_local)
    # The probe grid samples the interior, so its extent is one cell short of
    # the real wall on each side; growing it back would push the region into
    # the wall, so keep the conservative value and apply the inset as well.
    interior_size = (max(interior_size[0] - 2 * interior_inset, 0.02),
                     max(interior_size[1] - 2 * interior_inset, 0.02))
    outer_centre = tuple(np.asarray(outer_centre_local) @ R.T + centre_xy)
    interior_centre = tuple(np.asarray(interior_centre_local) @ R.T + centre_xy)
    return {
        "yaw": yaw,
        "outer_center_xy": [float(v) for v in outer_centre],
        "outer_size_xy": [float(v) for v in outer_size],
        "interior_center_xy": [float(v) for v in interior_centre],
        "interior_size_xy": [float(v) for v in interior_size],
        "base_z": base_z,
        "floor_z": floor_z,
        "rim_z": rim_z,
        "n_interior_probes": int(len(interior_local)),
        "interior_region": {
            "center_xy": [float(v) for v in interior_centre],
            "yaw": yaw,
            "size_xy": [float(interior_size[0]), float(interior_size[1])],
            "z_min": floor_z,
            "z_max": float(rim_z + top_clearance),
        },
    }


def container_obstacles(measure: Mapping[str, object], *, prefix: str = "box",
                        include_floor: bool = True) -> Dict[str, dict]:
    """Wall and floor cuboids filling the outer footprint minus the interior.

    Each wall spans from the base to the rim, so the hand is kept out of it for
    the whole climb, not only above the floor.
    """
    yaw = float(measure["yaw"])
    quat = yaw_quat(yaw)
    R = yaw_rot(yaw)
    outer_c = np.asarray(measure["outer_center_xy"], dtype=np.float64)
    outer = np.asarray(measure["outer_size_xy"], dtype=np.float64)
    inner_c = np.asarray(measure["interior_center_xy"], dtype=np.float64)
    inner = np.asarray(measure["interior_size_xy"], dtype=np.float64)
    base_z = float(measure["base_z"])
    rim_z = float(measure["rim_z"])
    floor_z = float(measure["floor_z"])
    height = rim_z - base_z
    if height <= 0:
        raise ValueError("the box rim is not above its base")

    # Interior edges in the box frame, relative to the OUTER rectangle centre,
    # so an off-centre cavity still produces four correctly sized walls.
    offset = R.T @ (inner_c - outer_c)
    inner_lo = offset - inner / 2.0
    inner_hi = offset + inner / 2.0
    outer_lo = -outer / 2.0
    outer_hi = outer / 2.0
    bands = {
        "x_pos": (inner_hi[0], outer_hi[0], 0),
        "x_neg": (outer_lo[0], inner_lo[0], 0),
        "y_pos": (inner_hi[1], outer_hi[1], 1),
        "y_neg": (outer_lo[1], inner_lo[1], 1),
    }
    out: Dict[str, dict] = {}
    for tag, (lo, hi, axis) in bands.items():
        thickness = float(hi - lo)
        if thickness <= 1e-4:
            continue
        local = np.zeros(2)
        local[axis] = float((lo + hi) / 2.0)
        dims = [0.0, 0.0, height]
        dims[axis] = thickness
        dims[1 - axis] = float(outer[1 - axis])
        world = outer_c + R @ local
        out[f"{prefix}_wall_{tag}"] = {
            "dims": [float(d) for d in dims],
            "pose": [float(world[0]), float(world[1]),
                     float(base_z + height / 2.0)] + quat,
        }
    if not out:
        raise ValueError("the measured container has no wall band; check the pose and mesh")
    if include_floor:
        floor_thickness = max(floor_z - base_z, 0.005)
        out[f"{prefix}_floor"] = {
            "dims": [float(inner[0]), float(inner[1]), float(floor_thickness)],
            "pose": [float(inner_c[0]), float(inner_c[1]),
                     float(floor_z - floor_thickness / 2.0)] + quat,
        }
    return out


@dataclass
class ContainerFixture:
    """One measured open container, for the planner and for ``result.json``."""

    name: str
    obj: str
    mesh_path: str
    pose_robot: np.ndarray
    measure: dict
    obstacles: Dict[str, dict]
    perception: dict

    @property
    def floor_z(self) -> float:
        return float(self.measure["floor_z"])

    @property
    def rim_z(self) -> float:
        return float(self.measure["rim_z"])

    @property
    def interior_region(self) -> dict:
        return dict(self.measure["interior_region"])

    def contains(self, pose_robot: np.ndarray) -> bool:
        xyz = np.asarray(pose_robot, dtype=np.float64)[:3, 3]
        return point_in_region(xyz, self.interior_region)

    def to_json(self) -> dict:
        data = asdict(self)
        data["pose_robot"] = np.asarray(self.pose_robot, dtype=np.float64).tolist()
        return data

    @property
    def center_xy(self) -> Tuple[float, float]:
        cx, cy = self.measure["interior_center_xy"]
        return float(cx), float(cy)

    @property
    def bearing_deg(self) -> float:
        """Robot-frame direction of the container centre (+x forward, +CCW)."""
        cx, cy = self.center_xy
        return float(math.degrees(math.atan2(cy, cx)))

    def release_surface_z(self, gap: float = 0.02) -> float:
        """Height an object is released at when it is put *into* this container.

        The object is lowered until its own mesh bottom is just above the rim
        and let go there, so it drops the last centimetre into the container
        instead of the hand descending inside it.
        """
        return float(self.rim_z + float(gap))

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ContainerFixture":
        return cls(name=str(data.get("name", data["obj"])),
                   obj=str(data["obj"]), mesh_path=str(data["mesh_path"]),
                   pose_robot=np.asarray(data["pose_robot"], dtype=np.float64),
                   measure=dict(data["measure"]),
                   obstacles={k: dict(v) for k, v in dict(data["obstacles"]).items()},
                   perception=dict(data.get("perception") or {}))


def build_fixture(name: str, obj: str, mesh_path: str, pose_robot: np.ndarray, *,
                  mesh=None, perception: Optional[dict] = None,
                  interior_inset: float = DEFAULT_INTERIOR_INSET,
                  rim_tolerance: float = RIM_TOLERANCE_M,
                  floor_percentile: float = 5.0,
                  include_floor: bool = True) -> ContainerFixture:
    """Measure a posed container mesh and build its planner obstacles."""
    mesh = load_mesh(mesh_path) if mesh is None else mesh
    measure = measure_container(mesh, pose_robot, interior_inset=interior_inset,
                                rim_tolerance=rim_tolerance,
                                floor_percentile=floor_percentile)
    return ContainerFixture(
        name=name, obj=obj, mesh_path=str(mesh_path),
        pose_robot=np.asarray(pose_robot, dtype=np.float64),
        measure=measure,
        obstacles=container_obstacles(measure, prefix=name,
                                      include_floor=include_floor),
        perception=dict(perception or {}))


def add_container_obstacles(scene_cfg: dict, fixtures, *, model: str = "mesh") -> dict:
    """Merge every measured container into a ``scene_cfg`` copy.

    ``model="mesh"`` puts the container's own mesh in the planning world, which
    is exact — a round bowl is a round bowl, not the square band a cuboid fit
    can express.  The approach and grasp are planned against it.  cuRobo drops
    *every* mesh from the worlds built with ``include_obj_obstacle=False``
    (lift, carry, retreat), so in mesh mode the container is not an obstacle
    during those phases; they are a vertical lift, a transfer above the rim and
    a vertical retreat, which is why that is acceptable.

    ``model="cuboid"`` uses the fitted wall/floor boxes instead, which survive
    those phases but over-approximate a round container.  ``model="both"`` adds
    each, at the cost of the cuboids' over-approximation during the approach.
    """
    import copy

    from autodex.utils.conversion import se32cart

    if model not in {"mesh", "cuboid", "both"}:
        raise ValueError("model must be 'mesh', 'cuboid' or 'both'")
    if isinstance(fixtures, ContainerFixture):
        fixtures = [fixtures]
    cfg = copy.deepcopy(scene_cfg)
    cuboids = dict(cfg.get("cuboid") or {})
    meshes = dict(cfg.get("mesh") or {})
    for fixture in fixtures:
        if model in ("mesh", "both"):
            if fixture.name in meshes:
                raise ValueError(f"container mesh {fixture.name!r} collides with an existing entry")
            meshes[fixture.name] = {
                "pose": se32cart(np.asarray(fixture.pose_robot, dtype=np.float64)).tolist(),
                "file_path": str(fixture.mesh_path),
            }
        if model in ("cuboid", "both"):
            for name, cuboid in fixture.obstacles.items():
                if name in cuboids:
                    raise ValueError(f"container cuboid {name!r} collides with an existing entry")
                cuboids[name] = {"dims": list(cuboid["dims"]), "pose": list(cuboid["pose"])}
    cfg["cuboid"] = cuboids
    cfg["mesh"] = meshes
    return cfg


def add_container_cuboids(scene_cfg: dict, fixtures) -> dict:
    """Backwards-compatible alias for the cuboid-only model."""
    return add_container_obstacles(scene_cfg, fixtures, model="cuboid")
