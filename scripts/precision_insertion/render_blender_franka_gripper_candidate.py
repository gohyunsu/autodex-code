"""Blender-side still renderer for one analytic Franka-gripper candidate.

The bundle contains exact exported Franka hand visual meshes plus the metric
cylindrical-key mesh and the transforms produced by the runtime analytic grasp
constructor.  Invoke through Blender, not a normal Python interpreter.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("oblique", "side", "top"),
                        required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    return parser.parse_args(sys.argv[separator + 1:])


def _matrix(value: np.ndarray) -> Matrix:
    return Matrix(np.asarray(value, dtype=float).tolist())


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _import_ply(path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_mesh.ply(filepath=str(path))
    imported = list(set(bpy.data.objects) - before)
    if len(imported) != 1:
        raise RuntimeError(f"expected one mesh from {path}, got {len(imported)}")
    obj = imported[0]
    obj.name = name
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(35.0)
    return obj


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat(
        "-Z", "Y").to_euler()


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner)
              for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[index] for point in points)
                       for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points)
                        for index in range(3)))
    return low, high


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.bundle.expanduser().resolve(), allow_pickle=False) as data:
        hand_transform = np.asarray(data["T_world_hand"], dtype=np.float64)
        key_transform = np.asarray(data["T_world_key"], dtype=np.float64)
        hand_mesh_dir = Path(str(data["hand_mesh_dir"].item()))
        hand_names = [str(name) for name in data["hand_names"].tolist()]
        key_mesh = Path(str(data["key_mesh"].item()))
        table_z = float(data["table_z"].item())

    _clear_scene()
    focus = []
    for index, name in enumerate(hand_names):
        obj = _import_ply(hand_mesh_dir / f"geometry_{index:03d}.ply", name)
        obj.matrix_world = _matrix(hand_transform)
        obj.color = (0.78, 0.80, 0.84, 1.0)
        focus.append(obj)

    key = _import_ply(key_mesh, "precision_key")
    key.matrix_world = _matrix(key_transform)
    key.color = (0.04, 0.30, 0.92, 1.0)
    focus.append(key)

    low, high = _world_bounds(focus)
    target = (low + high) * 0.5
    extent = high - low
    span = max(float(extent.x), float(extent.y), float(extent.z), 0.18)

    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(target.x, target.y, table_z - 0.0125))
    table = bpy.context.object
    table.name = "table"
    table.dimensions = (0.55, 0.55, 0.025)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    table.color = (0.20, 0.22, 0.25, 1.0)
    bevel = table.modifiers.new(name="table_edges", type="BEVEL")
    bevel.width = 0.006
    bevel.segments = 3

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.sensor_width = 36.0
    camera.data.lens = 56.0 if args.view != "top" else 60.0
    direction = {
        "oblique": Vector((1.25, -1.40, 0.90)),
        "side": Vector((0.10, -1.75, 0.25)),
        "top": Vector((0.05, -0.08, 1.80)),
    }[args.view].normalized()
    camera.location = target + direction * (span * 3.25)
    _look_at(camera, target)

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.curvature_ridge_factor = 1.35
    scene.display.shading.curvature_valley_factor = 0.85
    scene.display.shading.show_specular_highlight = True
    scene.display.shading.show_object_outline = False
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0.055, 0.065, 0.080)
    scene.display.render_aa = "32"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    bpy.ops.render.render(write_still=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
