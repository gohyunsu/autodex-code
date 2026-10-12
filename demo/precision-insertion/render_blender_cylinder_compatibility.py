"""Render all six exact cylinder socket/key pairs in one 16:9 animation.

Invoke with Blender and pass script arguments after ``--``.  This is a CAD
compatibility illustration; it does not contain a robot or physical result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


GAPS = ("01", "03", "05", "10", "15", "20")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--still-frame", type=int)
    at = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    return parser.parse_args(sys.argv[at + 1:])


def _reset() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_obj(path: Path, name: str, color) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.obj(filepath=str(path), axis_forward="Y", axis_up="Z")
    imported = [obj for obj in bpy.context.scene.objects
                if obj not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"no mesh imported from {path}")
    if len(imported) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in imported:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
    obj = imported[0]
    obj.name = name
    obj.color = color
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _look_at(obj: bpy.types.Object, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat(
        "-Z", "Y").to_euler()


def _text(body: str, location, size: float = 0.021) -> None:
    curve = bpy.data.curves.new(f"text_{body}", type="FONT")
    curve.body = body
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = 0.0003
    obj = bpy.data.objects.new(f"label_{body}", curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.color = (0.07, 0.09, 0.14, 1.0)


def _configure(args: argparse.Namespace, frames: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.curvature_ridge_factor = 1.25
    scene.display.shading.curvature_valley_factor = 0.75
    scene.display.shading.show_specular_highlight = True
    scene.world.color = (0.82, 0.85, 0.90)
    scene.display.shading.background_type = "WORLD"
    scene.display.render_aa = "32"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.render.filepath = str(args.output)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    if args.still_frame is None:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        bpy.ops.render.render(animation=True)
    else:
        if not 1 <= args.still_frame <= frames:
            raise ValueError("still frame is outside the animation")
        scene.render.image_settings.file_format = "PNG"
        scene.frame_set(args.still_frame)
        bpy.ops.render.render(write_still=True)


def main() -> int:
    args = _args()
    root = args.shared_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _reset()
    key_path = (root / "object_processing/precision_key_cylinder_r15_h80/"
                "raw_mesh/precision_key_cylinder_r15_h80.obj")
    positions = (
        (-0.25, 0.14), (0.0, 0.14), (0.25, 0.14),
        (-0.25, -0.14), (0.0, -0.14), (0.25, -0.14),
    )
    frames = 132
    for gap, (x, y) in zip(GAPS, positions):
        object_id = f"precision_socket_cylinder_gap_{gap}mm"
        socket_path = root / "object_processing" / object_id / "raw_mesh" / f"{object_id}.obj"
        geometry_path = (root / "AutoDex/precision_insertion/fixtures" /
                         object_id / "task_geometry.json")
        if not key_path.is_file() or not socket_path.is_file() or not geometry_path.is_file():
            raise FileNotFoundError(f"missing exact cylinder asset for gap {gap}")
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        socket = _import_obj(socket_path, f"socket_{gap}",
                             (0.78, 0.035, 0.025, 1.0))
        socket.location = (x, y, 0.012)
        key = _import_obj(key_path, f"key_{gap}",
                          (0.025, 0.25, 0.92, 1.0))
        preinsert = np.asarray(geometry["T_socket_key_preinsert"], dtype=float)
        seated = np.asarray(geometry["T_socket_key_seated"], dtype=float)
        for frame, local in ((1, preinsert), (24, preinsert), (76, seated),
                             (104, seated), (132, preinsert)):
            world = np.eye(4)
            world[:3, 3] = (x, y, 0.012)
            key.matrix_world = Matrix((world @ local).tolist())
            key.keyframe_insert(data_path="location", frame=frame)
            key.keyframe_insert(data_path="rotation_euler", frame=frame)
        if key.animation_data and key.animation_data.action:
            for fcurve in key.animation_data.action.fcurves:
                for point in fcurve.keyframe_points:
                    point.interpolation = "BEZIER"
        _text(f"{int(gap)} mm", (x, y - 0.085, 0.013), 0.018)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    table = bpy.context.object
    table.name = "presentation_table"
    table.dimensions = (0.90, 0.55, 0.024)
    table.color = (0.58, 0.62, 0.68, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = table.modifiers.new(name="soft_edges", type="BEVEL")
    bevel.width = 0.012
    bevel.segments = 3
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.72, -1.10, 0.68)
    camera.data.lens = 49.0
    camera.data.sensor_width = 36.0
    _look_at(camera, (0.0, -0.015, 0.065))
    bpy.context.scene.camera = camera
    _configure(args, frames)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
