"""Blender-side renderer for clean, original-mesh precision-insertion videos.

Invoke through Blender, not the normal Python interpreter:

    blender --background --python render_blender_actual_mesh_animation.py -- \
      BUNDLE.npz --output VIDEO.mp4
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
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--view",
        choices=("task", "key-socket", "reorient", "overview"),
        default="task",
        help=(
            "task shows the workspace from the socket side; key-socket uses "
            "the opposite oblique view to reduce palm occlusion; reorient "
            "frames the pickup and reset regions; overview "
            "keeps the full arm"
        ),
    )
    parser.add_argument(
        "--still-frame",
        type=int,
        help="render one 1-based frame to --output instead of an MP4",
    )
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    return parser.parse_args(sys.argv[separator + 1:])


def _matrix(array: np.ndarray) -> Matrix:
    return Matrix(np.asarray(array, dtype=float).tolist())


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _set_color(obj: bpy.types.Object, rgba: tuple[float, float, float, float]) -> None:
    obj.color = rgba


def _import_ply(path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_mesh.ply(filepath=str(path))
    imported = list(set(bpy.data.objects) - before)
    if len(imported) != 1:
        raise RuntimeError(f"expected one object from {path}, got {len(imported)}")
    obj = imported[0]
    obj.name = name
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(35.0)
    return obj


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


def _keyframe_transform(
    obj: bpy.types.Object,
    transforms: np.ndarray,
) -> None:
    for frame, transform in enumerate(transforms, start=1):
        obj.matrix_world = _matrix(transform)
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)
    if obj.animation_data and obj.animation_data.action:
        for curve in obj.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


def main() -> int:
    args = _arguments()
    bundle = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(bundle, allow_pickle=False) as data:
        robot_transforms = np.asarray(
            data["robot_geometry_transforms"], dtype=np.float64
        )
        object_poses = np.asarray(data["object_poses"], dtype=np.float64)
        socket_pose = np.asarray(data["socket_pose"], dtype=np.float64)
        mesh_dir = Path(str(data["robot_mesh_dir"].item()))
        geometry_names = [str(value) for value in data["geometry_names"].tolist()]
        key_path = Path(str(data["object_mesh_path"].item()))
        socket_path = Path(str(data["socket_mesh_path"].item()))

    _clear_scene()
    for index, name in enumerate(geometry_names):
        obj = _import_ply(mesh_dir / f"geometry_{index:03d}.ply", name)
        _set_color(obj, (0.64, 0.67, 0.72, 1.0))
        _keyframe_transform(obj, robot_transforms[:, index])

    key = _import_ply(key_path, "precision_key")
    _set_color(key, (0.025, 0.22, 0.90, 1.0))
    _keyframe_transform(key, object_poses)

    socket = _import_ply(socket_path, "precision_socket")
    _set_color(socket, (0.72, 0.025, 0.02, 1.0))
    socket.matrix_world = _matrix(socket_pose)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.31, 0.0, 0.0))
    table = bpy.context.object
    table.name = "table"
    table.dimensions = (1.05, 0.95, 0.08)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _set_color(table, (0.34, 0.37, 0.42, 1.0))
    bevel = table.modifiers.new(name="soft_table_edges", type="BEVEL")
    bevel.width = 0.012
    bevel.segments = 3

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.sensor_width = 36.0
    if args.view == "task":
        # Close fixed view of the entire key-to-socket workspace.  The key
        # starts near (0.40, 0.18), the socket is near (0.45, -0.10), and the
        # 12 cm lift remains in frame.  Some proximal arm links are allowed to
        # leave frame because contact geometry is the primary evidence here.
        camera.location = (0.90, -0.78, 0.62)
        camera.data.lens = 52.0
        _look_at(camera, Vector((0.425, 0.035, 0.15)))
    elif args.view == "key-socket":
        camera.location = (0.94, 0.58, 0.48)
        camera.data.lens = 55.0
        _look_at(camera, Vector((0.425, 0.02, 0.14)))
    elif args.view == "reorient":
        camera.location = (0.98, -0.72, 0.63)
        camera.data.lens = 52.0
        _look_at(camera, Vector((0.46, 0.10, 0.15)))
    else:
        camera.location = (1.22, -1.40, 0.94)
        camera.data.lens = 43.0
        _look_at(camera, Vector((0.28, -0.015, 0.275)))

    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_start = 1
    scene.frame_end = len(object_poses)
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
    scene.display.shading.show_object_outline = False
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.92, 0.94, 0.97)
    scene.display.render_aa = "32"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.render.filepath = str(output)
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.use_file_extension = True

    if args.still_frame is not None:
        if not 1 <= args.still_frame <= len(object_poses):
            raise ValueError(
                f"--still-frame must be in [1, {len(object_poses)}]"
            )
        scene.render.image_settings.file_format = "PNG"
        scene.frame_set(args.still_frame)
        bpy.ops.render.render(write_still=True)
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        bpy.ops.render.render(animation=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
