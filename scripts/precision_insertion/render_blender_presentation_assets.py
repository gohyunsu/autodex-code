#!/usr/bin/env python3
"""Render exact-mesh precision-insertion presentation assets in Blender.

Invoke through Blender and place script arguments after ``--``.  The modes
share a 16:9 camera, materials, lighting, and uncluttered background.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


SHARED = Path.home() / "shared_data"
DEFAULT_PRESENTATION = (
    SHARED / "AutoDex/precision_insertion/presentation_assets"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "family", "compatibility", "tabletop-key", "tabletop-socket",
            "contact-policy", "reorient-concept",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-file", type=Path)
    parser.add_argument("--target-pose-file", type=Path)
    parser.add_argument("--pose-id", default="000")
    parser.add_argument("--hand-pregrasp", type=Path)
    parser.add_argument("--hand-grasp", type=Path)
    parser.add_argument("--contact-points", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def _reset() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights):
        # Do not mutate while iterating.
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def _material(name: str, color: tuple[float, float, float, float], *,
              metallic: float = 0.0, roughness: float = 0.45) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Alpha"].default_value = color[3]
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if color[3] < 1.0:
        material.blend_method = "BLEND"
        material.use_screen_refraction = True
        material.show_transparent_back = True
    return material


def _load_mesh(path: Path, name: str, material: bpy.types.Material) -> bpy.types.Object:
    before = set(bpy.context.scene.objects)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix == ".obj":
        # The AutoDex meshes are already metric Z-up robot coordinates.
        bpy.ops.import_scene.obj(filepath=str(path), axis_forward="Y", axis_up="Z")
    else:
        raise ValueError(f"unsupported mesh format: {path}")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path}")
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    obj = meshes[0]
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def _matrix(array: np.ndarray) -> Matrix:
    return Matrix(np.asarray(array, dtype=float).tolist())


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _camera(location: tuple[float, float, float], target: tuple[float, float, float],
            lens: float = 52.0) -> bpy.types.Object:
    data = bpy.data.cameras.new("Camera")
    data.lens = lens
    data.sensor_width = 36.0
    obj = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)
    bpy.context.scene.camera = obj
    return obj


def _plate(size: float = 0.24) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.006))
    plate = bpy.context.object
    plate.name = "tabletop"
    plate.dimensions = (size, size, 0.012)
    plate.data.materials.append(
        _material("table", (0.004, 0.007, 0.012, 1.0), metallic=0.02, roughness=0.76)
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return plate


def _text(label: str, location: tuple[float, float, float], size: float,
          *, align: str = "CENTER", rotation_x_deg: float = 74.0,
          color: tuple[float, float, float, float] = (0.92, 0.94, 0.98, 1.0),
          ) -> bpy.types.Object:
    curve = bpy.data.curves.new(f"text_{label}", type="FONT")
    curve.body = label
    curve.align_x = align
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = size * 0.015
    obj = bpy.data.objects.new(f"text_{label}", curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (math.radians(rotation_x_deg), 0.0, 0.0)
    obj.data.materials.append(_material(f"textmat_{label}", color))
    return obj


def _lighting() -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.018, 0.024, 0.034, 1.0
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.10
    for name, location, energy, size in (
        ("key", (0.2, -0.3, 0.5), 42.0, 0.35),
        ("fill", (-0.35, -0.1, 0.3), 20.0, 0.30),
        ("rim", (0.0, 0.35, 0.45), 32.0, 0.25),
    ):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        _look_at(obj, (0.0, 0.0, 0.04))


def _configure(args: argparse.Namespace, *, video: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG" if video else "PNG"
    scene.render.filepath = str(args.output)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.7
    if video:
        scene.render.fps = args.fps
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"


def _paths() -> dict[str, Path]:
    objects = SHARED / "object_processing"
    result = {
        "0.1": DEFAULT_PRESENTATION / "00_geometry/meshes/plug_gap_0p1.stl",
        "0.3": objects / "precision_key_0p3mm/raw_mesh/precision_key_0p3mm.obj",
        "0.5": objects / "precision_key_0p5mm/raw_mesh/precision_key_0p5mm.obj",
        "1.0": objects / "precision_key_1p0mm/raw_mesh/precision_key_1p0mm.obj",
        "1.5": objects / "precision_key_1p5mm/raw_mesh/precision_key_1p5mm.obj",
        "socket": objects / "precision_socket_unified/raw_mesh/precision_socket_unified.obj",
        "allowed": objects / "precision_key_1p5mm/processed_data/mesh/contact_allowed.obj",
        "forbidden": objects / "precision_key_1p5mm/processed_data/mesh/contact_forbidden.obj",
        "geometry": SHARED / "AutoDex/precision_insertion/fixtures/unified_socket/task_geometry.json",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing presentation inputs: " + ", ".join(missing))
    return result


def _family(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    blue = _material("key blue", (0.035, 0.24, 0.95, 1.0), metallic=0.15, roughness=0.3)
    red = _material("socket red", (0.86, 0.035, 0.055, 1.0), metallic=0.08, roughness=0.35)
    xs = [-0.22, -0.13, -0.04, 0.05, 0.14]
    for x, gap in zip(xs, ("0.1", "0.3", "0.5", "1.0", "1.5")):
        key = _load_mesh(paths[gap], f"key_{gap}", blue)
        key.location = (x, 0.0, 0.0)
        _text(f"{gap} mm", (x, -0.055, 0.0), 0.012)
    socket = _load_mesh(paths["socket"], "socket", red)
    socket.location = (0.255, 0.0, 0.0)
    _text("socket", (0.255, -0.065, 0.0), 0.012)
    _camera((0.02, -0.92, 0.36), (0.02, 0.0, 0.035), 50.0)
    _configure(args, video=False)
    bpy.ops.render.render(write_still=True)


def _compatibility(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    blue = _material("key blue", (0.025, 0.25, 0.95, 1.0), metallic=0.1, roughness=0.3)
    red = _material("socket red", (0.90, 0.03, 0.05, 1.0), metallic=0.08, roughness=0.35)
    socket = _load_mesh(paths["socket"], "socket", red)
    key = _load_mesh(paths["1.5"], "key", blue)
    geometry = json.loads(paths["geometry"].read_text(encoding="utf-8"))
    pre = np.asarray(geometry["T_socket_key_preinsert"], dtype=float)
    seated = np.asarray(geometry["T_socket_key_seated"], dtype=float)
    start = pre.copy()
    start[2, 3] += 0.055
    key.matrix_world = _matrix(start)
    for frame, pose in ((1, start), (35, pre), (80, seated), (110, seated), (155, start)):
        key.matrix_world = _matrix(pose)
        key.keyframe_insert(data_path="location", frame=frame)
        key.keyframe_insert(data_path="rotation_euler", frame=frame)
    for curve in key.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 155
    _camera((0.18, -0.27, 0.18), (0.0, 0.0, 0.065), 58.0)
    _configure(args, video=True)
    bpy.ops.render.render(animation=True)


def _tabletop_key(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.pose_file is None:
        raise ValueError("tabletop-key requires --pose-file")
    blue = _material("key blue", (0.025, 0.25, 0.95, 1.0), metallic=0.1, roughness=0.32)
    key = _load_mesh(paths["1.5"], "key", blue)
    key.matrix_world = _matrix(np.load(args.pose_file))
    _plate()
    _text(args.pose_id, (-0.055, -0.060, 0.001), 0.018, rotation_x_deg=0.0)
    _camera((0.17, -0.28, 0.19), (0.0, 0.0, 0.025), 55.0)
    _configure(args, video=False)
    bpy.ops.render.render(write_still=True)


def _tabletop_socket(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    red = _material("socket red", (0.90, 0.03, 0.05, 1.0), metallic=0.08, roughness=0.35)
    _load_mesh(paths["socket"], "socket", red)
    _plate(0.25)
    _text("000", (-0.055, -0.060, 0.001), 0.018, rotation_x_deg=0.0)
    _camera((0.18, -0.30, 0.20), (0.0, 0.0, 0.035), 56.0)
    _configure(args, video=False)
    bpy.ops.render.render(write_still=True)


def _reorient_concept(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """Render an exact-mesh task primitive without claiming a robot plan."""
    if args.pose_file is None or args.target_pose_file is None:
        raise ValueError(
            "reorient-concept requires --pose-file and --target-pose-file"
        )
    blue = _material(
        "key blue", (0.025, 0.25, 0.95, 1.0), metallic=0.1, roughness=0.32
    )
    key = _load_mesh(paths["1.5"], "key", blue)
    start = np.asarray(np.load(args.pose_file), dtype=float)
    target = np.asarray(np.load(args.target_pose_file), dtype=float)
    lifted_start = start.copy()
    lifted_start[2, 3] += 0.085
    lifted_target = target.copy()
    lifted_target[2, 3] += 0.085
    key.rotation_mode = "QUATERNION"
    for frame, pose in (
        (1, start), (25, lifted_start), (70, lifted_target), (95, target),
        (115, target),
    ):
        translation, rotation, scale = _matrix(pose).decompose()
        key.location = translation
        key.rotation_quaternion = rotation
        key.scale = scale
        key.keyframe_insert(data_path="location", frame=frame)
        key.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    for curve in key.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"
    _plate()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 115
    _camera((0.32, -0.55, 0.34), (0.0, 0.0, 0.04), 54.0)
    _configure(args, video=True)
    bpy.ops.render.render(animation=True)


def _contact_policy(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.hand_grasp is None or args.contact_points is None:
        raise ValueError("contact-policy requires --hand-grasp and --contact-points")
    green = _material("allowed", (0.02, 0.80, 0.23, 1.0), roughness=0.38)
    red = _material("forbidden", (0.95, 0.025, 0.035, 1.0), roughness=0.38)
    white = _material("hand", (0.24, 0.29, 0.36, 1.0), metallic=0.1, roughness=0.42)
    yellow = _material("contacts", (1.0, 0.62, 0.02, 1.0), metallic=0.05, roughness=0.25)
    _load_mesh(paths["allowed"], "allowed contact", green)
    _load_mesh(paths["forbidden"], "forbidden contact", red)
    hand = _load_mesh(args.hand_grasp, "Inspire hand", white)
    contacts = np.load(args.contact_points)
    for index, point in enumerate(contacts):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.0018,
                                             location=tuple(float(v) for v in point))
        sphere = bpy.context.object
        sphere.name = f"contact_{index}"
        sphere.data.materials.append(yellow)
    target = np.eye(4)
    start = np.eye(4)
    start[1, 3] = -0.07
    hand.matrix_world = _matrix(start)
    for frame, pose in ((1, start), (55, target), (95, target), (145, start)):
        hand.matrix_world = _matrix(pose)
        hand.keyframe_insert(data_path="location", frame=frame)
        hand.keyframe_insert(data_path="rotation_euler", frame=frame)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 145
    _camera((0.25, -0.43, 0.22), (0.0, 0.0, 0.04), 52.0)
    _configure(args, video=True)
    bpy.ops.render.render(animation=True)


def main() -> int:
    args = _arguments()
    args.output = args.output.expanduser().resolve()
    for field in (
        "pose_file", "target_pose_file", "hand_pregrasp", "hand_grasp",
        "contact_points",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _reset()
    _lighting()
    paths = _paths()
    {
        "family": _family,
        "compatibility": _compatibility,
        "tabletop-key": _tabletop_key,
        "tabletop-socket": _tabletop_socket,
        "contact-policy": _contact_policy,
        "reorient-concept": _reorient_concept,
    }[args.mode](args, paths)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
