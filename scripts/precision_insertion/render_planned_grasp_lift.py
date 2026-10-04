#!/usr/bin/env python3
"""Render an exported FR3+Inspire grasp/lift plan with the real visual meshes.

This script is deliberately visualization-only.  It consumes the NPZ from
``export_planned_grasp_lift.py`` and never invokes the robot or planner.  The
key remains fixed during approach/closure, then is rigidly attached to the
Inspire base link for the already validated vertical lift.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from yourdfpy import URDF


ROBOT_URDF = (
    Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot" /
    "fr3_inspire_description" / "fr3_inspire.urdf"
)


def _rows(array: np.ndarray, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("every rendered phase needs at least two frames")
    indices = np.linspace(0, len(array) - 1, count).round().astype(int)
    return np.asarray(array[indices], dtype=np.float64)


def _o3d_mesh(mesh: trimesh.Trimesh, color: tuple[float, float, float]):
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    result.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    result.compute_vertex_normals()
    result.paint_uniform_color(color)
    return result


def _robot_mesh(urdf: URDF, q: np.ndarray, joint_names: list[str]) -> trimesh.Trimesh:
    urdf.update_cfg(dict(zip(joint_names, np.asarray(q, dtype=float))))
    mesh = urdf.scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected combined Trimesh, got {type(mesh)!r}")
    return mesh


def _add_caption(path: Path, phase: str, index: int, count: int) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not read rendered frame {path}")
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 82), (248, 248, 248), thickness=-1)
    cv2.putText(
        image,
        f"FR3 + Inspire | {phase}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (36, 36, 36),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "PLANNING PREVIEW - NOT ROBOT EXECUTION; insertion/reorientation not included",
        (24, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 80, 180),
        1,
        cv2.LINE_AA,
    )
    progress_x = int((w - 1) * (index + 1) / count)
    cv2.rectangle(image, (0, h - 7), (progress_x, h - 1), (60, 120, 230), -1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write captioned frame {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, default=ROBOT_URDF)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--approach-frames", type=int, default=65)
    parser.add_argument("--close-frames", type=int, default=24)
    parser.add_argument("--lift-frames", type=int, default=55)
    parser.add_argument("--final-hold-frames", type=int, default=15)
    parser.add_argument("--keep-frames", type=Path)
    args = parser.parse_args()

    args.trajectory = args.trajectory.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.robot_urdf = args.robot_urdf.expanduser().resolve()
    with np.load(args.trajectory, allow_pickle=False) as data:
        approach = _rows(data["approach_q"], args.approach_frames)
        close = _rows(data["close_q"], args.close_frames)
        lift = _rows(data["lift_q"], args.lift_frames)
        hold = np.repeat(lift[-1][None, :], args.final_hold_frames, axis=0)
        q_frames = np.concatenate([approach, close, lift, hold], axis=0)
        phases = (
            ["approach"] * len(approach)
            + ["close hand"] * len(close)
            + ["10 cm validated lift"] * len(lift)
            + ["lift complete"] * len(hold)
        )
        object_pose = np.asarray(data["object_world_se3"], dtype=np.float64)
        object_mesh_path = Path(str(data["object_mesh_path"].item()))
        joint_names = [str(name) for name in data["joint_names"].tolist()]

    urdf = URDF.load(str(args.robot_urdf), build_scene_graph=True, load_meshes=True)
    missing = sorted(set(joint_names) - set(urdf.actuated_joint_names))
    if missing:
        raise RuntimeError(f"trajectory joints absent from URDF: {missing}")
    key_local = trimesh.load(object_mesh_path, force="mesh", process=False)
    if not isinstance(key_local, trimesh.Trimesh):
        raise TypeError(f"expected key Trimesh, got {type(key_local)!r}")

    lift_start_index = len(approach) + len(close)
    _robot_mesh(urdf, q_frames[lift_start_index], joint_names)
    hand_start = urdf.get_transform("base_link", urdf.base_link)
    object_in_hand = np.linalg.inv(hand_start) @ object_pose

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.view.set_post_processing(False)
    renderer.scene.set_background([0.96, 0.97, 0.98, 1.0])
    renderer.scene.scene.set_sun_light(
        [0.45, -0.60, -0.65], [1.0, 1.0, 1.0], 80000
    )
    renderer.scene.scene.enable_sun_light(True)
    robot_material = o3d.visualization.rendering.MaterialRecord()
    robot_material.shader = "defaultLit"
    object_material = o3d.visualization.rendering.MaterialRecord()
    object_material.shader = "defaultLit"
    table_material = o3d.visualization.rendering.MaterialRecord()
    table_material.shader = "defaultLit"

    # Show only the cell-relevant tabletop patch.  Its top remains at the
    # exact planner table height, z=0.04 m.
    table = trimesh.creation.box(extents=[1.25, 1.35, 0.08])
    table.apply_translation([0.42, 0.0, 0.0])
    renderer.scene.add_geometry(
        "table", _o3d_mesh(table, (0.50, 0.52, 0.55)), table_material
    )
    renderer.setup_camera(
        43.0,
        np.asarray([0.20, 0.0, 0.47]),
        np.asarray([1.45, -1.55, 1.18]),
        np.asarray([0.0, 0.0, 1.0]),
    )

    frame_dir = Path(tempfile.mkdtemp(prefix="precision_grasp_lift_"))
    try:
        for index, (q, phase) in enumerate(zip(q_frames, phases)):
            robot = _robot_mesh(urdf, q, joint_names)
            hand_world = urdf.get_transform("base_link", urdf.base_link)
            key_world_pose = (
                object_pose
                if index < lift_start_index
                else hand_world @ object_in_hand
            )
            key = key_local.copy()
            key.apply_transform(key_world_pose)

            if index:
                renderer.scene.remove_geometry("robot")
                renderer.scene.remove_geometry("key")
            renderer.scene.add_geometry(
                "robot", _o3d_mesh(robot, (0.83, 0.84, 0.88)), robot_material
            )
            renderer.scene.add_geometry(
                "key", _o3d_mesh(key, (0.10, 0.38, 0.86)), object_material
            )
            output = frame_dir / f"frame_{index:04d}.png"
            # Open3D interprets PNG quality as compression level [0, 9], not
            # the JPEG-style [0, 100] scale.
            o3d.io.write_image(str(output), renderer.render_to_image(), 9)
            _add_caption(output, phase, index, len(q_frames))
            if index % 10 == 0 or index + 1 == len(q_frames):
                print(f"rendered {index + 1}/{len(q_frames)}", flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ]
        subprocess.run(command, check=True)
        if args.keep_frames is not None:
            keep = args.keep_frames.expanduser().resolve()
            if keep.exists():
                raise FileExistsError(f"refusing to replace frame directory: {keep}")
            shutil.copytree(frame_dir, keep)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
