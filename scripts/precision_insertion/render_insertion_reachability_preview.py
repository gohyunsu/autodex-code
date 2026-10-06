#!/usr/bin/env python3
"""Render the actual-mesh lift-to-insertion diagnostic as an MP4.

Grey robot frames are not observed to penetrate the socket in the sampled
hand-surface diagnostic.  Red robot frames contain a proven hand/socket
collision.  Only approach, close, and lift came from cuRobo; all later frames
are visibly labelled as a non-executable IK diagnostic.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from yourdfpy import URDF


DEFAULT_INPUT = (
    Path.home() / "shared_data/AutoDex/precision_insertion/visualizations/"
    "common_grasp_78_lift_to_insertion_preview.npz"
)
DEFAULT_OUTPUT = (
    Path.home() / "shared_data/AutoDex/precision_insertion/visualizations/"
    "common_grasp_78_lift_to_insertion_preview.mp4"
)


def _simplify(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """CPU quadric simplification used only for the MP4 renderer."""
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    simplified = legacy.simplify_quadric_decimation(max(4, int(target_faces)))
    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )


def _simplified_robot_geometries(
    urdf: URDF, total_target_faces: int,
) -> dict[str, trimesh.Trimesh]:
    total_faces = sum(len(mesh.faces) for mesh in urdf.scene.geometry.values())
    result: dict[str, trimesh.Trimesh] = {}
    for name, mesh in urdf.scene.geometry.items():
        share = max(8, round(total_target_faces * len(mesh.faces) / total_faces))
        result[name] = _simplify(mesh, share)
    return result


def _robot_mesh(
    urdf: URDF,
    geometries: dict[str, trimesh.Trimesh],
    q: np.ndarray,
    names: list[str],
) -> trimesh.Trimesh:
    urdf.update_cfg(dict(zip(names, np.asarray(q, dtype=float))))
    parts: list[trimesh.Trimesh] = []
    for name, mesh in geometries.items():
        transform, _ = urdf.scene.graph.get(name)
        moved = mesh.copy()
        moved.apply_transform(transform)
        parts.append(moved)
    return trimesh.util.concatenate(parts)


def _triangles(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.vertices)[np.asarray(mesh.faces)]


def _render_cpu(
    output: Path,
    robot: trimesh.Trimesh,
    key: trimesh.Trimesh,
    socket: trimesh.Trimesh,
    table: trimesh.Trimesh,
    seated_ghost: trimesh.Trimesh,
    *,
    collision: bool,
    closeup: bool,
    width: int,
    height: int,
) -> None:
    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    collections = (
        # Matplotlib sorts each Poly3D collection as one unit, so a translucent
        # table preserves the robot view instead of incorrectly painting the
        # whole tabletop over links that are physically above it.
        (table, (0.48, 0.50, 0.53), 0.13),
        (socket, (0.64, 0.05, 0.04), 1.0),
        (seated_ghost, (0.10, 0.78, 0.24), 0.24),
        (
            robot,
            (0.92, 0.14, 0.12) if collision else
            (0.50, 0.53, 0.59) if closeup else (0.82, 0.83, 0.87),
            1.0,
        ),
        (key, (0.08, 0.35, 0.88), 1.0),
    )
    for mesh, color, alpha in collections:
        poly = Poly3DCollection(
            _triangles(mesh), facecolor=color, edgecolor="none",
            linewidth=0.0, alpha=alpha,
        )
        axis.add_collection3d(poly)
    axis.set_xlim(-0.20, 0.78)
    axis.set_ylim(-0.62, 0.45)
    axis.set_zlim(0.0, 0.92)
    axis.set_box_aspect((0.98, 1.07, 0.92))
    axis.view_init(elev=23.0, azim=-58.0)
    axis.set_axis_off()
    figure.patch.set_facecolor((0.96, 0.97, 0.98))
    axis.set_facecolor((0.96, 0.97, 0.98))
    if closeup:
        inset = figure.add_axes([0.665, 0.085, 0.315, 0.43], projection="3d")
        centre = np.asarray(key.centroid)
        radius = 0.18
        local_collections = (
            (socket, (0.64, 0.05, 0.04), 1.0),
            (seated_ghost, (0.10, 0.78, 0.24), 0.20),
            (robot, (0.92, 0.14, 0.12) if collision else (0.38, 0.41, 0.48), 1.0),
            (key, (0.08, 0.35, 0.88), 1.0),
        )
        for mesh, color, alpha in local_collections:
            triangles = _triangles(mesh)
            keep = np.linalg.norm(triangles.mean(axis=1) - centre, axis=1) < radius
            if not np.any(keep):
                continue
            inset.add_collection3d(Poly3DCollection(
                triangles[keep], facecolor=color, edgecolor="none",
                linewidth=0.0, alpha=alpha,
            ))
        extent = 0.13
        inset.set_xlim(centre[0] - extent, centre[0] + extent)
        inset.set_ylim(centre[1] - extent, centre[1] + extent)
        inset.set_zlim(max(0.0, centre[2] - extent), centre[2] + extent)
        inset.set_box_aspect((1.0, 1.0, 1.0))
        inset.view_init(elev=22.0, azim=-54.0)
        inset.set_axis_off()
        inset.set_facecolor((0.90, 0.92, 0.95))
        inset.set_title("hand/key close-up", fontsize=9, color=(0.18, 0.18, 0.20))
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def _caption(
    path: Path,
    phase: str,
    collision: bool,
    index: int,
    count: int,
    *,
    preview_kind: str,
) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not read frame {path}")
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 105), (248, 248, 248), -1)
    cv2.putText(
        image, f"FR3 + Inspire + real key/socket meshes | {phase}",
        (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (35, 35, 35), 2,
        cv2.LINE_AA,
    )
    is_geometric_success = preview_kind == "sampled_geometric_success"
    is_validated = phase.startswith("validated")
    if is_geometric_success:
        contract = "SAMPLED GEOMETRIC IK PREVIEW - NOT CUROBO OR ROBOT EXECUTION"
    elif is_validated:
        contract = "CUROBO-VALIDATED PICK/LIFT INPUT"
    else:
        contract = "DIAGNOSTIC IK ONLY - NOT COLLISION-PLANNED OR ROBOT-EXECUTABLE"
    cv2.putText(
        image, contract, (22, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
        (15, 105, 35) if (is_validated or is_geometric_success)
        else (25, 80, 180), 1, cv2.LINE_AA,
    )
    if collision:
        state = "REJECTED: sampled hand/key/environment constraint violation"
    elif is_geometric_success:
        state = "Sampled policy, table, key/socket and hand/socket checks pass"
    else:
        state = "No sampled hand/socket penetration at this frame"
    cv2.putText(
        image, state, (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
        (25, 25, 210) if collision else (70, 70, 70), 1, cv2.LINE_AA,
    )
    progress = int((width - 1) * (index + 1) / count)
    cv2.rectangle(image, (0, height - 7), (progress, height - 1),
                  (65, 120, 225), -1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write captioned frame {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--robot-faces", type=int, default=10000)
    parser.add_argument("--keep-frames", type=Path)
    args = parser.parse_args()

    source = args.trajectory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        phases = [str(value) for value in data["phase"].tolist()]
        object_poses = np.asarray(data["object_pose"], dtype=np.float64)
        socket_pose = np.asarray(data["socket_pose"], dtype=np.float64)
        additional_socket_poses = (
            np.asarray(data["additional_socket_poses"], dtype=np.float64)
            if "additional_socket_poses" in data.files else
            np.empty((0, 4, 4), dtype=np.float64)
        )
        seated_pose = np.asarray(data["desired_seated_key_pose"], dtype=np.float64)
        collisions = np.asarray(data["collision_counts"], dtype=np.int64) > 0
        joint_names = [str(value) for value in data["joint_names"].tolist()]
        object_mesh_path = Path(str(data["object_mesh_path"].item()))
        socket_mesh_path = Path(str(data["socket_mesh_path"].item()))
        robot_urdf_path = Path(str(data["robot_urdf_path"].item()))
        preview_kind = (
            str(data["preview_kind"].item())
            if "preview_kind" in data.files else "collision_diagnostic"
        )
    if not (len(qpos) == len(phases) == len(object_poses) == len(collisions)):
        raise RuntimeError("preview arrays have inconsistent lengths")

    urdf = URDF.load(str(robot_urdf_path), build_scene_graph=True, load_meshes=True)
    robot_geometries = _simplified_robot_geometries(urdf, args.robot_faces)
    key_local = trimesh.load(object_mesh_path, force="mesh", process=False)
    socket = trimesh.load(socket_mesh_path, force="mesh", process=False)
    socket.apply_transform(socket_pose)
    if len(additional_socket_poses):
        sockets = [socket]
        socket_local = trimesh.load(socket_mesh_path, force="mesh", process=False)
        for pose in additional_socket_poses:
            extra = socket_local.copy()
            extra.apply_transform(pose)
            sockets.append(extra)
        socket = trimesh.util.concatenate(sockets)
    seated_ghost = key_local.copy()
    seated_ghost.apply_transform(seated_pose)

    table = trimesh.creation.box(extents=[1.25, 1.35, 0.08])
    table.apply_translation([0.42, 0.0, 0.0])

    frame_dir = Path(tempfile.mkdtemp(prefix="precision_insertion_preview_"))
    try:
        for index, (q, phase, key_pose, collision) in enumerate(zip(
            qpos, phases, object_poses, collisions
        )):
            robot = _robot_mesh(urdf, robot_geometries, q, joint_names)
            key = key_local.copy()
            key.apply_transform(key_pose)
            frame = frame_dir / f"frame_{index:04d}.png"
            _render_cpu(
                frame, robot, key, socket, table, seated_ghost,
                collision=bool(collision),
                closeup=preview_kind == "sampled_geometric_success",
                width=args.width, height=args.height,
            )
            _caption(
                frame, phase, bool(collision), index, len(qpos),
                preview_kind=preview_kind,
            )
            if index % 20 == 0 or index + 1 == len(qpos):
                print(f"rendered {index + 1}/{len(qpos)}", flush=True)

        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "warning",
            "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True)
        thumbnail = output.with_suffix(".png")
        shutil.copy2(frame_dir / f"frame_{len(qpos) - 1:04d}.png", thumbnail)
        if args.keep_frames is not None:
            keep = args.keep_frames.expanduser().resolve()
            if keep.exists():
                raise FileExistsError(f"refusing to replace frame directory: {keep}")
            shutil.copytree(frame_dir, keep)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    print(output)
    print(output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
