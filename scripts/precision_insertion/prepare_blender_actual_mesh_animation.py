#!/usr/bin/env python3
"""Prepare original FR3/Inspire visual meshes and animation transforms for Blender.

Run this script in the visualization environment.  It evaluates the URDF at
every saved trajectory frame, exports each original (non-decimated) visual
mesh once, and writes only the per-frame world transforms to an NPZ bundle.
Blender then renders that bundle without needing yourdfpy in its bundled
Python interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from yourdfpy import URDF


def _export_original_geometries(
    urdf: URDF,
    mesh_dir: Path,
    source_urdf: Path,
) -> list[str]:
    manifest_path = mesh_dir / "manifest.json"
    geometry_names = list(urdf.scene.geometry)
    expected = {
        "source_urdf": str(source_urdf),
        "geometry_names": geometry_names,
        "total_faces": int(sum(
            len(urdf.scene.geometry[name].faces) for name in geometry_names
        )),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_exist = all(
            (mesh_dir / f"geometry_{index:03d}.ply").is_file()
            for index in range(len(geometry_names))
        )
        if existing == expected and files_exist:
            return geometry_names
        raise RuntimeError(
            f"visual mesh cache does not match {source_urdf}: {mesh_dir}"
        )

    mesh_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(geometry_names):
        mesh = urdf.scene.geometry[name]
        mesh.export(mesh_dir / f"geometry_{index:03d}.ply")
    manifest_path.write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )
    return geometry_names


def _export_task_mesh(source: Path, output: Path) -> Path:
    if not output.is_file():
        mesh = trimesh.load(source, force="mesh", process=False)
        mesh.export(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mesh-cache", type=Path)
    args = parser.parse_args()

    trajectory = args.trajectory.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trajectory.with_name(trajectory.stem + "_blender_bundle.npz")
    )
    mesh_cache = (
        args.mesh_cache.expanduser().resolve()
        if args.mesh_cache is not None
        else trajectory.parent / "blender_original_robot_visuals"
    )

    with np.load(trajectory, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        object_poses = np.asarray(data["object_pose"], dtype=np.float64)
        socket_pose = np.asarray(data["socket_pose"], dtype=np.float64)
        joint_names = [str(value) for value in data["joint_names"].tolist()]
        object_mesh_path = Path(str(data["object_mesh_path"].item())).resolve()
        socket_mesh_path = Path(str(data["socket_mesh_path"].item())).resolve()
        robot_urdf_path = Path(str(data["robot_urdf_path"].item())).resolve()

    urdf = URDF.load(
        str(robot_urdf_path), build_scene_graph=True, load_meshes=True
    )
    geometry_names = _export_original_geometries(
        urdf, mesh_cache, robot_urdf_path
    )
    key_blender_mesh = _export_task_mesh(
        object_mesh_path, mesh_cache / "precision_key.ply"
    )
    socket_blender_mesh = _export_task_mesh(
        socket_mesh_path, mesh_cache / "precision_socket.ply"
    )
    transforms = np.empty(
        (len(qpos), len(geometry_names), 4, 4), dtype=np.float32
    )
    for frame, q in enumerate(qpos):
        urdf.update_cfg(dict(zip(joint_names, q)))
        for index, name in enumerate(geometry_names):
            transform, _ = urdf.scene.graph.get(name)
            transforms[frame, index] = transform

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        robot_geometry_transforms=transforms,
        object_poses=object_poses.astype(np.float32),
        socket_pose=socket_pose.astype(np.float32),
        robot_mesh_dir=np.asarray(str(mesh_cache)),
        geometry_names=np.asarray(geometry_names),
        object_mesh_path=np.asarray(str(key_blender_mesh)),
        socket_mesh_path=np.asarray(str(socket_blender_mesh)),
        source_trajectory=np.asarray(str(trajectory)),
    )
    print(output)
    print(json.dumps({
        "frames": len(qpos),
        "robot_visual_geometries": len(geometry_names),
        "original_robot_faces": int(sum(
            len(urdf.scene.geometry[name].faces) for name in geometry_names
        )),
        "mesh_cache": str(mesh_cache),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
