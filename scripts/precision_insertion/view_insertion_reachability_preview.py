#!/usr/bin/env python3
"""Interactively inspect the lift-to-insertion diagnostic in viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


DEFAULT_INPUT = (
    Path.home() / "shared_data/AutoDex/precision_insertion/visualizations/"
    "common_grasp_78_lift_to_insertion_preview.npz"
)


def _pose_wxyz(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3], np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    source = args.trajectory.expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        phases = [str(value) for value in data["phase"].tolist()]
        object_poses = np.asarray(data["object_pose"], dtype=np.float64)
        socket_pose = np.asarray(data["socket_pose"], dtype=np.float64)
        seated_pose = np.asarray(data["desired_seated_key_pose"], dtype=np.float64)
        collisions = np.asarray(data["collision_counts"], dtype=np.int64)
        distances = np.asarray(
            data["min_hand_socket_signed_distance_m"], dtype=np.float64
        )
        joint_names = [str(value) for value in data["joint_names"].tolist()]
        key_mesh_path = Path(str(data["object_mesh_path"].item()))
        socket_mesh_path = Path(str(data["socket_mesh_path"].item()))
        robot_urdf_path = Path(str(data["robot_urdf_path"].item()))
        status = str(data["preview_status"].item())

    robot = URDF.load(str(robot_urdf_path), build_scene_graph=True, load_meshes=True)
    key_mesh = trimesh.load(key_mesh_path, force="mesh", process=False)
    socket_mesh = trimesh.load(socket_mesh_path, force="mesh", process=False)
    socket_mesh.visual.face_colors = [210, 40, 35, 255]
    key_mesh.visual.face_colors = [30, 90, 225, 255]
    goal_mesh = key_mesh.copy()
    goal_mesh.visual.face_colors = [35, 210, 75, 90]
    table = trimesh.creation.box(extents=[1.25, 1.35, 0.08])
    table.apply_translation([0.42, 0.0, 0.0])
    table.visual.face_colors = [125, 130, 140, 255]

    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_trimesh("/table", table)
    socket_position, socket_wxyz = _pose_wxyz(socket_pose)
    server.scene.add_mesh_trimesh(
        "/socket", socket_mesh, position=socket_position, wxyz=socket_wxyz
    )
    goal_position, goal_wxyz = _pose_wxyz(seated_pose)
    server.scene.add_mesh_trimesh(
        "/desired_seated_key", goal_mesh,
        position=goal_position, wxyz=goal_wxyz,
    )
    key_handle = server.scene.add_mesh_trimesh("/key", key_mesh)

    phase_text = server.gui.add_text("phase", initial_value="", disabled=True)
    contract_text = server.gui.add_text(
        "contract", initial_value=status, disabled=True
    )
    collision_text = server.gui.add_text(
        "hand/socket check", initial_value="", disabled=True
    )
    sample_text = server.gui.add_text("sample", initial_value="", disabled=True)
    slider = server.gui.add_slider(
        "sample", min=0, max=len(qpos) - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("autoplay", initial_value=True)
    looping = server.gui.add_checkbox("loop", initial_value=True)
    speed = server.gui.add_slider(
        "samples/frame", min=1, max=8, step=1, initial_value=2
    )

    def show(index: int) -> None:
        index = int(np.clip(index, 0, len(qpos) - 1))
        robot.update_cfg(dict(zip(joint_names, qpos[index])))
        robot_mesh = robot.scene.to_geometry()
        robot_mesh.visual.face_colors = (
            [230, 35, 30, 255] if collisions[index] else [205, 208, 218, 255]
        )
        server.scene.add_mesh_trimesh("/robot", robot_mesh)
        key_position, key_wxyz = _pose_wxyz(object_poses[index])
        key_handle.position = key_position
        key_handle.wxyz = key_wxyz
        phase_text.value = phases[index]
        sample_text.value = f"{index + 1}/{len(qpos)}"
        collision_text.value = (
            f"COLLISION: {collisions[index]} penetrating samples, "
            f"min={distances[index] * 1000.0:.2f} mm"
            if collisions[index] else
            f"no penetrating sample, min={distances[index] * 1000.0:.2f} mm"
        )

    @slider.on_update
    def _on_slider(_event) -> None:
        show(int(slider.value))

    show(0)
    try:
        port = server.get_port()
    except Exception:
        port = args.port
    print(f"[viser] http://localhost:{port}")
    print(f"[viser] source={source}")
    print("[viser] red robot = proven sampled hand/socket penetration")
    try:
        while True:
            if bool(playing.value):
                next_index = int(slider.value) + int(speed.value)
                if next_index >= len(qpos):
                    if bool(looping.value):
                        next_index = 0
                    else:
                        next_index = len(qpos) - 1
                        playing.value = False
                slider.value = next_index
                show(next_index)
            time.sleep(1.0 / 24.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
