#!/usr/bin/env python3
"""Replay a saved lift-test animation without cameras, planner, or CUDA."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


ARM_VISUALS = {
    "franka": {
        "label": "Franka FR3 + Inspire hand",
        "urdf": ("fr3_inspire_description", "fr3_inspire.urdf"),
    },
    "xarm": {
        "label": "XArm6 + Inspire hand",
        "urdf": ("inspire_description", "xarm_inspire.urdf"),
    },
}


def _arm_from_episode(episode: Path) -> str:
    """Read the arm selected by run_session, preserving old FR3 episodes."""
    request_path = episode / "request.json"
    try:
        with request_path.open() as f:
            arm = json.load(f).get("arm", "franka")
    except (OSError, json.JSONDecodeError):
        arm = "franka"
    if arm not in ARM_VISUALS:
        raise SystemExit(
            f"unsupported arm {arm!r} in {request_path}; expected one of "
            f"{', '.join(sorted(ARM_VISUALS))}")
    return str(arm)


def _pose_wxyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    return np.asarray(T)[:3, 3], q_xyzw[[3, 0, 1, 2]]


def _interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate a saved rigid pose for browser frames between q samples."""
    from scipy.spatial.transform import Rotation, Slerp

    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.eye(4)
    out[:3, 3] = ((1.0 - alpha) * np.asarray(T0)[:3, 3]
                  + alpha * np.asarray(T1)[:3, 3])
    rotations = Rotation.from_matrix(np.stack([
        np.asarray(T0)[:3, :3], np.asarray(T1)[:3, :3]], axis=0))
    out[:3, :3] = Slerp([0.0, 1.0], rotations)(alpha).as_matrix()
    return out


def _add_box(server, name: str, cfg: dict) -> None:
    import trimesh
    from autodex.utils.conversion import cart2se3

    box = trimesh.creation.box(extents=np.asarray(cfg["dims"], dtype=float))
    box.apply_transform(cart2se3(np.asarray(cfg["pose"], dtype=float)))
    box.visual.vertex_colors = np.tile([145, 155, 170, 100], (len(box.vertices), 1))
    server.scene.add_mesh_trimesh(f"/world/{name}", box)


def _add_proxy(server, proxy: dict) -> None:
    """Show the Charuco proxy as a cyan closed XY line, when Viser supports it."""
    vertices = np.asarray(proxy["vertices_xy_m"], dtype=float)
    z = float(proxy["table_surface_z_m"]) + 0.002
    pts = np.c_[vertices, np.full(4, z)]
    starts, ends = pts, np.roll(pts, -1, axis=0)
    try:
        server.scene.add_line_segments(
            "/board_proxy", points=np.stack([starts, ends], axis=1),
            colors=np.tile([80, 220, 230], (4, 2, 1)), line_width=3.0)
    except Exception:
        # Older viser builds lack line segments; the board is still present as
        # the collision table and replay remains usable.
        pass


def _as_mesh(trimesh, path: Path):
    mesh = trimesh.load(path, process=False)
    return mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh


def _load_object_render_mesh(trimesh, episode: Path, target_cfg: dict):
    """Prefer a colorized raw mesh while retaining planning geometry for FK.

    ``scene_cfg`` intentionally stores the simplified mesh used by cuRobo.
    That collision mesh carries neither UVs nor texture/material data, so it
    is not an informative visual for common objects such as Pepsi.  A raw
    mesh with per-vertex RGB is render-only: it never changes the pose,
    collision model, or Jacobian calculation.  Require near-identical bounds
    before using it, preventing a pretty but misregistered visualization.
    """
    planning_path = Path(target_cfg["file_path"])
    planning = _as_mesh(trimesh, planning_path)
    try:
        request = json.loads((episode / "request.json").read_text())
        obj_name = str(request["object"])
        # .../<obj>/processed_data/mesh/simplified.obj → .../<obj>/raw_mesh/<obj>.obj
        raw_path = planning_path.parents[2] / "raw_mesh" / f"{obj_name}.obj"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw = _as_mesh(trimesh, raw_path)
        centroid_delta = float(np.linalg.norm(raw.centroid - planning.centroid))
        extent_delta = float(np.max(np.abs(raw.extents - planning.extents)))
        colors = getattr(raw.visual, "vertex_colors", None)
        has_color = colors is not None and len(colors) == len(raw.vertices)
        if centroid_delta > 0.005 or extent_delta > 0.005 or not has_color:
            raise ValueError(
                f"raw mesh frame/color validation failed: centroid={centroid_delta:.4f}m "
                f"extent={extent_delta:.4f}m color={has_color}")
        return raw, {
            "kind": "raw_vertex_color",
            "path": str(raw_path),
            "centroid_delta_m": centroid_delta,
            "max_extent_delta_m": extent_delta,
        }
    except Exception as exc:
        return planning, {
            "kind": "planning_mesh_fallback",
            "path": str(planning_path),
            "reason": repr(exc),
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", required=True, type=Path)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--arm", choices=sorted(ARM_VISUALS),
                   help="override the arm recorded in request.json")
    args = p.parse_args()
    episode = args.episode.expanduser().resolve()
    animation_path = episode / "animation.npz"
    if not animation_path.exists():
        raise SystemExit(f"animation not found: {animation_path}")
    with (episode / "scene_cfg.json").open() as f:
        scene_cfg = json.load(f)
    with (episode / "board_proxy.json").open() as f:
        board_proxy = json.load(f)
    anim = np.load(animation_path)
    qpos = np.asarray(anim["qpos"], dtype=float)
    phases = np.asarray(anim["phase"])
    object_poses = np.asarray(anim["object_pose"], dtype=float)
    if len(qpos) != len(object_poses):
        raise SystemExit("animation qpos/object_pose length mismatch")
    if "time_s" in anim:
        time_s = np.asarray(anim["time_s"], dtype=float).reshape(-1)
        trajectory_kind = str(np.asarray(
            anim["trajectory_kind"] if "trajectory_kind" in anim
            else "timestamped_execution_reference").item())
    else:
        # Episodes written before timestamped execution support only preserve
        # geometric samples.  Retain a usable viewer but label the limitation.
        time_s = np.arange(len(qpos), dtype=float) * 0.05
        trajectory_kind = "legacy_geometric_replay"
    if (len(time_s) != len(qpos) or not np.isfinite(time_s).all()
            or (len(time_s) > 1 and np.any(np.diff(time_s) <= 0.0))):
        raise SystemExit("animation time_s must match qpos and be strictly increasing")
    arm = args.arm or _arm_from_episode(episode)
    arm_visual = ARM_VISUALS[arm]

    import trimesh
    import viser
    import yourdfpy

    robot_urdf = (Path.home() / "shared_data/AutoDex/content/assets/robot"
                  / arm_visual["urdf"][0] / arm_visual["urdf"][1])
    if not robot_urdf.exists():
        raise SystemExit(f"{arm_visual['label']} URDF missing: {robot_urdf}")
    urdf = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True,
                              build_collision_scene_graph=False)
    robot_joint_names = list(urdf.actuated_joint_names)
    n_q = min(len(robot_joint_names), qpos.shape[1])
    target_cfg = scene_cfg["mesh"]["target"]
    mesh, visual_info = _load_object_render_mesh(trimesh, episode, target_cfg)

    server = viser.ViserServer(port=args.port)
    for name, cfg in scene_cfg.get("cuboid", {}).items():
        _add_box(server, name, cfg)
    _add_proxy(server, board_proxy)
    object_handle = server.scene.add_mesh_trimesh("/object", mesh)
    phase_text = server.gui.add_text("phase", initial_value="", disabled=True)
    attachment_text = server.gui.add_text("object state", initial_value="", disabled=True)
    render_text = server.gui.add_text(
        "object visual", initial_value=visual_info["kind"], disabled=True)
    robot_text = server.gui.add_text(
        "robot visual", initial_value=f"{arm_visual['label']} ({n_q} joints)", disabled=True)
    step_text = server.gui.add_text("sample", initial_value="", disabled=True)
    time_text = server.gui.add_text("trajectory time", initial_value="", disabled=True)
    contract_text = server.gui.add_text(
        "replay contract", initial_value=trajectory_kind, disabled=True)
    slider = server.gui.add_slider("sample", min=0, max=len(qpos) - 1,
                                   step=1, initial_value=0)
    # A saved lift is most naturally watched as an animation.  Keep a visible
    # pause control, but start and repeat by default so opening the page shows
    # the full approach→grasp→lift sequence without another click.
    playing = server.gui.add_checkbox("autoplay", initial_value=True)
    looping = server.gui.add_checkbox("loop", initial_value=True)
    speed = server.gui.add_slider("playback speed", min=0.1, max=2.0,
                                  step=0.1, initial_value=1.0)
    playhead_s = float(time_s[0])

    def show(i: int, *, q_override: np.ndarray | None = None,
             object_pose_override: np.ndarray | None = None,
             clock_s: float | None = None) -> None:
        i = int(np.clip(i, 0, len(qpos) - 1))
        q = qpos[i] if q_override is None else np.asarray(q_override, dtype=float)
        T_object = (object_poses[i] if object_pose_override is None
                    else np.asarray(object_pose_override, dtype=float))
        urdf.update_cfg({robot_joint_names[j]: float(q[j]) for j in range(n_q)})
        server.scene.add_mesh_trimesh("/robot", urdf.scene.to_geometry())
        pos, wxyz = _pose_wxyz(T_object)
        try:
            object_handle.position = pos
            object_handle.wxyz = wxyz
        except Exception:
            # Very old viser handles did not expose transform attributes.
            moved = mesh.copy()
            moved.apply_transform(object_poses[i])
            server.scene.add_mesh_trimesh("/object", moved)
        phase_text.value = str(phases[i])
        attachment_text.value = (
            "table-fixed (approach/squeeze; attachment begins at lift)"
            if str(phases[i]) != "lift"
            else "rigidly attached to wrist (lift)"
        )
        step_text.value = f"{i + 1}/{len(qpos)}"
        shown_time = time_s[i] if clock_s is None else float(clock_s)
        time_text.value = f"{shown_time:.3f}s / {time_s[-1]:.3f}s"

    def show_at_time(clock_s: float) -> int:
        """Render the reference q(t), linearly between its saved 10 ms nodes."""
        upper = int(np.searchsorted(time_s, clock_s, side="right"))
        index = int(np.clip(upper - 1, 0, len(qpos) - 1))
        if index >= len(qpos) - 1:
            show(index, clock_s=clock_s)
            return index
        interval = float(time_s[index + 1] - time_s[index])
        alpha = 0.0 if interval <= 0.0 else (clock_s - time_s[index]) / interval
        q = (1.0 - alpha) * qpos[index] + alpha * qpos[index + 1]
        T_object = _interpolate_pose(object_poses[index], object_poses[index + 1], alpha)
        show(index, q_override=q, object_pose_override=T_object, clock_s=clock_s)
        return index

    updating_slider_from_playback = False

    @slider.on_update
    def _on_step(_event) -> None:
        nonlocal playhead_s
        if not updating_slider_from_playback:
            playhead_s = float(time_s[int(slider.value)])
        show(int(slider.value))

    show(0)
    print(f"[viser] object visual={visual_info['kind']}: {visual_info['path']}")
    if visual_info["kind"] == "raw_vertex_color":
        print("[viser] raw mesh frame check: "
              f"centroid={visual_info['centroid_delta_m'] * 1000:.2f}mm "
              f"extent={visual_info['max_extent_delta_m'] * 1000:.2f}mm")
    else:
        print(f"[viser] colorized raw mesh unavailable: {visual_info.get('reason')}")
    print(f"[viser] robot visual: {arm_visual['label']} ({n_q} animated joints)")
    print(f"[viser] replay={trajectory_kind}, duration={time_s[-1]:.3f}s, samples={len(qpos)}")
    print(f"[viser] http://localhost:{args.port} — saved replay: {episode}")
    last_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            elapsed = now - last_tick
            last_tick = now
            if bool(playing.value):
                playhead_s += elapsed * float(speed.value)
                if playhead_s > float(time_s[-1]):
                    if bool(looping.value):
                        # Preserve excess wall time so a long browser stall
                        # does not make the replay visibly pause on frame 0.
                        playhead_s = float(time_s[0]) + (
                            (playhead_s - float(time_s[0])) %
                            (float(time_s[-1]) - float(time_s[0])))
                    else:
                        playhead_s = float(time_s[-1])
                        playing.value = False
                next_idx = show_at_time(playhead_s)
                if next_idx != int(slider.value):
                    updating_slider_from_playback = True
                    try:
                        slider.value = next_idx
                    finally:
                        updating_slider_from_playback = False
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
