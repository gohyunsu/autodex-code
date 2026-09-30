#!/usr/bin/env python3
"""Render greedy replay trajectories with full URDF and object meshes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _sample(data: np.lib.npyio.NpzFile, count: int) -> dict[str, np.ndarray]:
    indices = np.rint(np.linspace(0, len(data["qpos"]) - 1, count)).astype(int)
    return {name: np.asarray(data[name])[indices]
            for name in ("qpos", "object_poses", "ee_position", "phases")}


def _resolve_assets(trajectory_dir: Path, urdf: Path | None,
                    object_mesh: Path | None) -> tuple[Path, Path]:
    if urdf is None:
        urdf = next((path for path in (
            Path.home() / "shared_data/AutoDex/content/assets/robot/inspire_description/xarm_inspire.urdf",
            _REPO / "autodex/planner/src/curobo/content/assets/robot/inspire_description/xarm_inspire.urdf",
        ) if path.is_file()), None)
    replay = json.loads((trajectory_dir / "replay_manifest.json").read_text())
    source_run = Path(replay["source_run"])
    if not (source_run / "manifest.json").is_file():
        source_run = trajectory_dir.parent
    source = json.loads((source_run / "manifest.json").read_text())
    if object_mesh is None:
        from autodex.utils.path import get_obj_root
        object_mesh = (Path(get_obj_root(source["version"])) / source["object"]
                       / "processed_data/mesh/simplified.obj")
    if urdf is None or not urdf.is_file():
        raise FileNotFoundError(f"xarm-inspire URDF not found: {urdf}")
    if not object_mesh.is_file():
        raise FileNotFoundError(f"object planning mesh not found: {object_mesh}")
    return urdf.resolve(), object_mesh.resolve()


class MeshRenderer:
    """Persistent Open3D scene; only FK transforms change between frames."""

    def __init__(self, urdf_path: Path, object_path: Path, width: int,
                 height: int, *, camera_eye=None, camera_center=None,
                 table_alpha: float = 1.0, hand_only: bool = False,
                 show_table: bool = True):
        import open3d as o3d
        import trimesh
        from yourdfpy import URDF

        self.o3d = o3d
        self.robot = URDF.load(str(urdf_path), load_meshes=True,
                               build_scene_graph=True)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background(np.array([1, 1, 1, 1], np.float32))
        self.robot_names = []
        for index, (source_name, source_mesh) in enumerate(
                self.robot.scene.geometry.items()):
            is_hand = source_name.startswith("right_") or source_name == "base_link.STL"
            if hand_only and not is_hand:
                continue
            name = f"robot_{index:02d}"
            color = [0.45, 0.18, 0.68, 1.0] if is_hand else [0.08, 0.38, 0.72, 1.0]
            self.renderer.scene.add_geometry(
                name, self._convert(source_mesh), self._material(color))
            self.robot_names.append((name, source_name))

        source_object = trimesh.load(str(object_path), force="mesh", process=False)
        self.renderer.scene.add_geometry(
            "object", self._convert(source_object),
            self._material([0.12, 0.68, 0.28, 1.0]))
        if show_table:
            table = o3d.geometry.TriangleMesh.create_box(1.44, 1.44, 0.04)
            table.translate([-0.72, -0.72, 0.0])
            table.compute_vertex_normals()
            table_shader = ("defaultLit" if table_alpha >= 1.0
                            else "defaultLitTransparency")
            self.renderer.scene.add_geometry(
                "table", table,
                self._material([0.72, 0.72, 0.72, table_alpha], table_shader))
        center = ([0.0, 0.0, 0.38] if camera_center is None else camera_center)
        eye = ([1.35, -1.55, 1.10] if camera_eye is None else camera_eye)
        self.set_camera(eye, center)

    def set_camera(self, eye, center, field_of_view: float = 43.0) -> None:
        """Move the camera while preserving the persistent mesh scene."""
        self.renderer.setup_camera(
            field_of_view, np.asarray(center, np.float32),
            np.asarray(eye, np.float32),
            np.array([0.0, 0.0, 1.0], np.float32))

    def _convert(self, mesh):
        converted = self.o3d.geometry.TriangleMesh(
            self.o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            self.o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        converted.compute_vertex_normals()
        return converted

    def _material(self, color, shader="defaultLit"):
        material = self.o3d.visualization.rendering.MaterialRecord()
        material.shader = shader
        material.base_color = color
        material.base_roughness = 0.72
        material.base_reflectance = 0.22
        return material

    def _add_line(self, name: str, points: np.ndarray, color,
                  width: float) -> None:
        self.renderer.scene.remove_geometry(name)
        points = np.asarray(points)
        if len(points) > 1:
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-9]
            points = points[keep]
        if len(points) < 2:
            return
        trail = self.o3d.geometry.LineSet(
            self.o3d.utility.Vector3dVector(points),
            self.o3d.utility.Vector2iVector(
                np.column_stack((np.arange(len(points) - 1),
                                 np.arange(1, len(points))))))
        trail.colors = self.o3d.utility.Vector3dVector(
            np.tile(color[:3], (len(points) - 1, 1)))
        material = self._material(color, "unlitLine")
        material.line_width = width
        self.renderer.scene.add_geometry(name, trail, material)

    def render(self, sample: dict[str, np.ndarray], frame: int,
               reference: np.ndarray | None = None) -> np.ndarray:
        self.robot.update_cfg(sample["qpos"][frame])
        for render_name, source_name in self.robot_names:
            self.renderer.scene.set_geometry_transform(
                render_name, self.robot.scene.graph.get(source_name)[0])
        self.renderer.scene.set_geometry_transform(
            "object", sample["object_poses"][frame])

        points = np.asarray(sample["ee_position"][:frame + 1])
        self._add_line("ee_trail", points, [1.0, 0.38, 0.02, 1.0], 3.0)
        if reference is not None:
            self._add_line("ideal_path", np.asarray(reference),
                           [0.05, 0.82, 0.90, 1.0], 2.0)
        return np.asarray(self.renderer.render_to_image())


def _caption(image, title: str, phase: str):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(image).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 72), "white")
    canvas.paste(image, (0, 72))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15)
    phase_font = ImageFont.truetype(font_path, 14)
    draw.multiline_text((12, 8), title, fill="#151515", font=font, spacing=3)
    draw.text((canvas.width - 155, 49), f"phase: {phase}", fill="#c64c00",
              font=phase_font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--columns", type=int, default=None,
                        help="columns in combined GIF (default: up to 3)")
    args = parser.parse_args()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    paths = sorted(trajectory_dir.glob("[0-9][0-9]_*.npz"))
    if not paths:
        raise SystemExit(f"no trajectory npz files in {trajectory_dir}")
    urdf, object_mesh = _resolve_assets(trajectory_dir, args.urdf,
                                        args.object_mesh)
    print(f"[mesh] robot={urdf}\n[mesh] object={object_mesh}", flush=True)
    loaded = [np.load(path) for path in paths]
    samples = [_sample(data, args.frames) for data in loaded]
    titles = [
        f"Greedy #{i + 1}: {'/'.join(map(str, data['candidate_key']))}\n"
        f"newly-covered representative {str(data['cell_id'])} | "
        f"r={float(data['r_m']):.2f} m, theta={float(data['theta_deg']):.0f} deg | "
        f"gain +{int(data['marginal_gain'])} ({int(data['cumulative_coverage'])}/108)"
        for i, data in enumerate(loaded)]

    renderer = MeshRenderer(urdf, object_mesh, args.width, args.height)
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                  else trajectory_dir / "animations_mesh")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for path, sample, title in zip(paths, samples, titles):
        frames = [_caption(renderer.render(sample, frame), title,
                           str(sample["phases"][frame]))
                  for frame in range(args.frames)]
        output = output_dir / f"{path.stem}_mesh.gif"
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=round(1000 / args.fps), loop=0, optimize=False)
        all_frames.append(frames)
        print(f"[saved] {output}", flush=True)

    from PIL import Image
    tile_size = (480, 408)
    columns = (min(3, len(all_frames)) if args.columns is None
               else min(args.columns, len(all_frames)))
    if columns < 1:
        raise SystemExit("--columns must be positive")
    rows = int(np.ceil(len(all_frames) / columns))
    combined_frames = []
    for frame in range(args.frames):
        canvas = Image.new(
            "RGB", (tile_size[0] * columns, tile_size[1] * rows), "white")
        for index, frames in enumerate(all_frames):
            canvas.paste(frames[frame].resize(tile_size, Image.Resampling.LANCZOS),
                         ((index % columns) * tile_size[0],
                          (index // columns) * tile_size[1]))
        combined_frames.append(canvas)
    combined = output_dir / f"greedy_{len(all_frames)}_mesh_trajectories.gif"
    combined_frames[0].save(
        combined, save_all=True, append_images=combined_frames[1:],
        duration=round(1000 / args.fps), loop=0, optimize=False)
    print(f"[saved] {combined}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    raise SystemExit(main())
