#!/usr/bin/env python3
"""Render a rotating view of each greedy grasp beside cumulative coverage.

The replay files contain the solved xArm + Inspire configuration and object
pose for every selected greedy grasp.  This utility picks the first lift
configuration (the closed, object-attached grasp), rotates a close hand/object
mesh view through 360 degrees, and pairs it with accumulated coverage.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validation.planning.pipeline_lift_reachability.core import read_jsonl
from src.validation.planning.pipeline_lift_reachability.render_greedy_trajectories import (
    MeshRenderer,
    _resolve_assets,
)


def _grasp_frame(phases: np.ndarray) -> int:
    """Return the first lifted frame, where the fingers are closed on the object."""
    lift = np.flatnonzero(np.asarray(phases) == "lift")
    return int(lift[0]) if len(lift) else len(phases) - 1


def _coverage_image(xs: np.ndarray, ys: np.ndarray, covered: np.ndarray,
                    gained: np.ndarray, *, rank: int, total: int,
                    key: str, gain: int) -> "Image.Image":
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    figure, axis = plt.subplots(figsize=(6.4, 5.7), constrained_layout=True)
    axis.scatter(xs[~covered], ys[~covered], c="#d9d9d9", s=46,
                 label="not covered")
    axis.scatter(xs[covered & ~gained], ys[covered & ~gained], c="#2171b5",
                 s=46, label="covered earlier")
    axis.scatter(xs[gained], ys[gained], c="#2ca25f", edgecolors="#006d2c",
                 linewidths=0.8, s=74, label="newly covered")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(xs.min() - 0.06, xs.max() + 0.06)
    axis.set_ylim(ys.min() - 0.06, ys.max() + 0.06)
    axis.set_xlabel("robot x (m)")
    axis.set_ylabel("robot y (m)")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=9)
    axis.set_title(
        f"Cumulative workspace coverage: {int(covered.sum())}/{len(covered)}\n"
        f"Greedy grasp {rank}/{total}: +{gain} cells")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def _compose(mesh: np.ndarray, coverage, *, rank: int, total: int,
             key: str, cell_id: str, radius: float, theta: float):
    from PIL import Image, ImageDraw, ImageFont

    mesh_image = Image.fromarray(mesh).convert("RGB")
    plot = coverage.resize((640, mesh_image.height), Image.Resampling.LANCZOS)
    header = 74
    canvas = Image.new("RGB", (mesh_image.width + plot.width, mesh_image.height + header),
                       "white")
    canvas.paste(mesh_image, (0, header))
    canvas.paste(plot, (mesh_image.width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    subfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((16, 10), f"Greedy grasp {rank}/{total}  •  {key}", fill="#151515", font=font)
    draw.text((16, 40),
              f"attached_container | representative base cell {cell_id} "
              f"(r={radius:.2f} m, θ={theta:.0f}°) | purple: Inspire hand; green: object",
              fill="#404040", font=subfont)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--object-mesh", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--orbit-frames", type=int, default=16,
                        help="camera views in one full turn for each grasp")
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=570)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.orbit_frames < 4:
        raise SystemExit("--orbit-frames must be at least 4")

    run_dir = args.run_dir.expanduser().resolve()
    trajectory_dir = (args.trajectory_dir.expanduser().resolve() if args.trajectory_dir
                      else run_dir / "greedy_trajectory_replays")
    paths = sorted(trajectory_dir.glob("[0-9][0-9]_*.npz"))
    if not paths:
        raise SystemExit(f"no greedy replay files in {trajectory_dir}")
    matrix = np.load(run_dir / "coverage_matrix.npz")
    values = np.asarray(matrix["pipeline_success"], dtype=bool)
    cell_ids = [str(cell) for cell in matrix["cell_ids"]]
    metadata = {str(row["cell_id"]): row for row in read_jsonl(run_dir / "per_grasp.jsonl")}
    xs = np.asarray([float(metadata[cell]["object_x_m"]) for cell in cell_ids])
    ys = np.asarray([float(metadata[cell]["object_y_m"]) for cell in cell_ids])
    summary = json.loads((run_dir / "summary.json").read_text())
    greedy = summary["greedy"]["pipeline_success"]
    indices = greedy["candidate_indices"]
    if len(paths) != len(indices):
        raise SystemExit(f"{len(paths)} replay files but {len(indices)} greedy grasps")
    urdf, object_mesh = _resolve_assets(trajectory_dir, args.urdf, args.object_mesh)
    import trimesh
    object_geometry = trimesh.load(str(object_mesh), force="mesh", process=False)
    object_centroid_local = np.r_[np.asarray(object_geometry.centroid), 1.0]

    covered = np.zeros(values.shape[1], dtype=bool)
    frames = []
    for rank, (path, candidate_index) in enumerate(zip(paths, indices), start=1):
        data = np.load(path)
        grasp_index = _grasp_frame(data["phases"])
        gained = values[candidate_index] & ~covered
        covered |= values[candidate_index]
        object_pose = np.asarray(data["object_poses"][grasp_index])
        object_center = object_pose[:3, 3]
        object_centroid = (object_pose @ object_centroid_local)[:3]
        wrist_center = np.asarray(data["ee_position"][grasp_index])
        grasp_center = 0.5 * (object_centroid + wrist_center)
        radial = object_center[:2].copy()
        radial /= max(np.linalg.norm(radial), 1.0e-8)
        start_azimuth = np.arctan2(radial[1], radial[0])
        start_eye = grasp_center + np.array([
            0.38 * np.cos(start_azimuth),
            0.38 * np.sin(start_azimuth),
            0.22,
        ])
        # Arm links and the table are omitted so the orbit only shows the
        # hand/object contact geometry.
        renderer = MeshRenderer(
            urdf, object_mesh, args.width, args.height,
            camera_center=grasp_center,
            camera_eye=start_eye,
            hand_only=True, show_table=False)
        sample = {"qpos": np.asarray(data["qpos"]),
                  "object_poses": np.asarray(data["object_poses"]),
                  # A repeated point deliberately suppresses the trajectory trail:
                  # this is a grasp/coverage summary, not a motion replay.
                  "ee_position": np.repeat(
                      data["ee_position"][grasp_index][None, :],
                      len(data["ee_position"]), axis=0)}
        coverage = _coverage_image(
            xs, ys, covered, gained, rank=rank, total=len(paths),
            key="/".join(map(str, data["candidate_key"])), gain=int(gained.sum()))
        for orbit_frame in range(args.orbit_frames):
            azimuth = start_azimuth + 2.0 * np.pi * orbit_frame / args.orbit_frames
            eye = grasp_center + np.array([
                0.38 * np.cos(azimuth),
                0.38 * np.sin(azimuth),
                0.22,
            ])
            renderer.set_camera(eye, grasp_center)
            mesh = renderer.render(sample, grasp_index)
            frames.append(_compose(
                mesh, coverage, rank=rank, total=len(paths),
                key="/".join(map(str, data["candidate_key"])),
                cell_id=str(data["cell_id"].item()), radius=float(data["r_m"]),
                theta=float(data["theta_deg"])))
        del renderer

    output = (args.output.expanduser().resolve() if args.output
              else run_dir / "plots" / "greedy_grasp_turntable_coverage.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(1000 / args.fps), loop=0, optimize=False)
    print(output)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    raise SystemExit(main())
