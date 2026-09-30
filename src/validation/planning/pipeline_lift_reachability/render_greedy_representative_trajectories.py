#!/usr/bin/env python3
"""Render greedy representative trajectories as compact kinematic replays."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _sample(data: np.lib.npyio.NpzFile, count: int) -> dict[str, np.ndarray]:
    indices = np.rint(np.linspace(0, len(data["qpos"]) - 1, count)).astype(int)
    return {name: np.asarray(data[name])[indices]
            for name in ("link_positions", "object_poses", "phases")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.frames < 2 or args.fps <= 0:
        raise SystemExit("--frames must be >= 2 and --fps must be positive")

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    paths = sorted(trajectory_dir.glob("[0-9][0-9]_*.npz"))
    if not paths:
        raise SystemExit(f"no greedy replay NPZ files in {trajectory_dir}")
    loaded = [np.load(path) for path in paths]
    samples = [_sample(data, args.frames) for data in loaded]
    titles = [
        f"{rank}. {'/'.join(map(str, data['candidate_key']))}\n"
        f"{str(data['cell_id'])}, r={float(data['r_m']):.2f}, "
        f"theta={float(data['theta_deg']):.0f} deg"
        for rank, data in enumerate(loaded, start=1)
    ]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex_matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    columns = min(3, len(paths))
    rows = int(np.ceil(len(paths) / columns))
    images = []
    for frame in range(args.frames):
        figure = plt.figure(figsize=(5.25 * columns, 4.62 * rows))
        for index, (sample, title) in enumerate(zip(samples, titles), start=1):
            axis = figure.add_subplot(rows, columns, index, projection="3d")
            links = sample["link_positions"][frame]
            arm = links[:7]
            hand = links[7:]
            axis.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o",
                      color="#2171b5", linewidth=3, markersize=4)
            axis.scatter(hand[:, 0], hand[:, 1], hand[:, 2], s=20,
                         color="#7560a8")
            object_xyz = sample["object_poses"][frame, :3, 3]
            axis.scatter(*object_xyz, marker="s", s=105, color="#2ca25f")
            axis.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(0.0, 0.95),
                     xlabel="x", ylabel="y", zlabel="z")
            axis.view_init(elev=25, azim=-55)
            axis.set_title(f"{title}\n{sample['phases'][frame]}", fontsize=10)
        figure.tight_layout()
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        image = Image.fromarray(np.frombuffer(
            figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4))
        images.append(image.convert("RGB"))
        plt.close(figure)

    output_dir = trajectory_dir / "animations"
    output = (args.output.expanduser().resolve() if args.output else
              output_dir / f"greedy_{len(paths)}_representative_trajectories.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:],
                   duration=round(1000 / args.fps), loop=0, optimize=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
