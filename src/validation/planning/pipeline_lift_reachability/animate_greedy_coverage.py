#!/usr/bin/env python3
"""Animate the greedy coverage sequence from an existing reachability run."""
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

from src.validation.planning.pipeline_lift_reachability.core import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a GIF of greedily accumulated reachability coverage.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", default="pipeline_success",
                        choices=["both_endpoint_same_variant_success",
                                 "jacobian_lift_success", "pipeline_success"])
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    run_dir = args.run_dir.resolve()
    with (run_dir / "summary.json").open() as stream:
        summary = json.load(stream)
    matrix = np.load(run_dir / "coverage_matrix.npz")
    candidate_indices = summary["greedy"][args.stage]["candidate_indices"]
    candidate_keys = summary["greedy"][args.stage]["candidate_keys"]
    values = np.asarray(matrix[args.stage], dtype=bool)
    cell_ids = [str(value) for value in matrix["cell_ids"]]
    records = read_jsonl(run_dir / "per_grasp.jsonl")
    metadata = {str(row["cell_id"]): row for row in records}
    xs = np.asarray([float(metadata[cell]["object_x_m"]) for cell in cell_ids])
    ys = np.asarray([float(metadata[cell]["object_y_m"]) for cell in cell_ids])
    if not candidate_indices:
        raise SystemExit(f"no greedy coverage for {args.stage}: {run_dir}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex_matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output = args.output or run_dir / "plots" / f"greedy_{args.stage}.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    covered = np.zeros(values.shape[1], dtype=bool)
    frames: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    for rank, (index, key) in enumerate(zip(candidate_indices, candidate_keys), start=1):
        gained = values[index] & ~covered
        covered |= values[index]
        frames.append((covered.copy(), gained, key, rank))

    fig, axis = plt.subplots(figsize=(7.3, 6.3), constrained_layout=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("robot x (m)")
    axis.set_ylabel("robot y (m)")
    axis.grid(alpha=0.25)
    axis.set_xlim(xs.min() - 0.06, xs.max() + 0.06)
    axis.set_ylim(ys.min() - 0.06, ys.max() + 0.06)

    def draw(frame: int):
        covered_now, gained, key, rank = frames[frame]
        axis.clear()
        axis.scatter(xs[~covered_now], ys[~covered_now], c="#d9d9d9", s=62,
                     marker="o", label="not covered")
        axis.scatter(xs[covered_now & ~gained], ys[covered_now & ~gained],
                     c="#2171b5", s=62, marker="o", label="covered earlier")
        axis.scatter(xs[gained], ys[gained], c="#2ca25f", edgecolors="#006d2c",
                     linewidths=0.8, s=96, marker="o", label="newly covered")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("robot x (m)")
        axis.set_ylabel("robot y (m)")
        axis.grid(alpha=0.25)
        axis.set_xlim(xs.min() - 0.06, xs.max() + 0.06)
        axis.set_ylim(ys.min() - 0.06, ys.max() + 0.06)
        axis.set_title(
            f"Greedy grasp {rank}/{len(frames)}: {key}\n"
            f"coverage {int(covered_now.sum())}/{len(covered_now)} "
            f"(+{int(gained.sum())})")
        axis.legend(loc="upper left")

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / args.fps,
                              repeat=True)
    animation.save(output, writer=PillowWriter(fps=args.fps), dpi=150)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
