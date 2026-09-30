#!/usr/bin/env python3
"""Combine a run's ordered per-grasp coverage plots into one GIF."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=2.5)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 0 < args.scale <= 1:
        raise SystemExit("--scale must be in (0, 1]")

    run_dir = args.run_dir.expanduser().resolve()
    paths = sorted((run_dir / "plots" / "per_grasp").glob("*.png"))
    if not paths:
        raise SystemExit(f"no per-grasp PNG files under {run_dir}")

    from PIL import Image
    frames = []
    for path in paths:
        with Image.open(path) as source:
            size = (round(source.width * args.scale),
                    round(source.height * args.scale))
            frames.append(source.convert("RGB").resize(
                size, Image.Resampling.LANCZOS))
    output = (args.output.expanduser().resolve() if args.output
              else run_dir / "plots" / "per_grasp_all.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(1000 / args.fps), loop=0, optimize=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
