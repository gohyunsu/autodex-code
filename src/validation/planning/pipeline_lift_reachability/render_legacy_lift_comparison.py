#!/usr/bin/env python3
"""Render the selected legacy endpoint lift beside the Jacobian lift."""
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

from src.validation.planning.pipeline_lift_reachability.render_greedy_trajectories import (
    MeshRenderer, _resolve_assets)


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    return values[np.rint(np.linspace(0, len(values) - 1, count)).astype(int)]


def _panel(image: np.ndarray, heading: str, color: str):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(image).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 48), "white")
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 22)
    bbox = draw.textbbox((0, 0), heading, font=title_font)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, 9), heading,
              fill=color, font=title_font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--width", type=int, default=680)
    parser.add_argument("--height", type=int, default=500)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir.expanduser().resolve()
    trajectory_dir = (args.trajectory_dir.expanduser().resolve()
                      if args.trajectory_dir else comparison_dir.parent)
    report = json.loads((comparison_dir / "comparison_summary.json").read_text())
    selected = report["selected"]
    data = np.load(comparison_dir / selected["comparison_npz"])
    urdf, object_mesh = _resolve_assets(trajectory_dir, None, None)

    legacy = {
        "qpos": _resample(data["legacy_qpos"], args.frames),
        "object_poses": _resample(data["legacy_object_poses"], args.frames),
        "ee_position": _resample(data["legacy_ee_position"], args.frames),
    }
    current = {
        "qpos": _resample(data["current_qpos"], args.frames),
        "object_poses": _resample(data["current_object_poses"], args.frames),
        "ee_position": _resample(data["current_ee_position"], args.frames),
    }
    start = np.asarray(data["start_wrist_pose"])[:3, 3]
    ideal = np.stack((start, start + np.array([0.0, 0.0, 0.10])), axis=0)
    # Look from the selected object's radial side of the table so the
    # hand/object are in front of the arm instead of hidden behind it.
    theta = np.radians(float(selected["theta_deg"]))
    radial = np.array([np.cos(theta), np.sin(theta)])
    renderer = MeshRenderer(
        urdf, object_mesh, args.width, args.height,
        camera_eye=[1.55 * radial[0], 1.55 * radial[1], 1.08],
        camera_center=[0.10 * radial[0], 0.10 * radial[1], 0.34])
    from PIL import Image
    legacy_images = [renderer.render(legacy, frame, ideal)
                     for frame in range(args.frames)]
    current_images = [renderer.render(current, frame, ideal)
                      for frame in range(args.frames)]
    frames = []
    for frame in range(args.frames):
        left = _panel(
            legacy_images[frame],
            "Legacy endpoint lift", "#b3261e")
        right = _panel(
            current_images[frame],
            "Current lift", "#137333")
        canvas = Image.new("RGB", (left.width + right.width, left.height), "white")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        frames.append(canvas)

    output = (args.output.expanduser().resolve() if args.output else
              comparison_dir / "legacy_vs_jacobian_lift.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(1000 / args.fps), loop=0, optimize=False)
    print(output)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    raise SystemExit(main())
