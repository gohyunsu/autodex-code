#!/usr/bin/env python3
"""Render 16:9 VLM checkpoint storyboards from actual robot/task assets.

Usage from the repository root after activating ``autodex_bodex``::

    python scripts/precision_insertion/build_vlm_checkpoint_states.py
    python scripts/precision_insertion/render_vlm_checkpoint_storyboards.py

The script prepares the original FR3/Inspire visual meshes for Blender, renders
each state without labels, and assembles four 1920x1080 PNGs.  The output is an
illustration set; see ``render_manifest.json`` for per-panel provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
SHARED = Path.home() / "shared_data"
DEFAULT_TRAJECTORY = (
    SHARED / "AutoDex/precision_insertion/presentation_assets/08_vlm_checkpoints/"
    "actual_asset_states.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_TRAJECTORY.parent
DEFAULT_MESH_CACHE = (
    SHARED / "AutoDex/precision_insertion/presentation_assets/04_planning/"
    "robot_visual_mesh_cache"
)


STORYBOARDS = [
    {
        "file": "checkpoint_01_lift.png",
        "checkpoint": "after_lift",
        "panels": [
            ("lift_retained", "task", "retained"),
            ("lift_miss", "task", "miss"),
            ("lift_slip", "task", "slip"),
        ],
    },
    {
        "file": "checkpoint_02_preinsert.png",
        "checkpoint": "preinsertion_hold",
        "panels": [
            ("preinsert_aligned", "key-socket", "aligned"),
            ("preinsert_misaligned", "key-socket", "gross_misalignment"),
            ("preinsert_occluded", "task", "occluded"),
        ],
    },
    {
        "file": "checkpoint_03_insertion.png",
        "checkpoint": "insertion_abort_or_final_hold",
        "panels": [
            ("insertion_20mm", "key-socket", "verification_depth_reached"),
            ("insertion_partial", "key-socket", "partial_insertion"),
            ("insertion_rim_jam", "key-socket", "rim_jam_illustration"),
        ],
    },
    {
        "file": "checkpoint_04_finish.png",
        "checkpoint": "after_optional_finish",
        "panels": [
            ("finish_seated", "key-socket", "seated"),
            ("finish_press_jam", "key-socket", "press_jam_illustration"),
        ],
    },
]


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\n"
            + result.stdout[-8000:]
        )


def _compose(panel_paths: list[Path], output: Path) -> None:
    width, height, divider = 1920, 1080, 4
    count = len(panel_paths)
    available = width - divider * (count - 1)
    widths = [available // count] * count
    for index in range(available % count):
        widths[index] += 1
    canvas = Image.new("RGB", (width, height), (238, 241, 246))
    cursor = 0
    for index, (path, panel_width) in enumerate(zip(panel_paths, widths)):
        with Image.open(path) as image:
            fitted = image.convert("RGB").resize(
                (panel_width, height), Image.Resampling.LANCZOS
            )
        canvas.paste(fitted, (cursor, 0))
        cursor += panel_width
        if index + 1 < count:
            canvas.paste((245, 247, 250), (cursor, 0, cursor + divider, height))
            cursor += divider
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mesh-cache", type=Path, default=DEFAULT_MESH_CACHE)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    trajectory = args.trajectory.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mesh_cache = args.mesh_cache.expanduser().resolve()
    blender = (
        args.blender.expanduser().resolve()
        if args.blender is not None
        else Path(shutil.which("blender") or "/snap/bin/blender")
    )
    if not trajectory.is_file():
        parser.error(
            f"missing state trajectory: {trajectory}; run "
            "build_vlm_checkpoint_states.py first"
        )
    if not blender.is_file():
        parser.error(f"Blender executable not found: {blender}")

    with np.load(trajectory, allow_pickle=False) as data:
        state_names = [str(value) for value in data["state_name"].tolist()]
        object_mesh = str(data["object_mesh_path"].item())
        socket_mesh = str(data["socket_mesh_path"].item())
        robot_urdf = str(data["robot_urdf_path"].item())
    state_index = {name: index + 1 for index, name in enumerate(state_names)}

    bundle = trajectory.with_name(trajectory.stem + "_blender_bundle.npz")
    if not args.skip_prepare:
        _run([
            sys.executable,
            str(REPO / "scripts/precision_insertion/prepare_blender_actual_mesh_animation.py"),
            str(trajectory),
            "--output", str(bundle),
            "--mesh-cache", str(mesh_cache),
        ])
    if not bundle.is_file():
        parser.error(f"missing Blender bundle: {bundle}")

    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    renderer = REPO / "scripts/precision_insertion/render_blender_actual_mesh_animation.py"
    rendered_manifest: list[dict] = []

    for storyboard in STORYBOARDS:
        panel_width = 640 if len(storyboard["panels"]) == 3 else 960
        panel_paths: list[Path] = []
        panel_records: list[dict] = []
        for state_name, view, semantic in storyboard["panels"]:
            if state_name not in state_index:
                raise KeyError(f"missing state {state_name!r} in {trajectory}")
            panel_path = panel_dir / f"{state_name}_{view}.png"
            _run([
                str(blender), "--background", "--python", str(renderer), "--",
                str(bundle), "--output", str(panel_path),
                "--width", str(panel_width), "--height", "1080",
                "--view", view,
                "--still-frame", str(state_index[state_name]),
            ])
            panel_paths.append(panel_path)
            panel_records.append({
                "state": state_name,
                "semantic": semantic,
                "view": view,
                "frame_1based": state_index[state_name],
                "panel": str(panel_path),
            })

        output = output_dir / storyboard["file"]
        _compose(panel_paths, output)
        rendered_manifest.append({
            "checkpoint": storyboard["checkpoint"],
            "output": str(output),
            "panels": panel_records,
        })
        print(output)

    manifest = {
        "schema_version": 1,
        "scope": "actual_asset_vlm_checkpoint_storyboards",
        "trajectory": str(trajectory),
        "blender_bundle": str(bundle),
        "robot_urdf": robot_urdf,
        "key_mesh": object_mesh,
        "socket_mesh": socket_mesh,
        "resolution": [1920, 1080],
        "embedded_text": False,
        "physical_validation": False,
        "warning": (
            "These are deterministic actual-mesh illustrations. Recorded panels "
            "reuse saved trajectory states; failure panels are explicitly staged "
            "counterfactual or endpoint-IK illustrations."
        ),
        "storyboards": rendered_manifest,
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
