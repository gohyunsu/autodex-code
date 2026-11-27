#!/usr/bin/env python3
"""Render the two actual-mesh analytic Franka-gripper cylinder candidates.

The output contains three clean 1080p views for each symmetry-reduced tabletop
class and one labelled 1920x1080 comparison grid.  Candidate transforms are
recomputed with the same runtime constructor and checked against the stored
catalogue transforms before rendering.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from precision_insertion.analytic_gripper import analytic_franka_gripper_grasp
from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.endpoint import _hand_link_meshes


REPO = Path(__file__).resolve().parents[2]
BLENDER_RENDERER = (
    REPO / "scripts/precision_insertion/"
    "render_blender_franka_gripper_candidate.py")
VIEWS = ("oblique", "side", "top")
POSES = (("000", "end_down", "End-down (upright)"),
         ("001", "side_down", "Side-down (lying)"))


def _font(size: int):
    choices = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in choices:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _compose_grid(images: dict[tuple[str, str], Path], output: Path) -> None:
    canvas = Image.new("RGB", (1920, 1080), (15, 19, 25))
    draw = ImageDraw.Draw(canvas)
    header_font = _font(30)
    row_font = _font(29)
    cell_w, cell_h = 545, 307
    x_values = (180, 760, 1340)
    y_values = (160, 620)
    for x, view in zip(x_values, VIEWS):
        label = view.capitalize()
        box = draw.textbbox((0, 0), label, font=header_font)
        draw.text((x + (cell_w - (box[2] - box[0])) / 2, 90), label,
                  fill=(228, 232, 238), font=header_font)
    for y, (stem, _kind, row_label) in zip(y_values, POSES):
        draw.multiline_text((30, y + 110), row_label.replace(" ", "\n", 1),
                            fill=(228, 232, 238), font=row_font, spacing=8)
        for x, view in zip(x_values, VIEWS):
            with Image.open(images[(stem, view)]) as source:
                frame = source.convert("RGB")
                frame.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                px = x + (cell_w - frame.width) // 2
                py = y + (cell_h - frame.height) // 2
                canvas.paste(frame, (px, py))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def render(*, shared_root: Path, output_root: Path, blender: Path,
           width: int, height: int) -> dict:
    os.environ["PRECISION_END_EFFECTOR"] = "franka_gripper"
    root = shared_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    mode = select_mode("cylinder", 5)
    paths = AssetPaths(root, mode)
    mesh_cache = output / "mesh_cache"
    mesh_cache.mkdir()
    key_mesh = mesh_cache / "key.ply"
    import trimesh
    trimesh.load(str(paths.raw_mesh(mode.key_object)), force="mesh",
                 process=False).export(key_mesh)

    manifest = {
        "schema": "precision_insertion_franka_gripper_candidate_visuals_v1",
        "shared_root": str(root),
        "key_object": mode.key_object,
        "candidate_count": 2,
        "candidate_scope": "symmetry_reduced_end_down_and_side_down",
        "runtime_constructor": "analytic_franka_gripper_grasp",
        "render_scope": "actual_key_and_franka_hand_visual_meshes_at_nominal_contact",
        "physical_validation": False,
        "candidates": [],
        "images": [],
    }
    images: dict[tuple[str, str], Path] = {}
    for scene_id, (stem, expected_class, label) in enumerate(POSES):
        key_pose = np.load(paths.key_tabletop_dir / f"{stem}.npy",
                           allow_pickle=False)
        grasp = analytic_franka_gripper_grasp(
            key_pose, table_normal_robot=np.asarray([0.0, 0.0, 1.0]))
        if grasp.tabletop_class != expected_class:
            raise ValueError(f"tabletop class changed for {stem}")
        candidate = paths.candidate_dir / "table" / str(scene_id) / "000"
        stored = np.load(candidate / "wrist_se3.npy", allow_pickle=False)
        if not np.allclose(stored, grasp.T_key_hand, atol=1e-10):
            raise ValueError(f"stored/runtime grasp transform changed: {candidate}")
        manifest["candidates"].append({
            "tabletop_pose_stem": stem,
            "tabletop_class": expected_class,
            "label": label,
            "candidate_dir": str(candidate),
            "T_key_hand": grasp.T_key_hand.tolist(),
            "grasp_q_m": grasp.grasp_q.tolist(),
        })

        candidate_root = output / f"pose_{stem}_{expected_class}"
        individual = candidate_root / "individual"
        individual.mkdir(parents=True)
        hand_mesh_dir = candidate_root / "hand_meshes"
        hand_mesh_dir.mkdir()
        hand = _hand_link_meshes(paths.robot_urdf, grasp.grasp_q)
        hand_names = sorted(hand)
        for index, name in enumerate(hand_names):
            hand[name].export(hand_mesh_dir / f"geometry_{index:03d}.ply")
        bundle = candidate_root / "candidate_bundle.npz"
        np.savez_compressed(
            bundle,
            T_world_key=key_pose,
            T_world_hand=grasp.T_robot_hand,
            table_z=np.asarray(0.0),
            hand_mesh_dir=np.asarray(str(hand_mesh_dir)),
            hand_names=np.asarray(hand_names),
            key_mesh=np.asarray(str(key_mesh)),
        )
        for view in VIEWS:
            image = individual / f"{view}.png"
            subprocess.run([
                str(blender), "--background", "--python", str(BLENDER_RENDERER),
                "--", str(bundle), "--output", str(image), "--view", view,
                "--width", str(width), "--height", str(height),
            ], check=True)
            if not image.is_file():
                raise RuntimeError(f"Blender produced no image: {image}")
            images[(stem, view)] = image
            manifest["images"].append({
                "tabletop_pose_stem": stem,
                "tabletop_class": expected_class,
                "label": label,
                "view": view,
                "path": str(image),
            })

    grid = output / "franka_gripper_candidate_grid.png"
    _compose_grid(images, grid)
    manifest["grid"] = str(grid)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path(os.environ.get(
        "PRECISION_SHARED_ROOT",
        "/mnt/paradex2/hyunsu/precision_insertion/runtime/active_shared_data")))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path,
                        default=Path(shutil.which("blender") or ""))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if not args.blender.is_file() or not BLENDER_RENDERER.is_file():
        parser.error("Blender or candidate renderer is missing")
    result = render(shared_root=args.shared_root, output_root=args.output_root,
                    blender=args.blender, width=args.width, height=args.height)
    print(json.dumps({"grid": result["grid"],
                      "individual_count": len(result["images"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
