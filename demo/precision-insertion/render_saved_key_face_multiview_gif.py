#!/usr/bin/env python3
"""Animate the saved key-face proxy and socket-axis XY alignment in all views.

Only reads saved, hash-checked images and frozen calibration. The displayed
translation is a geometric illustration, not a command to the robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from compare_saved_key_face_methods import camera_projection, point


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE = 384
STAGES = (
    ("1  Original camera frames", "original"),
    ("2  Qwen key boxes", "qwen_box"),
    ("3  SAM3 key masks", "sam_mask"),
    ("4  Observed face ellipses", "observed_ellipse"),
    ("5  Multiview key-face point", "spatial_proxy"),
    ("6  XY shift toward socket axis", "xy_alignment"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def crop_from_grid(root: Path, run_id: str, name: str,
                   index: int, views: int) -> Image.Image:
    path = root / "role_images" / name / f"{run_id}.png"
    grid = Image.open(path).convert("RGB")
    columns = min(3, views)
    rows = math.ceil(views / columns)
    if grid.size != (512 * columns, 554 * rows):
        raise ValueError(f"saved role grid size changed: {path}")
    x = (index % columns) * 512
    y = (index // columns) * 554 + 42
    return grid.crop((x, y, x+512, y+512))


def dot(draw: ImageDraw.ImageDraw, xy: tuple[float, float],
        color: str, radius: int = 8) -> None:
    x, y = xy
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=color, outline="#09131e", width=2)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float],
          end: tuple[float, float]) -> None:
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length < 1e-3:
        return
    direction = delta/length
    side = np.array([-direction[1], direction[0]])
    head_length = min(14., max(5., length*.35))
    head_half_width = max(4., head_length*.55)
    draw.line((tuple(a), tuple(b)), fill="#09131e", width=9)
    draw.line((tuple(a), tuple(b)), fill="#ff9f35", width=5)
    head = [tuple(b),
            tuple(b-direction*head_length+side*head_half_width),
            tuple(b-direction*head_length-side*head_half_width)]
    draw.polygon(head, fill="#ff9f35", outline="#09131e")


def dashed_line(draw: ImageDraw.ImageDraw, start: tuple[float, float],
                end: tuple[float, float]) -> None:
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    span = float(np.linalg.norm(b-a))
    if span < 1e-3:
        return
    direction = (b-a)/span
    for distance in np.arange(0., span, 15.):
        first = a+direction*distance
        second = a+direction*min(distance+9., span)
        draw.line((tuple(first), tuple(second)), fill="#09131e", width=7)
        draw.line((tuple(first), tuple(second)), fill="#00daf3", width=4)


def axis_heights(run: Path, proxy_z_m: float,
                 fixture_root: Path | None) -> tuple[float, float]:
    calibration = read(run / "frozen_session_calibration.json")
    mode = select_mode("cylinder", float(calibration["mode"]["gap_mm"]))
    shared = (fixture_root if fixture_root is not None else
              Path(read(run / "endpoint_catalog.json")["shared_root"]))
    geometry = read(AssetPaths(shared, mode).task_geometry)
    rim_z = float(geometry["socket_entry_plane_z_m"])
    return rim_z - .02, max(rim_z + .09, proxy_z_m + .02)


def compose(stage: str, cameras: list[str], panels: list[Image.Image],
            footer: tuple[str, str] | None = None) -> Image.Image:
    if len(cameras) != len(panels):
        raise ValueError("camera labels and image panels differ")
    canvas = Image.new("RGB", (TILE*len(panels), TILE+164), "#111927")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT, 24)
    camera_font = ImageFont.truetype(FONT, 19)
    draw.text((14, 9), stage, font=title_font, fill="white")
    for i, (camera, panel) in enumerate(zip(cameras, panels)):
        x = i*TILE
        draw.text((x+12, 50), f"camera {camera}", font=camera_font,
                  fill="white")
        canvas.paste(panel.resize((TILE, TILE), Image.Resampling.LANCZOS),
                     (x, 90))
    if footer is not None:
        footer_font = ImageFont.truetype(FONT, 18)
        for line_number, line in enumerate(footer):
            draw.text((14, TILE+101+line_number*28), line,
                      font=footer_font, fill="#f4f6f8")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path,
                        help="Current location of the session fixture assets if moved")
    args = parser.parse_args()
    root = args.root.resolve()
    runs_root = args.runs_root.resolve()
    manifest = read(root / "evidence_manifest.json")
    comparison = read(root / "method_comparison.json")
    ellipse_rows = read(root / "sam_ellipse_core.json")["rows"] + \
                   read(root / "sam_ellipse_manual.json")["rows"]
    ellipses = {(row["run_id"], row["camera_id"]): row
                for row in ellipse_rows}
    grouped: dict[str, list[dict]] = {}
    for view in manifest["views"]:
        grouped.setdefault(view["run_id"], []).append(view)
    selected = root / "methods" / "sam_half_ellipse"
    gif_dir = selected / "animations_multiview"
    frame_root = selected / "animation_frames_multiview"
    gif_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "saved_key_face_multiview_xy_animation_v1",
               "scope": "read_only_projected_proxy_not_robot_motion_command",
               "runs": {}}
    for run_id, views in grouped.items():
        cameras = [v["camera_id"] for v in views]
        method = comparison["runs"][run_id]["methods"]["sam_half_ellipse_center"]
        if method["status"] != "estimated" or method["camera_count"] != len(views):
            raise ValueError(f"accepted run lacks all camera face centers: {run_id}")
        xyz = np.asarray(method["point_socket_m"], dtype=float)
        axis_xyz = np.array([0., 0., xyz[2]])
        axis_start_z, axis_end_z = axis_heights(
            runs_root/run_id, float(xyz[2]), args.fixture_root)
        panels_by_stage: dict[str, list[Image.Image]] = {
            name: [] for _title, name in STAGES}
        view_evidence = []
        for index, view in enumerate(views):
            camera = view["camera_id"]
            row = ellipses[(run_id, camera)]
            if (row["frame_id"] != view["frame_id"] or
                    row["source_sha256"] != view["source_sha256"] or
                    row["mask_sha256"] != view["mask_sha256"]):
                raise ValueError("animation evidence does not match saved frame")
            source = Path(view["source"])
            mask = Path(view["mask_path"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != view["source_sha256"]:
                raise ValueError("animation source camera image hash changed")
            mask_present = mask.is_file()
            if (mask_present and hashlib.sha256(mask.read_bytes()).hexdigest() !=
                    view["mask_sha256"]):
                raise ValueError("animation source SAM mask hash changed")
            for stage_name in ("original", "qwen_box", "sam_mask",
                               "observed_ellipse", "spatial_proxy"):
                panels_by_stage[stage_name].append(
                    crop_from_grid(root, run_id, stage_name, index, len(views)))
            P, _socket, _calibration = camera_projection(runs_root/run_id, camera)
            current = np.asarray(method["per_camera_proxy_reprojection_px"][camera])
            projected_current = np.asarray(point(P, xyz))
            if np.linalg.norm(current-projected_current) > 1e-3:
                raise ValueError("saved proxy projection and calibration differ")
            target = np.asarray(point(P, axis_xyz))
            origin = np.asarray(row["crop_xyxy"][:2], dtype=float)
            current_local = tuple(current-origin)
            target_local = tuple(target-origin)
            axis_start = tuple(np.asarray(point(P, (0., 0., axis_start_z))) - origin)
            axis_end = tuple(np.asarray(point(P, (0., 0., axis_end_z))) - origin)
            proxy_panel = panels_by_stage["spatial_proxy"][-1].copy()
            dashed_line(ImageDraw.Draw(proxy_panel), axis_start, axis_end)
            xy_panel = proxy_panel.copy()
            draw = ImageDraw.Draw(xy_panel)
            dot(draw, current_local, "#f85dc7", radius=7)
            dot(draw, target_local, "#ffe15b", radius=8)
            arrow(draw, current_local, target_local)
            x, y = target_local
            draw.ellipse((x-11, y-11, x+11, y+11),
                         outline="#ffe15b", width=3)
            side = 256
            middle = (np.asarray(current_local)+np.asarray(target_local))*.5
            left = int(np.clip(round(middle[0]-side*.5), 0, 512-side))
            top = int(np.clip(round(middle[1]-side*.5), 0, 512-side))
            focus = (left, top, left+side, top+side)
            panels_by_stage["spatial_proxy"][-1] = proxy_panel.crop(focus)
            panels_by_stage["xy_alignment"].append(xy_panel.crop(focus))
            view_evidence.append({
                "camera_id": camera, "frame_id": view["frame_id"],
                "timestamp_s": view["timestamp_s"],
                "source_sha256": view["source_sha256"],
                "mask_sha256": view["mask_sha256"],
                "mask_file_reverified_at_animation": mask_present,
                "proxy_full_px": current.tolist(),
                "socket_axis_same_z_full_px": target.tolist(),
                "proxy_to_axis_crop_px": [list(current_local), list(target_local)],
                "socket_axis_segment_crop_px": [list(axis_start), list(axis_end)],
                "final_two_stages_focus_crop_xyxy": list(focus),
            })
        frames = []
        run_frames = frame_root / run_id
        run_frames.mkdir(parents=True, exist_ok=True)
        delta_xy_mm = -xyz[:2]*1000.
        proxy_line = ("Socket-frame key proxy (mm): "
                      f"({xyz[0]*1000:+.2f}, {xyz[1]*1000:+.2f}, "
                      f"{xyz[2]*1000:+.2f})")
        shift_line = ("XY to axis (mm): "
                      f"({delta_xy_mm[0]:+.2f}, {delta_xy_mm[1]:+.2f}); "
                      f"distance {np.linalg.norm(delta_xy_mm):.2f} mm")
        for i, (title, name) in enumerate(STAGES, 1):
            footer = ((proxy_line, "Axis target at the same key height: (0, 0, z)")
                      if name == "spatial_proxy" else
                      (proxy_line, shift_line) if name == "xy_alignment" else None)
            frame = compose(title, cameras, panels_by_stage[name], footer)
            frame.save(run_frames / f"{i:02d}_{name}.png")
            frames.append(frame)
        gif_path = gif_dir / f"{run_id}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=[1300]*len(frames), loop=0, optimize=False)
        summary["runs"][run_id] = {
            "gif": str(gif_path),
            "final_frame": str(run_frames / "06_xy_alignment.png"),
            "camera_ids": cameras,
            "key_face_proxy_socket_m": xyz.tolist(),
            "socket_axis_same_z_m": axis_xyz.tolist(),
            "socket_axis_segment_z_m": [axis_start_z, axis_end_z],
            "illustrative_delta_socket_xy_mm": (-xyz[:2]*1000.).tolist(),
            "views": view_evidence,
        }
        print(run_id, len(cameras), gif_path)
    (selected / "multiview_xy_animation.json").write_text(
        json.dumps(summary, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
