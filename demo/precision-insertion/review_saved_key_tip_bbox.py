#!/usr/bin/env python3
"""Render read-only Qwen key boxes against saved calibrated preinsert frames.

The annotations are hypotheses on real undistorted AutoDex captures. A box
bottom midpoint is an image-space approximation, not a measured 3D key tip.
This tool never connects to a camera or robot and never creates a retry step.
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


COLORS = {
    "socket": "#00daf3", "x": "#fa514d", "y": "#56e066",
    "z": "#5c9df7", "planned": "#12e995", "qwen": "#fb56d0",
    "candidate": "#ffb23f", "sam3": "#ffe15b", "proxy": "#bd92ff",
    "previous": "#ffffff",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _project(K: np.ndarray, T: np.ndarray, xyz: tuple[float, float, float]) -> np.ndarray:
    point = T @ np.array([*xyz, 1.])
    if point[2] <= 0:
        raise ValueError("saved point lies behind the camera")
    homogeneous = K @ point[:3]
    return homogeneous[:2] / homogeneous[2]


def _circle(draw: ImageDraw.ImageDraw, point: tuple[float, float],
            color: str, radius: int = 7) -> None:
    x, y = point
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=color, outline="#111827", width=2)


def _cross(draw: ImageDraw.ImageDraw, point: tuple[float, float],
           color: str, radius: int = 9) -> None:
    x, y = point
    draw.line((x-radius, y-radius, x+radius, y+radius),
              fill="#111827", width=6)
    draw.line((x-radius, y+radius, x+radius, y-radius),
              fill="#111827", width=6)
    draw.line((x-radius, y-radius, x+radius, y+radius),
              fill=color, width=3)
    draw.line((x-radius, y+radius, x+radius, y-radius),
              fill=color, width=3)


def render_row(row: dict, *, runs_root: Path,
               font: ImageFont.ImageFont, tip_row: dict | None = None,
               proxy_view: dict | None = None) -> tuple[Image.Image, dict]:
    source = Path(row["source"]).resolve()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["source_sha256"]:
        raise ValueError(f"saved camera image changed: {source}")
    run = runs_root / row["run_id"]
    attempt = next((part for part in source.parents
                    if part.name.startswith("attempt_")), None)
    if attempt is None or not (run / "attempts" / attempt.name).is_dir():
        raise ValueError("saved image does not belong to this run's attempt")
    calibration = _read(run / "frozen_session_calibration.json")
    mode = select_mode("cylinder", float(calibration["mode"]["gap_mm"]))
    shared = Path(_read(run / "endpoint_catalog.json")["shared_root"])
    geometry = _read(AssetPaths(shared, mode).task_geometry)
    postlift = _read(run / "attempts" / attempt.name /
                     "postlift_preflights" / "000" / "report.json")
    camera = row["camera_id"]
    snapshot = calibration["camera_calibration"]
    K = np.asarray(snapshot["intrinsics_full"][camera]["K_undist"], dtype=float)
    world_to_camera = np.asarray(snapshot["extrinsics_full"][camera], dtype=float)
    robot_to_world = np.asarray(calibration["c2r"], dtype=float)
    socket_to_robot = np.asarray(calibration["socket_pose_robot"], dtype=float)
    key_to_robot = np.asarray(postlift["targets"]["T_robot_key_preinsert"], dtype=float)
    socket_to_camera = world_to_camera @ robot_to_world @ socket_to_robot
    key_to_camera = world_to_camera @ robot_to_world @ key_to_robot
    rim_z = float(geometry["socket_entry_plane_z_m"])
    bore_radius = float(geometry["socket_bore_radius_m"])
    key_tip_z = float(geometry["key_frame"]["tip_z_m"])
    rim = _project(K, socket_to_camera, (0., 0., rim_z))
    tip = _project(K, key_to_camera, (0., 0., key_tip_z))
    if (np.linalg.norm(rim - row["socket_rim_px"]) > 1.0 or
            np.linalg.norm(tip - row["nominal_tip_px"]) > 1.0):
        raise ValueError("saved overlay points differ from frozen geometry")
    image = Image.open(source).convert("RGB")
    x0, y0, x1, y1 = map(int, row["crop_xyxy"])
    if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height:
        raise ValueError("key ROI lies outside saved image")
    panel = image.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(panel)
    local = lambda point: (float(point[0]-x0), float(point[1]-y0))
    ring = [local(_project(K, socket_to_camera,
                           (bore_radius*math.cos(a),
                            bore_radius*math.sin(a), rim_z)))
            for a in np.linspace(0., 2.*math.pi, 96)]
    draw.line(ring + ring[:1], fill=COLORS["socket"], width=3)
    origin = local(rim)
    _cross(draw, origin, COLORS["socket"])
    for name, delta in (("x", (.03, 0., rim_z)),
                        ("y", (0., .03, rim_z)),
                        ("z", (0., 0., rim_z+.04))):
        endpoint = local(_project(K, socket_to_camera, delta))
        draw.line((*origin, *endpoint), fill=COLORS[name], width=3)
        draw.text((endpoint[0]+2, endpoint[1]-12), name.upper(),
                  fill=COLORS[name], font=font, stroke_fill="black",
                  stroke_width=2)
    _circle(draw, local(tip), COLORS["planned"])
    if row.get("visible_end_candidate_px") is not None:
        _cross(draw, local(np.asarray(row["visible_end_candidate_px"])),
               COLORS["candidate"])
    if row.get("human_approx_tip_full_px") is not None:
        _cross(draw, local(np.asarray(row["human_approx_tip_full_px"])),
               COLORS["candidate"])
    box = row.get("bbox_crop_px")
    if box is not None:
        draw.rectangle(tuple(map(float, box)), outline=COLORS["qwen"], width=3)
        _circle(draw, ((box[0]+box[2])/2, box[3]), COLORS["qwen"])
    if tip_row is not None:
        if (tip_row["frame_id"] != row["frame_id"] or
                tip_row["source_sha256"] != digest):
            raise ValueError("SAM3 point does not match saved Qwen frame")
        if tip_row.get("tip_full_px") is not None:
            _circle(draw, local(np.asarray(tip_row["tip_full_px"])),
                    COLORS["sam3"], radius=8)
        if tip_row.get("previous_tip_full_px") is not None:
            previous = local(np.asarray(tip_row["previous_tip_full_px"]))
            draw.ellipse((previous[0]-8, previous[1]-8,
                          previous[0]+8, previous[1]+8),
                         outline=COLORS["previous"], width=2)
    if proxy_view is not None:
        if (proxy_view["frame_id"] != row["frame_id"] or
                proxy_view["source_sha256"] != digest):
            raise ValueError("spatial proxy does not match saved Qwen frame")
        if proxy_view.get("proxy_reprojection_px") is not None:
            _cross(draw, local(np.asarray(proxy_view["proxy_reprojection_px"])),
                   COLORS["proxy"], radius=11)
    draw.rectangle((0, 0, panel.width, 27), fill="#152030")
    draw.text((7, 4), f'{camera}  frame {row["frame_id"]}',
              fill="white", font=font)
    metadata = {
        "run_id": row["run_id"], "attempt_id": attempt.name,
        "camera_id": camera, "frame_id": row["frame_id"],
        "timestamp_s": row["timestamp_s"],
        "source_image": str(source), "source_sha256": digest,
        "crop_xyxy_full_px": [x0, y0, x1, y1],
        "frozen_socket_pose_source": calibration["socket_pose_source"],
        "socket_rim_full_px": rim.tolist(),
        "planned_key_tip_full_px": tip.tolist(),
        "qwen_box_crop_px": box,
        "qwen_bottom_midpoint_full_px": row.get("bbox_tip_full_px"),
        "other_image_candidate_full_px": row.get("visible_end_candidate_px"),
        "human_approx_tip_full_px": row.get("human_approx_tip_full_px"),
        "sam3_distal_edge_full_px": None if tip_row is None else tip_row.get("tip_full_px"),
        "previous_rectangle_edge_full_px": None if tip_row is None else tip_row.get("previous_tip_full_px"),
        "proxy_reprojection_full_px": None if proxy_view is None else proxy_view.get("proxy_reprojection_px"),
        "proxy_reprojection_error_px": None if proxy_view is None else proxy_view.get("proxy_reprojection_error_px"),
        "qwen_prompt": row["prompt"], "qwen_raw_answer": row["answer"],
        "qwen_inference_s": row["inference_s"],
        "scope": "read_only_annotation_not_3d_tip_ground_truth_or_robot_command",
    }
    return panel, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tip-results", type=Path,
                        help="saved SAM3 mask endpoint results for the same frames")
    parser.add_argument("--proxy-results", type=Path,
                        help="saved multi-view socket-frame spatial proxy")
    args = parser.parse_args()
    rows = _read(args.results)["rows"]
    tips = {} if args.tip_results is None else {
        (r["run_id"], r["camera_id"]): r
        for r in _read(args.tip_results)["rows"]}
    proxies = {} if args.proxy_results is None else {
        (run_id, r["camera_id"]): r
        for run_id, run in _read(args.proxy_results)["runs"].items()
        for r in run["views"]}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    grouped: dict[str, list] = {}
    for row in rows:
        view_id = (row["run_id"], row["camera_id"])
        panel, metadata = render_row(
            row, runs_root=args.runs_root.expanduser().resolve(), font=font,
            tip_row=tips.get(view_id), proxy_view=proxies.get(view_id))
        grouped.setdefault(row["run_id"], []).append((panel, metadata))
    summary = {"schema": "precision_saved_key_bbox_review_v1",
               "scope": "saved_real_undistorted_camera_images_with_derived_overlays",
               "method": "Qwen whole-key bbox; bottom midpoint is approximate",
               "runs": {}}
    for run_id, panels in grouped.items():
        width = max(panel.width for panel, _ in panels)
        height = max(panel.height for panel, _ in panels)
        legend = [("cyan ring + XYZ", "frozen measured socket pose"),
                  ("green dot", "planned key tip"),
                  ("magenta box/dot", "Qwen key box / bottom midpoint"),
                  ("yellow dot", "SAM3 mask boundary center"),
                  ("white ring", "previous rotated-box end"),
                  ("purple X", "multi-view 3D proxy reprojected"),
                  ("orange X", "independent image/human approximate point")]
        columns = 2 if len(panels) >= 2 else 1
        legend_rows = math.ceil(len(legend)/columns)
        canvas = Image.new(
            "RGB", (width*len(panels), height+50+33*legend_rows), "#0f1825")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, _) in enumerate(panels):
            canvas.paste(panel, (index*width, 35))
        draw.text((8, 7), f'{run_id} | saved undistorted preinsert views',
                  fill="white", font=font)
        for index, (label, meaning) in enumerate(legend):
            col = index % columns
            line = index // columns
            draw.text((8+col*width, height+45+line*33),
                      f'{label}: {meaning}', fill="white", font=font)
        path = output / f'{run_id}_multiview.png'
        canvas.save(path)
        summary["runs"][run_id] = {
            "image": str(path),
            "views": [metadata for _, metadata in panels],
        }
        print(path)
    metadata_path = output / "review_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
