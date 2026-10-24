#!/usr/bin/env python3
"""Explain a saved Qwen/SAM3 key-tip proxy with six evidence-linked image panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from precision_insertion.held_key_tip import padded_key_box_roi
from review_saved_key_tip_bbox import COLORS, _circle, _cross, render_row


PANEL = 512
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
STAGES = (
    "1  Original undistorted frame",
    "2  Planned-axis crop",
    "3  Qwen key box",
    "4  SAM3 key mask",
    "5  Fit actual end boundary",
    "6  3D proxy + socket pose",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(FONT, size)


def _local(point: list[float] | tuple[float, float], crop: list[int]) -> tuple[float, float]:
    return float(point[0]-crop[0]), float(point[1]-crop[1])


def render_process(
    bbox: dict, tip: dict, proxy: dict, *, runs_root: Path,
) -> tuple[Image.Image, dict]:
    if (bbox["run_id"], bbox["camera_id"], bbox["frame_id"],
            bbox["source_sha256"]) != (
            tip["run_id"], tip["camera_id"], tip["frame_id"],
            tip["source_sha256"]):
        raise ValueError("Qwen and SAM3 rows refer to different camera frames")
    if (bbox["frame_id"] != proxy["frame_id"] or
            bbox["source_sha256"] != proxy["source_sha256"]):
        raise ValueError("3D proxy refers to a different camera frame")
    source = Path(bbox["source"]).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != bbox["source_sha256"]:
        raise ValueError(f"original camera frame changed: {source}")
    original = Image.open(source).convert("RGB")
    crop_xyxy = bbox["crop_xyxy"]
    crop = original.crop(tuple(crop_xyxy))
    if crop.size != (PANEL, PANEL):
        raise ValueError("method illustration expects the saved 512 px crop")

    panels: list[Image.Image] = []
    frame = original.copy()
    draw = ImageDraw.Draw(frame)
    draw.rectangle(tuple(crop_xyxy), outline="#ffe15b", width=8)
    frame.thumbnail((PANEL, PANEL), Image.Resampling.LANCZOS)
    first = Image.new("RGB", (PANEL, PANEL), "#172332")
    first.paste(frame, ((PANEL-frame.width)//2, (PANEL-frame.height)//2))
    panels.append(first)

    second = crop.copy()
    draw = ImageDraw.Draw(second)
    rear = _local(bbox["planned_rear_full_px"], crop_xyxy)
    planned_tip = _local(bbox["nominal_tip_px"], crop_xyxy)
    rim = _local(bbox["socket_rim_px"], crop_xyxy)
    draw.line((*rear, *planned_tip), fill=COLORS["planned"], width=4)
    _circle(draw, planned_tip, COLORS["planned"], radius=8)
    _cross(draw, rim, COLORS["socket"], radius=10)
    panels.append(second)

    third = crop.copy()
    draw = ImageDraw.Draw(third)
    box = bbox["bbox_crop_px"]
    draw.rectangle(tuple(box), outline=COLORS["qwen"], width=5)
    _circle(draw, ((box[0]+box[2])/2, box[3]), COLORS["qwen"], radius=8)
    panels.append(third)

    fourth = crop.copy()
    fifth = crop.copy()
    mask_path = tip.get("mask_path")
    if mask_path and tip.get("tip_full_px") is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        roi = tip.get("roi_within_crop") or padded_key_box_roi(
            box, canvas_size_wh=crop.size)
        x0, y0, x1, y1 = map(int, roi)
        if mask.shape != (y1-y0, x1-x0):
            raise ValueError("saved SAM3 mask differs from its Qwen crop ROI")
        alpha = np.zeros((PANEL, PANEL, 4), dtype=np.uint8)
        alpha[y0:y1, x0:x1][mask] = (255, 225, 91, 90)
        rgba = Image.fromarray(alpha)
        fourth = Image.alpha_composite(fourth.convert("RGBA"), rgba).convert("RGB")
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_draw = ImageDraw.Draw(fifth)
        for contour in contours:
            coords = [(float(pt[0][0]+x0), float(pt[0][1]+y0))
                      for pt in contour]
            if len(coords) >= 2:
                edge_draw.line(coords + coords[:1], fill=COLORS["sam3"], width=3)
        edge_draw.line((*rear, *planned_tip), fill=COLORS["planned"], width=3)
        fit = tip.get("boundary_fit")
        if fit is not None:
            for sample in fit["boundary_samples_full_px"]:
                _circle(edge_draw, _local(sample, crop_xyxy), COLORS["sam3"],
                        radius=3)
            edge_draw.line([_local(p, crop_xyxy) for p in
                            fit["fitted_boundary_full_px"]],
                           fill=COLORS["sam3"], width=5)
        if tip.get("previous_tip_full_px") is not None:
            previous = _local(tip["previous_tip_full_px"], crop_xyxy)
            edge_draw.ellipse((previous[0]-10, previous[1]-10,
                               previous[0]+10, previous[1]+10),
                              outline="white", width=3)
        _circle(edge_draw, _local(tip["tip_full_px"], crop_xyxy),
                COLORS["sam3"], radius=10)
        if bbox.get("human_approx_tip_full_px") is not None:
            _cross(edge_draw, _local(bbox["human_approx_tip_full_px"], crop_xyxy),
                   COLORS["candidate"], radius=9)
    else:
        for panel in (fourth, fifth):
            ImageDraw.Draw(panel).text((18, 465), "SAM3: no mask in saved replay",
                                       fill="white", font=_font(18),
                                       stroke_fill="#101927", stroke_width=3)
    panels.extend((fourth, fifth))

    sixth, _meta = render_row(bbox, runs_root=runs_root, font=_font(16),
                             tip_row=tip, proxy_view=proxy)
    panels.append(sixth)
    canvas = Image.new("RGB", (PANEL*6, PANEL+154), "#101927")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, STAGES)):
        x = index*PANEL
        canvas.paste(panel, (x, 56))
        draw.text((x+8, 12), title, fill="white", font=_font(19))
    header = f'{bbox["run_id"]} | camera {bbox["camera_id"]} | frame {bbox["frame_id"]}'
    draw.text((10, 582), header, fill="white", font=_font(19))
    draw.text((10, 616),
              "cyan: measured socket  green: planned tip  magenta: Qwen  "
              "yellow: fitted SAM3 end boundary  white ring: old box edge  purple: 3D proxy  orange: human approx",
              fill="white", font=_font(18))
    draw.text((10, 642),
              "All landmarks are estimates on a real saved frame; the 3D point is a visual proxy, not a robot target.",
              fill="#d7ddec", font=_font(16))
    metadata = {
        "run_id": bbox["run_id"], "camera_id": bbox["camera_id"],
        "frame_id": bbox["frame_id"], "timestamp_s": bbox["timestamp_s"],
        "original_image": str(source), "original_sha256": digest,
        "qwen_prompt": bbox["prompt"], "qwen_response": bbox["answer"],
        "qwen_bbox_crop_px": box,
        "sam3_prompt": tip["prompt"] if "prompt" in tip else "white cylinder",
        "sam3_mask": mask_path, "sam3_distal_edge_full_px": tip.get("tip_full_px"),
        "previous_rectangle_edge_full_px": tip.get("previous_tip_full_px"),
        "fitted_boundary": tip.get("boundary_fit"),
        "sam3_mask_sha256": None if not mask_path else
        hashlib.sha256(Path(mask_path).read_bytes()).hexdigest(),
        "socket_rim_full_px": bbox["socket_rim_px"],
        "planned_tip_full_px": bbox["nominal_tip_px"],
        "human_approx_tip_full_px": bbox.get("human_approx_tip_full_px"),
        "proxy_reprojection_full_px": proxy.get("proxy_reprojection_px"),
        "proxy_reprojection_error_px": proxy.get("proxy_reprojection_error_px"),
    }
    return canvas, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bbox_results", type=Path)
    parser.add_argument("--tip-results", type=Path, required=True)
    parser.add_argument("--proxy-results", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-name", default="process_metadata.json")
    args = parser.parse_args()
    bboxes = _read(args.bbox_results)["rows"]
    tips = {(row["run_id"], row["camera_id"]): row
            for row in _read(args.tip_results)["rows"]}
    proxy_runs = _read(args.proxy_results)["runs"]
    proxies = {(run_id, row["camera_id"]): row
               for run_id, run in proxy_runs.items() for row in run["views"]}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for bbox in bboxes:
        key = (bbox["run_id"], bbox["camera_id"])
        tip = tips[key]
        proxy = proxies[key]
        canvas, item = render_process(bbox, tip, proxy,
                                      runs_root=args.runs_root.expanduser().resolve())
        path = output / f"{key[0]}_{key[1]}_method.png"
        canvas.save(path)
        item["process_image"] = str(path)
        metadata.append(item)
        print(path)
    if Path(args.metadata_name).name != args.metadata_name:
        raise ValueError("metadata name must be a filename")
    (output / args.metadata_name).write_text(
        json.dumps({"schema": "saved_key_tip_method_process_v1",
                    "views": metadata}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
