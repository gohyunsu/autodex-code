#!/usr/bin/env python3
"""Extend the saved key-face GIF with multiview key-axis correction stages.

The first seven frames are copied pixel-for-pixel from the accepted 1080p
presentation GIF.  The three appended frames use the same saved images,
SAM3 masks, camera crops, socket pose, and colour vocabulary.  This is a
read-only visualization: it neither connects to the robot nor writes a
runtime correction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw, ImageFont

from compare_saved_key_axis_methods import (
    _largest_component_points,
    _load_rows,
    _minimum_rectangle,
)
from saved_key_face_palette import (
    AXIS_TARGET,
    KEY_PROXY,
    MARKER_OUTLINE,
    OBSERVED_KEY,
    PANEL_BACKGROUND,
)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE = 384
PANEL_TOP = 90
SOURCE_SIZE = (1152, 548)
OUTPUT_SIZE = (1920, 1080)
OUTPUT_Y = 83
OBSERVED_AXIS = "#ff4d57"
TARGET_AXIS = AXIS_TARGET
FUSED_AXIS = KEY_PROXY


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resampling(name: str):
    namespace = getattr(Image, "Resampling", Image)
    return getattr(namespace, name)


def panel_from(frame: Image.Image, index: int) -> Image.Image:
    return frame.crop((index*TILE, PANEL_TOP,
                       (index+1)*TILE, PANEL_TOP+TILE)).copy()


def compose(title: str, cameras: list[str], panels: list[Image.Image],
            footer: tuple[str, str] | None = None) -> Image.Image:
    canvas = Image.new("RGB", SOURCE_SIZE, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 9), title, font=ImageFont.truetype(FONT, 24), fill="white")
    camera_font = ImageFont.truetype(FONT, 19)
    for index, (camera, panel) in enumerate(zip(cameras, panels)):
        draw.text((index*TILE+12, 50), f"camera {camera}",
                  font=camera_font, fill="white")
        canvas.paste(panel, (index*TILE, PANEL_TOP))
    if footer:
        footer_font = ImageFont.truetype(FONT, 18)
        for line_number, line in enumerate(footer):
            draw.text((14, TILE+101+line_number*28), line,
                      font=footer_font, fill="#f4f6f8")
    return canvas


def line(draw: ImageDraw.ImageDraw, first, second, color: str,
         width: int = 4) -> None:
    first = tuple(np.asarray(first, dtype=float))
    second = tuple(np.asarray(second, dtype=float))
    draw.line((first, second), fill=MARKER_OUTLINE, width=width+4)
    draw.line((first, second), fill=color, width=width)


def dot(draw: ImageDraw.ImageDraw, xy, color: str, radius: int = 6) -> None:
    x, y = np.asarray(xy, dtype=float)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=color, outline=MARKER_OUTLINE, width=2)


def project(view: dict, point_socket: np.ndarray) -> np.ndarray:
    K = np.asarray(view["intrinsics"], dtype=float)
    E = np.asarray(view["T_camera_socket"], dtype=float)
    camera = E[:3, :3] @ point_socket + E[:3, 3]
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def source_to_panel(full_xy: np.ndarray, crop_xyxy: list[int]) -> np.ndarray:
    return (np.asarray(full_xy, dtype=float)-np.asarray(crop_xyxy[:2]))*.75


def draw_observed_axis(panel: Image.Image, centre: np.ndarray,
                       direction: np.ndarray, length: float = 120.) -> None:
    draw = ImageDraw.Draw(panel)
    line(draw, centre-length*direction, centre+length*direction,
         OBSERVED_AXIS)
    dot(draw, centre, OBSERVED_KEY)


def upscale(frame: Image.Image) -> Image.Image:
    resized = frame.resize((OUTPUT_SIZE[0], 913), resampling("BICUBIC"))
    output = Image.new("RGB", OUTPUT_SIZE, PANEL_BACKGROUND)
    output.paste(resized, (0, OUTPUT_Y))
    return output


def exact_palette(image: Image.Image) -> Image.Image:
    # Every accepted old frame contains <=256 RGB colours.  Per-frame local
    # palettes therefore retain those pixels exactly, unlike one global GIF
    # palette that visibly shifts the camera colours.
    return image.quantize(colors=256, method=Image.MEDIANCUT,
                          dither=Image.NONE)


def save_local_palette_gif(path: Path, frames: list[Image.Image],
                           durations: list[int]) -> None:
    paletted = [exact_palette(frame) for frame in frames]
    with path.open("wb") as handle:
        header, _used = GifImagePlugin.getheader(
            paletted[0], info={"loop": 0, "background": 0})
        for block in header:
            handle.write(block)
        # Pillow 9.0 marks the stream GIF89a but its legacy getheader helper
        # does not emit the loop extension itself.
        handle.write(b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
        for frame, duration in zip(paletted, durations):
            blocks = GifImagePlugin.getdata(
                frame, duration=duration, disposal=2,
                include_color_table=True)
            for block in blocks:
                handle.write(block)
        handle.write(b";")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gif", type=Path, required=True)
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--refined-summary", type=Path, required=True)
    parser.add_argument("--axis-results", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--run-id", default="20261022_105828_126084")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-output", type=Path, required=True)
    args = parser.parse_args()

    summary = read(args.refined_summary)["runs"][args.run_id]
    axis_record = read(args.axis_results)["runs"][args.run_id]
    method = axis_record["methods"]["min_area_rectangle"]
    axis = np.asarray(method["axis_socket"], dtype=float)
    tip = np.asarray(axis_record["tip_socket_m"], dtype=float)
    target_tip = np.array([0., 0., tip[2]])
    cameras = summary["camera_ids"]
    views_by_camera = {row["camera_id"]: row for row in summary["views"]}
    calibrated = {row["camera_id"]: row for row in
                  read(args.calibration_manifest)["views"]}
    rows, qwen = _load_rows(args.evidence_root)

    # Recreate the same minimum-area-rectangle shaft direction in all three
    # displayed views.  Only cameras present in `calibrated` participate in
    # the 3D fusion recorded in axis_results.
    observed: dict[str, dict[str, np.ndarray]] = {}
    for camera in cameras:
        row = rows[(args.run_id, camera)]
        box = qwen[(args.run_id, camera)]
        points = _largest_component_points(row, box, args.evidence_root)
        planned = (np.asarray(box["nominal_tip_px"], dtype=float)-
                   np.asarray(box["planned_rear_full_px"], dtype=float))
        _rectangle_centre, direction = _minimum_rectangle(points, planned)
        centre = np.asarray(row["sam_half_ellipse"]["center_full_px"],
                            dtype=float)
        crop = views_by_camera[camera]["all_stages_source_crop_full_xyxy"]
        observed[camera] = {
            "centre": source_to_panel(centre, crop),
            "direction": direction,
        }

    base_mask = Image.open(args.source_frames/"03_sam_mask.png").convert("RGB")
    base_xy = Image.open(args.source_frames/"07_socket_xy_alignment.png").convert("RGB")

    # Stage 8: direct 2D evidence in every displayed view.
    stage8_panels = []
    for index, camera in enumerate(cameras):
        panel = panel_from(base_mask, index)
        record = observed[camera]
        draw_observed_axis(panel, record["centre"], record["direction"])
        stage8_panels.append(panel)
    stage8 = compose("8  Per-view key shaft axes", cameras, stage8_panels)

    # Stage 9: show the fused 3D axis reprojected into the two calibrated
    # views.  The third panel deliberately remains a red 2D observation.
    stage9_panels = []
    for index, camera in enumerate(cameras):
        panel = panel_from(base_mask, index)
        record = observed[camera]
        draw_observed_axis(panel, record["centre"], record["direction"])
        if camera in calibrated:
            crop = views_by_camera[camera]["all_stages_source_crop_full_xyxy"]
            tip_px = source_to_panel(project(calibrated[camera], tip), crop)
            rear_px = source_to_panel(
                project(calibrated[camera], tip-.08*axis), crop)
            draw = ImageDraw.Draw(panel)
            line(draw, tip_px, rear_px, FUSED_AXIS, width=5)
            dot(draw, tip_px, FUSED_AXIS, radius=7)
        stage9_panels.append(panel)
    stage9 = compose(
        "9  Fused multiview key axis", cameras, stage9_panels,
        ("Socket-frame key axis: "
         f"({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})",
         "Tilt to socket -Z: "
         f"{method['tilt_to_socket_axis_deg']:.2f} deg  |  "
         "two calibrated views fused"))

    # Stage 10: preserve the accepted socket/XY panel and add the current
    # shaft axis plus its desired socket-axis-aligned orientation.
    stage10_panels = []
    for index, camera in enumerate(cameras):
        panel = panel_from(base_xy, index)
        view = views_by_camera[camera]
        current = np.asarray(view["proxy_to_axis_crop_px"][0])*0.75
        target = np.asarray(view["proxy_to_axis_crop_px"][1])*0.75
        axis_high = np.asarray(view["socket_axis_segment_crop_px"][1])*0.75
        draw = ImageDraw.Draw(panel)
        if camera in calibrated:
            crop = view["all_stages_source_crop_full_xyxy"]
            current_rear = source_to_panel(
                project(calibrated[camera], tip-.08*axis), crop)
        else:
            current_rear = current-120.*observed[camera]["direction"]
        line(draw, current, current_rear, FUSED_AXIS, width=5)
        line(draw, target, axis_high, TARGET_AXIS, width=5)
        dot(draw, current, FUSED_AXIS, radius=7)
        dot(draw, target, TARGET_AXIS, radius=7)
        stage10_panels.append(panel)
    delta_mm = -tip[:2]*1000.
    stage10 = compose(
        "10  Next-trial XY and key-axis correction", cameras, stage10_panels,
        (f"XY correction: ({delta_mm[0]:+.2f}, {delta_mm[1]:+.2f}) mm  |  "
         f"key tilt correction: {method['tilt_to_socket_axis_deg']:.2f} deg",
         "Accumulate the translation and rotation, then apply them to the next trial"))

    args.frame_output.mkdir(parents=True, exist_ok=True)
    appended = [stage8, stage9, stage10]
    for number, (name, frame) in enumerate(zip(
            ("per_view_shaft_axes", "fused_key_axis", "xy_axis_correction"),
            appended), start=8):
        frame.save(args.frame_output/f"{number:02d}_{name}.png")
        upscale(frame).save(args.frame_output/f"{number:02d}_{name}_1080p.png")

    base = Image.open(args.base_gif)
    frames, durations = [], []
    for index in range(base.n_frames):
        base.seek(index)
        frames.append(base.convert("RGB"))
        durations.append(int(base.info.get("duration", 2500)))
    frames.extend(upscale(frame) for frame in appended)
    durations.extend((2800, 3400, 4400))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_local_palette_gif(args.output, frames, durations)

    manifest = {
        "schema": "saved_key_face_xy_axis_animation_v1",
        "scope": "read_only_saved_evidence_not_robot_command",
        "run_id": args.run_id,
        "output_gif": str(args.output),
        "frame_count": len(frames),
        "durations_ms": durations,
        "displayed_cameras": cameras,
        "axis_fusion_cameras": list(axis_record["camera_ids"]),
        "third_view_role": "2d_consistency_check_only",
        "key_tip_socket_m": tip.tolist(),
        "xy_correction_mm": delta_mm.tolist(),
        "key_axis_socket": axis.tolist(),
        "target_axis_socket": [0., 0., -1.],
        "tilt_correction_deg": method["tilt_to_socket_axis_deg"],
        "minimal_alignment_rotvec_socket_deg":
            method["minimal_alignment_rotvec_socket_deg"],
        "old_frames_preserved_with_exact_per_frame_local_palettes": True,
    }
    (args.frame_output/"animation_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
