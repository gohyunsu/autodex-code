#!/usr/bin/env python3
"""Render one physical same-trial closed-loop episode from saved evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_saved_key_face_methods import camera_projection, point, socket_ring
from saved_key_face_palette import (
    AXIS_TARGET, KEY_PROXY, MARKER_OUTLINE, OBSERVED_BOUNDARY_SAMPLE,
    OBSERVED_KEY, OBSERVED_KEY_MASK_RGBA, PANEL_BACKGROUND, SOCKET_POSE,
    XY_SHIFT,
)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CANVAS = (1920, 1080)
PANEL = 438
PANEL_Y = 170
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def panel_positions(count: int) -> list[int]:
    if count <= 0:
        raise ValueError("visualization needs at least one camera")
    gap = 26
    width = count*PANEL + (count-1)*gap
    left = (CANVAS[0]-width)//2
    return [left+i*(PANEL+gap) for i in range(count)]


def local(xy, crop):
    return float(xy[0] - crop[0]), float(xy[1] - crop[1])


def dot(draw: ImageDraw.ImageDraw, xy, color: str, radius: int = 8) -> None:
    x, y = xy
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color,
                 outline=MARKER_OUTLINE, width=2)


def cross(draw: ImageDraw.ImageDraw, xy, color: str, radius: int = 11,
          width: int = 4) -> None:
    x, y = xy
    draw.line((x-radius, y-radius, x+radius, y+radius), fill=MARKER_OUTLINE,
              width=width+4)
    draw.line((x-radius, y+radius, x+radius, y-radius), fill=MARKER_OUTLINE,
              width=width+4)
    draw.line((x-radius, y-radius, x+radius, y+radius), fill=color, width=width)
    draw.line((x-radius, y+radius, x+radius, y-radius), fill=color, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color: str = XY_SHIFT) -> None:
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta/length
    side = np.array([-direction[1], direction[0]])
    head = min(18., max(9., length*.3))
    draw.line((tuple(a), tuple(b)), fill=MARKER_OUTLINE, width=10)
    draw.line((tuple(a), tuple(b)), fill=color, width=6)
    polygon = [tuple(b), tuple(b-direction*head+side*head*.55),
               tuple(b-direction*head-side*head*.55)]
    draw.polygon(polygon, fill=color, outline=MARKER_OUTLINE)


def mask_overlay(panel: Image.Image, record: dict, crop: list[int],
                 observation: dict) -> Image.Image:
    path = record.get("mask_file")
    roi = record.get("roi_within_crop")
    if path:
        source = Path(path)
        if not source.is_absolute():
            source = Path(observation["raw_bundle"]) / path
    else:
        source = None
    if source is None or not roi or not source.is_file():
        return panel
    mask = np.asarray(Image.open(source).convert("L")) > 0
    x0, y0, x1, y1 = map(int, roi)
    if mask.shape != (y1-y0, x1-x0):
        return panel
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[y0:y1, x0:x1][mask] = OBSERVED_KEY_MASK_RGBA
    return Image.alpha_composite(panel.convert("RGBA"),
                                 Image.fromarray(rgba)).convert("RGB")


def draw_panel(image_path: Path, record: dict, camera: str, observation: dict,
               run_dir: Path, fixture_root: Path | None) -> Image.Image:
    crop = list(map(int, record["crop_xyxy"]))
    panel = Image.open(image_path).convert("RGB").crop(tuple(crop))
    panel = mask_overlay(panel, record, crop, observation)
    draw = ImageDraw.Draw(panel)

    # Frozen session socket geometry.
    ring = [local(p, crop) for p in socket_ring(run_dir, camera, fixture_root)]
    draw.line(ring + ring[:1], fill=SOCKET_POSE, width=4)
    P, _socket, _calibration = camera_projection(run_dir, camera)
    alignment = observation["alignment"]
    raw_tip = alignment.get("tip_socket_m")
    tip = (None if raw_tip is None else np.asarray(raw_tip, dtype=float))
    entry_z = float(alignment["socket_entry_plane_z_m"])
    axis_a = local(point(P, (0., 0., entry_z)), crop)
    axis_top = entry_z + .10 if tip is None else max(entry_z+.10, tip[2]+.025)
    axis_b = local(point(P, (0., 0., axis_top)), crop)
    draw.line((axis_a, axis_b), fill=SOCKET_POSE, width=4)

    # Qwen's key proposal is part of the saved estimator evidence.  Red means
    # that the proposal overlapped the frozen socket projection and was
    # therefore identified as a socket/key association error by the runtime.
    box = record.get("qwen_box_crop_px")
    if box is not None:
        confusion = record.get("reason") == "key_box_matches_frozen_socket"
        draw.rectangle(tuple(float(v) for v in box),
                       outline="#ff4d5e" if confusion else "#ffb347", width=4)
    rear = record.get("planned_rear_full_px")
    nominal = record.get("nominal_tip_full_px")
    if rear is not None and nominal is not None:
        rear_local, nominal_local = local(rear, crop), local(nominal, crop)
        draw.line((rear_local, nominal_local), fill="#75a7ff", width=3)
        dot(draw, nominal_local, "#75a7ff", radius=6)

    fit = record.get("sam_half_ellipse")
    if fit is not None:
        ellipse = [local(p, crop) for p in fit["ellipse_full_px"]]
        draw.line(ellipse + ellipse[:1], fill=OBSERVED_KEY, width=4)
        for sample in fit["distal_boundary_samples_full_px"]:
            dot(draw, local(sample, crop), OBSERVED_BOUNDARY_SAMPLE, radius=2)
        dot(draw, local(fit["center_full_px"], crop), OBSERVED_KEY, radius=8)

    # The magenta point is the fused socket-frame key-tip estimate.  The
    # orange arrow ends at the immediate waypoint actually requested in this
    # iteration, not at the full residual-to-axis target.
    increment_raw = observation["choice"].get("increment_socket_m")
    if tip is not None:
        axis_same_z = local(point(P, (0., 0., tip[2])), crop)
        cross(draw, axis_same_z, AXIS_TARGET)
        fused = local(point(P, tip), crop)
        dot(draw, fused, KEY_PROXY, radius=7)
        if increment_raw is not None:
            increment = np.asarray(increment_raw, dtype=float)
            waypoint = tip.copy()
            waypoint[:2] += increment
            projected_waypoint = local(point(P, waypoint), crop)
            arrow(draw, fused, projected_waypoint)

    used = camera in alignment.get("inlier_cameras", [])
    reason = record.get("reason", "unknown").replace("_", " ")
    status = "used" if used else ("observed" if fit is not None else reason)
    draw.rounded_rectangle((8, 8, 247, 50), radius=10,
                           fill=(9, 19, 30, 215))
    label = f"{camera}  |  {status}"
    if len(label) > 38:
        label = f"{camera}  |  {record.get('reason', 'no face')}"
    draw.text((20, 16), label, font=font(17 if len(label) > 25 else 21,
                                        bold=used),
              fill=("#ffffff" if used else "#cbd5df"))
    return panel.resize((PANEL, PANEL), LANCZOS)


def convergence_plot(observations: list[dict], current: int) -> Image.Image:
    width, height = 1090, 300
    image = Image.new("RGB", (width, height), "#0b131d")
    draw = ImageDraw.Draw(image)
    errors = [None if o["choice"].get("xy_error_m") is None else
              float(o["choice"]["xy_error_m"])*1000. for o in observations]
    if not any(value is not None for value in errors):
        draw.text((45, 75), "No metric key-tip estimate", font=font(32, bold=True),
                  fill="#ffb347")
        draw.text((45, 132),
                  "No XY correction was applied in this observation.",
                  font=font(24), fill="#e5edf4")
        draw.text((45, 184),
                  "The nominal pre-insertion pose was used for descent.",
                  font=font(22), fill="#aebbc7")
        return image
    x0, y0, x1, y1 = 78, 32, width-30, height-52
    draw.line((x0, y1, x1, y1), fill="#5b6875", width=2)
    draw.line((x0, y0, x0, y1), fill="#5b6875", width=2)
    finite = [value for value in errors if value is not None]
    maximum = max(finite)*1.08
    points = []
    valid_rows = [(i, value) for i, value in enumerate(errors)
                  if value is not None]
    for valid_index, (i, value) in enumerate(valid_rows):
        x = x0 + (x1-x0)*(i/max(1, len(errors)-1))
        y = y1 - (y1-y0)*(value/maximum)
        points.append((i, x, y, value))
    if len(points) > 1:
        draw.line([(x, y) for _i, x, y, _value in points],
                  fill="#7fa8ff", width=4)
    for i, x, y, value in points:
        color = XY_SHIFT if i == current else KEY_PROXY
        dot(draw, (x, y), color, radius=7 if i == current else 5)
        draw.text((x-26, y-31), f"{value:.2f}", font=font(17), fill="#ffffff")
        draw.text((x-4, y1+13), str(i), font=font(17), fill="#cbd5df")
    draw.text((12, 8), "XY error (mm)", font=font(18, bold=True), fill="#ffffff")
    draw.text((x1-85, y1+13), "iteration", font=font(16), fill="#cbd5df")
    return image


def compose(observations: list[dict], index: int, run_dir: Path,
            fixture_root: Path | None, shift: dict | None,
            excluded_cameras: set[str]) -> Image.Image:
    observation = observations[index]
    canvas = Image.new("RGB", CANVAS, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    choice = observation["choice"]
    increment_raw = choice.get("increment_socket_m")
    increment = (None if increment_raw is None else
                 np.asarray(increment_raw, dtype=float)*1000.)
    inliers = observation["alignment"].get("inlier_cameras", [])
    draw.text((42, 28), f"Closed-loop iteration {index}", font=font(38, bold=True),
              fill="#ffffff")
    measured = (None if shift is None else
                np.asarray(shift["measured_increment_socket_m"], dtype=float)*1000.)
    move_text = ("commanded XY  unavailable"
                 if increment is None else
                 f"commanded XY  ({increment[0]:+.2f}, {increment[1]:+.2f}) mm"
                 if measured is None else
                 f"commanded / measured XY  ({increment[0]:+.2f}, {increment[1]:+.2f}) / "
                 f"({measured[0]:+.2f}, {measured[1]:+.2f}) mm")
    if index == len(observations)-1 and shift is not None:
        move_text += (f"     axial-height restore  "
                      f"{shift['measured_axis_increment_socket_m']*1000.:+.2f} mm")
    error = choice.get("xy_error_m")
    error_text = ("not observed" if error is None else f"{error*1000.:.2f} mm")
    draw.text((42, 82),
              f"estimated XY error  {error_text}     {move_text}",
              font=font(23), fill="#dce5ed")
    draw.text((42, 120), f"estimator views  {', '.join(inliers)}",
              font=font(18), fill="#aebbc7")

    records = {r["camera_id"]: r
               for r in observation["alignment"]["sam_half_ellipse_views"]}
    cameras = [camera for camera in sorted(records)
               if camera not in excluded_cameras]
    raw = Path(observation["raw_bundle"]) / "images"
    for x, camera in zip(panel_positions(len(cameras)), cameras):
        panel = draw_panel(raw/f"{camera}.png", records[camera], camera,
                           observation, run_dir, fixture_root)
        canvas.paste(panel, (x, PANEL_Y))

    plot = convergence_plot(observations, index)
    canvas.paste(plot, (42, 654))
    legend_x = 1170
    draw.text((legend_x, 664), "Overlay", font=font(24, bold=True), fill="#ffffff")
    legend = [
        (OBSERVED_KEY, "observed key face / SAM mask"),
        (KEY_PROXY, "fused key-tip estimate"),
        (SOCKET_POSE, "measured socket rim and axis"),
        (AXIS_TARGET, "socket-axis target at key-tip height"),
        (XY_SHIFT, "waypoint applied in this iteration"),
    ]
    for row, (color, label) in enumerate(legend):
        y = 712 + row*46
        draw.ellipse((legend_x, y, legend_x+20, y+20), fill=color,
                     outline=MARKER_OUTLINE, width=2)
        draw.text((legend_x+34, y-3), label, font=font(20), fill="#e5edf4")
    reason = observation["alignment"]["reason"].replace("_", " ")
    draw.text((1170, 958), f"source: {reason}", font=font(18), fill="#aebbc7")
    return canvas


def final_frame(attempt_dir: Path,
                excluded_cameras: set[str]) -> Image.Image | None:
    roots = sorted((attempt_dir/"raw_captures"/"final_clear_view").glob("*/images"))
    if not roots:
        return None
    root = roots[-1]
    cameras = [p.stem for p in sorted(root.glob("*.png"))
               if p.stem not in excluded_cameras]
    if not cameras:
        return None
    canvas = Image.new("RGB", CANVAS, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 30), "Insertion result", font=font(38, bold=True), fill="#ffffff")
    result_path = attempt_dir / "result.json"
    result = read(result_path) if result_path.is_file() else {}
    insertion_success = result.get("labels", {}).get("insertion_success")
    result_text = ("released key is inside the socket (manual label)"
                   if insertion_success is True else
                   "released key is outside the socket (manual label)"
                   if insertion_success is False else
                   "released-key result is not labelled")
    draw.text((42, 84), result_text,
              font=font(24), fill="#dce5ed")
    for x, camera in zip(panel_positions(len(cameras)), cameras):
        image = Image.open(root/f"{camera}.png").convert("RGB")
        # Preserve the complete camera frame for the result; the correction
        # frames above use the recorded 512 px estimator crop.
        image.thumbnail((PANEL, PANEL), LANCZOS)
        tile = Image.new("RGB", (PANEL, PANEL), "#0b131d")
        tile.paste(image, ((PANEL-image.width)//2, (PANEL-image.height)//2))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rounded_rectangle((8, 8, 230, 50), radius=10,
                                    fill=(9, 19, 30, 215))
        tile_draw.text((20, 16), camera, font=font(21, bold=True), fill="#ffffff")
        canvas.paste(tile, (x, PANEL_Y))
    draw.rounded_rectangle((42, 690, 1878, 930), radius=24, fill="#0b131d")
    outcome = ("INSERTION SUCCESS" if insertion_success is True else
               "INSERTION FAILED" if insertion_success is False else
               "INSERTION RESULT UNKNOWN")
    draw.text((86, 738), outcome, font=font(48, bold=True),
              fill=OBSERVED_KEY if insertion_success is True else "#ff6b6b")
    draw.text((86, 815),
              "The hand was released and the final clear-view images were saved.",
              font=font(25), fill="#e5edf4")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--session-run-dir", type=Path, required=True,
                        help="Directory containing frozen_session_calibration.json")
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exclude-camera", action="append", default=[],
                        help="Omit one camera from the rendered grids only")
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve()
    run_dir = args.session_run_dir.resolve()
    excluded_cameras = set(args.exclude_camera)
    output = (args.output_dir.resolve() if args.output_dir else
              attempt/"visualizations"/"closed_loop_multiview")
    output.mkdir(parents=True, exist_ok=True)
    observations = [read(p) for p in sorted(
        (attempt/"same_trial_xy"/"observations").glob("*.json"))]
    if not observations:
        raise ValueError("attempt has no same-trial XY observations")
    # New runs bind endpoint estimation to the socket-pose inliers selected
    # during this session's socket-only measurement. Use that same view set
    # in the review GIF; explicit CLI exclusions remain available for replay
    # of older evidence that predates the recorded selection.
    socket_selection = observations[0].get(
        "socket_geometry_view_selection", {})
    excluded_cameras.update(socket_selection.get(
        "excluded_camera_ids", []))

    shift_paths = sorted((attempt/"same_trial_xy"/"shifts").glob("*/report.json"))
    # Active-observation runs also contain descriptive auxiliary moves such
    # as ``000_lateral_0`` and ``001_post_lateral_correction``.  The frame
    # grid associates only the canonical, numeric correction following each
    # observation; keep the auxiliary reports on disk without trying to cast
    # their names to iteration numbers.
    shifts = {int(path.parent.name): read(path) for path in shift_paths
              if path.parent.name.isdigit()}
    frames = []
    records = []
    for index, observation in enumerate(observations):
        shift_index = index if index < len(observations)-1 else index+1
        shift = shifts.get(shift_index)
        frame = compose(observations, index, run_dir, args.fixture_root, shift,
                        excluded_cameras)
        path = output/f"iteration_{index:02d}.png"
        frame.save(path)
        frames.append(frame)
        records.append({
            "iteration": index,
            "image": str(path),
            "raw_bundle": observation["raw_bundle"],
            "xy_error_mm": (None if observation["choice"].get("xy_error_m")
                            is None else
                            observation["choice"]["xy_error_m"]*1000.),
            "command_socket_xy_mm": (None if observation["choice"].get(
                "increment_socket_m") is None else (np.asarray(
                observation["choice"]["increment_socket_m"])*1000.).tolist()),
            "measured_socket_xy_mm": None if shift is None else (
                np.asarray(shift["measured_increment_socket_m"])*1000.).tolist(),
            "measured_axis_increment_mm": None if shift is None else (
                shift["measured_axis_increment_socket_m"]*1000.),
            "inlier_cameras": observation["alignment"].get("inlier_cameras", []),
            "estimator_reason": observation["alignment"]["reason"],
        })
    final = final_frame(attempt, excluded_cameras)
    if final is not None:
        final_path = output/"insertion_result.png"
        final.save(final_path)
        frames.append(final)
    gif = output/"closed_loop_iterations.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=[1800]*len(frames), loop=0, optimize=False)
    report = {
        "schema": "precision_insertion_same_trial_closed_loop_visualization_v1",
        "scope": "saved physical episode evidence; no robot command",
        "attempt_dir": str(attempt),
        "session_run_dir": str(run_dir),
        "excluded_cameras": sorted(excluded_cameras),
        "socket_geometry_view_selection": socket_selection,
        "gif": str(gif),
        "final_result_frame": None if final is None else str(final_path),
        "iterations": records,
    }
    report_path = output/"visualization.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"gif": str(gif), "frames": len(observations),
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
