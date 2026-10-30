#!/usr/bin/env python3
"""Render a saved four-view tabletop key observation without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS = (1920, 1080)
PANEL = (900, 430)
BACKGROUND = "#101a27"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def project(K: np.ndarray, T_camera_world: np.ndarray,
            T_world_key: np.ndarray, xyz) -> tuple[float, float] | None:
    camera = T_camera_world @ T_world_key @ np.r_[xyz, 1.]
    if camera[2] <= 0:
        return None
    pixel = K @ camera[:3]
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])


def render(bundle: Path, run: Path, output: Path) -> dict:
    observation = read(bundle / "key_observation.json")
    calibration = read(run / "frozen_session_calibration.json")
    camera_calibration = calibration["camera_calibration"]
    pose = np.asarray(observation["pose_world"], dtype=float)
    cameras = sorted(observation["per_view"])
    if len(cameras) != 4:
        raise ValueError("this renderer expects the capture4 four-view bundle")

    canvas = Image.new("RGB", CANVAS, BACKGROUND)
    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((44, 28), "Initial tabletop key perception",
                     font=font(38, True), fill="white")
    draw_canvas.text((44, 80),
                     "green: SAM key mask    blue: selected 6D key axis    cyan: frozen socket region",
                     font=font(21), fill="#cad5df")
    rows = []
    positions = ((40, 130), (980, 130), (40, 590), (980, 590))
    for camera, (left, top) in zip(cameras, positions):
        source = cv2.imread(str(bundle / "initial_tabletop" / "images" /
                                f"{camera}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(bundle / "masks" / f"{camera}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if source is None or mask is None or mask.shape != source.shape[:2]:
            raise FileNotFoundError(f"missing aligned image/mask for {camera}")
        overlay = source.copy()
        selected = mask > 0
        tint = np.zeros_like(source)
        tint[:, :, 1] = 255
        overlay[selected] = np.round(
            .58 * source[selected] + .42 * tint[selected]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (80, 255, 80), 4)

        socket = observation["consistency"]["socket_exclusion"][
            "per_camera"][camera]["projected_hull_bbox_px"]
        cv2.rectangle(overlay, (socket[0], socket[1]),
                      (socket[2], socket[3]), (255, 230, 40), 4)
        intrinsics = camera_calibration["intrinsics_full"][camera]
        K = np.asarray(intrinsics["K_undist"], dtype=float)
        E = np.asarray(camera_calibration["extrinsics_full"][camera],
                       dtype=float)
        axis = [project(K, E, pose, (0., 0., z)) for z in (0., .04, .08)]
        if all(point is not None for point in axis):
            cv2.line(overlay, tuple(np.rint(axis[0]).astype(int)),
                     tuple(np.rint(axis[2]).astype(int)), (255, 130, 30), 5)
            for point, radius in zip(axis, (9, 6, 9)):
                cv2.circle(overlay, tuple(np.rint(point).astype(int)), radius,
                           (255, 130, 30), -1)
                cv2.circle(overlay, tuple(np.rint(point).astype(int)), radius,
                           (20, 20, 20), 2)

        image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        image.thumbnail((PANEL[0], PANEL[1]-48), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", PANEL, "#0b131d")
        panel.paste(image, ((PANEL[0]-image.width)//2, 48))
        panel_draw = ImageDraw.Draw(panel)
        view = observation["per_view"][camera]
        panel_draw.text((14, 10),
                        f"{camera}   mask {view['mask_pixels']} px   FoundPose inliers {view['foundpose_inliers']}",
                        font=font(19, True), fill="white")
        canvas.paste(panel, (left, top))
        rows.append({"camera_id": camera, "accepted": view["accepted"],
                     "mask_pixels": view["mask_pixels"],
                     "foundpose_inliers": view["foundpose_inliers"]})

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    report = {
        "schema": "precision_insertion_initial_key_perception_visualization_v1",
        "scope": "saved images, masks, FoundPose output and calibration; no inference or robot command",
        "bundle": str(bundle), "session_run": str(run),
        "image": str(output), "selected_camera_id": observation[
            "selected_camera_id"], "views": rows,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.bundle.resolve(), args.session_run.resolve(),
                            args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
