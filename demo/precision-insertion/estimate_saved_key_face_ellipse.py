#!/usr/bin/env python3
"""Compare SAM3 mask-derived key-face ellipses on saved AutoDex images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.held_key_tip import (
    distal_key_mask_half_ellipse, padded_key_box_roi,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nominal_circle(box: dict, run: Path, *, samples: int = 96) -> dict:
    calibration = _read(run / "frozen_session_calibration.json")
    mode = select_mode("cylinder", float(calibration["mode"]["gap_mm"]))
    root = Path(_read(run / "endpoint_catalog.json")["shared_root"])
    geometry = _read(AssetPaths(root, mode).task_geometry)
    source = Path(box["source"])
    attempt = next((part for part in source.parents
                    if part.name.startswith("attempt_")), None)
    if attempt is None:
        raise ValueError("saved image has no attempt directory")
    postlift = _read(run / "attempts" / attempt.name /
                     "postlift_preflights" / "000" / "report.json")
    camera = box["camera_id"]
    snapshot = calibration["camera_calibration"]
    K = np.asarray(snapshot["intrinsics_full"][camera]["K_undist"], dtype=float)
    E = np.asarray(snapshot["extrinsics_full"][camera], dtype=float)
    C = np.asarray(calibration["c2r"], dtype=float)
    key = np.asarray(postlift["targets"]["T_robot_key_preinsert"], dtype=float)
    M = E @ C @ key
    radius = float(geometry["key_radius_m"])
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    positions = [(0., 0.)] + [
        (radius*np.cos(angle), radius*np.sin(angle))
        for angle in np.linspace(0., 2.*np.pi, samples, endpoint=False)]
    pixels = []
    for x, y in positions:
        point = M @ np.array([x, y, tip_z, 1.])
        if point[2] <= 0:
            raise ValueError("planned key face is behind the saved camera")
        h = K @ point[:3]
        pixels.append((h[:2]/h[2]).tolist())
    return {
        "center_full_px": pixels[0],
        "ellipse_full_px": pixels[1:],
        "physical_key_radius_m": radius,
        "source": "frozen_calibration_and_planned_held_key_pose_not_observed_face",
    }


def _unconstrained_cv_ellipse(
    mask: np.ndarray, *, origin: tuple[int, int],
    planned_direction: np.ndarray,
) -> dict | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    contour += np.asarray(origin)
    direction = planned_direction / np.linalg.norm(planned_direction)
    axial = contour @ direction
    selected = contour[axial >= np.percentile(axial, 70.)]
    if len(selected) < 5:
        return None
    center, size, angle = cv2.fitEllipse(
        selected.astype(np.float32).reshape(-1, 1, 2))
    return {"center_full_px": [float(center[0]), float(center[1])],
            "axes_px": list(map(float, size)), "angle_deg": float(angle),
            "source": "unconstrained_opencv_fit_to_distal_30pct_of_sam3_contour"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bbox_results", type=Path)
    parser.add_argument("--sam-results", type=Path, required=True)
    parser.add_argument("--socket-qc", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted = set(_read(args.socket_qc)["accepted_runs"])
    boxes = {(row["run_id"], row["camera_id"]): row
             for row in _read(args.bbox_results)["rows"]}
    rows = []
    for original in _read(args.sam_results)["rows"]:
        if original["run_id"] not in accepted:
            continue
        box = boxes[(original["run_id"], original["camera_id"])]
        source = Path(box["source"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if (digest != original["source_sha256"] or
                digest != box["source_sha256"] or
                original["frame_id"] != box["frame_id"]):
            raise ValueError("saved frame ID or original image hash changed")
        mask_path = Path(original["mask_path"]).resolve()
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        roi = original.get("roi_within_crop") or padded_key_box_roi(
            box["bbox_crop_px"], canvas_size_wh=(512, 512))
        x0, y0, x1, y1 = map(int, roi)
        if mask.shape != (y1-y0, x1-x0):
            raise ValueError("saved SAM3 mask differs from its Qwen ROI")
        origin = (box["crop_xyxy"][0]+x0, box["crop_xyxy"][1]+y0)
        direction = np.asarray(box["nominal_tip_px"], dtype=float) - np.asarray(
            box["planned_rear_full_px"], dtype=float)
        sam_ellipse = distal_key_mask_half_ellipse(
            mask, planned_rear_px=box["planned_rear_full_px"],
            planned_tip_px=box["nominal_tip_px"], mask_origin_px=origin)
        nominal = _nominal_circle(
            box, args.runs_root.expanduser().resolve() / original["run_id"])
        direct = _unconstrained_cv_ellipse(
            mask, origin=origin, planned_direction=direction)
        row = {
            "run_id": original["run_id"],
            "camera_id": original["camera_id"],
            "frame_id": original["frame_id"],
            "timestamp_s": original["timestamp_s"],
            "source": str(source), "source_sha256": digest,
            "mask_path": str(mask_path),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "crop_xyxy": box["crop_xyxy"],
            "sam3_prompt": original.get("prompt", "white cylinder"),
            "qwen_prompt": box["prompt"], "qwen_response": box["answer"],
            "sam_rectangle_tip_full_px": original.get("tip_full_px"),
            "sam_half_ellipse": sam_ellipse,
            "unconstrained_cv_ellipse": direct,
            "planned_physical_circle": nominal,
            "human_approx_tip_full_px": box.get("human_approx_tip_full_px"),
        }
        rows.append(row)
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "schema": "saved_sam_key_face_ellipse_comparison_v1",
        "scope": "read_only_saved_images; only visually accepted socket poses",
        "socket_qc": str(args.socket_qc), "rows": rows,
    }, indent=2)+"\n", encoding="utf-8")
    print(target)
    for row in rows:
        print(row["run_id"], row["camera_id"], "SAM ellipse",
              None if row["sam_half_ellipse"] is None else
              row["sam_half_ellipse"]["center_full_px"])


if __name__ == "__main__":
    main()
