#!/usr/bin/env python3
"""Replay partial-cylinder image alignment from one saved physical attempt.

This command is read only: it consumes the already saved SAM masks/landmarks
and frozen session calibration.  It never connects to a camera or robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.held_key_tip import visible_cylinder_mask_profile
from precision_insertion.partial_shape_alignment import (
    PartialCylinderView, estimate_partial_cylinder_alignment,
    partial_shape_measurement_weights,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--session-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robust-pixel-scale", type=float, default=8.)
    args = parser.parse_args()

    attempt = args.attempt_dir.expanduser().resolve()
    run = args.session_run_dir.expanduser().resolve()
    calibration = read(run / "frozen_session_calibration.json")
    snapshot = calibration["camera_calibration"]
    C = np.asarray(calibration["c2r"], dtype=float)
    socket = np.asarray(calibration["socket_pose_robot"], dtype=float)
    catalog = read(run / "endpoint_catalog.json")
    mode_row = catalog["mode"]
    mode = select_mode(mode_row["family"], float(mode_row["gap_mm"]))
    geometry = read(AssetPaths(Path(catalog["shared_root"]), mode).task_geometry)
    key_length_m = float(geometry["key_frame"]["tip_z_m"])

    rows = []
    for observation_file in sorted(
            (attempt / "same_trial_xy" / "observations").glob("*.json")):
        observation = read(observation_file)
        alignment = observation["alignment"]
        views = []
        profiles = {}
        for record in alignment.get("sam_half_ellipse_views", []):
            fit = record.get("sam_half_ellipse")
            camera = record.get("camera_id")
            if fit is None or camera not in snapshot["intrinsics_full"]:
                continue
            mask_file = record.get("mask_file")
            mask = (None if mask_file is None else
                    cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE))
            rear = record.get("planned_rear_full_px")
            tip = record.get("nominal_tip_full_px")
            roi = record.get("roi_within_crop")
            crop = record.get("crop_xyxy")
            if (mask is not None and rear is not None and tip is not None and
                    roi is not None and crop is not None):
                profile = visible_cylinder_mask_profile(
                    mask, axis_unit_full_px=fit["axis_unit_full_px"],
                    mask_origin_px=(crop[0] + roi[0], crop[1] + roi[1]),
                    planned_axis_span_px=math.dist(rear, tip))
                weights = partial_shape_measurement_weights(fit, profile)
                profiles[camera] = {**profile,
                                    "partial_shape_weights": weights}
                if (weights["distal_center_weight"] +
                        weights["shaft_axis_weight"] > 0.):
                    K = np.asarray(snapshot["intrinsics_full"][camera][
                        "K_undist"], dtype=float)
                    E = np.asarray(
                        snapshot["extrinsics_full"][camera], dtype=float)
                    views.append(PartialCylinderView(
                        camera_id=camera, intrinsics=K,
                        T_camera_socket=E @ C @ socket,
                        distal_center_uv_px=tuple(fit["center_full_px"]),
                        shaft_axis_uv=tuple(fit["axis_unit_full_px"]),
                        distal_center_weight=weights[
                            "distal_center_weight"],
                        shaft_axis_weight=weights["shaft_axis_weight"]))
        result = estimate_partial_cylinder_alignment(
            views,
            nominal_tip_z_socket_m=float(
                alignment["nominal_tip_z_socket_m"]),
            key_length_m=key_length_m,
            robust_pixel_scale=args.robust_pixel_scale)
        rows.append({
            "iteration": int(observation_file.stem),
            "source_observation": str(observation_file),
            "endpoint_status": alignment["status"],
            "endpoint_reason": alignment["reason"],
            "visible_shape_profiles": profiles,
            "partial_shape_alignment": result,
        })

    report = {
        "schema": "precision_insertion_saved_partial_shape_review_v1",
        "scope": "saved_evidence_read_only_no_camera_or_robot_command",
        "attempt_dir": str(attempt),
        "session_run_dir": str(run),
        "key_length_m": key_length_m,
        "robust_pixel_scale": args.robust_pixel_scale,
        "iterations": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "iterations": len(rows)},
                     indent=2))


if __name__ == "__main__":
    main()
