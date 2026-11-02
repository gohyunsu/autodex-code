#!/usr/bin/env python3
"""Replay partial-cylinder image alignment from saved physical attempts.

This command is read only: it consumes the already saved SAM masks/landmarks
and frozen session calibration.  It never connects to a camera or robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from precision_insertion.same_trial_xy import select_same_trial_alignment


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def review_attempt(
    attempt: Path, run: Path, *, robust_pixel_scale: float,
    max_axis_tilt_deg: float,
) -> dict:
    """Review one attempt using its immutable session calibration."""
    attempt = attempt.expanduser().resolve()
    run = run.expanduser().resolve()
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
            if (fit is None or "axis_unit_full_px" not in fit or
                    camera not in snapshot["intrinsics_full"]):
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
                if profile is None:
                    continue
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
            robust_pixel_scale=robust_pixel_scale)
        selection = select_same_trial_alignment(
            {**alignment, "partial_shape_alignment": result},
            source="partial_shape",
            max_axis_tilt_deg=max_axis_tilt_deg)["alignment_selection"]
        rows.append({
            "iteration": int(observation_file.stem),
            "source_observation": str(observation_file),
            "endpoint_status": alignment["status"],
            "endpoint_reason": alignment["reason"],
            "visible_shape_profiles": profiles,
            "partial_shape_alignment": result,
            "partial_shape_control_selection": selection,
        })

    return {
        "schema": "precision_insertion_saved_partial_shape_review_v2",
        "scope": "saved_evidence_read_only_no_camera_or_robot_command",
        "attempt_dir": str(attempt),
        "session_run_dir": str(run),
        "key_length_m": key_length_m,
        "robust_pixel_scale": robust_pixel_scale,
        "max_axis_tilt_deg": max_axis_tilt_deg,
        "iterations": rows,
    }


def _bound_session_run(attempt: Path) -> Path:
    binding = read(attempt / "preflight_binding.json")
    preflight = Path(binding["preflight_report"]).expanduser().resolve()
    # <run>/trial_preflights/<index>/report.json
    return preflight.parents[2]


def _finite_stats(values: list[float]) -> dict:
    """Return descriptive evidence without creating a control threshold."""
    finite = sorted(float(value) for value in values
                    if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "minimum": None, "median": None,
                "maximum": None}
    middle = len(finite) // 2
    median = (finite[middle] if len(finite) % 2 else
              (finite[middle - 1] + finite[middle]) / 2.)
    return {
        "count": len(finite),
        "minimum": finite[0],
        "median": median,
        "maximum": finite[-1],
    }


def _batch_summary(attempts: list[dict]) -> dict:
    """Summarize the last valid saved fit for each physical attempt.

    An attempt can end with an iteration that has only one observable view.
    For that reason, this report retains both the final iteration status and
    the last earlier iteration that had a multiview shape fit.  The values are
    offline correlations with human labels, never live admission criteria.
    """
    rows = []
    iteration_statuses = Counter()
    control_selections = Counter()
    fallback_reasons = Counter()
    for attempt in attempts:
        iterations = attempt["review"]["iterations"]
        iteration_statuses.update(
            str(row["partial_shape_alignment"].get("status"))
            for row in iterations)
        for row in iterations:
            selection = row["partial_shape_control_selection"]
            control_selections[str(selection.get("selected"))] += 1
            if selection.get("selected") != "partial_shape":
                fallback_reasons[str(selection.get("reason"))] += 1
        fitted = [row for row in iterations
                  if row["partial_shape_alignment"].get("status") ==
                  "diagnostic_partial_shape_pose"]
        last = fitted[-1] if fitted else None
        fit = last["partial_shape_alignment"] if last is not None else {}
        tip = fit.get("tip_socket_m")
        xy_mm = ([float(tip[0]) * 1000., float(tip[1]) * 1000.]
                 if isinstance(tip, list) and len(tip) == 3 else None)
        per_view = fit.get("per_view")
        if not isinstance(per_view, list):
            per_view = []
        rows.append({
            "attempt_id": attempt["attempt_id"],
            "candidate_id": attempt.get("candidate_id"),
            "execution_status": attempt.get("status"),
            "iteration_count": len(iterations),
            "final_iteration_shape_status": (
                iterations[-1]["partial_shape_alignment"].get("status")
                if iterations else None),
            "last_valid_fit_iteration": (
                last["iteration"] if last is not None else None),
            "last_valid_tip_xy_mm": xy_mm,
            "last_valid_tip_xy_norm_mm": (
                math.hypot(*xy_mm) if xy_mm is not None else None),
            "last_valid_axis_tilt_deg": fit.get("axis_tilt_deg"),
            "last_valid_camera_ids": fit.get("camera_ids", []),
            "last_valid_max_center_residual_px": (
                max(float(view["center_residual_norm_px"])
                    for view in per_view) if per_view else None),
            "last_valid_max_axis_residual_deg": (
                max(float(view["shaft_orientation_residual_deg"])
                    for view in per_view) if per_view else None),
        })

    by_status = []
    for status in sorted({str(row["execution_status"]) for row in rows}):
        selected = [row for row in rows
                    if str(row["execution_status"]) == status]
        fitted = [row for row in selected
                  if row["last_valid_tip_xy_norm_mm"] is not None]
        by_status.append({
            "execution_status": status,
            "attempt_count": len(selected),
            "attempts_with_valid_multiview_shape_fit": len(fitted),
            "last_valid_tip_xy_norm_mm": _finite_stats([
                row["last_valid_tip_xy_norm_mm"] for row in fitted]),
            "last_valid_axis_tilt_deg": _finite_stats([
                row["last_valid_axis_tilt_deg"] for row in fitted
                if row["last_valid_axis_tilt_deg"] is not None]),
        })
    return {
        "attempt_count": len(rows),
        "attempts_with_any_valid_multiview_shape_fit": sum(
            row["last_valid_fit_iteration"] is not None for row in rows),
        "iteration_shape_status_counts": dict(sorted(
            iteration_statuses.items())),
        "iteration_control_selection_counts": dict(sorted(
            control_selections.items())),
        "iteration_control_fallback_reason_counts": dict(sorted(
            fallback_reasons.items())),
        "by_execution_status": by_status,
        "attempt_last_fit_rows": rows,
        "interpretation": (
            "Offline saved-evidence summary only. The last valid fit can "
            "precede the final iteration and is not an insertion-success "
            "criterion or a robot command."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--attempt-dir", type=Path)
    source.add_argument(
        "--attempt-root", type=Path,
        help="Review every attempt_* directory with saved XY observations")
    parser.add_argument("--session-run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robust-pixel-scale", type=float, default=8.)
    parser.add_argument(
        "--max-axis-tilt-deg", type=float, default=10.,
        help=("Existing alignment-model range used by the live selector; "
              "out-of-range fits remain diagnostic and fall back"))
    args = parser.parse_args()

    if args.attempt_dir is not None:
        attempt = args.attempt_dir.expanduser().resolve()
        run = (args.session_run_dir.expanduser().resolve()
               if args.session_run_dir is not None else
               _bound_session_run(attempt))
        report = review_attempt(
            attempt, run, robust_pixel_scale=args.robust_pixel_scale,
            max_axis_tilt_deg=args.max_axis_tilt_deg)
        count = len(report["iterations"])
    else:
        if args.session_run_dir is not None:
            parser.error("--session-run-dir applies only to --attempt-dir")
        root = args.attempt_root.expanduser().resolve()
        attempts = []
        for result_file in sorted(root.glob("attempt_*/result.json"),
                                  key=lambda path: path.stat().st_mtime):
            attempt = result_file.parent
            if not any((attempt / "same_trial_xy" / "observations").glob(
                    "*.json")) or not (attempt / "preflight_binding.json").is_file():
                continue
            review = review_attempt(
                attempt, _bound_session_run(attempt),
                robust_pixel_scale=args.robust_pixel_scale,
                max_axis_tilt_deg=args.max_axis_tilt_deg)
            result = read(result_file)
            attempts.append({
                "attempt_id": attempt.name,
                "candidate_id": result.get("candidate_id"),
                "status": result.get("status"),
                "labels": result.get("labels"),
                "review": review,
            })
        report = {
            "schema": "precision_insertion_saved_partial_shape_batch_v2",
            "scope": "saved_evidence_read_only_no_camera_or_robot_command",
            "attempt_root": str(root),
            "robust_pixel_scale": args.robust_pixel_scale,
            "max_axis_tilt_deg": args.max_axis_tilt_deg,
            "attempts": attempts,
            "summary": _batch_summary(attempts),
        }
        count = len(attempts)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "reviewed": count},
                     indent=2))


if __name__ == "__main__":
    main()
