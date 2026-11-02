#!/usr/bin/env python3
"""Summarize fixed-grasp pre-insertion repeatability from saved attempts.

This is an offline analysis tool.  Wrist endpoint error and visually observed
key-tip error remain separate measurements; a planar single-view hypothesis is
never counted as a metric key-tip observation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _latest(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _matrix(value: Any) -> list[list[float]] | None:
    if (not isinstance(value, list) or len(value) != 4 or
            any(not isinstance(row, list) or len(row) != 4 for row in value)):
        return None
    try:
        matrix = [[float(cell) for cell in row] for row in value]
    except (TypeError, ValueError):
        return None
    return matrix if all(math.isfinite(cell) for row in matrix for cell in row) else None


def _digest(value: Any) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partial_shape_attempts(
    path: Path | None, *, attempt_root: Path,
) -> tuple[dict[str, dict], dict | None]:
    if path is None:
        return {}, None
    source = path.expanduser().resolve()
    review = _load(source)
    if (review.get("schema") != "precision_insertion_saved_partial_shape_batch_v2" or
            not isinstance(review.get("attempts"), list)):
        raise ValueError("partial-shape review needs the saved batch v2 schema")
    recorded_root = review.get("attempt_root")
    if (not isinstance(recorded_root, str) or
            Path(recorded_root).expanduser().resolve() != attempt_root):
        raise ValueError("partial-shape review belongs to another attempt root")
    by_attempt = {}
    for row in review["attempts"]:
        if not isinstance(row, dict) or not isinstance(row.get("attempt_id"), str):
            raise ValueError("partial-shape review has an invalid attempt row")
        iterations = row.get("review", {}).get("iterations")
        if not isinstance(iterations, list):
            raise ValueError("partial-shape review lacks iteration records")
        first = next((item for item in iterations
                      if isinstance(item, dict) and item.get("iteration") == 0), None)
        if first is not None:
            by_attempt[row["attempt_id"]] = first
    return by_attempt, {
        "path": str(source), "sha256": _file_sha256(source),
        "schema": review["schema"],
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "q1": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.5),
        "q3": _quantile(finite, 0.75),
        "maximum": max(finite) if finite else None,
    }


def _subtract_translation(measured: list[list[float]] | None,
                          target: list[list[float]] | None) -> list[float] | None:
    if measured is None or target is None:
        return None
    return [measured[index][3] - target[index][3] for index in range(3)]


def _partial_shape_row(iteration: dict | None) -> dict[str, Any]:
    if not isinstance(iteration, dict):
        return {
            "status": "not_replayed", "metric_observation": False,
            "selected_for_live_control": False,
            "selection_reason": None, "tip_socket_m": None,
            "xy_residual_norm_m": None, "axis_tilt_deg": None,
            "camera_ids": [], "jacobian_rank": None,
        }
    shape = iteration.get("partial_shape_alignment")
    selection = iteration.get("partial_shape_control_selection")
    if not isinstance(shape, dict):
        shape = {}
    if not isinstance(selection, dict):
        selection = {}
    cameras = shape.get("camera_ids")
    cameras = ([str(camera) for camera in cameras]
               if isinstance(cameras, list) else [])
    tip = shape.get("tip_socket_m")
    if (not isinstance(tip, list) or len(tip) != 3 or
            any(not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) for value in tip)):
        tip = None
    else:
        tip = [float(value) for value in tip]
    tilt = shape.get("axis_tilt_deg")
    tilt = (float(tilt) if isinstance(tilt, (int, float)) and
            math.isfinite(float(tilt)) else None)
    rank = shape.get("jacobian_rank")
    metric = bool(
        shape.get("status") == "diagnostic_partial_shape_pose" and
        len(set(cameras)) >= 2 and tip is not None and tilt is not None and
        type(rank) is int and rank == 4)
    return {
        "status": str(shape.get("status", "not_replayed")),
        "metric_observation": metric,
        "selected_for_live_control": selection.get("selected") == "partial_shape",
        "selection_reason": selection.get("reason"),
        "tip_socket_m": tip if metric else None,
        "diagnostic_tip_socket_m": tip,
        "xy_residual_norm_m": (
            math.hypot(tip[0], tip[1]) if metric else None),
        "axis_tilt_deg": tilt if metric else None,
        "camera_ids": cameras,
        "jacobian_rank": rank if type(rank) is int else None,
    }


def _attempt_row(attempt: Path, partial_shape: dict | None = None,
                 ) -> dict[str, Any] | None:
    result = _load(attempt / "result.json")
    preinsert_path = _latest(attempt, "preinsert_assessments/*/report.json")
    preinsert = _load(preinsert_path)
    if not preinsert:
        return None
    preflight_path = _latest(attempt, "postlift_preflights/*/report.json")
    preflight = _load(preflight_path)
    observation_path = attempt / "same_trial_xy" / "observations" / "000.json"
    observation = _load(observation_path)
    alignment = observation.get("alignment")
    if not isinstance(alignment, dict):
        alignment = {}

    prediction = preinsert.get("comparison", {}).get("prediction", {})
    if not isinstance(prediction, dict):
        prediction = {}
    targets = preflight.get("targets")
    if not isinstance(targets, dict):
        targets = {}
    measured_wrist = _matrix(prediction.get("T_robot_hand_measured_fk"))
    target_wrist = _matrix(targets.get("T_robot_hand_preinsert"))
    wrist_vector = _subtract_translation(measured_wrist, target_wrist)

    tip = alignment.get("tip_socket_m")
    if (not isinstance(tip, list) or len(tip) != 3 or
            any(not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) for value in tip)):
        tip = None
    else:
        tip = [float(value) for value in tip]
    inlier_cameras = alignment.get("inlier_cameras")
    if not isinstance(inlier_cameras, list):
        inlier_cameras = []
    inlier_cameras = [str(camera) for camera in inlier_cameras]
    alignment_status = str(alignment.get("status", "not_observed"))
    metric_tip = bool(
        tip is not None and len(inlier_cameras) >= 2 and
        "metric" in alignment_status)

    context = preinsert.get("nominal_source_context")
    if not isinstance(context, dict):
        context = {}
    labels = result.get("labels") if isinstance(result.get("labels"), dict) else {}
    diagnostic = (result.get("diagnostic_labels")
                  if isinstance(result.get("diagnostic_labels"), dict) else {})
    translation_error = preinsert.get("hand_translation_error_m")
    rotation_error = preinsert.get("hand_rotation_error_deg")
    target_digest = _digest(target_wrist)
    return {
        "attempt_id": attempt.name,
        "candidate_id": result.get("candidate_id", preinsert.get("candidate_id")),
        "session_calibration_sha256": preinsert.get(
            "session_calibration_sha256"),
        "task_family": context.get("family"),
        "gap_mm": context.get("gap_mm"),
        "preinsert_target_sha256": target_digest,
        "wrist_endpoint": {
            "translation_error_robot_m": wrist_vector,
            "translation_error_norm_m": (
                float(translation_error)
                if isinstance(translation_error, (int, float)) else None),
            "rotation_error_deg": (
                float(rotation_error)
                if isinstance(rotation_error, (int, float)) else None),
            "measured_pose_available": measured_wrist is not None,
            "target_pose_available": target_wrist is not None,
        },
        "first_preinsert_key_tip": {
            "status": alignment_status,
            "reason": alignment.get("reason"),
            "metric_observation": metric_tip,
            "tip_socket_m": tip if metric_tip else None,
            "diagnostic_tip_socket_m": tip,
            "xy_residual_norm_m": (
                math.hypot(tip[0], tip[1]) if metric_tip else None),
            "axis_tilt_deg": (float(alignment["axis_tilt_deg"])
                              if metric_tip and isinstance(
                                  alignment.get("axis_tilt_deg"), (int, float))
                              else None),
            "inlier_cameras": inlier_cameras,
            "source_report": str(observation_path) if observation else None,
        },
        "first_preinsert_partial_shape": _partial_shape_row(partial_shape),
        "labels": {
            "grasp_success": labels.get("grasp_success"),
            "preinsert_reached": labels.get("preinsert_reached"),
            "tip_entry_success": diagnostic.get("tip_entry_success"),
            "insertion_success": labels.get("insertion_success"),
        },
        "sources": {
            "preinsert_report": str(preinsert_path),
            "preflight_report": str(preflight_path) if preflight_path else None,
        },
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrist_translation = [
        row["wrist_endpoint"]["translation_error_norm_m"] * 1000.0
        for row in rows
        if row["wrist_endpoint"]["translation_error_norm_m"] is not None]
    wrist_rotation = [
        row["wrist_endpoint"]["rotation_error_deg"] for row in rows
        if row["wrist_endpoint"]["rotation_error_deg"] is not None]
    metric = [row["first_preinsert_key_tip"] for row in rows
              if row["first_preinsert_key_tip"]["metric_observation"]]
    shapes = [row["first_preinsert_partial_shape"] for row in rows
              if row["first_preinsert_partial_shape"]["metric_observation"]]
    selected_shapes = [item for item in shapes
                       if item["selected_for_live_control"]]
    targets = sorted({row["preinsert_target_sha256"] for row in rows
                      if row["preinsert_target_sha256"]})
    return {
        "attempt_count": len(rows),
        "same_nominal_preinsert_target": len(targets) == 1,
        "preinsert_target_sha256_values": targets,
        "wrist_translation_error_mm": _stats(wrist_translation),
        "wrist_rotation_error_deg": _stats(wrist_rotation),
        "metric_key_tip_count": len(metric),
        "metric_tip_x_mm": _stats([
            item["tip_socket_m"][0] * 1000.0 for item in metric]),
        "metric_tip_y_mm": _stats([
            item["tip_socket_m"][1] * 1000.0 for item in metric]),
        "metric_tip_xy_residual_mm": _stats([
            item["xy_residual_norm_m"] * 1000.0 for item in metric]),
        "metric_axis_tilt_deg": _stats([
            item["axis_tilt_deg"] for item in metric
            if item["axis_tilt_deg"] is not None]),
        "partial_shape_metric_count": len(shapes),
        "partial_shape_tip_x_mm": _stats([
            item["tip_socket_m"][0] * 1000.0 for item in shapes]),
        "partial_shape_tip_y_mm": _stats([
            item["tip_socket_m"][1] * 1000.0 for item in shapes]),
        "partial_shape_tip_xy_residual_mm": _stats([
            item["xy_residual_norm_m"] * 1000.0 for item in shapes]),
        "partial_shape_axis_tilt_deg": _stats([
            item["axis_tilt_deg"] for item in shapes]),
        "partial_shape_control_selected_count": len(selected_shapes),
        "partial_shape_selected_tip_xy_residual_mm": _stats([
            item["xy_residual_norm_m"] * 1000.0
            for item in selected_shapes]),
        "partial_shape_selected_axis_tilt_deg": _stats([
            item["axis_tilt_deg"] for item in selected_shapes]),
        "partial_shape_status_counts": dict(sorted(Counter(
            row["first_preinsert_partial_shape"]["status"]
            for row in rows).items())),
        "partial_shape_control_selection_counts": dict(sorted(Counter(
            ("selected" if row["first_preinsert_partial_shape"]
             ["selected_for_live_control"] else "fallback_or_not_replayed")
            for row in rows).items())),
        "first_tip_status_counts": dict(sorted(Counter(
            row["first_preinsert_key_tip"]["status"] for row in rows).items())),
        "interpretation": (
            "wrist error is robot FK versus the commanded preinsert target; "
            "endpoint and partial-shape key errors are separate independent "
            "multi-view measurements. "
            "A full-rank partial-shape fit remains diagnostic unless its "
            "saved control selection says selected; selected-only statistics "
            "are reported separately. "
            "Their difference is not reported as object-in-hand error because "
            "that requires composing the measured wrist with one consistent "
            "symmetry-reduced key pose, not subtracting scalar norms."),
    }


def summarize(attempt_root: Path, candidate_id: str | None = None,
              partial_shape_review: Path | None = None) -> dict[str, Any]:
    root = attempt_root.expanduser().resolve()
    partial_shapes, partial_shape_source = _partial_shape_attempts(
        partial_shape_review, attempt_root=root)
    rows = []
    for attempt in sorted(path for path in root.glob("attempt_*") if path.is_dir()):
        row = _attempt_row(attempt, partial_shapes.get(attempt.name))
        if row is not None and (candidate_id is None or
                                row["candidate_id"] == candidate_id):
            rows.append(row)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["candidate_id"], row["session_calibration_sha256"],
               row["task_family"], row["gap_mm"])
        grouped.setdefault(key, []).append(row)
    groups = []
    for key, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
        groups.append({
            "candidate_id": key[0],
            "session_calibration_sha256": key[1],
            "task_family": key[2],
            "gap_mm": key[3],
            **_group_summary(selected),
            "attempt_ids": [row["attempt_id"] for row in selected],
        })
    return {
        "schema": "precision_insertion_repeatability_summary_v1",
        "attempt_root": str(root),
        "candidate_filter": candidate_id,
        "partial_shape_review": partial_shape_source,
        "attempts_with_preinsert_measurement": len(rows),
        "groups": groups,
        "attempts": rows,
        "scope": (
            "offline_saved_evidence_only_no_robot_camera_or_model_access_"
            "single_view_planar_points_excluded_from_metric_tip_statistics"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument(
        "--partial-shape-review", type=Path,
        help="optional saved batch-v2 replay joined by attempt and iteration 0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.attempt_root, args.candidate_id, args.partial_shape_review)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
