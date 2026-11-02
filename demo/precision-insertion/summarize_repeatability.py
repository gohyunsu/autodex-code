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


def _attempt_row(attempt: Path) -> dict[str, Any] | None:
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
        "first_tip_status_counts": dict(sorted(Counter(
            row["first_preinsert_key_tip"]["status"] for row in rows).items())),
        "interpretation": (
            "wrist error is robot FK versus the commanded preinsert target; "
            "metric key-tip error is independent multi-view evidence. "
            "Their difference is not reported as object-in-hand error because "
            "the current tip-only observation is not a full key SE(3) pose."),
    }


def summarize(attempt_root: Path, candidate_id: str | None = None) -> dict[str, Any]:
    root = attempt_root.expanduser().resolve()
    rows = []
    for attempt in sorted(path for path in root.glob("attempt_*") if path.is_dir()):
        row = _attempt_row(attempt)
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.attempt_root, args.candidate_id)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
