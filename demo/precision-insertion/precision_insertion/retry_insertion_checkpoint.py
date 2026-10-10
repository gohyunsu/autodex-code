"""Replay a fresh before/after VLM observation of one guarded XY retry.

The controller's metric claim and all image bytes are source-bound. A depth
number in that external claim is still unadmitted until an independent
physical key-depth producer has been commissioned and replay-verified.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import cv2

from .config import TaskMode
from .frame_provenance import bounded_capture_skew_s
from .grounded_lateral import GroundedLateralPreflight
from .insertion_checkpoint import (
    _admitted_measurement, _phase_rows, verify_final_insertion_capture,
    verify_preinsert_raw_capture, write_insertion_checkpoint,
)
from .observer import ImageVLM, observe_insertion_visual
from .outcome import InsertionEvidence, judge_insertion
from .postshift_arrival_checkpoint import PostShiftArrivalCheckpoint
from .postshift_arrival_replan import PostShiftArrivalReplan
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .retry_guarded_execution import verify_retry_guarded_execution
from .retry_guarded_metric import verify_retry_guarded_metric
from .session_bootstrap import _safe_id


_SCHEMA = "precision_insertion_retry_observed_checkpoint_v1"
_SCOPE = "retry_raw_images_and_external_metrics_not_physical_depth_success"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess_retry_insertion_checkpoint(
    *, execution_log_path: Path, handoff_report_path: Path,
    expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint,
    checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
    preinsert_bundle: Path, final_bundle: Path,
    decision_timestamp_s: float, backend: ImageVLM,
    max_phase_skew_s: float, max_preinsert_age_s: float,
    max_final_observation_gap_s: float,
    minimum_visual_views: int = 2,
) -> dict:
    """Observe a second stroke, not a replay of the first insertion images."""
    limits = (max_phase_skew_s, max_preinsert_age_s,
              max_final_observation_gap_s)
    if (not isinstance(mode, TaskMode) or mode.family != "cylinder" or
            not math.isclose(mode.target_depth_m, .020, abs_tol=1e-9) or
            not all(type(value) in (float, int) and
                    math.isfinite(value) and value > 0 for value in limits) or
            type(minimum_visual_views) is not int or
            minimum_visual_views < 2 or
            type(decision_timestamp_s) not in (float, int) or
            not math.isfinite(decision_timestamp_s)):
        raise ValueError("retry checkpoint needs commissioned 20 mm limits")
    execution_path = Path(execution_log_path).expanduser().resolve()
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    execution = verify_retry_guarded_execution(
        execution_path, handoff_report_path=handoff_path,
        expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    metric_path = Path(execution["metric_record"]["path"])
    metric, started, completed = verify_retry_guarded_metric(
        metric_path, handoff_report_path=handoff_path,
        expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    before_root = Path(preinsert_bundle).expanduser().resolve()
    after_root = Path(final_bundle).expanduser().resolve()
    if before_root == after_root or before_root == arrival.capture_dir.resolve():
        raise ValueError("retry needs a new pre-contact camera capture")
    before = verify_preinsert_raw_capture(before_root)
    after = verify_final_insertion_capture(after_root)
    arrival_record = verify_preinsert_raw_capture(arrival.capture_dir)
    admitted = set(arrival.alignment["inlier_cameras"])
    cameras = sorted(admitted & set(before["frame_evidence"]) &
                     set(after["frame_evidence"]))
    if len(cameras) < minimum_visual_views:
        raise ValueError("retry lacks paired previously grounded camera views")
    old_rows = arrival_record["frame_evidence"]
    before_rows = {camera: before["frame_evidence"][camera]
                   for camera in cameras}
    after_rows = {camera: after["frame_evidence"][camera]
                  for camera in cameras}
    if (before["request_id"] <= arrival_record["request_id"] or
            after["request_id"] <= before["request_id"] or
            any(before_rows[camera]["frame_id"] <=
                old_rows[camera]["frame_id"] or
                after_rows[camera]["frame_id"] <=
                before_rows[camera]["frame_id"] for camera in cameras) or
            bounded_capture_skew_s(before_rows) > max_phase_skew_s or
            bounded_capture_skew_s(after_rows) > max_phase_skew_s):
        raise ValueError("retry camera frames are reused or unsynchronized")
    handoff_decision = float(json.loads(
        handoff_path.read_text(encoding="utf-8"))["decision_timestamp_s"])
    before_lower = min(row["timestamp_s"] - row["max_error_s"]
                       for row in before_rows.values())
    before_upper = max(row["timestamp_s"] + row["max_error_s"]
                       for row in before_rows.values())
    after_lower = min(row["timestamp_s"] - row["max_error_s"]
                      for row in after_rows.values())
    after_upper = max(row["timestamp_s"] + row["max_error_s"]
                      for row in after_rows.values())
    if (not handoff_decision < before_lower <= before_upper < started <
            completed < after_lower <= after_upper <= decision_timestamp_s or
            started - before_upper > max_preinsert_age_s or
            after_lower - completed > max_final_observation_gap_s):
        raise ValueError("retry frames do not bracket the guarded contact")
    frames, inputs = _phase_rows(before_root, before, cameras, "preinsert")
    later, later_inputs = _phase_rows(
        after_root, after, cameras, "final_or_abort")
    frames.extend(later)
    inputs.extend(later_inputs)
    for camera in cameras:
        name = f"{_safe_id(camera, 'camera ID')}.png"
        arrival_image = cv2.imread(str(
            arrival.capture_dir / "images" / name))
        before_image = cv2.imread(str(before_root / "images" / name))
        after_image = cv2.imread(str(after_root / "images" / name))
        if (arrival_image is None or before_image is None or
                after_image is None or
                arrival_image.shape != before_image.shape or
                before_image.shape != after_image.shape):
            raise ValueError("retry image pair changed camera geometry")
    visual = observe_insertion_visual(backend, frames)
    supported = (visual.parse_error is None and
                 len(set(visual.parsed["evidence_views"]) & set(cameras)) >=
                 minimum_visual_views)
    visual_class = (visual.parsed["visual_class"] if supported else
                    "unobservable")
    sensor, admissibility = _admitted_measurement(metric)
    evidence = InsertionEvidence(
        visual_class,
        tuple(sensor["key_depth_interval_m"])
        if sensor["key_depth_interval_m"] is not None else None,
        sensor["key_depth_source"], sensor["alignment_within_limits"],
        sensor["safety_abort"], sensor["grasp_held"])
    outcome = judge_insertion(evidence, target_depth_m=mode.target_depth_m)
    return {
        "schema": _SCHEMA,
        "attempt_id": expected.attempt_id,
        "candidate_id": expected.candidate_id,
        "session_calibration_sha256":
            execution["session_calibration_sha256"],
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "target_depth_m": mode.target_depth_m,
        "handoff_report_path": str(handoff_path),
        "handoff_report_sha256": _sha(handoff_path),
        "execution_log_path": str(execution_path),
        "execution_log_sha256": _sha(execution_path),
        "metric_record_path": str(metric_path),
        "metric_record_sha256": _sha(metric_path),
        "preinsert_bundle": str(before_root),
        "preinsert_manifest_sha256": _sha(before_root / "manifest.json"),
        "final_bundle": str(after_root),
        "final_manifest_sha256": _sha(after_root / "manifest.json"),
        "guarded_started_at_s": started,
        "guarded_completed_at_s": completed,
        "decision_timestamp_s": float(decision_timestamp_s),
        "frames": inputs,
        "visual": visual.to_record(),
        "minimum_visual_views": minimum_visual_views,
        "max_phase_skew_s": float(max_phase_skew_s),
        "max_preinsert_age_s": float(max_preinsert_age_s),
        "max_final_observation_gap_s":
            float(max_final_observation_gap_s),
        "effective_vlm_class": visual_class,
        "key_depth_admissibility": admissibility,
        "evidence": asdict(evidence),
        "outcome": outcome.to_record(),
        "scope": _SCOPE,
        "robot_ready": False,
    }


class _RecordedAnswer:
    def __init__(self, visual: dict):
        self.model = visual["backend_model"]
        self.answer = visual["raw_answer"]

    def infer(self, _images, _prompt):
        return self.answer


def verify_retry_insertion_checkpoint(
    report_path: Path, *, expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint,
    checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Replay saved VLM text, pixels, metric and label without inference."""
    saved = json.loads(Path(report_path).expanduser().resolve().read_text(
        encoding="utf-8"))
    if (not isinstance(saved, dict) or saved.get("schema") != _SCHEMA or
            saved.get("scope") != _SCOPE or
            saved.get("robot_ready") is not False or
            not isinstance(saved.get("visual"), dict) or
            type(saved["visual"].get("latency_s")) not in (int, float) or
            not math.isfinite(saved["visual"]["latency_s"]) or
            saved["visual"]["latency_s"] < 0):
        raise ValueError("invalid retry insertion checkpoint")
    rebuilt = assess_retry_insertion_checkpoint(
        execution_log_path=Path(saved["execution_log_path"]),
        handoff_report_path=Path(saved["handoff_report_path"]),
        expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan,
        mode=mode, shared_root=shared_root, calibration=calibration,
        preinsert_bundle=Path(saved["preinsert_bundle"]),
        final_bundle=Path(saved["final_bundle"]),
        decision_timestamp_s=saved["decision_timestamp_s"],
        backend=_RecordedAnswer(saved["visual"]),
        max_phase_skew_s=saved["max_phase_skew_s"],
        max_preinsert_age_s=saved["max_preinsert_age_s"],
        max_final_observation_gap_s=saved["max_final_observation_gap_s"],
        minimum_visual_views=saved["minimum_visual_views"])
    rebuilt["visual"]["latency_s"] = saved["visual"]["latency_s"]
    if saved != rebuilt:
        raise ValueError("retry insertion checkpoint differs from replay")
    return saved


def write_retry_insertion_checkpoint(report: dict, output_dir: Path) -> Path:
    """Use the same exclusive artifact writer as the first checkpoint."""
    if not isinstance(report, dict) or report.get("schema") != _SCHEMA:
        raise ValueError("retry checkpoint writer needs its own report")
    return write_insertion_checkpoint(report, output_dir)
