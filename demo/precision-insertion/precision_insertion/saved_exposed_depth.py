"""Bind an optional VLM/CAD depth diagnostic to final AutoDex camera bytes.

This is a saved-data producer for evaluating visible-rear grounding. It does
not assert that its user-supplied worst-case error bounds are commissioned,
and its output is deliberately *not* admitted as a physical task-success
source by ``insertion_checkpoint``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .calibration import (
    SessionCalibration, _canonical_sha256,
    validate_session_camera_calibration,
)
from .config import TaskMode
from .exposed_depth import ExposedDepthLimits, estimate_exposed_depth_for_mode
from .frame_provenance import bounded_capture_skew_s
from .grounded_alignment import observe_exposed_key_rear_axis
from .observer import ImageVLM
from .raw_camera_capture import verify_raw_camera_capture
from .session_bootstrap import _safe_id
from .xy_overlay import CalibratedXYFrame, camera_socket_transform
from .world import validated_frozen_socket_pose


_SCHEMA = "precision_insertion_saved_exposed_depth_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_frames(
    *, mode: TaskMode, shared_root: Path,
    calibration: SessionCalibration, calibration_path: Path,
    final_bundle: Path, completed_at_s: float, decision_timestamp_s: float,
    max_capture_skew_s: float, max_final_gap_s: float,
    max_frame_age_s: float,
) -> tuple[list[CalibratedXYFrame], dict]:
    limits = (completed_at_s, decision_timestamp_s,
              max_capture_skew_s, max_final_gap_s, max_frame_age_s)
    if (not all(type(v) in (int, float) and math.isfinite(v)
                for v in limits) or
            min(max_capture_skew_s, max_final_gap_s, max_frame_age_s) <= 0):
        raise ValueError("depth capture needs finite commissioned time limits")
    session_path = Path(calibration_path).expanduser().resolve()
    stored = json.loads(session_path.read_text(encoding="utf-8"))
    if (not isinstance(calibration, SessionCalibration) or
            stored != calibration.record or
            stored.get("schema") !=
            "precision_insertion_session_calibration_v1"):
        raise ValueError("depth capture differs from frozen session record")
    snapshot = calibration.record.get("camera_calibration")
    if not isinstance(snapshot, dict):
        raise ValueError("depth capture lacks frozen camera calibration")
    intrinsics = snapshot.get("intrinsics_full")
    extrinsics = snapshot.get("extrinsics_full")
    if not isinstance(intrinsics, dict) or not isinstance(extrinsics, dict):
        raise ValueError("depth capture lacks calibrated camera matrices")
    cameras = set(intrinsics)
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={camera: value["K_undist"]
                           for camera, value in intrinsics.items()},
        extrinsics_full=extrinsics, calibrated_camera_ids=cameras)
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=Path(shared_root).expanduser().resolve(),
        calibration=calibration)
    bundle = Path(final_bundle).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="final_or_abort")
    rows = capture["frame_evidence"]
    if (len(rows) < 2 or not set(rows) <= cameras or
            bounded_capture_skew_s(rows) > max_capture_skew_s):
        raise ValueError("final depth capture lacks synchronized calibrated views")
    first = min(row["timestamp_s"] - row["max_error_s"]
                for row in rows.values())
    last = max(row["timestamp_s"] + row["max_error_s"]
               for row in rows.values())
    if (not completed_at_s < first or
            first - completed_at_s > max_final_gap_s or
            decision_timestamp_s < last or
            decision_timestamp_s - first > max_frame_age_s):
        raise ValueError("final depth images do not follow this completed stroke")
    c2r = np.asarray(calibration.record["c2r"], dtype=float)
    frames = []
    for camera in sorted(rows):
        image = cv2.imread(str(bundle / "images" /
                               f"{_safe_id(camera, 'camera ID')}.png"),
                           cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("final depth camera image cannot be decoded")
        frames.append(CalibratedXYFrame(
            camera, rows[camera]["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            camera_socket_transform(
                T_camera_world=np.asarray(extrinsics[camera], dtype=float),
                T_world_robot=c2r, T_robot_socket=socket),
            np.asarray(intrinsics[camera]["K_undist"], dtype=float)))
    sources = {
        "session_calibration_path": str(session_path),
        "session_calibration_file_sha256": _sha(session_path),
        "session_calibration_sha256": _canonical_sha256(calibration.record),
        "camera_calibration_sha256": calibration.record[
            "camera_calibration_sha256"],
        "final_bundle": str(bundle),
        "final_manifest_sha256": _sha(bundle / "manifest.json"),
        "capture_id": capture["capture_id"],
        "request_id": capture["request_id"],
        "frame_evidence": rows,
        "image_file_sha256": capture["image_file_sha256"],
        "completed_at_s": float(completed_at_s),
        "decision_timestamp_s": float(decision_timestamp_s),
        "max_capture_skew_s": float(max_capture_skew_s),
        "max_final_gap_s": float(max_final_gap_s),
        "max_frame_age_s": float(max_frame_age_s),
    }
    return frames, sources


def assess_saved_exposed_depth(
    *, attempt_id: str, candidate_id: str, mode: TaskMode,
    shared_root: Path, calibration: SessionCalibration,
    calibration_path: Path, final_bundle: Path,
    completed_at_s: float, decision_timestamp_s: float,
    max_capture_skew_s: float, max_final_gap_s: float,
    max_frame_age_s: float, backend: ImageVLM,
    limits: ExposedDepthLimits,
) -> dict:
    _safe_id(attempt_id, "attempt ID")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("depth capture needs a selected candidate ID")
    limits.validate()
    root = Path(shared_root).expanduser().resolve()
    frames, source = _source_frames(
        mode=mode, shared_root=root, calibration=calibration,
        calibration_path=calibration_path, final_bundle=final_bundle,
        completed_at_s=completed_at_s,
        decision_timestamp_s=decision_timestamp_s,
        max_capture_skew_s=max_capture_skew_s,
        max_final_gap_s=max_final_gap_s,
        max_frame_age_s=max_frame_age_s)
    grounded, observations = observe_exposed_key_rear_axis(backend, frames)
    diagnostic = estimate_exposed_depth_for_mode(
        grounded, mode=mode, shared_root=root, limits=limits)
    return {
        "schema": _SCHEMA,
        "attempt_id": attempt_id, "candidate_id": candidate_id,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "shared_root": str(root), "source": source,
        "task_geometry_path": str(AssetPaths(root, mode).task_geometry.resolve()),
        "task_geometry_sha256": diagnostic["task_geometry_sha256"],
        "vlm_observations": [item.to_record() for item in observations],
        "diagnostic": diagnostic,
        "depth_source_admissible_for_task_label": False,
        "scope": "saved_final_image_depth_diagnostic_not_physical_success_source",
        "robot_ready": False,
    }


def write_saved_exposed_depth(report: dict, output_dir: Path) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


class _RecordedAnswers:
    """Replay saved VLM text through the same strict parser, without inference."""

    native_pixel_coordinates = True

    def __init__(self, observations: list[dict]):
        self.answers = [row["raw_answer"] for row in observations]
        self.model = observations[0]["backend_model"]

    def infer(self, _images, _prompt):
        return self.answers.pop(0)


def verify_saved_exposed_depth(
    report_path: Path, *, mode: TaskMode, shared_root: Path,
    calibration: SessionCalibration,
) -> dict:
    """Recheck saved pixels/calibration, parse raw VLM text and recompute CAD.

    This does not rerun the model or validate its landmark error bound.
    """
    saved = json.loads(Path(report_path).read_text(encoding="utf-8"))
    root = Path(shared_root).expanduser().resolve()
    if (not isinstance(saved, dict) or saved.get("schema") != _SCHEMA or
            saved.get("mode") != {"family": mode.family, "gap_mm": mode.gap_mm} or
            saved.get("shared_root") != str(root) or
            saved.get("robot_ready") is not False or
            saved.get("depth_source_admissible_for_task_label") is not False or
            saved.get("scope") !=
            "saved_final_image_depth_diagnostic_not_physical_success_source" or
            not isinstance(saved.get("vlm_observations"), list) or
            not saved["vlm_observations"]):
        raise ValueError("invalid saved exposed-depth diagnostic")
    source = saved["source"]
    frames, expected_source = _source_frames(
        mode=mode, shared_root=root, calibration=calibration,
        calibration_path=Path(source["session_calibration_path"]),
        final_bundle=Path(source["final_bundle"]),
        completed_at_s=source["completed_at_s"],
        decision_timestamp_s=source["decision_timestamp_s"],
        max_capture_skew_s=source["max_capture_skew_s"],
        max_final_gap_s=source["max_final_gap_s"],
        max_frame_age_s=source["max_frame_age_s"])
    if source != expected_source:
        raise ValueError("saved exposed-depth capture or session changed")
    recorded = saved["vlm_observations"]
    if len(recorded) != len(frames) or any(
            not isinstance(row, dict) or
            row.get("backend_model") != recorded[0].get("backend_model") or
            type(row.get("latency_s")) not in (int, float) or
            not math.isfinite(row["latency_s"]) or row["latency_s"] < 0
            for row in recorded):
        raise ValueError("saved depth VLM observation list is invalid")
    grounded, replay = observe_exposed_key_rear_axis(
        _RecordedAnswers(recorded), frames)
    for old, new in zip(recorded, replay):
        expected = new.to_record()
        expected["latency_s"] = old["latency_s"]
        if old != expected:
            raise ValueError("saved depth VLM text/landmarks differ from replay")
    diagnostic = estimate_exposed_depth_for_mode(
        grounded, mode=mode, shared_root=root,
        limits=ExposedDepthLimits(**saved["diagnostic"]["limits"]))
    if (saved["task_geometry_path"] != diagnostic["task_geometry_path"] or
            saved["task_geometry_sha256"] != diagnostic[
                "task_geometry_sha256"] or
            saved["diagnostic"] != diagnostic):
        raise ValueError("saved exposed-depth calculation differs from replay")
    return saved
