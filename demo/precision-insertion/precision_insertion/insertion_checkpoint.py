"""Bind a read-only insertion VLM verdict to saved, time-ordered camera pixels.

Either an admitted held-key capture or a raw same-camera capture can supply
the pre-insertion image. The final image is also raw: an occluded key need not
have a valid post-grasp FoundPose estimate. A separate controller/sensor record supplies metric depth,
alignment, grasp and abort assertions. None of these records commands motion
or certifies the camera/controller producers themselves.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import cv2
from PIL import Image

from .frame_provenance import bounded_capture_skew_s, image_sha256
from .guarded_trace import verify_guarded_contact_trace
from .key_perception import verify_key_capture_artifacts
from .observer import ImageVLM, LabeledFrame, observe_insertion_visual
from .outcome import InsertionEvidence, judge_insertion
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .session_bootstrap import _safe_id


FinalInsertionCapture = RawCameraCapture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_preinsert_raw_capture(
    capture: FinalInsertionCapture, output_dir: Path,
) -> Path:
    return write_raw_camera_capture(capture, output_dir, phase="preinsert")


def write_final_insertion_capture(
    capture: FinalInsertionCapture, output_dir: Path,
) -> Path:
    return write_raw_camera_capture(capture, output_dir,
                                    phase="final_or_abort")


def verify_preinsert_raw_capture(output_dir: Path) -> dict:
    return verify_raw_camera_capture(output_dir, phase="preinsert")


def verify_final_insertion_capture(output_dir: Path) -> dict:
    return verify_raw_camera_capture(output_dir, phase="final_or_abort")


def _metric_record(
    path: Path, *, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str,
) -> tuple[dict, float, float]:
    metric = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(metric, dict) or
            metric.get("schema") != "precision_insertion_guarded_execution_v1" or
            metric.get("attempt_id") != attempt_id or
            metric.get("candidate_id") != candidate_id or
            metric.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            not isinstance(metric.get("measurement"), dict)):
        raise ValueError("guarded execution metrics differ from this attempt")
    try:
        started = float(metric["started_at_s"])
        completed = float(metric["completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("guarded execution times are missing") from exc
    if not all(map(math.isfinite, (started, completed))) or started >= completed:
        raise ValueError("guarded execution times are invalid")
    expected_fields = {
        "key_depth_interval_m", "key_depth_source",
        "alignment_within_limits", "safety_abort", "grasp_held",
    }
    if set(metric["measurement"]) != expected_fields:
        raise ValueError("guarded execution metric fields are incomplete")
    for name in ("alignment_within_limits", "safety_abort", "grasp_held"):
        if metric["measurement"][name] is not None and type(
                metric["measurement"][name]) is not bool:
            raise ValueError(f"guarded execution {name} must be boolean or unknown")
    source_records = metric.get("source_records")
    if (not isinstance(source_records, dict) or
            set(source_records) != {
                "key_depth", "alignment", "force_trace", "grasp_state"}):
        raise ValueError("guarded metrics need four named source references")
    for name, source in source_records.items():
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError(f"invalid guarded metric source: {name}")
        source_path = Path(source["path"])
        if (not source_path.is_absolute() or not source_path.is_file() or
                _sha(source_path) != source["sha256"]):
            raise ValueError(f"guarded metric source changed: {name}")
    trace = verify_guarded_contact_trace(
        Path(source_records["force_trace"]["path"]))
    if (trace["attempt_id"] != attempt_id or
            trace["candidate_id"] != candidate_id or
            trace["session_calibration_sha256"] !=
            session_calibration_sha256 or
            trace["started_at_s"] != started or
            trace["terminal_decision_time_s"] > completed or
            metric["measurement"]["safety_abort"] is not trace["safety_abort"]):
        raise ValueError("guarded metric conflicts with replayed contact trace")
    return metric, started, completed


def _phase_rows(root: Path, report: dict, cameras: list[str], phase: str) -> tuple[list[LabeledFrame], list[dict]]:
    frames, inputs = [], []
    for camera in cameras:
        path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        row = report["frame_evidence"][camera]
        if image is None or image_sha256(image) != row["image_sha256"]:
            raise ValueError("insertion VLM image differs from saved frame")
        frames.append(LabeledFrame(
            camera, phase, row["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
        inputs.append({
            "phase": phase, "camera_id": camera, "path": str(path),
            "file_sha256": _sha(path), "frame_id": row["frame_id"],
            "image_sha256": row["image_sha256"],
            "timestamp_s": row["timestamp_s"],
            "max_error_s": row["max_error_s"],
        })
    return frames, inputs


def _preinsert_capture(root: Path) -> tuple[dict, str, str, set[str]]:
    """Admit a pose-bound capture or a raw image without inventing FoundPose."""
    key_manifest = root / "evidence_manifest.json"
    raw_manifest = root / "manifest.json"
    if key_manifest.is_file() == raw_manifest.is_file():
        raise ValueError("pre-insertion bundle must contain exactly one capture kind")
    if key_manifest.is_file():
        verify_key_capture_artifacts(root)
        record = json.loads((root / "key_observation.json").read_text(
            encoding="utf-8"))
        if record.get("phase") != "held_preinsert":
            raise ValueError("pre-insertion key bundle is not held-preinsert")
        cameras = set(record["consistency"]["accepted_views"])
        return record, "held_key_pose", "evidence_manifest.json", cameras
    record = verify_preinsert_raw_capture(root)
    return record, "raw_images", "manifest.json", set(record["frame_evidence"])


def assess_insertion_checkpoint(
    *, attempt_id: str, candidate_id: str, target_depth_m: float,
    session_calibration_sha256: str,
    preinsert_bundle: Path, final_bundle: Path, metric_record_path: Path,
    preinsert_reached_at_s: float, decision_timestamp_s: float,
    backend: ImageVLM, max_phase_skew_s: float,
    max_preinsert_age_s: float, max_final_observation_gap_s: float,
    minimum_visual_views: int = 2,
) -> dict:
    """Fuse a saved multiview comparison with independent metric assertions."""
    _safe_id(attempt_id, "attempt ID")
    if (not isinstance(session_calibration_sha256, str) or
            len(session_calibration_sha256) != 64 or
            any(ch not in "0123456789abcdef"
                for ch in session_calibration_sha256)):
        raise ValueError("insertion needs a frozen session calibration digest")
    limits = (target_depth_m, max_phase_skew_s, max_preinsert_age_s,
              max_final_observation_gap_s)
    if not math.isclose(target_depth_m, .020, rel_tol=0, abs_tol=1e-9):
        raise ValueError("insertion checkpoint target must be 20 mm")
    if (not all(math.isfinite(float(x)) and float(x) > 0 for x in limits) or
            type(minimum_visual_views) is not int or minimum_visual_views < 2 or
            not all(math.isfinite(float(x)) for x in (
                preinsert_reached_at_s, decision_timestamp_s))):
        raise ValueError("insertion checkpoint needs positive commissioned limits")
    before_root = Path(preinsert_bundle).expanduser().resolve()
    after_root = Path(final_bundle).expanduser().resolve()
    metric_path = Path(metric_record_path).expanduser().resolve()
    before, before_kind, before_manifest, before_cameras = _preinsert_capture(
        before_root)
    after = verify_final_insertion_capture(after_root)
    metric, started, completed = _metric_record(
        metric_path, attempt_id=attempt_id, candidate_id=candidate_id,
        session_calibration_sha256=session_calibration_sha256)
    cameras = sorted(before_cameras & set(after["frame_evidence"]))
    if len(cameras) < minimum_visual_views:
        raise ValueError("insertion checkpoint lacks paired camera views")
    before_rows = {camera: before["frame_evidence"][camera] for camera in cameras}
    after_rows = {camera: after["frame_evidence"][camera] for camera in cameras}
    if (bounded_capture_skew_s(before_rows) > max_phase_skew_s or
            bounded_capture_skew_s(after_rows) > max_phase_skew_s):
        raise ValueError("insertion camera exposure skew exceeds limit")
    before_lower = min(row["timestamp_s"] - row["max_error_s"]
                       for row in before_rows.values())
    before_upper = max(row["timestamp_s"] + row["max_error_s"]
                       for row in before_rows.values())
    after_lower = min(row["timestamp_s"] - row["max_error_s"]
                      for row in after_rows.values())
    after_upper = max(row["timestamp_s"] + row["max_error_s"]
                      for row in after_rows.values())
    if (not preinsert_reached_at_s < started < completed < after_lower or
            before_upper >= started or
            before_lower < preinsert_reached_at_s - max_preinsert_age_s or
            started - before_upper > max_preinsert_age_s or
            after_lower - completed > max_final_observation_gap_s or
            decision_timestamp_s < after_upper):
        raise ValueError("insertion frames do not bracket this guarded attempt")
    frames, inputs = _phase_rows(before_root, before, cameras, "preinsert")
    later, later_inputs = _phase_rows(after_root, after, cameras,
                                      "final_or_abort")
    frames.extend(later)
    inputs.extend(later_inputs)
    for camera in cameras:
        before_image = cv2.imread(str(before_root / "images" /
                                      f"{_safe_id(camera, 'camera ID')}.png"))
        after_image = cv2.imread(str(after_root / "images" /
                                     f"{_safe_id(camera, 'camera ID')}.png"))
        if before_image.shape != after_image.shape:
            raise ValueError("insertion temporal pair changed image geometry")
    visual = observe_insertion_visual(backend, frames)
    supported = (visual.parse_error is None and
                 len(set(visual.parsed["evidence_views"]) & set(cameras)) >=
                 minimum_visual_views)
    visual_class = (visual.parsed["visual_class"] if supported else
                    "unobservable")
    sensor = metric["measurement"]
    evidence = InsertionEvidence(
        visual_class, tuple(sensor["key_depth_interval_m"])
        if sensor["key_depth_interval_m"] is not None else None,
        sensor["key_depth_source"], sensor["alignment_within_limits"],
        sensor["safety_abort"], sensor["grasp_held"])
    outcome = judge_insertion(evidence, target_depth_m=target_depth_m)
    return {
        "schema": "precision_insertion_observed_checkpoint_v1",
        "attempt_id": attempt_id, "candidate_id": candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "target_depth_m": target_depth_m,
        "preinsert_reached_at_s": preinsert_reached_at_s,
        "decision_timestamp_s": decision_timestamp_s,
        "guarded_started_at_s": started,
        "guarded_completed_at_s": completed,
        "preinsert_bundle": str(before_root),
        "preinsert_capture_kind": before_kind,
        "preinsert_manifest_sha256": _sha(before_root / before_manifest),
        "final_bundle": str(after_root),
        "final_manifest_sha256": _sha(after_root / "manifest.json"),
        "metric_record_path": str(metric_path),
        "metric_record_sha256": _sha(metric_path),
        "frames": inputs, "visual": visual.to_record(),
        "minimum_visual_views": minimum_visual_views,
        "max_phase_skew_s": max_phase_skew_s,
        "max_preinsert_age_s": max_preinsert_age_s,
        "max_final_observation_gap_s": max_final_observation_gap_s,
        "effective_vlm_class": visual_class,
        "evidence": asdict(evidence), "outcome": outcome.to_record(),
        "scope": "saved_images_and_external_metrics_not_contact_control_or_producer_certification",
        "robot_ready": False,
    }


def write_insertion_checkpoint(report: dict, output_dir: Path) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return target / "report.json"


def verify_insertion_checkpoint(report_path: Path) -> dict:
    """Recheck image bytes, timing provenance, metric bytes and fused label."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") !=
            "precision_insertion_observed_checkpoint_v1" or
            report.get("visual", {}).get("stage") != "insertion_visual"):
        raise ValueError("invalid insertion checkpoint")
    digest = report.get("session_calibration_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError("insertion checkpoint lacks session binding")
    before_root = Path(report["preinsert_bundle"]).expanduser().resolve()
    after_root = Path(report["final_bundle"]).expanduser().resolve()
    before, before_kind, before_manifest, before_cameras = _preinsert_capture(
        before_root)
    after = verify_final_insertion_capture(after_root)
    if (before_kind != report.get("preinsert_capture_kind") or
            _sha(before_root / before_manifest) !=
            report["preinsert_manifest_sha256"] or
            _sha(after_root / "manifest.json") !=
            report["final_manifest_sha256"]):
        raise ValueError("insertion source capture changed")
    metric_path = Path(report["metric_record_path"]).expanduser().resolve()
    if _sha(metric_path) != report["metric_record_sha256"]:
        raise ValueError("insertion metric record changed")
    metric, started, completed = _metric_record(
        metric_path, attempt_id=report["attempt_id"],
        candidate_id=report["candidate_id"],
        session_calibration_sha256=digest)
    if (started != report["guarded_started_at_s"] or
            completed != report["guarded_completed_at_s"]):
        raise ValueError("insertion execution time changed")
    for name in ("max_phase_skew_s", "max_preinsert_age_s",
                 "max_final_observation_gap_s", "target_depth_m"):
        if not math.isfinite(float(report[name])) or float(report[name]) <= 0:
            raise ValueError("invalid insertion checkpoint limits")
    if not math.isclose(float(report["target_depth_m"]), .020,
                        rel_tol=0, abs_tol=1e-9):
        raise ValueError("insertion checkpoint target is not 20 mm")
    expected_order = []
    grouped = {"preinsert": set(), "final_or_abort": set()}
    for row in report["frames"]:
        phase, camera = row.get("phase"), row.get("camera_id")
        if phase not in grouped or camera in grouped[phase]:
            raise ValueError("duplicate or unknown insertion image input")
        grouped[phase].add(camera)
        root, source = ((before_root, before) if phase == "preinsert" else
                        (after_root, after))
        path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        evidence = source["frame_evidence"][camera]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if (row.get("path") != str(path) or image is None or
                row.get("file_sha256") != _sha(path) or
                row.get("image_sha256") != image_sha256(image) or
                any(row.get(field) != evidence[field] for field in (
                    "frame_id", "image_sha256", "timestamp_s", "max_error_s"))):
            raise ValueError("insertion VLM input differs from saved frame")
        expected_order.append(f"{phase}/{camera}@{row['timestamp_s']:.6f}")
    minimum = report["minimum_visual_views"]
    if type(minimum) is not int or minimum < 2:
        raise ValueError("insertion checkpoint needs two visual views")
    if (grouped["preinsert"] != grouped["final_or_abort"] or
            len(grouped["preinsert"]) < minimum or
            report["visual"].get("image_order") != expected_order):
        raise ValueError("insertion VLM camera pairing or order changed")
    for camera in grouped["preinsert"]:
        first = cv2.imread(str(before_root / "images" /
                               f"{_safe_id(camera, 'camera ID')}.png"))
        last = cv2.imread(str(after_root / "images" /
                              f"{_safe_id(camera, 'camera ID')}.png"))
        if first is None or last is None or first.shape != last.shape:
            raise ValueError("insertion temporal image geometry changed")
    if (not grouped["preinsert"] <= before_cameras or
            not grouped["final_or_abort"] <= set(after["frame_evidence"])):
        raise ValueError("insertion VLM used unadmitted cameras")
    before_rows = {camera: before["frame_evidence"][camera]
                   for camera in grouped["preinsert"]}
    after_rows = {camera: after["frame_evidence"][camera]
                  for camera in grouped["final_or_abort"]}
    if (bounded_capture_skew_s(before_rows) > report["max_phase_skew_s"] or
            bounded_capture_skew_s(after_rows) > report["max_phase_skew_s"]):
        raise ValueError("insertion camera exposure skew changed")
    before_lower = min(row["timestamp_s"] - row["max_error_s"]
                       for row in before_rows.values())
    before_upper = max(row["timestamp_s"] + row["max_error_s"]
                       for row in before_rows.values())
    after_lower = min(row["timestamp_s"] - row["max_error_s"]
                      for row in after_rows.values())
    after_upper = max(row["timestamp_s"] + row["max_error_s"]
                      for row in after_rows.values())
    reached = report["preinsert_reached_at_s"]
    if (not reached < started < completed < after_lower or
            before_upper >= started or
            before_lower < reached - report["max_preinsert_age_s"] or
            started - before_upper > report["max_preinsert_age_s"] or
            after_lower - completed > report["max_final_observation_gap_s"] or
            report["decision_timestamp_s"] < after_upper):
        raise ValueError("insertion frame timing no longer brackets attempt")
    prefix = ("Images are supplied in this exact order:\n" +
              "\n".join(f"{index + 1}: {label}" for index, label in
                        enumerate(expected_order)) + "\n\n")
    if not str(report["visual"].get("prompt", "")).startswith(prefix):
        raise ValueError("insertion VLM prompt differs from saved image order")
    visual = report["visual"]
    parsed = visual.get("parsed", {})
    if visual.get("parse_error") is None:
        try:
            if json.loads(visual["raw_answer"]) != parsed:
                raise ValueError("insertion VLM parsed answer changed")
        except json.JSONDecodeError as exc:
            raise ValueError("insertion VLM raw answer is invalid") from exc
    if (not isinstance(parsed, dict) or
            parsed.get("visual_class") not in {
                "normal_appearance", "partial", "rim_jam", "slip",
                "unobservable"} or
            not isinstance(parsed.get("evidence_views"), list) or
            not all(isinstance(view, str)
                    for view in parsed["evidence_views"]) or
            len(parsed["evidence_views"]) !=
            len(set(parsed["evidence_views"])) or
            not set(parsed["evidence_views"]) <= grouped["preinsert"]):
        raise ValueError("invalid insertion VLM parsed response")
    supported = (visual.get("parse_error") is None and
                 isinstance(parsed.get("evidence_views"), list) and
                 len(set(parsed["evidence_views"]) &
                     grouped["preinsert"]) >= minimum)
    vlm_class = parsed.get("visual_class") if supported else "unobservable"
    if vlm_class != report["effective_vlm_class"]:
        raise ValueError("insertion effective visual class changed")
    sensor = metric["measurement"]
    evidence = InsertionEvidence(
        vlm_class, tuple(sensor["key_depth_interval_m"])
        if sensor["key_depth_interval_m"] is not None else None,
        sensor["key_depth_source"], sensor["alignment_within_limits"],
        sensor["safety_abort"], sensor["grasp_held"])
    outcome = judge_insertion(evidence, target_depth_m=report["target_depth_m"])
    serialized_evidence = json.loads(json.dumps(asdict(evidence)))
    if (serialized_evidence != report["evidence"] or
            outcome.to_record() != report["outcome"]):
        raise ValueError("insertion label conflicts with bound evidence")
    return report
