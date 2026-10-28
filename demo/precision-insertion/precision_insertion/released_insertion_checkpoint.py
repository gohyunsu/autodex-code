"""VLM label from post-release clear-view socket images.

The held pre-insertion snapshot remains a separate metric-XY source.  It is
bound here only to prove that it preceded the guarded stroke; it is not shown
to the final success classifier. Initial and final raw captures remain paired
in the checkpoint for provenance and manual review; the model receives only
the final socket crop for each camera.
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
from .insertion_checkpoint import (
    _admitted_measurement, _metric_record, verify_preinsert_raw_capture,
)
from .initial_visual import initial_visual_source
from .observer import (
    ImageVLM, LabeledFrame, RELEASED_SOCKET_IMAGE_SIZE, _parse_object,
    observe_released_insertion_visual, verify_per_view_consensus,
    verify_released_socket_token_consensus, resize_classification_images,
)
from .outcome import InsertionEvidence, resolve_insertion_outcome
from .post_insertion_release import verify_post_insertion_release
from .raw_camera_capture import verify_raw_camera_capture
from .session_bootstrap import _safe_id
from .visual_views import final_assessment_cameras


SCHEMA = "precision_insertion_released_checkpoint_v1"
FINAL_MIN_LONG_SIDE_PX = 384


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_frames(root: Path, record: dict, cameras: list[str], phase: str):
    frames, rows = [], []
    for camera in cameras:
        name = _safe_id(camera, "camera ID")
        path = root / "images" / f"{name}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        evidence = record["frame_evidence"][camera]
        if image is None or image_sha256(image) != evidence["image_sha256"]:
            raise ValueError("released-insertion VLM image differs from capture")
        frames.append(LabeledFrame(
            camera, phase, evidence["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
        rows.append({
            "phase": phase, "camera_id": camera, "path": str(path),
            "file_sha256": _sha(path), "frame_id": evidence["frame_id"],
            "image_sha256": evidence["image_sha256"],
            "timestamp_s": evidence["timestamp_s"],
            "max_error_s": evidence["max_error_s"],
        })
    return frames, rows


def _socket_focus_roi(
    image_size: tuple[int, int], bbox: list[int],
) -> tuple[int, int, int, int]:
    """Keep the socket and protruding key visible at classifier resolution."""
    width, height = image_size
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(type(value) is not int for value in bbox)):
        raise ValueError("socket projection needs four integer pixels")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("socket projection is outside the final camera image")
    socket_width, socket_height = x1 - x0, y1 - y0
    left = max(0, int(x0 - .5 * socket_width))
    top = max(0, int(y0 - 1.2 * socket_height))
    right = min(width, int(x1 + .5 * socket_width))
    bottom = min(height, int(y1 + .3 * socket_height))
    return left, top, right, bottom


def _focus_final_frames(
    frames: list[LabeledFrame], initial: dict, *,
    allow_partial_projection: bool = False,
) -> tuple[list[LabeledFrame], dict[str, dict]]:
    projections = ((initial.get("consistency", {})
                    .get("socket_exclusion") or {}).get("per_camera", {}))
    focused, records = [], {}
    for frame in frames:
        projection = projections.get(frame.camera_id)
        if projection is None:
            focused.append(frame)
            continue
        bbox = projection.get("projected_hull_bbox_px")
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(value) is not int for value in bbox)):
            focused.append(frame)
            continue
        width, height = frame.image.size
        if (not allow_partial_projection and
                not (0 <= bbox[0] < bbox[2] <= width and
                     0 <= bbox[1] < bbox[3] <= height)):
            focused.append(frame)
            continue
        # A socket close to the image boundary can have a projected mesh
        # hull partly outside the frame. Use its visible portion instead of
        # shrinking the socket to a full-frame classifier input.
        visible_bbox = [max(0, bbox[0]), max(0, bbox[1]),
                        min(width, bbox[2]), min(height, bbox[3])]
        if (visible_bbox[0] >= visible_bbox[2] or
                visible_bbox[1] >= visible_bbox[3]):
            focused.append(frame)
            continue
        roi = _socket_focus_roi(frame.image.size, visible_bbox)
        crop = frame.image.crop(roi)
        focused.append(LabeledFrame(
            frame.camera_id, frame.phase, frame.timestamp_s, crop))
        records[frame.camera_id] = {
            "roi_xyxy_full_px": list(roi),
            "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        }
    return focused, records


def _focus_initial_key_frames(
    frames: list[LabeledFrame], initial_key_bundle: Path,
) -> tuple[list[LabeledFrame], dict[str, dict]]:
    """Replay the archived two-image classifier's initial key crop.

    New released-key labels use only the final socket crop. Old reports still
    bind this mask-derived crop and need the same function for verification.
    """
    focused, records = [], {}
    for frame in frames:
        mask_path = (initial_key_bundle / "masks" /
                     f"{_safe_id(frame.camera_id, 'camera ID')}.png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (frame.image.height, frame.image.width):
            focused.append(frame)
            continue
        points = cv2.findNonZero(mask)
        if points is None:
            focused.append(frame)
            continue
        x, y, width, height = cv2.boundingRect(points)
        # A square crop keeps the narrow cylinder visible after the
        # classification resize and stays above Qwen's minimum pixel area.
        side = max(3 * width, 3 * height, 256)
        cx, cy = x + width // 2, y + height // 2
        left = max(0, min(cx - side // 2, frame.image.width - side))
        top = max(0, min(cy - side // 2, frame.image.height - side))
        roi = (left, top, min(frame.image.width, left + side),
               min(frame.image.height, top + side))
        crop = frame.image.crop(roi)
        focused.append(LabeledFrame(
            frame.camera_id, frame.phase, frame.timestamp_s, crop))
        records[frame.camera_id] = {
            "roi_xyxy_full_px": list(roi),
            "mask_file_sha256": _sha(mask_path),
            "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        }
    return focused, records


def assess_released_insertion_checkpoint(
    *, attempt_id: str, candidate_id: str, target_depth_m: float,
    session_calibration_sha256: str, initial_key_bundle: Path,
    preinsert_bundle: Path, final_bundle: Path, metric_record_path: Path,
    release_record_path: Path, preinsert_reached_at_s: float,
    decision_timestamp_s: float, backend: ImageVLM,
    max_phase_skew_s: float, minimum_visual_views: int = 2,
    label_mode: str = "vlm_only", manual_annotation: dict | None = None,
    final_camera_ids: list[str] | None = None,
) -> dict:
    """Create the released-key task label while retaining controller metrics."""
    _safe_id(attempt_id, "attempt ID")
    if not math.isclose(target_depth_m, .020, rel_tol=0, abs_tol=1e-9):
        raise ValueError("released insertion target must be 20 mm")
    limits = (max_phase_skew_s,)
    if (not all(math.isfinite(float(value)) and float(value) > 0
                for value in limits) or
            type(minimum_visual_views) is not int or minimum_visual_views < 2):
        raise ValueError("released checkpoint needs positive commissioned limits")
    initial_root = Path(initial_key_bundle).expanduser().resolve()
    preinsert_root = Path(preinsert_bundle).expanduser().resolve()
    final_root = Path(final_bundle).expanduser().resolve()
    metric_path = Path(metric_record_path).expanduser().resolve()
    release_path = Path(release_record_path).expanduser().resolve()
    initial_image_root, initial = initial_visual_source(initial_root)
    if initial.get("phase") != "tabletop":
        raise ValueError("success comparison needs the initial tabletop key capture")
    preinsert = verify_preinsert_raw_capture(preinsert_root)
    final = verify_raw_camera_capture(final_root, phase="final_clear_view")
    metric, started, completed = _metric_record(
        metric_path, attempt_id=attempt_id, candidate_id=candidate_id,
        session_calibration_sha256=session_calibration_sha256)
    release = verify_post_insertion_release(
        release_path, attempt_id=attempt_id, candidate_id=candidate_id,
        session_calibration_sha256=session_calibration_sha256,
        guarded_execution_path=metric_path)
    # FoundPose pose-admission views do not limit the independent raw visual
    # comparison.  Both captures retain their own frame provenance below.
    cameras = final_assessment_cameras(
        set(initial["frame_evidence"]) & set(final["frame_evidence"]),
        selected=final_camera_ids)
    if len(cameras) < minimum_visual_views:
        raise ValueError("released insertion lacks paired initial/final cameras")
    initial_rows = {camera: initial["frame_evidence"][camera]
                    for camera in cameras}
    preinsert_rows = preinsert["frame_evidence"]
    final_rows = {camera: final["frame_evidence"][camera]
                  for camera in cameras}
    if (bounded_capture_skew_s(initial_rows) > max_phase_skew_s or
            bounded_capture_skew_s(preinsert_rows) > max_phase_skew_s or
            bounded_capture_skew_s(final_rows) > max_phase_skew_s):
        raise ValueError("released checkpoint camera skew exceeds limit")
    initial_end = max(row["timestamp_s"] + row["max_error_s"]
                      for row in initial_rows.values())
    preinsert_end = max(row["timestamp_s"] + row["max_error_s"]
                        for row in preinsert_rows.values())
    final_begin = min(row["timestamp_s"] - row["max_error_s"]
                      for row in final_rows.values())
    final_end = max(row["timestamp_s"] + row["max_error_s"]
                    for row in final_rows.values())
    if (not initial_end < preinsert_reached_at_s < started < completed or
            preinsert_end >= started or
            release["clear_view_completed_at_s"] >= final_begin or
            decision_timestamp_s < final_end):
        raise ValueError(
            "initial, preinsert, stroke, release and final images are not ordered")
    frames, inputs = _image_frames(
        initial_image_root, initial, cameras, "initial_tabletop")
    later, later_inputs = _image_frames(
        final_root, final, cameras, "final_clear_view")
    initial_sizes = {frame.camera_id: frame.image.size for frame in frames}
    if any(initial_sizes[frame.camera_id] != frame.image.size
           for frame in later):
        raise ValueError("initial/final temporal pair changed image geometry")
    later, final_visual_crops = _focus_final_frames(
        later, initial, allow_partial_projection=True)
    frames.extend(later)
    inputs.extend(later_inputs)
    visual = observe_released_insertion_visual(
        backend, frames, min_long_side_px=FINAL_MIN_LONG_SIDE_PX)
    supported = (visual.parse_error is None and
                 len(set(visual.parsed["evidence_views"]) & set(cameras)) >=
                 minimum_visual_views)
    visual_class = visual.parsed["visual_class"] if supported else "unobservable"
    sensor, depth_admissibility = _admitted_measurement(metric)
    evidence = InsertionEvidence(
        visual_class,
        tuple(sensor["key_depth_interval_m"])
        if sensor["key_depth_interval_m"] is not None else None,
        sensor["key_depth_source"], sensor["alignment_within_limits"],
        sensor["safety_abort"], sensor["grasp_held"])
    outcome = resolve_insertion_outcome(
        evidence, target_depth_m=target_depth_m, label_mode=label_mode,
        manual_annotation=manual_annotation)
    return {
        "schema": SCHEMA, "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "target_depth_m": target_depth_m,
        "preinsert_reached_at_s": preinsert_reached_at_s,
        "decision_timestamp_s": decision_timestamp_s,
        "guarded_started_at_s": started, "guarded_completed_at_s": completed,
        "initial_key_bundle": str(initial_root),
        "initial_manifest_sha256": _sha(initial_root / "evidence_manifest.json"),
        "preinsert_bundle": str(preinsert_root),
        "preinsert_manifest_sha256": _sha(preinsert_root / "manifest.json"),
        "final_bundle": str(final_root),
        "final_manifest_sha256": _sha(final_root / "manifest.json"),
        "metric_record_path": str(metric_path),
        "metric_record_sha256": _sha(metric_path),
        "release_record_path": str(release_path),
        "release_record_sha256": _sha(release_path),
        "frames": inputs, "visual": visual.to_record(),
        "visual_input_policy": "final_socket_crop_token_v4",
        "final_visual_crops": final_visual_crops,
        "minimum_visual_views": minimum_visual_views,
        "max_phase_skew_s": max_phase_skew_s,
        "effective_vlm_class": visual_class,
        "vlm_label": outcome["vlm_label"], "label_mode": label_mode,
        "manual_annotation": manual_annotation,
        "key_depth_admissibility": depth_admissibility,
        "evidence": asdict(evidence), "outcome": outcome,
        "scope": "released_clear_view_vlm_task_label_with_saved_preinsert_xy_source",
        "robot_ready": False,
    }


def write_released_insertion_checkpoint(report: dict, output_dir: Path) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    path = target / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    return path


def verify_released_insertion_checkpoint(path: Path) -> dict:
    """Replay immutable inputs and the deterministic per-view consensus."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") != SCHEMA or
            report.get("visual", {}).get("stage") !=
            "released_insertion_visual"):
        raise ValueError("invalid released insertion checkpoint")
    initial_root = Path(report["initial_key_bundle"])
    preinsert_root = Path(report["preinsert_bundle"])
    final_root = Path(report["final_bundle"])
    initial_image_root, initial = initial_visual_source(initial_root)
    preinsert = verify_preinsert_raw_capture(preinsert_root)
    final = verify_raw_camera_capture(final_root, phase="final_clear_view")
    if (_sha(initial_root / "evidence_manifest.json") !=
            report["initial_manifest_sha256"] or
            _sha(preinsert_root / "manifest.json") !=
            report["preinsert_manifest_sha256"] or
            _sha(final_root / "manifest.json") !=
            report["final_manifest_sha256"] or
            _sha(Path(report["metric_record_path"])) !=
            report["metric_record_sha256"] or
            _sha(Path(report["release_record_path"])) !=
            report["release_record_sha256"]):
        raise ValueError("released insertion source bytes changed")
    verify_post_insertion_release(
        Path(report["release_record_path"]),
        attempt_id=report["attempt_id"], candidate_id=report["candidate_id"],
        session_calibration_sha256=report["session_calibration_sha256"],
        guarded_execution_path=Path(report["metric_record_path"]))
    grouped = {"initial_tabletop": set(), "final_clear_view": set()}
    expected_order = []
    for row in report["frames"]:
        phase, camera = row["phase"], row["camera_id"]
        if phase not in grouped or camera in grouped[phase]:
            raise ValueError("duplicate released-insertion VLM frame")
        grouped[phase].add(camera)
        root, source = ((initial_image_root, initial) if phase == "initial_tabletop"
                        else (final_root, final))
        image_path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        evidence = source["frame_evidence"][camera]
        if (image is None or row["path"] != str(image_path) or
                row["file_sha256"] != _sha(image_path) or
                row["image_sha256"] != image_sha256(image) or
                any(row[field] != evidence[field] for field in (
                    "frame_id", "image_sha256", "timestamp_s", "max_error_s"))):
            raise ValueError("released-insertion VLM frame changed")
        expected_order.append(f"{phase}/{camera}@{row['timestamp_s']:.6f}")
    if (grouped["initial_tabletop"] != grouped["final_clear_view"] or
            len(grouped["initial_tabletop"]) < report["minimum_visual_views"] or
            report["visual"].get("image_order") != expected_order):
        raise ValueError("released-insertion camera pairs changed")
    policy = report.get("visual_input_policy")
    new_policy = policy in ("final_socket_crop_token_v1",
                            "final_socket_crop_token_v2",
                            "final_socket_crop_token_v3",
                            "final_socket_crop_token_v4")
    if policy not in (None, "final_socket_crop_token_v1",
                       "final_socket_crop_token_v2",
                       "final_socket_crop_token_v3",
                       "final_socket_crop_token_v4"):
        raise ValueError("unknown released-insertion visual input policy")
    if new_policy and "final_visual_crops" not in report:
        raise ValueError("released-insertion final socket crops are missing")
    if "initial_visual_crops" in report and not new_policy:
        replay_frames, _ = _image_frames(
            initial_image_root, initial, sorted(grouped["initial_tabletop"]),
            "initial_tabletop")
        _, expected_crops = _focus_initial_key_frames(replay_frames, initial_root)
        if report["initial_visual_crops"] != expected_crops:
            raise ValueError("released-insertion initial key crops changed")
        aggregation = report["visual"].get("aggregation") or {}
        for camera, row in zip(
                aggregation.get("completed_camera_ids", []),
                aggregation.get("per_view", [])):
            if camera in expected_crops:
                images = (row.get("input_transform") or {}).get("images") or []
                if (len(images) != 2 or
                        images[0].get("source_rgb_sha256") !=
                        expected_crops[camera]["crop_rgb_sha256"]):
                    raise ValueError("released-insertion initial VLM crop changed")
    if "final_visual_crops" in report:
        replay_frames, _ = _image_frames(
            final_root, final, sorted(grouped["final_clear_view"]),
            "final_clear_view")
        focused_frames, expected_crops = _focus_final_frames(
            replay_frames, initial,
            allow_partial_projection=policy in (
                "final_socket_crop_token_v3", "final_socket_crop_token_v4"))
        if report["final_visual_crops"] != expected_crops:
            raise ValueError("released-insertion socket crops changed")
        aggregation = report["visual"].get("aggregation") or {}
        focused_by_camera = {frame.camera_id: frame for frame in focused_frames}
        for camera, row in zip(
                aggregation.get("completed_camera_ids", []),
                aggregation.get("per_view", [])):
            if new_policy:
                _, expected_transform = resize_classification_images(
                    [focused_by_camera[camera].image],
                    RELEASED_SOCKET_IMAGE_SIZE,
                    min_long_side_px=(FINAL_MIN_LONG_SIDE_PX if policy in (
                        "final_socket_crop_token_v2",
                        "final_socket_crop_token_v3",
                        "final_socket_crop_token_v4") else None))
                if row.get("input_transform") != expected_transform:
                    raise ValueError("released-insertion final VLM input changed")
            if camera in expected_crops:
                transform = row.get("input_transform") or {}
                images = transform.get("images") or []
                index = 0 if new_policy else 1
                if (len(images) != (1 if new_policy else 2) or
                        images[index].get("source_rgb_sha256") !=
                        expected_crops[camera]["crop_rgb_sha256"]):
                    raise ValueError("released-insertion VLM crop input changed")
    if new_policy:
        verify_released_socket_token_consensus(
            report["visual"], grouped["initial_tabletop"],
            prompt_version=("v2" if policy == "final_socket_crop_token_v4"
                            else "v1"))
    else:
        verify_per_view_consensus(
            report["visual"], field="visual_class", fallback="unobservable",
            cameras=grouped["initial_tabletop"], before="initial_tabletop",
            after="final_clear_view")
    parsed = report["visual"].get("parsed", {})
    if report["visual"].get("parse_error") is None and report["visual"].get(
            "aggregation") is None:
        if _parse_object(report["visual"]["raw_answer"]) != parsed:
            raise ValueError("released-insertion raw VLM answer changed")
    if parsed.get("visual_class") not in {
            "inside_socket", "outside_socket", "unobservable"}:
        raise ValueError("invalid released-insertion visual class")
    supported = (report["visual"].get("parse_error") is None and
                 len(set(parsed.get("evidence_views", [])) &
                     grouped["initial_tabletop"]) >=
                 report["minimum_visual_views"])
    effective = parsed["visual_class"] if supported else "unobservable"
    if effective != report["effective_vlm_class"]:
        raise ValueError("released-insertion effective label changed")
    metric, _, _ = _metric_record(
        Path(report["metric_record_path"]), attempt_id=report["attempt_id"],
        candidate_id=report["candidate_id"],
        session_calibration_sha256=report["session_calibration_sha256"])
    sensor, depth_admissibility = _admitted_measurement(metric)
    evidence = InsertionEvidence(
        effective,
        tuple(sensor["key_depth_interval_m"])
        if sensor["key_depth_interval_m"] is not None else None,
        sensor["key_depth_source"], sensor["alignment_within_limits"],
        sensor["safety_abort"], sensor["grasp_held"])
    outcome = resolve_insertion_outcome(
        evidence, target_depth_m=report["target_depth_m"],
        label_mode=report["label_mode"],
        manual_annotation=report["manual_annotation"])
    serialized_evidence = json.loads(json.dumps(asdict(evidence)))
    if (serialized_evidence != report["evidence"] or
            depth_admissibility != report["key_depth_admissibility"] or
            outcome != report["outcome"]):
        raise ValueError("released-insertion result changed")
    return report
