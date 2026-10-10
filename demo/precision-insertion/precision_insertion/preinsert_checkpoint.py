"""Read-only observed arrival gate at the frozen socket's preinsert hold.

An external controller supplies a completed transfer log. The saved camera
bundle, measured stationary FR3/Inspire joints, post-lift grasp relation, and
per-view raw/CAD VLM comparison are checked together. This does not execute
motion, prove contact, or certify the external log's physical producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .bounded_postlift import (
    BoundedPostLiftPreflight, verify_bounded_postlift_preflight,
)
from .calibration import SessionCalibration, validate_session_camera_calibration
from .config import TaskMode, select_mode
from .frame_provenance import bounded_capture_skew_s, image_sha256
from .geometry import pose_angle_deg, validate_se3
from .held_scene_overlay import (
    HeldSceneComparison, build_held_scene_comparison,
    build_held_scene_prediction,
)
from .live_robot_state import LiveRobotState
from .observer import (
    HeldSceneView, ImageVLM, PreinsertVisualAssessment,
    observe_preinsert_hold_views,
)
from .postlift_preflight import PostLiftPreflight
from .raw_camera_capture import verify_raw_camera_capture
from .records import AttemptRecord
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _transfer_log(path: Path, attempt: AttemptRecord) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or
            data.get("schema") != "precision_insertion_transfer_execution_v1" or
            data.get("attempt_id") != attempt.attempt_id or
            data.get("candidate_id") != attempt.candidate_id):
        raise ValueError("transfer log does not belong to this attempt")
    if (not isinstance(data.get("measurement"), dict) or
            set(data["measurement"]) != {
            "trajectory_complete", "safety_abort", "grasp_held"} or
            any(value is not None and type(value) is not bool
                for value in data["measurement"].values())):
        raise ValueError("transfer measurement needs tri-state completion, abort and grip")
    try:
        started = float(data["started_at_s"])
        completed = float(data["completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transfer log needs finite execution times") from exc
    if not all(math.isfinite(v) and v > 0 for v in (started, completed)) or started >= completed:
        raise ValueError("transfer execution times are invalid")
    sources = data.get("source_records")
    if not isinstance(sources, dict) or set(sources) != {
            "trajectory_feedback", "safety", "grasp_state"}:
        raise ValueError("transfer log lacks named producer records")
    for name, source in sources.items():
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError(f"invalid transfer producer reference: {name}")
        if not isinstance(source["path"], str) or not isinstance(source["sha256"], str):
            raise ValueError(f"invalid transfer producer reference: {name}")
        source_path = Path(source["path"])
        if (not source_path.is_absolute() or not source_path.is_file() or
                _sha(source_path) != source["sha256"]):
            raise ValueError(f"transfer producer record changed: {name}")
    return data


def _arrival_label(
    *, measurement: dict, position_error: float, rotation_error: float,
    max_position_error: float, max_rotation_error: float, visual_status: str,
) -> tuple[bool | None, str]:
    if (measurement["trajectory_complete"] is False or
            measurement["safety_abort"] is True or
            measurement["grasp_held"] is False):
        return False, "transfer_or_grasp_failed"
    if (position_error > max_position_error or
            rotation_error > max_rotation_error):
        return False, "measured_hold_pose_outside_limits"
    if visual_status in {"gross_misalignment", "slip_or_miss"}:
        return False, "multiview_visible_preinsert_failure"
    if (visual_status == "coarse_match" and measurement == {
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True}):
        return True, "measured_arrival_and_visible_held_key"
    return None, "preinsert_evidence_incomplete_or_occluded"


def _verify_visual_record(visual: dict, views: dict) -> None:
    rows = visual.get("per_view")
    if (not isinstance(rows, list) or len(rows) != len(views) or
            len(rows) < 2):
        raise ValueError("preinsert visual report lacks its camera responses")
    decisive: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or
                row.get("stage") != "preinsert_visual" or
                not isinstance(row.get("image_order"), list) or
                len(row["image_order"]) != 2 or
                not isinstance(row.get("parsed"), dict)):
            raise ValueError("preinsert per-view response is invalid")
        label = row["image_order"][0]
        if not isinstance(label, str) or not label.startswith("raw_preinsert/"):
            raise ValueError("preinsert per-view raw image order changed")
        camera = label.removeprefix("raw_preinsert/").split("@", 1)[0]
        if (camera not in views or camera in seen or
                row["image_order"] != [
                    f"raw_preinsert/{camera}@{views[camera]['timestamp_s']:.6f}",
                    f"predicted_scene/{camera}@{views[camera]['timestamp_s']:.6f}"]):
            raise ValueError("preinsert VLM camera/frame order changed")
        seen.add(camera)
        parsed = row["parsed"]
        category = parsed.get("class")
        if category not in {
                "coarse_match", "gross_misalignment", "slip_or_miss",
                "unobservable"}:
            raise ValueError("preinsert per-view class is invalid")
        if (row.get("parse_error") is None and
                category != "unobservable" and
                isinstance(parsed.get("evidence"), str) and
                parsed["evidence"].strip() and
                parsed.get("evidence_views") == [camera]):
            decisive.setdefault(category, []).append(camera)
    if len(decisive) == 1 and len(next(iter(decisive.values()))) >= 2:
        status = next(iter(decisive))
        supporting = sorted(decisive[status])
    else:
        status, supporting = "unknown", []
    if (visual.get("status") != status or
            visual.get("supporting_cameras") != supporting):
        raise ValueError("preinsert visual consensus differs from per-view responses")


class _RecordedPreinsertAnswers:
    """Replay saved text through the inference parser without a model call."""

    def __init__(self, rows: list[dict]):
        self.model = rows[0].get("backend_model")
        self._answers = [row.get("raw_answer") for row in rows]

    def infer(self, _images, _prompt):
        return self._answers.pop(0)


def _verify_visual_replay(
    visual: dict, paired_pixels: list[HeldSceneView],
) -> None:
    """Bind parsed labels, prompts and consensus to exact saved VLM text."""
    recorded = visual.get("per_view")
    if (not isinstance(recorded, list) or
            len(recorded) != len(paired_pixels) or
            any(not isinstance(row, dict) or
                not isinstance(row.get("raw_answer"), str) or
                not isinstance(row.get("backend_model"), str) or
                not row["backend_model"].strip() or
                type(row.get("latency_s")) not in (float, int) or
                not math.isfinite(row["latency_s"]) or
                row["latency_s"] < 0 for row in recorded)):
        raise ValueError("preinsert VLM saved answer is invalid")
    stamps = [view.timestamp_s for view in paired_pixels]
    # This replay bound merely admits the *same* already verified captures;
    # it is not a new assertion about commissioned camera synchronization.
    replay_skew_s = max(max(stamps) - min(stamps) + 1e-6, 1e-6)
    try:
        replay = observe_preinsert_hold_views(
            _RecordedPreinsertAnswers(recorded), paired_pixels,
            max_capture_skew_s=replay_skew_s).to_record()
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("preinsert VLM saved answer cannot be replayed") from exc
    for old, new in zip(recorded, replay["per_view"]):
        new["latency_s"] = old["latency_s"]
    if visual != replay:
        raise ValueError("preinsert VLM text, prompt or parsed answer differs from replay")


@dataclass(frozen=True)
class PreinsertCheckpoint:
    attempt_id: str
    candidate_id: str
    preinsert_reached: bool | None
    reason: str
    hand_translation_error_m: float
    hand_rotation_error_deg: float
    visual: PreinsertVisualAssessment
    comparison: HeldSceneComparison
    source_record: dict

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_preinsert_checkpoint_v1",
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "preinsert_reached": self.preinsert_reached,
            "reason": self.reason,
            "hand_translation_error_m": self.hand_translation_error_m,
            "hand_rotation_error_deg": self.hand_rotation_error_deg,
            "visual": self.visual.to_record(),
            "comparison": self.comparison.to_record(),
            **self.source_record,
            "scope": "read_only_saved_evidence_not_controller_or_sensor_certification",
            "robot_ready": False,
        }


def assess_preinsert_checkpoint(
    *, attempt: AttemptRecord,
    postlift: PostLiftPreflight | BoundedPostLiftPreflight,
    postlift_report_path: Path, calibration: SessionCalibration,
    shared_root: Path, mode: TaskMode, raw_bundle: Path,
    transfer_execution_path: Path, joint_sample: LiveRobotState,
    backend: ImageVLM, max_capture_skew_s: float,
    max_joint_frame_skew_s: float, max_transfer_observation_gap_s: float,
    max_hand_translation_error_m: float, max_hand_rotation_error_deg: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    renderer_factory: Callable | None = None,
    robot_loader: Callable | None = None,
) -> PreinsertCheckpoint:
    """Fuse per-view coarse VLM with measured hold pose and execution evidence.

    ``coarse_match`` is necessary for a positive label but never sufficient
    alone. A hidden key yields unknown, not a predicted-pose success. The
    source log and camera timing are verified as files and claims; commissioning
    must separately establish sensor authenticity and physical error bounds.
    """
    limits = (max_capture_skew_s, max_joint_frame_skew_s,
              max_transfer_observation_gap_s, max_hand_translation_error_m,
              max_hand_rotation_error_deg, max_arm_hand_skew_s,
              max_hand_command_error_raw, max_arm_velocity_rad_s)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in limits):
        raise ValueError("preinsert arrival needs positive commissioned limits")
    if (not isinstance(attempt, AttemptRecord) or attempt.mode != mode or
            attempt.labels["grasp_success"] is not True or
            attempt.labels["preinsert_reached"] is not None or
            attempt.candidate_id is None or
            not isinstance(postlift, (PostLiftPreflight,
                                      BoundedPostLiftPreflight)) or
            postlift.status != "sampled_postlift_preflight_pass" or
            postlift.attempt_id != attempt.attempt_id or
            "/".join(postlift.candidate_key) != attempt.candidate_id or
            postlift.targets is None or postlift.planning is None or
            not postlift.planning.sampled_planning_pass):
        raise ValueError("preinsert arrival needs the matching post-lift transfer plan")
    if not isinstance(calibration, SessionCalibration):
        raise TypeError("preinsert arrival needs a frozen socket session")
    session_hash = _canonical_digest(calibration.record)
    if (postlift.session_calibration_sha256 != session_hash or
            attempt.session_calibration_sha256 != session_hash):
        raise ValueError("preinsert arrival changed the frozen session")
    postlift_path = Path(postlift_report_path).expanduser().resolve()
    saved_plan = json.loads(postlift_path.read_text(encoding="utf-8"))
    bounded = isinstance(postlift, BoundedPostLiftPreflight)
    relation_field = ("bounded_held_relation" if bounded else
                      "observed_held_relation")
    if bounded:
        candidate_dir = (AssetPaths(Path(shared_root), mode).candidate_dir /
                         Path(*postlift.candidate_key))
        verify_bounded_postlift_preflight(
            postlift_path, mode=mode, shared_root=shared_root,
            candidate_dir=candidate_dir)
    if (not isinstance(saved_plan, dict) or
            not isinstance(saved_plan.get(relation_field), dict) or
            not isinstance(saved_plan.get("targets"), dict) or
            saved_plan.get("status") != postlift.status or
            saved_plan.get("attempt_id") != attempt.attempt_id or
            saved_plan.get("candidate_key") != list(postlift.candidate_key) or
            saved_plan.get("session_calibration_sha256") != session_hash or
            saved_plan.get(relation_field, {}).get("T_key_hand") !=
            postlift.relation.T_key_hand.tolist() or
            saved_plan.get("targets", {}).get("T_robot_hand_preinsert") !=
            postlift.targets.T_robot_hand_preinsert.tolist()):
        raise ValueError("saved post-lift plan does not match this transfer")
    if not np.allclose(postlift.targets.T_key_hand,
                       postlift.relation.T_key_hand, rtol=0, atol=1e-8):
        raise ValueError("preinsert target changed the key-hand relation")
    transfer_path = Path(transfer_execution_path).expanduser().resolve()
    transfer = _transfer_log(transfer_path, attempt)
    started = float(transfer["started_at_s"])
    completed = float(transfer["completed_at_s"])
    if started <= max(postlift.joint_timestamp_s,
                      postlift.key_capture_timestamp_s):
        raise ValueError("transfer predates the observed post-lift replan")
    bundle = Path(raw_bundle).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="preinsert")
    frame_rows = capture["frame_evidence"]
    if (len(frame_rows) < 2 or
            bounded_capture_skew_s(frame_rows) > max_capture_skew_s):
        raise ValueError("preinsert cameras are not synchronized")
    first_exposure = min(row["timestamp_s"] - row["max_error_s"]
                         for row in frame_rows.values())
    last_exposure = max(row["timestamp_s"] + row["max_error_s"]
                        for row in frame_rows.values())
    if (completed >= first_exposure or
            first_exposure - completed > max_transfer_observation_gap_s):
        raise ValueError("preinsert images do not follow this transfer")
    if not isinstance(joint_sample, LiveRobotState):
        raise TypeError("preinsert arrival needs measured joint feedback")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    if (joint_sample.sample_timestamp_s <= completed or
            abs(joint_sample.sample_timestamp_s -
                (first_exposure + last_exposure) / 2) > max_joint_frame_skew_s):
        raise ValueError("preinsert joint feedback and exposure are not aligned")
    snapshot = calibration.record.get("camera_calibration", {})
    camera_ids = set(frame_rows)
    if (not isinstance(snapshot, dict) or
            not isinstance(snapshot.get("intrinsics_full"), dict) or
            not isinstance(snapshot.get("extrinsics_full"), dict) or
            not camera_ids <= set(snapshot.get("intrinsics_full", {})) or
            not camera_ids <= set(snapshot.get("extrinsics_full", {}))):
        raise ValueError("preinsert cameras differ from the frozen session")
    all_camera_ids = set(snapshot["intrinsics_full"])
    K_all = {serial: np.asarray(snapshot["intrinsics_full"][serial]["K_undist"],
                                dtype=float) for serial in all_camera_ids}
    camera_world_all = {
        serial: validate_se3(snapshot["extrinsics_full"][serial],
                             name=f"{serial} camera-world extrinsic")
        for serial in all_camera_ids}
    validate_session_camera_calibration(
        calibration, intrinsics_undist=K_all,
        extrinsics_full=camera_world_all, calibrated_camera_ids=all_camera_ids)
    K = {serial: K_all[serial] for serial in camera_ids}
    camera_world = {serial: camera_world_all[serial] for serial in camera_ids}
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    T_camera_robot = {serial: camera_world[serial] @ c2r
                      for serial in camera_ids}
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=shared_root, calibration=calibration)
    prediction_kwargs = {}
    if robot_loader is not None:
        prediction_kwargs["robot_loader"] = robot_loader
    prediction = build_held_scene_prediction(
        shared_root=shared_root, mode=mode,
        full_q_measured=joint_sample.full_q,
        T_robot_socket_frozen=socket,
        T_key_hand_hypothesis=postlift.relation.T_key_hand,
        relation_source=("verified_physical_grasp_calibration" if bounded else
                         "observed_multiview_key_plus_wrist"),
        **prediction_kwargs)
    target = validate_se3(postlift.targets.T_robot_hand_preinsert,
                          name="planned preinsert T_robot_hand")
    position_error = float(np.linalg.norm(
        prediction.T_robot_hand[:3, 3] - target[:3, 3]))
    rotation_error = pose_angle_deg(prediction.T_robot_hand, target)
    frames = {serial: cv2.imread(str(bundle / "images" / f"{serial}.png"),
                                   cv2.IMREAD_COLOR)
              for serial in camera_ids}
    if any(frame is None for frame in frames.values()):
        raise ValueError("saved preinsert image cannot be decoded")
    comparison = build_held_scene_comparison(
        prediction=prediction, frames_bgr=frames,
        frame_timestamps_s={serial: frame_rows[serial]["timestamp_s"]
                            for serial in camera_ids},
        intrinsics_undistorted=K, T_camera_robot=T_camera_robot,
        renderer_factory=renderer_factory)
    visual = observe_preinsert_hold_views(
        backend, comparison.views, max_capture_skew_s=max_capture_skew_s)
    measurement = transfer["measurement"]
    label, reason = _arrival_label(
        measurement=measurement, position_error=position_error,
        rotation_error=rotation_error,
        max_position_error=max_hand_translation_error_m,
        max_rotation_error=max_hand_rotation_error_deg,
        visual_status=visual.status)
    return PreinsertCheckpoint(
        attempt.attempt_id, attempt.candidate_id, label, reason,
        position_error, rotation_error, visual, comparison, {
            "session_calibration_sha256": session_hash,
            "postlift_report_path": str(postlift_path),
            "postlift_report_sha256": _sha(postlift_path),
            "transfer_execution_path": str(transfer_path),
            "transfer_execution_sha256": _sha(transfer_path),
            "raw_bundle": str(bundle),
            "raw_manifest_sha256": _sha(bundle / "manifest.json"),
            "transfer_measurement": measurement,
            "transfer_started_at_s": started,
            "transfer_completed_at_s": completed,
            "observation_completed_at_s": last_exposure,
            "joint_feedback": joint_sample.to_record(),
            "max_hand_translation_error_m": max_hand_translation_error_m,
            "max_hand_rotation_error_deg": max_hand_rotation_error_deg,
            **({"bounded_source_context": {
                "shared_root": str(Path(shared_root).expanduser().resolve()),
                "family": mode.family, "gap_mm": mode.gap_mm,
                "candidate_dir": str(candidate_dir.resolve()),
            }} if bounded else {}),
        })


def write_preinsert_checkpoint(
    checkpoint: PreinsertCheckpoint, output_dir: Path,
) -> Path:
    """Save the exact VLM overlay pixels alongside an immutable report."""
    if not isinstance(checkpoint, PreinsertCheckpoint):
        raise TypeError("expected a computed preinsert checkpoint")
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    image_dir = target / "overlays"
    image_dir.mkdir()
    record = checkpoint.to_record()
    png_hashes = {}
    for view in checkpoint.comparison.views:
        path = image_dir / f"{view.camera_id}.png"
        overlay_bgr = cv2.cvtColor(
            np.asarray(view.predicted_overlay), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), overlay_bgr):
            raise OSError(f"could not save preinsert overlay: {path}")
        if (image_sha256(overlay_bgr) != checkpoint.comparison.pixel_digests[
                view.camera_id]["overlay_image_sha256"]):
            raise ValueError("saved preinsert overlay differs from VLM input")
        png_hashes[view.camera_id] = _sha(path)
    record["overlay_png_sha256"] = png_hashes
    report = target / "report.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def verify_preinsert_checkpoint(report_path: Path) -> dict:
    """Recheck saved bytes; do not re-run VLM or certify external producers."""
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") !=
            "precision_insertion_preinsert_checkpoint_v1" or
            report.get("robot_ready") is not False or
            report.get("visual", {}).get("status") not in {
                "coarse_match", "gross_misalignment", "slip_or_miss", "unknown"}):
        raise ValueError("invalid preinsert checkpoint report")
    postlift = Path(report["postlift_report_path"])
    transfer = Path(report["transfer_execution_path"])
    raw = Path(report["raw_bundle"])
    if (not postlift.is_absolute() or not transfer.is_absolute() or
            not raw.is_absolute() or
            _sha(postlift) != report["postlift_report_sha256"] or
            _sha(transfer) != report["transfer_execution_sha256"] or
            _sha(raw / "manifest.json") != report["raw_manifest_sha256"]):
        raise ValueError("preinsert checkpoint source file changed")
    postlift_record = json.loads(postlift.read_text(encoding="utf-8"))
    if postlift_record.get("schema") == (
            "precision_insertion_bounded_postlift_preflight_v1"):
        context = report.get("bounded_source_context")
        if not isinstance(context, dict):
            raise ValueError("bounded preinsert source context is missing")
        mode = select_mode(context["family"], context["gap_mm"])
        root = Path(context["shared_root"]).expanduser().resolve()
        key = postlift_record["candidate_key"]
        candidate_dir = AssetPaths(root, mode).candidate_dir / Path(*key)
        if (context["candidate_dir"] != str(candidate_dir.resolve()) or
                report["candidate_id"] != "/".join(key)):
            raise ValueError("bounded preinsert candidate context changed")
        verify_bounded_postlift_preflight(
            postlift, mode=mode, shared_root=root,
            candidate_dir=candidate_dir)
    identity = type("AttemptRef", (), {
        "attempt_id": report["attempt_id"],
        "candidate_id": report["candidate_id"],
    })()
    source = _transfer_log(transfer, identity)
    if (source["measurement"] != report["transfer_measurement"] or
            float(source["started_at_s"]) != report["transfer_started_at_s"] or
            float(source["completed_at_s"]) != report["transfer_completed_at_s"]):
        raise ValueError("preinsert transfer report changed")
    capture = verify_raw_camera_capture(raw, phase="preinsert")
    views = report.get("comparison", {}).get("views", {})
    png_hashes = report.get("overlay_png_sha256", {})
    if (set(views) != set(capture["frame_evidence"]) or
            set(png_hashes) != set(views) or len(views) < 2):
        raise ValueError("preinsert overlay camera set changed")
    paired_pixels = []
    for serial in sorted(views):
        hashes = views[serial]
        source_image = cv2.imread(str(raw / "images" / f"{serial}.png"),
                                  cv2.IMREAD_COLOR)
        overlay_path = path.parent / "overlays" / f"{serial}.png"
        overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if (source_image is None or overlay_image is None or
                image_sha256(source_image) != hashes["raw_image_sha256"] or
                image_sha256(overlay_image) != hashes["overlay_image_sha256"] or
                _sha(overlay_path) != png_hashes[serial] or
                hashes["timestamp_s"] !=
                capture["frame_evidence"][serial]["timestamp_s"]):
            raise ValueError("preinsert VLM pixel evidence changed")
        paired_pixels.append(HeldSceneView(
            serial, hashes["timestamp_s"],
            Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))))
    _verify_visual_record(report["visual"], views)
    _verify_visual_replay(report["visual"], paired_pixels)
    expected_label, expected_reason = _arrival_label(
        measurement=report["transfer_measurement"],
        position_error=report["hand_translation_error_m"],
        rotation_error=report["hand_rotation_error_deg"],
        max_position_error=report["max_hand_translation_error_m"],
        max_rotation_error=report["max_hand_rotation_error_deg"],
        visual_status=report["visual"]["status"])
    if (report["preinsert_reached"] is not expected_label or
            report["reason"] != expected_reason):
        raise ValueError("preinsert label contradicts saved evidence")
    return report
