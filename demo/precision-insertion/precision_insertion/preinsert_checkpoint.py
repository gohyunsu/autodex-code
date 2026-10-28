"""Read-only observed arrival gate at the frozen socket's preinsert hold.

An external controller supplies a completed transfer log. The saved camera
bundle, measured stationary FR3/Inspire joints, post-lift grasp relation, and
per-view raw-only VLM check are evaluated together. CAD projections are saved
for deterministic geometry/audit only and are never passed to the VLM. This
does not execute motion, prove contact, or certify the external producer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
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
from .nominal_postlift import (
    NominalPostLiftPreflight, verify_nominal_postlift_preflight,
)
from .labeling import resolve_label, independent_vlm_label
from .lift_checkpoint import verify_lift_checkpoint
from .observer import (
    HeldSceneView, ImageVLM, LabeledFrame, PreinsertVisualAssessment,
    observe_preinsert_raw_views, observe_preinsert_temporal_views,
    resize_classification_images,
)
from .postlift_preflight import PostLiftPreflight
from .raw_camera_capture import verify_raw_camera_capture
from .records import AttemptRecord
from .runtime_evidence_cache import recalled, remember
from .timing_profile import timed
from .world import validated_frozen_socket_pose
from .visual_views import assessment_cameras


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _focus_temporal_frames(
    frames: list[LabeledFrame], *, calibration: dict,
    after_lift_q, preinsert_q, robot_urdf: Path,
    socket_bboxes: dict,
) -> tuple[list[LabeledFrame], dict]:
    """Keep the measured hand and this session's socket large in VLM pairs.

    The hand projection locates the crop; it is not a claim that the key is
    still held. The socket box comes from the frozen session's CAD projection.
    No overlay, resampling, or synthetic pixel is added to either image.
    """
    from .lift_visible_grip import (
        HAND_CROP_SIDE, _hand_foci_for_pair, _hand_roi,
    )

    geometry = _hand_foci_for_pair(after_lift_q, preinsert_q, robot_urdf)
    focused, records = [], {}
    for frame in frames:
        roi, hand_px = _hand_roi(
            image_size=frame.image.size, camera=frame.camera_id,
            calibration=calibration,
            T_robot_hand=geometry[frame.phase][0],
            focus_local=geometry[frame.phase][1])
        if frame.phase == "preinsert":
            socket = socket_bboxes.get(frame.camera_id, {}).get(
                "projected_hull_bbox_px")
            if (isinstance(socket, list) and len(socket) == 4 and
                    all(type(v) is int for v in socket) and
                    socket[0] < socket[2] and socket[1] < socket[3]):
                width, height = frame.image.size
                left = max(0, min(roi[0], socket[0]))
                top = max(0, min(roi[1], socket[1]))
                right = min(width, max(roi[2], socket[2]))
                bottom = min(height, max(roi[3], socket[3]))
                needed = max(HAND_CROP_SIDE, right - left + 64,
                             bottom - top + 64)
                if needed > min(width, height):
                    roi = (0, 0, width, height)
                else:
                    side = needed
                    cx, cy = (left + right) / 2, (top + bottom) / 2
                    x0 = max(0, min(width - side, round(cx - side / 2)))
                    y0 = max(0, min(height - side, round(cy - side / 2)))
                    roi = (x0, y0, x0 + side, y0 + side)
            else:
                # Without a projected socket, retain the raw full scene so
                # the model can still inspect the target in this camera.
                roi = (0, 0, *frame.image.size)
        crop = frame.image.crop(roi)
        focused.append(LabeledFrame(
            frame.camera_id, frame.phase, frame.timestamp_s, crop))
        records.setdefault(frame.camera_id, {})[frame.phase] = {
            "roi_xyxy_full_px": list(roi),
            "source_size_wh": list(frame.image.size),
            "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
            "projected_hand_px": hand_px,
        }
    return focused, {
        "schema": "precision_insertion_temporal_hand_socket_crop_v1",
        "per_camera": records,
    }


def _replay_temporal_crops(
    frames: list[LabeledFrame], crops: dict,
) -> list[LabeledFrame]:
    """Rebuild the exact VLM pixels from the immutable raw camera captures."""
    if crops.get("schema") != "precision_insertion_temporal_hand_socket_crop_v1":
        raise ValueError("unknown preinsert VLM crop policy")
    records = crops.get("per_camera", {})
    if set(records) != {frame.camera_id for frame in frames}:
        raise ValueError("preinsert VLM crop camera set changed")
    focused = []
    for frame in frames:
        row = records[frame.camera_id][frame.phase]
        roi = row["roi_xyxy_full_px"]
        if (row["source_size_wh"] != list(frame.image.size) or
                not isinstance(roi, list) or len(roi) != 4 or
                any(type(v) is not int for v in roi) or
                not (0 <= roi[0] < roi[2] <= frame.image.width and
                     0 <= roi[1] < roi[3] <= frame.image.height)):
            raise ValueError("preinsert VLM crop lies outside saved frame")
        image = frame.image.crop(tuple(roi))
        if row["crop_rgb_sha256"] != hashlib.sha256(image.tobytes()).hexdigest():
            raise ValueError("preinsert VLM crop pixels changed")
        focused.append(LabeledFrame(
            frame.camera_id, frame.phase, frame.timestamp_s, image))
    return focused


def _transfer_log(
    path: Path, attempt: AttemptRecord, *,
    schema: str = "precision_insertion_transfer_execution_v1",
) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or
            data.get("schema") != schema or
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
    *, measurement: dict, visual_status: str,
) -> tuple[bool | None, str]:
    if (measurement["trajectory_complete"] is False or
            measurement["safety_abort"] is True or
            measurement["grasp_held"] is False):
        return False, "transfer_or_grasp_failed"
    if visual_status in {"gross_misalignment", "slip_or_miss"}:
        return False, "multiview_visible_preinsert_failure"
    if (visual_status == "coarse_match" and measurement == {
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True}):
        return True, "measured_arrival_and_visible_held_key"
    return None, "preinsert_evidence_incomplete_or_occluded"


def visible_held_transfer_return(reason: str, visual_status: str) -> bool:
    """Admit only completed, visibly held preinsert failures to held return.

    The transfer controller's independent ``grasp_held`` record and measured
    endpoint are checked by the caller; a visual slip/miss never qualifies.
    """
    return (reason, visual_status) == (
        "multiview_visible_preinsert_failure", "gross_misalignment")


def observe_held_preinsert_scene(
    *, calibration: SessionCalibration, shared_root: Path, mode: TaskMode,
    raw_bundle: Path, frame_rows: dict, joint_sample: LiveRobotState,
    T_key_hand: np.ndarray, T_robot_hand_target: np.ndarray,
    relation_source: str, backend: ImageVLM, max_capture_skew_s: float,
    renderer_factory: Callable | None = None,
    robot_loader: Callable | None = None,
    after_lift_raw_bundle: Path | None = None,
    after_lift_frame_rows: dict | None = None,
    after_lift_joint_q=None,
    socket_bboxes: dict | None = None,
) -> tuple[float, float, HeldSceneComparison, PreinsertVisualAssessment]:
    """Render and label one measured held-key arrival for either transfer route."""
    snapshot = calibration.record.get("camera_calibration", {})
    camera_ids = set(frame_rows)
    if (not isinstance(snapshot, dict) or
            not isinstance(snapshot.get("intrinsics_full"), dict) or
            not isinstance(snapshot.get("extrinsics_full"), dict) or
            not camera_ids <= set(snapshot["intrinsics_full"]) or
            not camera_ids <= set(snapshot["extrinsics_full"])):
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
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    frames_bgr = {serial: cv2.imread(
        str(raw_bundle / "images" / f"{serial}.png"), cv2.IMREAD_COLOR)
        for serial in camera_ids}
    if any(frame is None for frame in frames_bgr.values()):
        raise ValueError("saved preinsert image cannot be decoded")

    def render_comparison():
        prediction_kwargs = {}
        if robot_loader is not None:
            prediction_kwargs["robot_loader"] = robot_loader
        prediction = build_held_scene_prediction(
            shared_root=shared_root, mode=mode,
            full_q_measured=joint_sample.full_q,
            T_robot_socket_frozen=validated_frozen_socket_pose(
                mode=mode, shared_root=shared_root, calibration=calibration),
            T_key_hand_hypothesis=T_key_hand,
            relation_source=relation_source, **prediction_kwargs)
        target = validate_se3(T_robot_hand_target,
                              name="planned preinsert T_robot_hand")
        position_error = float(np.linalg.norm(
            prediction.T_robot_hand[:3, 3] - target[:3, 3]))
        rotation_error = pose_angle_deg(prediction.T_robot_hand, target)
        comparison = build_held_scene_comparison(
            prediction=prediction, frames_bgr=frames_bgr,
            frame_timestamps_s={serial: frame_rows[serial]["timestamp_s"]
                                for serial in camera_ids},
            intrinsics_undistorted={serial: K_all[serial]
                                    for serial in camera_ids},
            T_camera_robot={serial: camera_world_all[serial] @ c2r
                            for serial in camera_ids},
            renderer_factory=renderer_factory)
        return position_error, rotation_error, comparison

    if after_lift_raw_bundle is None and after_lift_frame_rows is None:
        # Historical same-held-attempt XY retry utility; never assembled in
        # the cylinder live pipeline. Its single-frame evidence stays explicit.
        position_error, rotation_error, comparison = render_comparison()
        visual = observe_preinsert_raw_views(
            backend, comparison.views, max_capture_skew_s=max_capture_skew_s)
        return position_error, rotation_error, comparison, visual
    if after_lift_raw_bundle is None or after_lift_frame_rows is None:
        raise ValueError("preinsert temporal views need an after-lift bundle")
    paired = []
    for camera in sorted(camera_ids):
        after_lift = cv2.imread(
            str(after_lift_raw_bundle / "images" / f"{camera}.png"),
            cv2.IMREAD_COLOR)
        if after_lift is None:
            raise ValueError("saved after-lift image cannot be decoded")
        paired.extend((
            LabeledFrame(camera, "after_lift",
                         after_lift_frame_rows[camera]["timestamp_s"],
                         Image.fromarray(cv2.cvtColor(after_lift, cv2.COLOR_BGR2RGB))),
            LabeledFrame(camera, "preinsert", frame_rows[camera]["timestamp_s"],
                         Image.fromarray(cv2.cvtColor(
                             frames_bgr[camera], cv2.COLOR_BGR2RGB))),
        ))
    crop_record = None
    if after_lift_joint_q is not None and socket_bboxes is not None:
        paired, crop_record = _focus_temporal_frames(
            paired, calibration=calibration.record,
            after_lift_q=after_lift_joint_q,
            preinsert_q=joint_sample.full_q,
            robot_urdf=AssetPaths(Path(shared_root), mode).robot_urdf,
            socket_bboxes=socket_bboxes)
    # Only a remote backend can overlap inference with this PC's GPU renderer
    # without competing for the same GPU memory. Both use the same saved raw
    # camera pixels; the overlay is never a VLM input.
    if getattr(backend, "independent_remote_inference", False):
        with ThreadPoolExecutor(max_workers=1,
                                thread_name_prefix="precision-preinsert") as pool:
            visual_future = pool.submit(
                copy_context().run, observe_preinsert_temporal_views,
                backend, paired, max_capture_skew_s=max_capture_skew_s)
            position_error, rotation_error, comparison = render_comparison()
            visual = visual_future.result()
    else:
        position_error, rotation_error, comparison = render_comparison()
        visual = observe_preinsert_temporal_views(
            backend, paired, max_capture_skew_s=max_capture_skew_s)
    if crop_record is not None:
        visual = replace(visual, input_crops=crop_record)
    return position_error, rotation_error, comparison, visual


def _verify_visual_record(
    visual: dict, views: dict, after_rows: dict | None = None,
) -> None:
    rows = visual.get("per_view")
    if (not isinstance(rows, list) or len(rows) > len(views) or
            len(rows) < 2):
        raise ValueError("preinsert visual report lacks its camera responses")
    decisive: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or
                row.get("stage") != "preinsert_visual" or
                not isinstance(row.get("image_order"), list) or
                len(row["image_order"]) != (2 if after_rows is not None else 1) or
                not isinstance(row.get("parsed"), dict)):
            raise ValueError("preinsert per-view response is invalid")
        label = row["image_order"][0]
        prefix = "after_lift/" if after_rows is not None else "raw_preinsert/"
        if not isinstance(label, str) or not label.startswith(prefix):
            raise ValueError("preinsert per-view raw image order changed")
        camera = label.removeprefix(prefix).split("@", 1)[0]
        expected = ([
            f"after_lift/{camera}@{after_rows[camera]['timestamp_s']:.6f}",
            f"preinsert/{camera}@{views[camera]['timestamp_s']:.6f}"]
            if after_rows is not None and camera in after_rows else [
                f"raw_preinsert/{camera}@{views[camera]['timestamp_s']:.6f}"]
            if camera in views else [])
        if (camera not in views or camera in seen or
                row["image_order"] != expected):
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
        self._rows = rows
        self._prepared = 0
        self._answers = [row.get("raw_answer") for row in rows]

    def prepare_classification_images(self, images):
        row = self._rows[self._prepared]
        self._prepared += 1
        transform = row.get("input_transform")
        if transform is None:
            return images, None
        if not isinstance(transform, dict):
            raise ValueError("preinsert VLM input transform is invalid")
        return resize_classification_images(
            images, tuple(transform.get("max_size", ())))

    def infer(self, _images, _prompt):
        return self._answers.pop(0)


def _verify_visual_replay(
    visual: dict, paired_pixels: list[LabeledFrame] | list[HeldSceneView],
) -> None:
    """Bind parsed labels, prompts and consensus to exact saved VLM text."""
    recorded = visual.get("per_view")
    temporal = bool(paired_pixels and isinstance(paired_pixels[0], LabeledFrame))
    selected = {
        row["image_order"][0].split("/", 1)[1].split("@", 1)[0]
        for row in recorded
        if isinstance(row, dict) and isinstance(row.get("image_order"), list)
        and row["image_order"] and isinstance(row["image_order"][0], str)
        and "/" in row["image_order"][0]
    } if isinstance(recorded, list) else set()
    paired_pixels = [view for view in paired_pixels
                     if view.camera_id in selected]
    if (not isinstance(recorded, list) or
            len(recorded) * (2 if temporal else 1) != len(paired_pixels) or
            any(not isinstance(row, dict) or
                not isinstance(row.get("raw_answer"), str) or
                not isinstance(row.get("backend_model"), str) or
                not row["backend_model"].strip() or
                type(row.get("latency_s")) not in (float, int) or
                not math.isfinite(row["latency_s"]) or
                row["latency_s"] < 0 for row in recorded)):
        raise ValueError("preinsert VLM saved answer is invalid")
    if temporal:
        stamps_by_phase = {
            phase: [view.timestamp_s for view in paired_pixels
                    if view.phase == phase]
            for phase in ("after_lift", "preinsert")}
        replay_skew_s = max(
            max(stamps) - min(stamps) for stamps in stamps_by_phase.values()) + 1e-6
    else:
        stamps = [view.timestamp_s for view in paired_pixels]
        replay_skew_s = max(stamps) - min(stamps) + 1e-6
    # This replay bound merely admits the *same* already verified captures;
    # it is not a new assertion about commissioned camera synchronization.
    replay_skew_s = max(replay_skew_s, 1e-6)
    try:
        observer = (observe_preinsert_temporal_views if temporal
                    else observe_preinsert_raw_views)
        replay = observer(
            _RecordedPreinsertAnswers(recorded), paired_pixels,
            max_capture_skew_s=replay_skew_s)
        if visual.get("input_crops") is not None:
            replay = replace(replay, input_crops=visual["input_crops"])
        replay = replay.to_record()
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
            "vlm_label": independent_vlm_label(
                "preinsert_reached", self.visual.status),
            "comparison": self.comparison.to_record(),
            **self.source_record,
            "scope": "read_only_saved_evidence_not_controller_or_sensor_certification",
            "robot_ready": False,
        }


def assess_preinsert_checkpoint(
    *, attempt: AttemptRecord,
    postlift: PostLiftPreflight | BoundedPostLiftPreflight | NominalPostLiftPreflight,
    postlift_report_path: Path, calibration: SessionCalibration,
    shared_root: Path, mode: TaskMode, raw_bundle: Path,
    after_lift_raw_bundle: Path,
    transfer_execution_path: Path, joint_sample: LiveRobotState,
    backend: ImageVLM, max_capture_skew_s: float,
    max_joint_frame_skew_s: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    renderer_factory: Callable | None = None,
    robot_loader: Callable | None = None,
    label_mode: str = "evidence_fused",
    manual_annotation: dict | None = None,
    lift_checkpoint_report_path: Path | None = None,
    allow_held_misalignment: bool = False,
) -> PreinsertCheckpoint:
    """Fuse per-view coarse VLM with measured hold pose and execution evidence.

    ``coarse_match`` is necessary for a positive label but never sufficient
    alone. A hidden key yields unknown, not a predicted-pose success. The
    source log and camera timing are verified as files and claims; commissioning
    must separately establish sensor authenticity and physical error bounds.
    """
    limits = (max_capture_skew_s, max_joint_frame_skew_s,
              max_arm_hand_skew_s,
              max_hand_command_error_raw, max_arm_velocity_rad_s)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in limits):
        raise ValueError("preinsert arrival needs positive commissioned limits")
    if (not isinstance(attempt, AttemptRecord) or attempt.mode != mode or
            attempt.labels["grasp_success"] is not True or
            attempt.labels["preinsert_reached"] is not None or
            attempt.candidate_id is None or
            not isinstance(postlift, (PostLiftPreflight,
                                      BoundedPostLiftPreflight,
                                      NominalPostLiftPreflight)) or
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
    nominal = isinstance(postlift, NominalPostLiftPreflight)
    relation_field = ("bounded_held_relation" if bounded else
                      "nominal_held_relation" if nominal else
                      "observed_held_relation")
    if bounded or nominal:
        candidate_dir = (AssetPaths(Path(shared_root), mode).candidate_dir /
                         Path(*postlift.candidate_key))
        verifier = (verify_bounded_postlift_preflight if bounded else
                    verify_nominal_postlift_preflight)
        verifier(
            postlift_path, mode=mode, shared_root=shared_root,
            candidate_dir=candidate_dir)
    held_relation = (
        postlift.relation.T_key_hand_command
        if (bounded or nominal) and
        postlift.relation.T_key_hand_command is not None else
        postlift.relation.T_key_hand)
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
                       held_relation, rtol=0, atol=1e-8):
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
    after_lift_bundle = Path(after_lift_raw_bundle).expanduser().resolve()
    after_lift_capture = verify_raw_camera_capture(
        after_lift_bundle, phase="after_lift")
    lift_checkpoint_path = None
    lift_record = None
    if lift_checkpoint_report_path is not None:
        lift_checkpoint_path = Path(lift_checkpoint_report_path).expanduser().resolve()
        lift_record = verify_lift_checkpoint(lift_checkpoint_path)
        if (lift_record.get("schema") != "precision_insertion_raw_lift_checkpoint_v1" or
                lift_record.get("attempt_id") != attempt.attempt_id or
                lift_record.get("candidate_id") != attempt.candidate_id or
                lift_record.get("grasp_success") is not True or
                lift_record.get("after_bundle") != str(after_lift_bundle)):
            raise ValueError("preinsert pair differs from this attempt's lift checkpoint")
    frame_rows = capture["frame_evidence"]
    after_lift_rows = after_lift_capture["frame_evidence"]
    if (set(after_lift_rows) != set(frame_rows) or
            after_lift_capture["capture_id"] == capture["capture_id"] or
            bounded_capture_skew_s(after_lift_rows) > max_capture_skew_s or
            max(row["timestamp_s"] + row["max_error_s"]
                for row in after_lift_rows.values()) >= started or
            any(after_lift_rows[camera]["frame_id"] >= frame_rows[camera]["frame_id"]
                or after_lift_rows[camera]["timestamp_s"] >=
                frame_rows[camera]["timestamp_s"] for camera in frame_rows)):
        raise ValueError("after-lift/preinsert raw camera pairs are not ordered")
    if (len(frame_rows) < 2 or
            bounded_capture_skew_s(frame_rows) > max_capture_skew_s):
        raise ValueError("preinsert cameras are not synchronized")
    first_exposure = min(row["timestamp_s"] - row["max_error_s"]
                         for row in frame_rows.values())
    last_exposure = max(row["timestamp_s"] + row["max_error_s"]
                        for row in frame_rows.values())
    if completed >= first_exposure:
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
    selected_cameras = assessment_cameras(frame_rows)
    selected_rows = {camera: frame_rows[camera] for camera in selected_cameras}
    selected_after_rows = {
        camera: after_lift_rows[camera] for camera in selected_cameras}
    socket_bboxes = None
    if (lift_record is not None and
            isinstance(lift_record.get("before_bundle"), str) and
            isinstance(lift_record.get("joint_feedback"), dict)):
        initial_key = json.loads((Path(lift_record["before_bundle"]) /
                                  "key_observation.json").read_text(
                                      encoding="utf-8"))
        socket_bboxes = (initial_key.get("consistency", {})
                         .get("socket_exclusion", {}).get("per_camera"))
    position_error, rotation_error, comparison, visual = (
        observe_held_preinsert_scene(
            calibration=calibration, shared_root=shared_root, mode=mode,
            raw_bundle=bundle, after_lift_raw_bundle=after_lift_bundle,
            frame_rows=selected_rows,
            after_lift_frame_rows=selected_after_rows,
            joint_sample=joint_sample,
            T_key_hand=held_relation,
            T_robot_hand_target=postlift.targets.T_robot_hand_preinsert,
            relation_source=(
                "verified_physical_grasp_calibration" if bounded else
                "nominal_v8_grasp_hypothesis" if nominal else
                "observed_multiview_key_plus_wrist"),
            backend=backend, max_capture_skew_s=max_capture_skew_s,
            renderer_factory=renderer_factory, robot_loader=robot_loader,
            after_lift_joint_q=(lift_record["joint_feedback"].get("full_q")
                                if socket_bboxes else None),
            socket_bboxes=socket_bboxes))
    measurement = transfer["measurement"]
    label, reason = _arrival_label(
        measurement=measurement, visual_status=visual.status)
    label, reason = resolve_label(
        stage="preinsert_reached", mode=label_mode,
        visual_class=visual.status, fused_value=label,
        fused_reason=reason, annotation=manual_annotation)
    if (allow_held_misalignment and label_mode == "vlm_only" and
            visual.status == "gross_misalignment" and measurement == {
                "trajectory_complete": True, "safety_abort": False,
                "grasp_held": True}):
        label, reason = True, "held_arrival_xy_refinement_pending"
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
            "after_lift_raw_bundle": str(after_lift_bundle),
            "after_lift_manifest_sha256": _sha(after_lift_bundle / "manifest.json"),
            **({"lift_checkpoint_report_path": str(lift_checkpoint_path),
                "lift_checkpoint_report_sha256": _sha(lift_checkpoint_path)}
               if lift_checkpoint_path is not None else {}),
            "max_capture_skew_s": max_capture_skew_s,
            "vlm_frame_evidence": {
                camera: {
                    phase: {
                        "timestamp_s": source["frame_evidence"][camera]["timestamp_s"],
                        "frame_id": source["frame_evidence"][camera]["frame_id"],
                        "image_sha256": source["frame_evidence"][camera]["image_sha256"],
                        "png_sha256": source["image_file_sha256"][camera],
                    }
                    for phase, source in (("after_lift", after_lift_capture),
                                          ("preinsert", capture))}
                for camera in selected_cameras},
            "transfer_measurement": measurement,
            "label_mode": label_mode,
            "allow_held_misalignment": allow_held_misalignment,
            "manual_annotation": manual_annotation,
            "transfer_started_at_s": started,
            "transfer_completed_at_s": completed,
            "observation_completed_at_s": last_exposure,
            "joint_feedback": joint_sample.to_record(),
            **({"bounded_source_context" if bounded else
                "nominal_source_context": {
                "shared_root": str(Path(shared_root).expanduser().resolve()),
                "family": mode.family, "gap_mm": mode.gap_mm,
                "candidate_dir": str(candidate_dir.resolve()),
            }} if bounded or nominal else {}),
        })


def write_preinsert_checkpoint(
    checkpoint: PreinsertCheckpoint, output_dir: Path,
) -> Path:
    """Save raw VLM inputs plus separate geometry-audit renderings.

    The same live session can reuse this producer result immediately. Offline
    readers still run the full verifier against the saved PNGs and sources.
    """
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
            raise ValueError("saved preinsert geometry-audit overlay changed")
        png_hashes[view.camera_id] = _sha(path)
    record["overlay_png_sha256"] = png_hashes
    report = target / "report.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    remember("preinsert_checkpoint", report, "preinsert", {
        "sha256": _sha(report), "record": record})
    return report


def verify_preinsert_checkpoint(report_path: Path) -> dict:
    """Recheck saved bytes; do not re-run VLM or certify external producers."""
    path = Path(report_path).expanduser().resolve()
    report_bytes = path.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    cached = recalled("preinsert_checkpoint", path, "preinsert")
    if cached is not None and cached["sha256"] == report_sha256:
        with timed("evidence.live_preinsert_report_reuse", phase="evidence",
                   kind="io"):
            return cached["record"]
    report = json.loads(report_bytes)
    if (not isinstance(report, dict) or report.get("schema") !=
            "precision_insertion_preinsert_checkpoint_v1" or
            report.get("robot_ready") is not False or
            report.get("visual", {}).get("status") not in {
                "coarse_match", "gross_misalignment", "slip_or_miss", "unknown"}):
        raise ValueError("invalid preinsert checkpoint report")
    postlift = Path(report["postlift_report_path"])
    transfer = Path(report["transfer_execution_path"])
    raw = Path(report["raw_bundle"])
    after_lift = Path(report["after_lift_raw_bundle"])
    if (not postlift.is_absolute() or not transfer.is_absolute() or
            not raw.is_absolute() or not after_lift.is_absolute() or
            _sha(postlift) != report["postlift_report_sha256"] or
            _sha(transfer) != report["transfer_execution_sha256"] or
            _sha(raw / "manifest.json") != report["raw_manifest_sha256"] or
            _sha(after_lift / "manifest.json") !=
            report["after_lift_manifest_sha256"]):
        raise ValueError("preinsert checkpoint source file changed")
    postlift_record = json.loads(postlift.read_text(encoding="utf-8"))
    bounded_source = postlift_record.get("schema") == (
        "precision_insertion_bounded_postlift_preflight_v1")
    nominal_source = postlift_record.get("schema") == (
        "precision_insertion_nominal_postlift_preflight_v1")
    if bounded_source or nominal_source:
        context = report.get("bounded_source_context" if bounded_source else
                             "nominal_source_context")
        if not isinstance(context, dict):
            raise ValueError("held-relation preinsert source context is missing")
        mode = select_mode(context["family"], context["gap_mm"])
        root = Path(context["shared_root"]).expanduser().resolve()
        key = postlift_record["candidate_key"]
        candidate_dir = AssetPaths(root, mode).candidate_dir / Path(*key)
        if (context["candidate_dir"] != str(candidate_dir.resolve()) or
                report["candidate_id"] != "/".join(key)):
            raise ValueError("preinsert candidate context changed")
        verifier = (verify_bounded_postlift_preflight if bounded_source else
                    verify_nominal_postlift_preflight)
        verifier(
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
    after_capture = verify_raw_camera_capture(after_lift, phase="after_lift")
    lift_path_value = report.get("lift_checkpoint_report_path")
    if lift_path_value is not None:
        lift_path = Path(lift_path_value)
        if (not lift_path.is_absolute() or
                _sha(lift_path) != report.get("lift_checkpoint_report_sha256")):
            raise ValueError("preinsert lift checkpoint source changed")
        lift_record = verify_lift_checkpoint(lift_path)
        if (lift_record.get("schema") != "precision_insertion_raw_lift_checkpoint_v1" or
                lift_record.get("attempt_id") != report["attempt_id"] or
                lift_record.get("candidate_id") != report["candidate_id"] or
                lift_record.get("grasp_success") is not True or
                lift_record.get("after_bundle") != str(after_lift)):
            raise ValueError("preinsert lift checkpoint binding changed")
    after_rows = after_capture["frame_evidence"]
    pre_rows = capture["frame_evidence"]
    if (set(after_rows) != set(pre_rows) or
            after_capture["capture_id"] == capture["capture_id"] or
            bounded_capture_skew_s(after_rows) > report["max_capture_skew_s"] or
            bounded_capture_skew_s(pre_rows) > report["max_capture_skew_s"] or
            max(row["timestamp_s"] + row["max_error_s"]
                for row in after_rows.values()) >= report["transfer_started_at_s"] or
            min(row["timestamp_s"] - row["max_error_s"]
                for row in pre_rows.values()) <= report["transfer_completed_at_s"] or
            any(after_rows[camera]["frame_id"] >= pre_rows[camera]["frame_id"]
                or after_rows[camera]["timestamp_s"] >=
                pre_rows[camera]["timestamp_s"] for camera in pre_rows)):
        raise ValueError("preinsert raw camera pair chronology changed")
    views = report.get("comparison", {}).get("views", {})
    png_hashes = report.get("overlay_png_sha256", {})
    if (not set(views) <= set(capture["frame_evidence"]) or
            not set(views) <= set(after_capture["frame_evidence"]) or
            set(png_hashes) != set(views) or len(views) < 2):
        raise ValueError("preinsert overlay camera set changed")
    paired_pixels = []
    for serial in sorted(views):
        hashes = views[serial]
        source_image = cv2.imread(str(raw / "images" / f"{serial}.png"),
                                  cv2.IMREAD_COLOR)
        before_image = cv2.imread(
            str(after_lift / "images" / f"{serial}.png"), cv2.IMREAD_COLOR)
        overlay_path = path.parent / "overlays" / f"{serial}.png"
        overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if (source_image is None or before_image is None or
                overlay_image is None or
                image_sha256(source_image) != hashes["raw_image_sha256"] or
                image_sha256(overlay_image) != hashes["overlay_image_sha256"] or
                _sha(overlay_path) != png_hashes[serial] or
                hashes["timestamp_s"] !=
                capture["frame_evidence"][serial]["timestamp_s"] or
                report.get("vlm_frame_evidence", {}).get(serial) != {
                    phase: {
                        "timestamp_s": source["frame_evidence"][serial]["timestamp_s"],
                        "frame_id": source["frame_evidence"][serial]["frame_id"],
                        "image_sha256": source["frame_evidence"][serial]["image_sha256"],
                        "png_sha256": source["image_file_sha256"][serial],
                    }
                    for phase, source in (("after_lift", after_capture),
                                          ("preinsert", capture))}):
            raise ValueError("preinsert VLM pixel evidence changed")
        paired_pixels.extend((
            LabeledFrame(serial, "after_lift",
                         after_capture["frame_evidence"][serial]["timestamp_s"],
                         Image.fromarray(cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB))),
            LabeledFrame(serial, "preinsert", hashes["timestamp_s"],
                         Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))),
        ))
    _verify_visual_record(report["visual"], views,
                          after_capture["frame_evidence"])
    if report["visual"].get("input_crops") is not None:
        paired_pixels = _replay_temporal_crops(
            paired_pixels, report["visual"]["input_crops"])
    _verify_visual_replay(report["visual"], paired_pixels)
    expected_label, expected_reason = _arrival_label(
        measurement=report["transfer_measurement"],
        visual_status=report["visual"]["status"])
    expected_label, expected_reason = resolve_label(
        stage="preinsert_reached",
        mode=report.get("label_mode", "evidence_fused"),
        visual_class=report["visual"]["status"],
        fused_value=expected_label, fused_reason=expected_reason,
        annotation=report.get("manual_annotation"))
    if (report.get("allow_held_misalignment") is True and
            report.get("label_mode") == "vlm_only" and
            report["visual"]["status"] == "gross_misalignment" and
            report["transfer_measurement"] == {
                "trajectory_complete": True, "safety_abort": False,
                "grasp_held": True}):
        expected_label, expected_reason = (
            True, "held_arrival_xy_refinement_pending")
    if (report["preinsert_reached"] is not expected_label or
            report["reason"] != expected_reason):
        raise ValueError("preinsert label contradicts saved evidence")
    remember("preinsert_checkpoint", path, "preinsert", {
        "sha256": report_sha256, "record": report})
    return report
