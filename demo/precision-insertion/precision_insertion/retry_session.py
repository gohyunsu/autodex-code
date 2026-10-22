"""Bind a failed insertion to grounded next-trial XY advice.

This joins existing camera, held-relation, endpoint, ZeroDex-style voting and
cuRobo preflight helpers. Guarded withdrawal and its log are supplied by a
future commissioned executor; nothing here commands or authorizes motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .calibration import validate_session_camera_calibration
from .assets import AssetPaths
from .bounded_postlift import verify_bounded_postlift_preflight
from .nominal_postlift import verify_nominal_postlift_preflight
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .frame_provenance import image_sha256, verify_frame_provenance
from .geometry import validate_se3
from .grounded_alignment import (
    AlignmentLimits, estimate_grounded_alignment,
    estimate_grounded_line_alignment,
    observe_grounded_key_axis,
    observe_grounded_cylinder_axis, observe_grounded_cylinder_axis_cropped,
    resolve_xy_only_cylinder_alignment,
)
from .grounding_crop import (
    nominal_preinsert_rois, verify_native_crop_observation,
)
from .held_relation import HeldRelation, resolve_postlift_held_relation
from .key_perception import KeyPoseObservation, verify_key_capture_artifacts
from .live_robot_state import LiveRobotState
from .guarded_axial_handoff import _state_from_record
from .observer import ImageVLM, LabeledFrame
from .preinsert_checkpoint import verify_preinsert_checkpoint
from .raw_camera_capture import verify_raw_camera_capture
from .path_audit import PathAuditLimits
from .records import AttemptRecord
from .retry_preflight import (
    XYRetryPreflight, plan_xy_retry_from_withdrawn_hold,
    write_xy_retry_preflight,
)
from .session_bootstrap import _safe_id
from .tip_tracking import tracked_tip_xy_alignment, tracked_view
from .timing_profile import timed
from .xy_retry import XYRetryAssessment, assess_xy_retry
from .visual_views import assessment_cameras
from .xy_overlay import CalibratedXYFrame, camera_socket_transform
from .world import validated_frozen_socket_pose


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _cylinder_tip_and_rois(
    *, postlift_file: Path, socket: np.ndarray, tip_z_m: float,
    socket_rim_z_m: float, frames: Sequence[CalibratedXYFrame],
    backend: ImageVLM,
    roi_xyxy_by_camera: Mapping[str, tuple[int, int, int, int]] | None,
) -> tuple[np.ndarray, Mapping[str, tuple[int, int, int, int]] | None]:
    nominal = json.loads(postlift_file.read_text(encoding="utf-8"))
    T_robot_key = validate_se3(
        nominal["targets"]["T_robot_key_preinsert"],
        name="commanded preinsert key target")
    tip_robot = T_robot_key @ np.array([0., 0., tip_z_m, 1.])
    socket_from_robot = np.linalg.inv(socket)
    tip_socket = socket_from_robot @ tip_robot
    if (roi_xyxy_by_camera is None and
            getattr(backend, "native_pixel_coordinates", False)):
        rear_socket = socket_from_robot @ T_robot_key @ np.array([0., 0., 0., 1.])
        roi_xyxy_by_camera = nominal_preinsert_rois(
            frames, tip_socket_m=tip_socket[:3],
            rear_socket_m=rear_socket[:3],
            socket_rim_z_m=socket_rim_z_m)
    return tip_socket, roi_xyxy_by_camera


@dataclass(frozen=True)
class RetrySessionLimits:
    max_state_skew_s: float
    max_arm_hand_skew_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float
    max_grasp_translation_drift_m: float
    max_grasp_rotation_drift_deg: float
    minimum_anchor_separation_px: float
    crop_width_px: int
    max_frame_age_s: float
    max_capture_skew_s: float
    axial_waypoint_step_m: float
    path_limits: PathAuditLimits
    # Legacy same-attempt retry budget; inactive for next-trial XY learning.
    max_total_offset_m: float | None = None

    def validate(self) -> None:
        values = (
            self.max_state_skew_s, self.max_arm_hand_skew_s,
            self.max_hand_command_error_raw, self.max_arm_velocity_rad_s,
            self.max_grasp_translation_drift_m,
            self.max_grasp_rotation_drift_deg,
            self.minimum_anchor_separation_px, self.max_frame_age_s,
            self.max_capture_skew_s, self.axial_waypoint_step_m)
        if (not all(math.isfinite(float(v)) and float(v) > 0 for v in values) or
                type(self.crop_width_px) is not int or self.crop_width_px < 32):
            raise ValueError("retry limits must be finite and commissioned")
        if self.max_total_offset_m is not None and (
                not math.isfinite(self.max_total_offset_m) or
                self.max_total_offset_m <= 0):
            raise ValueError("legacy total offset limit must be positive")
        self.path_limits.validate()


@dataclass(frozen=True)
class RetrySessionResult:
    status: str
    attempt_id: str
    candidate_id: str
    held_key_observation_id: str
    held_key_evidence_dir: Path
    held_key_evidence_manifest_sha256: str
    withdrawal_completed_at_s: float
    withdrawal_evidence_path: Path
    withdrawal_evidence_sha256: str
    postlift_preflight_path: Path
    postlift_preflight_sha256: str
    frame_binding: dict
    joint_sample: LiveRobotState
    held_relation: HeldRelation
    assessment: XYRetryAssessment | None
    preflight: XYRetryPreflight | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_observed_xy_retry_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "held_key_observation_id": self.held_key_observation_id,
            "held_key_evidence_dir": str(self.held_key_evidence_dir),
            "held_key_evidence_manifest_sha256": (
                self.held_key_evidence_manifest_sha256),
            "withdrawal_completed_at_s": self.withdrawal_completed_at_s,
            "withdrawal_evidence_path": str(self.withdrawal_evidence_path),
            "withdrawal_evidence_sha256": self.withdrawal_evidence_sha256,
            "postlift_preflight_path": str(self.postlift_preflight_path),
            "postlift_preflight_sha256": self.postlift_preflight_sha256,
            "frame_binding": self.frame_binding,
            "joint_sample": self.joint_sample.to_record(),
            "held_relation": self.held_relation.to_record(),
            "assessment": (None if self.assessment is None
                           else self.assessment.to_record()),
            "preflight": (None if self.preflight is None
                          else self.preflight.to_record()),
            "scope": "read_only_evidence_vote_and_preflight_not_robot_motion",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class UnobservedXYDiagnostic:
    """Saved VLM direction after a failed attempt, never a pending retry."""

    status: str
    attempt_id: str
    candidate_id: str
    withdrawal_completed_at_s: float
    withdrawal_evidence_path: Path
    withdrawal_evidence_sha256: str
    postlift_preflight_path: Path
    postlift_preflight_sha256: str
    frame_binding: dict
    joint_sample: LiveRobotState
    assessment: XYRetryAssessment
    preflight: None = None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_unobserved_xy_diagnostic_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "withdrawal_completed_at_s": self.withdrawal_completed_at_s,
            "withdrawal_evidence_path": str(self.withdrawal_evidence_path),
            "withdrawal_evidence_sha256": self.withdrawal_evidence_sha256,
            "postlift_preflight_path": str(self.postlift_preflight_path),
            "postlift_preflight_sha256": self.postlift_preflight_sha256,
            "frame_binding": self.frame_binding,
            "joint_sample": self.joint_sample.to_record(),
            "assessment": self.assessment.to_record(),
            "preflight": None,
            "held_key_relation_source": "v8_nominal_unobserved_key",
            "scope": "saved_direction_only_not_pending_retry_or_robot_motion",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class GroundedXYDiagnostic:
    """Visual XY advice with explicit metric, planar or exploratory source."""

    status: str
    attempt_id: str
    candidate_id: str
    session_calibration_sha256: str
    camera_calibration_sha256: str
    task_geometry_sha256: str
    frame_request_id: int
    withdrawal_evidence_path: Path
    withdrawal_evidence_sha256: str
    postlift_preflight_path: Path
    postlift_preflight_sha256: str
    frame_binding: dict
    joint_sample: LiveRobotState
    alignment: dict
    vlm_observations: tuple

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_grounded_xy_diagnostic_v2",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "session_calibration_sha256": self.session_calibration_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "task_geometry_sha256": self.task_geometry_sha256,
            "frame_request_id": self.frame_request_id,
            "withdrawal_evidence_path": str(self.withdrawal_evidence_path),
            "withdrawal_evidence_sha256": self.withdrawal_evidence_sha256,
            "postlift_preflight_path": str(self.postlift_preflight_path),
            "postlift_preflight_sha256": self.postlift_preflight_sha256,
            "frame_binding": self.frame_binding,
            "joint_sample": self.joint_sample.to_record(),
            "alignment": self.alignment,
            "vlm_observations": [v.to_record() for v in self.vlm_observations],
            "preflight": None,
            "scope": "saved_xy_advice_only_not_pending_retry_or_robot_motion",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class PreinsertXYDiagnostic:
    """XY advice replayed from the saved pre-insertion snapshot."""

    status: str
    attempt_id: str
    candidate_id: str
    session_calibration_sha256: str
    camera_calibration_sha256: str
    task_geometry_sha256: str
    frame_request_id: int
    preinsert_checkpoint_path: Path
    preinsert_checkpoint_sha256: str
    preinsert_bundle_path: Path
    preinsert_manifest_sha256: str
    postlift_preflight_path: Path
    postlift_preflight_sha256: str
    frame_binding: dict
    joint_sample: LiveRobotState
    alignment: dict
    vlm_observations: tuple

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_saved_preinsert_xy_diagnostic_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "session_calibration_sha256": self.session_calibration_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "task_geometry_sha256": self.task_geometry_sha256,
            "frame_request_id": self.frame_request_id,
            "preinsert_checkpoint_path": str(self.preinsert_checkpoint_path),
            "preinsert_checkpoint_sha256": self.preinsert_checkpoint_sha256,
            "preinsert_bundle_path": str(self.preinsert_bundle_path),
            "preinsert_manifest_sha256": self.preinsert_manifest_sha256,
            "postlift_preflight_path": str(self.postlift_preflight_path),
            "postlift_preflight_sha256": self.postlift_preflight_sha256,
            "frame_binding": self.frame_binding,
            "joint_sample": self.joint_sample.to_record(),
            "alignment": self.alignment,
            "vlm_observations": [row.to_record() for row in self.vlm_observations],
            "preflight": None,
            "scope": "saved_preinsert_raw_pixels_next_attempt_xy_only",
            "robot_ready": False,
        }


def _validated_retry_trial_context(
    *, mode: TaskMode, shared_root: Path, calibration, catalog: dict,
    trial, attempt: AttemptRecord, postlift_preflight_report_path: Path,
) -> tuple[dict, dict, Path, Path]:
    """Shared failed-trial and saved-plan gate for both retry assessments."""
    if (not isinstance(attempt, AttemptRecord) or
            attempt.mode != mode or attempt.candidate_id is None or
            attempt.labels["insertion_success"] is not False or
            attempt._pending_retry):
        raise ValueError("retry needs an observed failed insertion with a held grasp")
    insertion = next((
        event for event in reversed(attempt.events)
        if event["stage"] == "insertion_success"), None)
    if insertion is None:
        raise ValueError("retry needs a saved insertion observation")
    insertion_input = insertion["detail"]["input"]
    if (insertion_input["grasp_held"] is not True or
            insertion_input["safety_abort"] is not False or
            attempt.failure_code in {"slip", "force_abort", "reset_failed"}):
        raise ValueError("retry cannot follow slip or a safety abort")
    preinsert = next((
        event for event in reversed(attempt.events)
        if event["stage"] == "preinsert_reached"), None)
    if preinsert is None or preinsert.get("value") is not True:
        raise ValueError("retry needs an observed pre-insertion arrival")
    postlift_file = Path(postlift_preflight_report_path).expanduser().resolve()
    if (not postlift_file.is_file() or
            Path(preinsert["evidence_refs"]["postlift_preflight"])
            .expanduser().resolve() != postlift_file):
        raise ValueError("retry needs the preinsert stage's saved post-lift preflight")
    postlift = json.loads(postlift_file.read_text(encoding="utf-8"))
    if not isinstance(postlift, dict):
        raise ValueError("post-lift preflight must be a JSON object")
    observed_route = (postlift.get("schema") ==
                      "precision_insertion_postlift_preflight_v1")
    bounded_route = (postlift.get("schema") ==
                     "precision_insertion_bounded_postlift_preflight_v1")
    nominal_route = (postlift.get("schema") ==
                     "precision_insertion_nominal_postlift_preflight_v1")
    relation_field = ("bounded_held_relation" if bounded_route else
                      "nominal_held_relation" if nominal_route else
                      "observed_held_relation")
    bounded_margin = postlift.get("uncertainty_margin")
    if (not isinstance(postlift.get("planning"), dict) or
            not isinstance(postlift.get(relation_field), dict) or
            not (observed_route or bounded_route or nominal_route) or
            postlift.get("status") != "sampled_postlift_preflight_pass" or
            postlift.get("attempt_id") != attempt.attempt_id or
            postlift.get("candidate_key") != attempt.candidate_id.split("/") or
            postlift["planning"].get("sampled_planning_pass") is not True or
            (bounded_route and (not isinstance(bounded_margin, dict) or
                                bounded_margin.get(
                                    "sampled_margin_pass") is not True)) or
            postlift.get("session_calibration_sha256") !=
            attempt.session_calibration_sha256):
        raise ValueError("post-lift preflight is not a passing plan for this attempt")
    if (getattr(trial, "status", None) != "sampled_planning_pass" or
            trial.selected_candidate_key is None or
            trial.insertion_plan is None or
            trial.insertion_plan.sampled_planning_pass is not True or
            "/".join(trial.selected_candidate_key) != attempt.candidate_id or
            trial.pose_class["stem"] != attempt.tabletop_pose_stem or
            trial.session_calibration_sha256 !=
            attempt.session_calibration_sha256):
        raise ValueError("retry trial and observed attempt differ")
    root = Path(shared_root).expanduser().resolve()
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if Path(catalog["shared_root"]).expanduser().resolve() != root:
        raise ValueError("retry catalogue uses another shared root")
    if (trial.catalog_sha256 != _digest(catalog) or
            trial.session_calibration_sha256 !=
            _digest(calibration.record)):
        raise ValueError("retry catalogue or frozen session changed since trial")
    if postlift.get("catalog_sha256") != trial.catalog_sha256:
        raise ValueError("post-lift preflight uses another endpoint catalogue")
    if bounded_route or nominal_route:
        selected = select_pose_candidates(
            catalog, expected_mode=mode,
            tabletop_pose_stem=attempt.tabletop_pose_stem)
        matches = [row for row in selected["candidates"]
                   if tuple(row["key"]) == trial.selected_candidate_key]
        if (selected["status"] != "candidates_available" or
                len(matches) != 1):
            raise ValueError("bounded retry grasp is no longer endpoint eligible")
        verifier = (verify_bounded_postlift_preflight if bounded_route else
                    verify_nominal_postlift_preflight)
        verifier(
            postlift_file, mode=mode, shared_root=root,
            candidate_dir=matches[0]["candidate_dir"])
    return insertion, postlift, postlift_file, root


def _validated_retry_withdrawal(
    *, attempt: AttemptRecord, insertion: dict,
    withdrawal_completed_at_s: float, withdrawal_evidence_path: Path,
    verified_frames: Mapping, joint_sample: LiveRobotState,
    held_capture_start_s: float | None = None,
) -> tuple[float, Path]:
    withdrawal_time = float(withdrawal_completed_at_s)
    withdrawal_file = Path(withdrawal_evidence_path).expanduser().resolve()
    if (not math.isfinite(withdrawal_time) or
            withdrawal_time <= insertion["timestamp_s"] or
            not withdrawal_file.is_file() or
            withdrawal_time >= min(
                row["timestamp_s"] - row["max_error_s"]
                for row in verified_frames.values()) or
            joint_sample.sample_timestamp_s <= withdrawal_time or
            (held_capture_start_s is not None and
             held_capture_start_s <= withdrawal_time)):
        raise ValueError("held-key images/state need a preceding logged withdrawal")
    withdrawal_log = json.loads(withdrawal_file.read_text(encoding="utf-8"))
    if (not isinstance(withdrawal_log, dict) or
            withdrawal_log.get("schema") !=
            "precision_insertion_guarded_withdrawal_v1" or
            withdrawal_log.get("attempt_id") != attempt.attempt_id or
            withdrawal_log.get("candidate_id") != attempt.candidate_id or
            withdrawal_log.get("status") != "withdrawn_to_preinsert_hold" or
            withdrawal_log.get("completed_at_s") != withdrawal_time or
            withdrawal_log.get("key_still_held") is not True or
            withdrawal_log.get("safety_abort") is not False or
            withdrawal_log.get("source") !=
            "commissioned_guarded_controller"):
        raise ValueError("guarded withdrawal log does not confirm a held safe return")
    return withdrawal_time, withdrawal_file


def assess_and_plan_observed_xy_retry(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    catalog: dict, trial, attempt: AttemptRecord,
    held_key_observation: KeyPoseObservation,
    held_key_evidence_dir: Path, joint_sample: LiveRobotState,
    frames: Sequence[LabeledFrame], intrinsics_full: Mapping,
    extrinsics_full: Mapping, frame_request_id: int,
    frame_ids: Mapping[str, int], acquisition_metadata: Mapping,
    backend: ImageVLM, withdrawal_completed_at_s: float,
    withdrawal_evidence_path: Path, postlift_preflight_report_path: Path,
    decision_timestamp_s: float,
    limits: RetrySessionLimits,
) -> RetrySessionResult:
    """Require one withdrawn, still-held key and its same-frame VLM images."""
    limits.validate()
    insertion, postlift, postlift_file, root = _validated_retry_trial_context(
        mode=mode, shared_root=shared_root, calibration=calibration,
        catalog=catalog, trial=trial, attempt=attempt,
        postlift_preflight_report_path=postlift_preflight_report_path)
    bounded_postlift = (
        postlift["schema"] == "precision_insertion_bounded_postlift_preflight_v1")
    if postlift["schema"] == "precision_insertion_nominal_postlift_preflight_v1":
        raise ValueError("historical observed-key retry does not use the nominal route")
    if (not isinstance(held_key_observation, KeyPoseObservation) or
            held_key_observation.phase != "held_preinsert" or
            held_key_observation.key_object != mode.key_object or
            held_key_observation.family != mode.family):
        raise ValueError("retry needs a phase-specific held-key observation")
    evidence_dir = Path(held_key_evidence_dir).expanduser().resolve()
    manifest = verify_key_capture_artifacts(evidence_dir)
    saved = json.loads((evidence_dir / "key_observation.json").read_text(
        encoding="utf-8"))
    if (saved != held_key_observation.to_record() or
            manifest["capture_id"] != held_key_observation.capture_id or
            manifest["request_id"] != held_key_observation.request_id):
        raise ValueError("held-key pose differs from saved camera evidence")
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("retry requires measured Franka and Inspire joints")
    joint_sample.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    if (not math.isfinite(float(decision_timestamp_s)) or
            decision_timestamp_s < joint_sample.sample_timestamp_s):
        raise ValueError("retry decision cannot precede measured robot feedback")
    held_key_observation.require_state_alignment(
        state_timestamp_s=joint_sample.sample_timestamp_s,
        maximum_skew_s=limits.max_state_skew_s)
    if (not frames or len({frame.camera_id for frame in frames}) != len(frames)
            or any(frame.phase != "preinsert_hold" or
                   not isinstance(frame.image, Image.Image) or
                   frame.image.mode != "RGB" for frame in frames)):
        raise ValueError("retry needs unique full-frame RGB preinsert views")
    images_bgr = {frame.camera_id: np.asarray(
        frame.image, dtype=np.uint8)[:, :, ::-1].copy() for frame in frames}
    verified = verify_frame_provenance(
        acquisition_metadata, request_id=frame_request_id,
        images_bgr=images_bgr, frame_ids=frame_ids)
    accepted = set(held_key_observation.consistency.get("accepted_views", []))
    if (frame_request_id != held_key_observation.request_id or
            len(verified) < 2 or not set(verified) <= accepted or
            any(held_key_observation.frame_evidence.get(serial) != row
                for serial, row in verified.items()) or
            any(frame.timestamp_s != verified[frame.camera_id]["timestamp_s"]
                for frame in frames)):
        raise ValueError("VLM images and held-key pose are not the same camera frames")
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={serial: row["K_undist"]
                           for serial, row in intrinsics_full.items()},
        extrinsics_full=extrinsics_full,
        calibrated_camera_ids=set(intrinsics_full))
    withdrawal_time, withdrawal_file = _validated_retry_withdrawal(
        attempt=attempt, insertion=insertion,
        withdrawal_completed_at_s=withdrawal_completed_at_s,
        withdrawal_evidence_path=withdrawal_evidence_path,
        verified_frames=verified, joint_sample=joint_sample,
        held_capture_start_s=held_key_observation.acquisition_interval_s[0])
    selected = select_pose_candidates(
        catalog, expected_mode=mode,
        tabletop_pose_stem=attempt.tabletop_pose_stem)
    if selected["status"] != "candidates_available":
        raise ValueError("retry grasp catalogue is unavailable")
    matches = [row for row in selected["candidates"]
               if tuple(row["key"]) == trial.selected_candidate_key]
    if len(matches) != 1:
        raise ValueError("retry grasp is not current endpoint eligible")
    candidate_dir = Path(matches[0]["candidate_dir"])
    nominal_relation = validate_se3(np.load(
        candidate_dir / "wrist_se3.npy", allow_pickle=False),
        name="selected v8 T_key_hand")
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    key_robot = validate_se3(
        np.linalg.inv(c2r) @ held_key_observation.pose_world,
        name="observed withdrawn T_robot_key")
    wrist_robot = validate_se3(
        planner.fk_wrist(joint_sample.full_q),
        name="measured withdrawn wrist FK")
    relation_field = ("bounded_held_relation" if bounded_postlift else
                      "observed_held_relation")
    prior_source = (
        "measured_wrist_plus_bounded_held_relation" if bounded_postlift else
        "measured_wrist_plus_observed_held_relation")
    previous_relation = validate_se3(
        postlift.get(relation_field, {}).get("T_key_hand"),
        name="saved post-lift T_key_hand")
    prior = held_key_observation.consistency.get("held_pose_prior")
    if (not isinstance(prior, dict) or
            prior.get("source") !=
            prior_source or
            not math.isfinite(float(prior.get("timestamp_s", math.nan))) or
            abs(prior["timestamp_s"] - joint_sample.sample_timestamp_s) >
            limits.max_state_skew_s):
        raise ValueError("held-key admission lacks the measured-wrist prior")
    predicted_world = validate_se3(
        c2r @ wrist_robot @ np.linalg.inv(previous_relation),
        name="post-lift-relation predicted key pose_world")
    if not np.allclose(
            validate_se3(prior.get("pose_world"),
                         name="held-key admitted pose prior_world"),
            predicted_world, rtol=0, atol=1e-6):
        raise ValueError("held-key admission prior differs from saved grasp/wrist")
    relation = resolve_postlift_held_relation(
        mode=mode, shared_root=root,
        T_robot_key_observed=key_robot,
        T_robot_hand_measured=wrist_robot,
        candidate_T_key_hand=(previous_relation if bounded_postlift else
                              nominal_relation),
        max_translation_drift_m=limits.max_grasp_translation_drift_m,
        max_rotation_drift_deg=limits.max_grasp_rotation_drift_deg)
    manifest_hash = hashlib.sha256(
        (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest()
    withdrawal_hash = hashlib.sha256(withdrawal_file.read_bytes()).hexdigest()
    postlift_hash = hashlib.sha256(postlift_file.read_bytes()).hexdigest()

    def result(status: str, assessment=None, preflight=None):
        return RetrySessionResult(
            status, attempt.attempt_id, attempt.candidate_id,
            held_key_observation.capture_id, evidence_dir, manifest_hash,
            withdrawal_time, withdrawal_file, withdrawal_hash,
            postlift_file, postlift_hash, verified,
            joint_sample, relation, assessment, preflight)

    if (relation.translation_drift_m >
            limits.max_grasp_translation_drift_m or
            relation.rotation_drift_deg >
            limits.max_grasp_rotation_drift_deg):
        return result("observed_grasp_relation_drift_exceeded")
    assessment = assess_xy_retry(
        shared_root=root, mode=mode, calibration=calibration,
        catalog=catalog, candidate_key=trial.selected_candidate_key,
        tabletop_pose_stem=attempt.tabletop_pose_stem,
        current_offset_socket_m=attempt.xy_offset_socket_m,
        observed_T_key_hand=relation.T_key_hand,
        held_hand_q_measured=joint_sample.full_q[7:],
        observed_key_hand_source="multiview_key_pose_plus_live_wrist",
        max_grasp_translation_drift_m=(
            limits.max_grasp_translation_drift_m),
        max_grasp_rotation_drift_deg=limits.max_grasp_rotation_drift_deg,
        failed_insertion_observed=True,
        guarded_withdrawal_complete=True, grasp_held=True, hard_abort=False,
        frames=frames, intrinsics_full=intrinsics_full,
        extrinsics_full=extrinsics_full,
        frame_timestamp_source="camera_acquisition",
        frame_request_id=frame_request_id, frame_ids=frame_ids,
        acquisition_metadata=acquisition_metadata, backend=backend,
        max_total_offset_m=limits.max_total_offset_m,
        minimum_anchor_separation_px=(
            limits.minimum_anchor_separation_px),
        crop_width_px=limits.crop_width_px,
        decision_timestamp_s=decision_timestamp_s,
        max_frame_age_s=limits.max_frame_age_s,
        max_capture_skew_s=limits.max_capture_skew_s)
    if assessment.status != "proposal_requires_live_preflight":
        return result(assessment.status, assessment)
    preflight = plan_xy_retry_from_withdrawn_hold(
        planner=planner, assessment=assessment, calibration=calibration,
        catalog=catalog, mode=mode, shared_root=root,
        candidate_key=trial.selected_candidate_key,
        tabletop_pose_stem=attempt.tabletop_pose_stem,
        observed_T_key_hand=relation.T_key_hand,
        observed_relation_timestamp_s=(
            held_key_observation.selected_acquisition_timestamp_s),
        live_start_q=joint_sample.full_q,
        start_q_timestamp_s=joint_sample.sample_timestamp_s,
        max_state_skew_s=limits.max_state_skew_s,
        held_hand_source="measured", trial_scene=trial.trial_scene,
        limits=limits.path_limits,
        axial_waypoint_step_m=limits.axial_waypoint_step_m)
    return result(
        "ready_to_record_pending_retry" if preflight.status ==
        "sampled_retry_preflight_pass" else "retry_preflight_failed",
        assessment, preflight)


def _validated_unobserved_retry_capture(
    *, mode: TaskMode, shared_root: Path, calibration, catalog: dict,
    trial, attempt: AttemptRecord, joint_sample: LiveRobotState,
    frames: Sequence[LabeledFrame], intrinsics_full: Mapping,
    extrinsics_full: Mapping, frame_request_id: int,
    frame_ids: Mapping[str, int], acquisition_metadata: Mapping,
    withdrawal_completed_at_s: float,
    withdrawal_evidence_path: Path, postlift_preflight_report_path: Path,
    decision_timestamp_s: float, limits: RetrySessionLimits,
) -> tuple[Path, Path, Path, dict]:
    """Common failed trial, measured state, camera and withdrawal gate."""
    limits.validate()
    insertion, _postlift, postlift_file, root = _validated_retry_trial_context(
        mode=mode, shared_root=shared_root, calibration=calibration,
        catalog=catalog, trial=trial, attempt=attempt,
        postlift_preflight_report_path=postlift_preflight_report_path)
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("retry requires measured Franka and Inspire joints")
    joint_sample.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    if (not math.isfinite(float(decision_timestamp_s)) or
            decision_timestamp_s < joint_sample.sample_timestamp_s):
        raise ValueError("retry decision cannot precede measured robot feedback")
    if (not frames or len({frame.camera_id for frame in frames}) != len(frames)
            or any(frame.phase != "preinsert_hold" or
                   not isinstance(frame.image, Image.Image) or
                   frame.image.mode != "RGB" for frame in frames)):
        raise ValueError("retry needs unique full-frame RGB preinsert views")
    images_bgr = {frame.camera_id: np.asarray(
        frame.image, dtype=np.uint8)[:, :, ::-1].copy() for frame in frames}
    verified = verify_frame_provenance(
        acquisition_metadata, request_id=frame_request_id,
        images_bgr=images_bgr, frame_ids=frame_ids)
    minimum_views = 1 if mode.family == "cylinder" else 2
    if (len(verified) < minimum_views or
            any(frame.timestamp_s != verified[frame.camera_id]["timestamp_s"]
                for frame in frames)):
        raise ValueError(
            "diagnostic needs bound same-request camera frames")
    exposure_begin = min(row["timestamp_s"] - row["max_error_s"]
                         for row in verified.values())
    exposure_end = max(row["timestamp_s"] + row["max_error_s"]
                       for row in verified.values())
    if (joint_sample.sample_timestamp_s <
            exposure_begin - limits.max_state_skew_s or
            joint_sample.sample_timestamp_s >
            exposure_end + limits.max_state_skew_s):
        raise ValueError("measured hand state is not synchronized with retry frames")
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={serial: row["K_undist"]
                           for serial, row in intrinsics_full.items()},
        extrinsics_full=extrinsics_full,
        calibrated_camera_ids=set(intrinsics_full))
    withdrawal_time, withdrawal_file = _validated_retry_withdrawal(
        attempt=attempt, insertion=insertion,
        withdrawal_completed_at_s=withdrawal_completed_at_s,
        withdrawal_evidence_path=withdrawal_evidence_path,
        verified_frames=verified, joint_sample=joint_sample)
    if (decision_timestamp_s - exposure_begin > limits.max_frame_age_s or
            exposure_end - exposure_begin > limits.max_capture_skew_s or
            exposure_end > decision_timestamp_s + limits.max_capture_skew_s):
        raise ValueError("retry camera capture is stale or asynchronous")
    return root, postlift_file, withdrawal_file, verified


def assess_unobserved_xy_diagnostic(
    *, mode: TaskMode, shared_root: Path, calibration, catalog: dict,
    trial, attempt: AttemptRecord, joint_sample: LiveRobotState,
    frames: Sequence[LabeledFrame], intrinsics_full: Mapping,
    extrinsics_full: Mapping, frame_request_id: int,
    frame_ids: Mapping[str, int], acquisition_metadata: Mapping,
    backend: ImageVLM, withdrawal_completed_at_s: float,
    withdrawal_evidence_path: Path, postlift_preflight_report_path: Path,
    decision_timestamp_s: float, limits: RetrySessionLimits,
    screen: Callable = screen_grasp_endpoint,
) -> UnobservedXYDiagnostic:
    """Bind a raw-camera XY direction to a failed attempt without key pose.

    This deliberately does not call ``plan_xy_retry_from_withdrawn_hold``:
    nominal key/socket collision says nothing decisive about a squeezed key's
    true lateral shift. The saved direction is not a pending robot retry.
    """
    root, postlift_file, withdrawal_file, verified = (
        _validated_unobserved_retry_capture(
            mode=mode, shared_root=shared_root, calibration=calibration,
            catalog=catalog, trial=trial, attempt=attempt,
            joint_sample=joint_sample, frames=frames,
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full, frame_request_id=frame_request_id,
            frame_ids=frame_ids, acquisition_metadata=acquisition_metadata,
            withdrawal_completed_at_s=withdrawal_completed_at_s,
            withdrawal_evidence_path=withdrawal_evidence_path,
            postlift_preflight_report_path=postlift_preflight_report_path,
            decision_timestamp_s=decision_timestamp_s, limits=limits))
    assessment = assess_xy_retry(
        shared_root=root, mode=mode, calibration=calibration,
        catalog=catalog, candidate_key=trial.selected_candidate_key,
        tabletop_pose_stem=attempt.tabletop_pose_stem,
        current_offset_socket_m=attempt.xy_offset_socket_m,
        observed_T_key_hand=None,
        held_hand_q_measured=joint_sample.full_q[7:],
        observed_key_hand_source="v8_nominal_unobserved_key",
        max_grasp_translation_drift_m=(
            limits.max_grasp_translation_drift_m),
        max_grasp_rotation_drift_deg=limits.max_grasp_rotation_drift_deg,
        failed_insertion_observed=True,
        guarded_withdrawal_complete=True, grasp_held=True, hard_abort=False,
        frames=frames, intrinsics_full=dict(intrinsics_full),
        extrinsics_full=dict(extrinsics_full),
        frame_timestamp_source="camera_acquisition",
        frame_request_id=frame_request_id, frame_ids=frame_ids,
        acquisition_metadata=acquisition_metadata, backend=backend,
        max_total_offset_m=limits.max_total_offset_m,
        minimum_anchor_separation_px=limits.minimum_anchor_separation_px,
        crop_width_px=limits.crop_width_px,
        decision_timestamp_s=decision_timestamp_s,
        max_frame_age_s=limits.max_frame_age_s,
        max_capture_skew_s=limits.max_capture_skew_s, screen=screen)
    if assessment.status == "proposal_requires_live_preflight":
        raise RuntimeError("unobserved diagnostic cannot authorize a retry")
    return UnobservedXYDiagnostic(
        assessment.status, attempt.attempt_id, attempt.candidate_id,
        float(withdrawal_completed_at_s), withdrawal_file,
        hashlib.sha256(withdrawal_file.read_bytes()).hexdigest(),
        postlift_file, hashlib.sha256(postlift_file.read_bytes()).hexdigest(),
        verified, joint_sample, assessment)


def _compute_grounded_alignment(
    *, mode: TaskMode, root: Path, calibration, postlift_file: Path,
    frames: Sequence[LabeledFrame], intrinsics_full: Mapping,
    extrinsics_full: Mapping, backend: ImageVLM,
    alignment_limits: AlignmentLimits,
    axis_reference_key_z_m: float | None,
    roi_xyxy_by_camera: Mapping[str, tuple[int, int, int, int]] | None,
    square_translation_only: bool, search_index: int,
    max_nominal_tip_height_error_m: float,
) -> tuple[str, dict, tuple]:
    """Shared raw-pixel geometry for live-withdrawal and saved-snapshot use."""
    geometry_path = AssetPaths(root, mode).task_geometry
    geometry_bytes = geometry_path.read_bytes()
    geometry = json.loads(geometry_bytes)
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    reference_z = None
    if mode.family != "cylinder":
        if axis_reference_key_z_m is None:
            reference_z = (geometry["key_frame"].get("handle_front_z_m")
                           if square_translation_only else 0.0)
        else:
            reference_z = float(axis_reference_key_z_m)
        if (reference_z is None or not math.isfinite(reference_z) or
                reference_z < 0 or reference_z >= tip_z):
            raise ValueError("axis reference must be behind the CAD tip")
        if square_translation_only and reference_z <= 0:
            raise ValueError("square shoulder needs a positive CAD z coordinate")
    if (float(geometry.get("verification_depth_m", mode.target_depth_m)) !=
            mode.target_depth_m or
            geometry["insertion_direction_socket"] != [0.0, 0.0, -1.0]):
        raise ValueError("CAD depth/axis differs from configured insertion task")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    selected_cameras = set(assessment_cameras(
        frame.camera_id for frame in frames))
    calibrated = [CalibratedXYFrame(
        frame.camera_id, frame.timestamp_s, frame.image,
        camera_socket_transform(
            T_camera_world=extrinsics_full[frame.camera_id],
            T_world_robot=c2r, T_robot_socket=socket),
        np.asarray(intrinsics_full[frame.camera_id]["K_undist"], dtype=float),
    ) for frame in frames if frame.camera_id in selected_cameras]
    if mode.yaw_relevant:
        if square_translation_only:
            grounded, observations = observe_grounded_key_axis(
                backend, calibrated, reference_kind="square_shoulder")
            alignment = estimate_grounded_alignment(
                grounded, landmark_spacing_m=tip_z - reference_z,
                socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
                verification_depth_m=mode.target_depth_m,
                limits=alignment_limits)
            alignment = dict(
                alignment, square_translation_only=True,
                square_yaw_observed=False,
                square_yaw_requires_independent_evidence=True,
                axis_reference_feature="insertion_shaft_handle_shoulder",
                scope="read_only_xy_axis_diagnostic_square_yaw_unobserved")
        else:
            observations = ()
            alignment = estimate_grounded_alignment(
                [], landmark_spacing_m=tip_z - reference_z,
                socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
                verification_depth_m=mode.target_depth_m,
                limits=alignment_limits, yaw_relevant=True)
    else:
        tip_socket, roi_xyxy_by_camera = _cylinder_tip_and_rois(
            postlift_file=postlift_file, socket=socket, tip_z_m=tip_z,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            frames=calibrated, backend=backend,
            roi_xyxy_by_camera=roi_xyxy_by_camera)
        if roi_xyxy_by_camera is None:
            grounded, observations = observe_grounded_cylinder_axis(
                backend, calibrated)
        else:
            grounded, observations = observe_grounded_cylinder_axis_cropped(
                backend, calibrated,
                roi_xyxy_by_camera={camera: roi_xyxy_by_camera[camera]
                                    for camera in selected_cameras})
            for frame, observation in zip(calibrated, observations):
                verify_native_crop_observation(
                    observation.to_record(), full_image_size_px=frame.raw.size)
        alignment = estimate_grounded_line_alignment(
            grounded,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            verification_depth_m=mode.target_depth_m,
            limits=alignment_limits)
        alignment = resolve_xy_only_cylinder_alignment(
            grounded, alignment,
            nominal_tip_z_socket_m=float(tip_socket[2]),
            search_index=search_index,
            max_reprojection_px=alignment_limits.max_reprojection_px,
            min_parallax_deg=alignment_limits.min_parallax_deg,
            max_nominal_tip_height_error_m=max_nominal_tip_height_error_m)
    alignment = dict(
        alignment, estimator_limits=asdict(alignment_limits),
        grounding_input=("lossless_raw_roi" if roi_xyxy_by_camera is not None
                         and not mode.yaw_relevant else
                         "full_undistorted_frames"),
        key_object=mode.key_object, socket_object=mode.socket_object,
        socket_entry_plane_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m,
        axis_reference_key_z_m=(reference_z if mode.family != "cylinder"
                                else None))
    return hashlib.sha256(geometry_bytes).hexdigest(), alignment, tuple(observations)


def _compute_tracked_alignment(
    *, mode: TaskMode, root: Path, calibration, postlift_file: Path,
    frames: Sequence[LabeledFrame], frame_rows: Mapping,
    reference_dir: Path, intrinsics_full: Mapping,
    extrinsics_full: Mapping, alignment_limits: AlignmentLimits,
    search_index: int, max_nominal_tip_height_error_m: float,
) -> tuple[str, dict]:
    """Use saved key appearance and raw preinsert frames; never call Qwen."""
    geometry_bytes = AssetPaths(root, mode).task_geometry.read_bytes()
    geometry = json.loads(geometry_bytes)
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    postlift = json.loads(postlift_file.read_text(encoding="utf-8"))
    key = validate_se3(postlift["targets"]["T_robot_key_preinsert"],
                       name="commanded preinsert key target")
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    nominal_tip = np.linalg.inv(socket) @ key @ np.array([0., 0., tip_z, 1.])
    reference = Path(reference_dir).expanduser().resolve()
    if not (reference / "initial_tabletop" / "manifest.json").is_file():
        raise ValueError("tracked tip needs this trial's initial key images")
    views, records = [], []
    for frame in frames:
        camera = frame.camera_id
        image = cv2.imread(str(reference / "initial_tabletop" / "images" /
                               f"{camera}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(reference / "masks" / f"{camera}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            records.append({"camera_id": camera, "frame_id":
                            frame_rows[camera]["frame_id"],
                            "admitted": False, "reason": "key_reference_unavailable"})
            continue
        T_camera_socket = camera_socket_transform(
            T_camera_world=extrinsics_full[camera],
            T_world_robot=c2r, T_robot_socket=socket)
        K = np.asarray(intrinsics_full[camera]["K_undist"], dtype=float)

        def project(z: float) -> np.ndarray:
            h = K @ (np.asarray(extrinsics_full[camera]) @ c2r @ key @
                     np.array([0., 0., z, 1.]))[:3]
            return h[:2] / h[2]

        bgr = cv2.cvtColor(np.asarray(frame.image), cv2.COLOR_RGB2BGR)
        measured, record = tracked_view(
            camera_id=camera, timestamp_s=frame.timestamp_s,
            frame_id=frame_rows[camera]["frame_id"],
            preinsert_bgr=bgr, reference_bgr=image,
            reference_mask=mask, T_camera_socket=T_camera_socket,
            intrinsics=K, projected_rear_px=project(0.),
            projected_tip_px=project(tip_z))
        record["image_sha256"] = frame_rows[camera]["image_sha256"]
        records.append(record)
        if measured is not None:
            views.append(measured)
    alignment = tracked_tip_xy_alignment(
        views, nominal_tip_z_socket_m=float(nominal_tip[2]),
        search_index=search_index,
        max_reprojection_px=alignment_limits.max_reprojection_px,
        min_parallax_deg=alignment_limits.min_parallax_deg,
        max_nominal_tip_height_error_m=max_nominal_tip_height_error_m)
    alignment.update(
        tracker_views=records,
        key_reference_dir=str(reference),
        grounding_input="saved_undistorted_frames_and_initial_key_masks",
        socket_entry_plane_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m,
        key_object=mode.key_object, socket_object=mode.socket_object)
    return hashlib.sha256(geometry_bytes).hexdigest(), alignment


def assess_saved_preinsert_xy_diagnostic(
    *, mode: TaskMode, shared_root: Path, calibration, catalog: dict,
    trial, attempt: AttemptRecord, preinsert_checkpoint_path: Path,
    postlift_preflight_report_path: Path, backend: ImageVLM,
    limits: RetrySessionLimits, alignment_limits: AlignmentLimits,
    axis_reference_key_z_m: float | None = None,
    roi_xyxy_by_camera: Mapping[str, tuple[int, int, int, int]] | None = None,
    square_translation_only: bool = False, search_index: int = 0,
    xy_estimator: str = "vlm", key_reference_dir: Path | None = None,
) -> PreinsertXYDiagnostic:
    """Compute next-attempt XY from the raw snapshot saved before insertion."""
    limits.validate()
    if type(square_translation_only) is not bool or (
            square_translation_only and mode.family != "square"):
        raise ValueError("square translation-only grounding needs square mode")
    if xy_estimator not in {"vlm", "tracked_tip"} or (
            xy_estimator == "tracked_tip" and
            (mode.family != "cylinder" or key_reference_dir is None)):
        raise ValueError("tracked tip needs cylinder mode and initial key evidence")
    root = Path(shared_root).expanduser().resolve()
    _insertion, _postlift, postlift_file, validated_root = (
        _validated_retry_trial_context(
            mode=mode, shared_root=root, calibration=calibration,
            catalog=catalog, trial=trial, attempt=attempt,
            postlift_preflight_report_path=postlift_preflight_report_path))
    if validated_root != root:
        raise ValueError("saved preinsert diagnostic changed shared root")
    checkpoint_path = Path(preinsert_checkpoint_path).expanduser().resolve()
    report = verify_preinsert_checkpoint(checkpoint_path)
    if (report.get("attempt_id") != attempt.attempt_id or
            report.get("candidate_id") != attempt.candidate_id or
            report.get("preinsert_reached") is not True or
            Path(report["postlift_report_path"]).resolve() != postlift_file):
        raise ValueError("saved preinsert snapshot belongs to another attempt")
    bundle = Path(report["raw_bundle"]).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="preinsert")
    joint_sample = _state_from_record(report["joint_feedback"])
    joint_sample.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    frame_rows = capture["frame_evidence"]
    selected_cameras = (sorted(frame_rows) if xy_estimator == "tracked_tip"
                        else assessment_cameras(frame_rows))
    frames = []
    for camera in selected_cameras:
        row = frame_rows[camera]
        image = cv2.imread(str(bundle / "images" / f"{camera}.png"),
                           cv2.IMREAD_COLOR)
        if image is None or image_sha256(image) != row["image_sha256"]:
            raise ValueError("saved preinsert raw image changed")
        frames.append(LabeledFrame(
            camera, "preinsert_hold", row["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
    selected_rows = [frame_rows[camera] for camera in selected_cameras]
    if (len(frames) < (1 if mode.family == "cylinder" else 2) or
            max(row["timestamp_s"] + row["max_error_s"]
                for row in selected_rows) -
            min(row["timestamp_s"] - row["max_error_s"]
                for row in selected_rows) > limits.max_capture_skew_s):
        raise ValueError("saved preinsert cameras are insufficient or asynchronous")
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={serial: row["K_undist"] for serial, row in
                           calibration.record["camera_calibration"]
                           ["intrinsics_full"].items()},
        extrinsics_full=calibration.record["camera_calibration"]
        ["extrinsics_full"],
        calibrated_camera_ids=set(calibration.record["camera_calibration"]
                                  ["intrinsics_full"]))
    snapshot = calibration.record["camera_calibration"]
    if xy_estimator == "tracked_tip":
        with timed("xy_retry.tracked_tip_alignment", phase="xy_retry",
                   cameras=len(frames)):
            geometry_hash, alignment = _compute_tracked_alignment(
                mode=mode, root=root, calibration=calibration,
                postlift_file=postlift_file, frames=frames,
                frame_rows=frame_rows, reference_dir=key_reference_dir,
                intrinsics_full=snapshot["intrinsics_full"],
                extrinsics_full=snapshot["extrinsics_full"],
                alignment_limits=alignment_limits,
                search_index=search_index,
                max_nominal_tip_height_error_m=(
                    limits.max_grasp_translation_drift_m))
        observations = ()
    else:
        geometry_hash, alignment, observations = _compute_grounded_alignment(
            mode=mode, root=root, calibration=calibration,
            postlift_file=postlift_file, frames=frames,
            intrinsics_full=snapshot["intrinsics_full"],
            extrinsics_full=snapshot["extrinsics_full"], backend=backend,
            alignment_limits=alignment_limits,
            axis_reference_key_z_m=axis_reference_key_z_m,
            roi_xyxy_by_camera=roi_xyxy_by_camera,
            square_translation_only=square_translation_only,
            search_index=search_index,
            max_nominal_tip_height_error_m=limits.max_grasp_translation_drift_m)
    return PreinsertXYDiagnostic(
        alignment["status"], attempt.attempt_id, attempt.candidate_id,
        attempt.session_calibration_sha256,
        calibration.record["camera_calibration_sha256"], geometry_hash,
        capture["request_id"], checkpoint_path,
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), bundle,
        hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        postlift_file, hashlib.sha256(postlift_file.read_bytes()).hexdigest(),
        frame_rows, joint_sample, alignment, observations)


def assess_grounded_xy_diagnostic(
    *, mode: TaskMode, shared_root: Path, calibration, catalog: dict,
    trial, attempt: AttemptRecord, joint_sample: LiveRobotState,
    frames: Sequence[LabeledFrame], intrinsics_full: Mapping,
    extrinsics_full: Mapping, frame_request_id: int,
    frame_ids: Mapping[str, int], acquisition_metadata: Mapping,
    backend: ImageVLM, withdrawal_completed_at_s: float,
    withdrawal_evidence_path: Path, postlift_preflight_report_path: Path,
    decision_timestamp_s: float, limits: RetrySessionLimits,
    alignment_limits: AlignmentLimits,
    axis_reference_key_z_m: float | None = None,
    roi_xyxy_by_camera: Mapping[str, tuple[int, int, int, int]] | None = None,
    square_translation_only: bool = False,
    search_index: int = 0,
) -> GroundedXYDiagnostic:
    """Observe raw key landmarks after a failed trial, never command XY.

    The cylinder route uses a tip point and the visible shaft axis line; its
    line endpoints need no cross-view point correspondence. An opt-in square
    diagnostic triangulates tip and CAD shoulder centres to estimate *only*
    XY/axis tilt. Square yaw remains unobserved and this diagnostic cannot
    authorize a retry or insertion. The default square route still abstains.
    """
    if type(square_translation_only) is not bool or (
            square_translation_only and mode.family != "square"):
        raise ValueError("square translation-only grounding needs square mode")
    if roi_xyxy_by_camera is not None and mode.yaw_relevant:
        raise ValueError("two-landmark ROI grounding cannot determine square-key yaw")
    root, postlift_file, withdrawal_file, verified = (
        _validated_unobserved_retry_capture(
            mode=mode, shared_root=shared_root, calibration=calibration,
            catalog=catalog, trial=trial, attempt=attempt,
            joint_sample=joint_sample, frames=frames,
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full, frame_request_id=frame_request_id,
            frame_ids=frame_ids, acquisition_metadata=acquisition_metadata,
            withdrawal_completed_at_s=withdrawal_completed_at_s,
            withdrawal_evidence_path=withdrawal_evidence_path,
            postlift_preflight_report_path=postlift_preflight_report_path,
            decision_timestamp_s=decision_timestamp_s, limits=limits))
    geometry_path = AssetPaths(root, mode).task_geometry
    geometry_bytes = geometry_path.read_bytes()
    geometry = json.loads(geometry_bytes)
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    if mode.family != "cylinder":
        if axis_reference_key_z_m is None:
            reference_z = (geometry["key_frame"].get("handle_front_z_m")
                           if square_translation_only else 0.0)
        else:
            reference_z = float(axis_reference_key_z_m)
        if (reference_z is None or not math.isfinite(reference_z) or
                reference_z < 0 or
                reference_z >= tip_z):
            raise ValueError("axis reference must be behind the CAD tip")
        if square_translation_only and reference_z <= 0:
            raise ValueError("square shoulder needs a positive CAD z coordinate")
    if (float(geometry.get("verification_depth_m", mode.target_depth_m)) !=
            mode.target_depth_m or
            geometry["insertion_direction_socket"] != [0.0, 0.0, -1.0]):
        raise ValueError("CAD depth/axis differs from configured insertion task")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    selected_cameras = set(assessment_cameras(
        frame.camera_id for frame in frames))
    calibrated = [CalibratedXYFrame(
        frame.camera_id, frame.timestamp_s, frame.image,
        camera_socket_transform(
            T_camera_world=extrinsics_full[frame.camera_id],
            T_world_robot=c2r, T_robot_socket=socket),
        np.asarray(intrinsics_full[frame.camera_id]["K_undist"], dtype=float),
    ) for frame in frames if frame.camera_id in selected_cameras]
    if mode.yaw_relevant:
        if square_translation_only:
            grounded, observations = observe_grounded_key_axis(
                backend, calibrated, reference_kind="square_shoulder")
            alignment = estimate_grounded_alignment(
                grounded, landmark_spacing_m=tip_z - reference_z,
                socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
                verification_depth_m=mode.target_depth_m,
                limits=alignment_limits)
            alignment = dict(
                alignment, square_translation_only=True,
                square_yaw_observed=False,
                square_yaw_requires_independent_evidence=True,
                axis_reference_feature="insertion_shaft_handle_shoulder",
                scope="read_only_xy_axis_diagnostic_square_yaw_unobserved")
        else:
            # Two collinear points/one axial line leave square-key yaw
            # unobservable. Do not spend an API call on a guaranteed abstention.
            observations = ()
            alignment = estimate_grounded_alignment(
                [], landmark_spacing_m=tip_z - reference_z,
                socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
                verification_depth_m=mode.target_depth_m,
                limits=alignment_limits, yaw_relevant=True)
    else:
        tip_socket, roi_xyxy_by_camera = _cylinder_tip_and_rois(
            postlift_file=postlift_file, socket=socket, tip_z_m=tip_z,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            frames=calibrated, backend=backend,
            roi_xyxy_by_camera=roi_xyxy_by_camera)
        if roi_xyxy_by_camera is None:
            grounded, observations = observe_grounded_cylinder_axis(
                backend, calibrated)
        else:
            grounded, observations = observe_grounded_cylinder_axis_cropped(
                backend, calibrated,
                roi_xyxy_by_camera={camera: roi_xyxy_by_camera[camera]
                                    for camera in selected_cameras})
            for frame, observation in zip(calibrated, observations):
                verify_native_crop_observation(
                    observation.to_record(),
                    full_image_size_px=frame.raw.size)
        alignment = estimate_grounded_line_alignment(
            grounded,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            verification_depth_m=mode.target_depth_m,
            limits=alignment_limits)
        alignment = resolve_xy_only_cylinder_alignment(
            grounded, alignment,
            nominal_tip_z_socket_m=float(tip_socket[2]),
            search_index=search_index,
            max_reprojection_px=alignment_limits.max_reprojection_px,
            min_parallax_deg=alignment_limits.min_parallax_deg,
            max_nominal_tip_height_error_m=limits.max_grasp_translation_drift_m)
    # The session runner applies the configured weight to the next attempt.
    alignment = dict(
        alignment, estimator_limits=asdict(alignment_limits),
        grounding_input=("lossless_raw_roi" if roi_xyxy_by_camera is not None
                         and not mode.yaw_relevant else
                         "full_undistorted_frames"),
        key_object=mode.key_object, socket_object=mode.socket_object,
        socket_entry_plane_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m,
        axis_reference_key_z_m=(reference_z if mode.family != "cylinder"
                                else None))
    return GroundedXYDiagnostic(
        alignment["status"], attempt.attempt_id, attempt.candidate_id,
        attempt.session_calibration_sha256,
        calibration.record["camera_calibration_sha256"],
        hashlib.sha256(geometry_bytes).hexdigest(), frame_request_id,
        withdrawal_file, hashlib.sha256(withdrawal_file.read_bytes()).hexdigest(),
        postlift_file, hashlib.sha256(postlift_file.read_bytes()).hexdigest(),
        verified, joint_sample, alignment, tuple(observations))


def write_retry_session_artifacts(
    result: RetrySessionResult | UnobservedXYDiagnostic | GroundedXYDiagnostic |
    PreinsertXYDiagnostic,
    frames: Sequence[LabeledFrame],
    output_dir: Path,
) -> Path:
    """Save complete source frames and every VLM/preflight artifact once."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    if {frame.camera_id for frame in frames} != set(result.frame_binding):
        raise ValueError("saved retry frames differ from verified VLM frames")
    files = {}
    frame_dir = target / "frames"
    frame_dir.mkdir()
    for frame in frames:
        camera = _safe_id(frame.camera_id, "retry camera ID")
        path = frame_dir / f"{camera}.png"
        frame.image.save(path, format="PNG")
        saved_bgr = np.asarray(Image.open(path).convert("RGB"),
                               dtype=np.uint8)[:, :, ::-1].copy()
        if image_sha256(saved_bgr) != result.frame_binding[camera]["image_sha256"]:
            raise ValueError("saved retry image differs from VLM source pixels")
        files[str(path.relative_to(target))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    if (not isinstance(result, (GroundedXYDiagnostic, PreinsertXYDiagnostic)) and
            result.assessment is not None and result.assessment.overlays is not None):
        overlay_dir = target / "overlays"
        overlay_dir.mkdir()
        for view in result.assessment.overlays.views:
            camera = _safe_id(view.camera_id, "overlay camera ID")
            for name, image in (("raw_crop", view.raw),
                                ("choices", view.overlay)):
                path = overlay_dir / f"{camera}_{name}.png"
                image.save(path, format="PNG")
                files[str(path.relative_to(target))] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
    if (not isinstance(result, (GroundedXYDiagnostic, PreinsertXYDiagnostic)) and
            result.preflight is not None):
        preflight_dir = write_xy_retry_preflight(
            result.preflight, target / "preflight")
        for path in preflight_dir.rglob("*"):
            if path.is_file():
                files[str(path.relative_to(target))] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
    report = result.to_record()
    report["artifacts_sha256"] = files
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def write_saved_preinsert_xy_artifacts(
    result: PreinsertXYDiagnostic, output_dir: Path,
) -> Path:
    """Persist a diagnostic by reloading its already verified raw bundle."""
    if not isinstance(result, PreinsertXYDiagnostic):
        raise TypeError("expected a saved preinsert XY diagnostic")
    frames = []
    for camera, row in sorted(result.frame_binding.items()):
        image = cv2.imread(str(result.preinsert_bundle_path / "images" /
                               f"{camera}.png"), cv2.IMREAD_COLOR)
        if image is None or image_sha256(image) != row["image_sha256"]:
            raise ValueError("saved preinsert image changed before artifact write")
        frames.append(LabeledFrame(
            camera, "preinsert_hold", row["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
    return write_retry_session_artifacts(result, frames, output_dir)
