"""Bind one observed insertion failure to a read-only 1 mm VLM retry.

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

import numpy as np
from PIL import Image

from .calibration import validate_session_camera_calibration
from .assets import AssetPaths
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .frame_provenance import image_sha256, verify_frame_provenance
from .geometry import validate_se3
from .grounded_alignment import (
    AlignmentLimits, estimate_grounded_alignment,
    estimate_grounded_line_alignment,
    observe_grounded_cylinder_axis,
)
from .held_relation import HeldRelation, resolve_postlift_held_relation
from .key_perception import KeyPoseObservation, verify_key_capture_artifacts
from .live_robot_state import LiveRobotState
from .observer import ImageVLM, LabeledFrame
from .path_audit import PathAuditLimits
from .records import AttemptRecord
from .retry_preflight import (
    XYRetryPreflight, plan_xy_retry_from_withdrawn_hold,
    write_xy_retry_preflight,
)
from .session_bootstrap import _safe_id
from .xy_retry import XYRetryAssessment, assess_xy_retry
from .xy_overlay import CalibratedXYFrame, camera_socket_transform
from .world import validated_frozen_socket_pose


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrySessionLimits:
    max_state_skew_s: float
    max_arm_hand_skew_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float
    max_grasp_translation_drift_m: float
    max_grasp_rotation_drift_deg: float
    max_total_offset_m: float
    minimum_anchor_separation_px: float
    crop_width_px: int
    max_frame_age_s: float
    max_capture_skew_s: float
    axial_waypoint_step_m: float
    path_limits: PathAuditLimits

    def validate(self) -> None:
        values = (
            self.max_state_skew_s, self.max_arm_hand_skew_s,
            self.max_hand_command_error_raw, self.max_arm_velocity_rad_s,
            self.max_grasp_translation_drift_m,
            self.max_grasp_rotation_drift_deg, self.max_total_offset_m,
            self.minimum_anchor_separation_px, self.max_frame_age_s,
            self.max_capture_skew_s, self.axial_waypoint_step_m)
        if (not all(math.isfinite(float(v)) and float(v) > 0 for v in values) or
                type(self.crop_width_px) is not int or self.crop_width_px < 32):
            raise ValueError("retry limits must be finite and commissioned")
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
    """Metric visual hypothesis, with no retry preflight or pending motion."""

    status: str
    attempt_id: str
    candidate_id: str
    session_calibration_sha256: str
    camera_calibration_sha256: str
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
            "schema": "precision_insertion_grounded_xy_diagnostic_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "session_calibration_sha256": self.session_calibration_sha256,
            "camera_calibration_sha256": self.camera_calibration_sha256,
            "withdrawal_evidence_path": str(self.withdrawal_evidence_path),
            "withdrawal_evidence_sha256": self.withdrawal_evidence_sha256,
            "postlift_preflight_path": str(self.postlift_preflight_path),
            "postlift_preflight_sha256": self.postlift_preflight_sha256,
            "frame_binding": self.frame_binding,
            "joint_sample": self.joint_sample.to_record(),
            "alignment": self.alignment,
            "vlm_observations": [v.to_record() for v in self.vlm_observations],
            "preflight": None,
            "scope": "saved_metric_hypothesis_only_not_pending_retry_or_robot_motion",
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
    if (not isinstance(postlift, dict) or
            not isinstance(postlift.get("planning"), dict) or
            not isinstance(postlift.get("observed_held_relation"), dict) or
            postlift.get("schema") !=
            "precision_insertion_postlift_preflight_v1" or
            postlift.get("status") != "sampled_postlift_preflight_pass" or
            postlift.get("attempt_id") != attempt.attempt_id or
            postlift.get("candidate_key") != attempt.candidate_id.split("/") or
            postlift["planning"].get("sampled_planning_pass") is not True or
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
    previous_relation = validate_se3(
        postlift.get("observed_held_relation", {}).get("T_key_hand"),
        name="saved post-lift T_key_hand")
    prior = held_key_observation.consistency.get("held_pose_prior")
    if (not isinstance(prior, dict) or
            prior.get("source") !=
            "measured_wrist_plus_observed_held_relation" or
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
        candidate_T_key_hand=nominal_relation,
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
    if (len(verified) < 2 or
            any(frame.timestamp_s != verified[frame.camera_id]["timestamp_s"]
                for frame in frames)):
        raise ValueError("diagnostic needs two bound same-request camera frames")
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
) -> GroundedXYDiagnostic:
    """Observe raw key landmarks after a failed trial, never command XY.

    The cylinder route uses a tip point and the visible shaft axis line; its
    line endpoints need no cross-view point correspondence. The square route
    needs a third non-collinear feature to estimate yaw and therefore abstains.
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
    geometry = json.loads(AssetPaths(root, mode).task_geometry.read_text(
        encoding="utf-8"))
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    if mode.family != "cylinder":
        if axis_reference_key_z_m is None:
            # The square diagnostic still records why a two-point axis
            # estimate cannot determine the key's insertion yaw.
            reference_z = 0.0
        else:
            reference_z = float(axis_reference_key_z_m)
        if (not math.isfinite(reference_z) or reference_z < 0 or
                reference_z >= tip_z):
            raise ValueError("axis reference must be behind the CAD tip")
    if (float(geometry.get("verification_depth_m", mode.target_depth_m)) !=
            mode.target_depth_m or
            geometry["insertion_direction_socket"] != [0.0, 0.0, -1.0]):
        raise ValueError("CAD depth/axis differs from configured insertion task")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    calibrated = [CalibratedXYFrame(
        frame.camera_id, frame.timestamp_s, frame.image,
        camera_socket_transform(
            T_camera_world=extrinsics_full[frame.camera_id],
            T_world_robot=c2r, T_robot_socket=socket),
        np.asarray(intrinsics_full[frame.camera_id]["K_undist"], dtype=float),
    ) for frame in frames]
    if mode.yaw_relevant:
        # Two collinear points/one axial line leave square-key yaw
        # unobservable. Do not spend an API call on a guaranteed abstention.
        observations = ()
        alignment = estimate_grounded_alignment(
            [], landmark_spacing_m=tip_z - reference_z,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            verification_depth_m=mode.target_depth_m,
            limits=alignment_limits, yaw_relevant=True)
    else:
        grounded, observations = observe_grounded_cylinder_axis(
            backend, calibrated)
        alignment = estimate_grounded_line_alignment(
            grounded,
            socket_rim_z_m=float(geometry["socket_entry_plane_z_m"]),
            verification_depth_m=mode.target_depth_m,
            limits=alignment_limits)
    step = alignment["step_socket_m"]
    if step is not None:
        proposed = np.asarray(attempt.xy_offset_socket_m) + np.asarray(step)
        if np.linalg.norm(proposed) > limits.max_total_offset_m:
            alignment = dict(alignment, status="abstain",
                             reason="proposed_offset_exceeds_session_budget",
                             step_socket_m=None)
    alignment = dict(
        alignment, estimator_limits=asdict(alignment_limits),
        key_object=mode.key_object, socket_object=mode.socket_object,
        socket_entry_plane_z_m=float(geometry["socket_entry_plane_z_m"]),
        verification_depth_m=mode.target_depth_m,
        axis_reference_key_z_m=(reference_z if mode.family != "cylinder"
                                else None))
    return GroundedXYDiagnostic(
        alignment["status"], attempt.attempt_id, attempt.candidate_id,
        attempt.session_calibration_sha256,
        calibration.record["camera_calibration_sha256"],
        withdrawal_file, hashlib.sha256(withdrawal_file.read_bytes()).hexdigest(),
        postlift_file, hashlib.sha256(postlift_file.read_bytes()).hexdigest(),
        verified, joint_sample, alignment, tuple(observations))


def write_retry_session_artifacts(
    result: RetrySessionResult | UnobservedXYDiagnostic | GroundedXYDiagnostic,
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
    if (not isinstance(result, GroundedXYDiagnostic) and
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
    if (not isinstance(result, GroundedXYDiagnostic) and
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
