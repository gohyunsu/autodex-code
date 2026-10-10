"""Combine exact XY endpoint gates, camera projections and ZeroDex votes.

This prepares a *proposal* after a failed insertion and a completed guarded
withdrawal. It never drives the robot. A proposed offset still needs fresh
live Franka/held-key path planning and commissioned contact control.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .assets import AssetPaths
from .calibration import validate_session_camera_calibration
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .frame_provenance import bounded_capture_skew_s, verify_frame_provenance
from .geometry import pose_angle_deg, validate_se3
from .observer import ImageVLM, LabeledFrame, VLMObservation, observe_xy_views
from .xy_endpoint import screen_axis_1mm_endpoint_choices
from .xy_overlay import (
    CalibratedXYFrame, XYOverlayBatch, build_xy_candidate_overlays,
    camera_socket_transform,
)
from .xy_voting import (
    ChoiceDecision, XYChoice, resolve_multiview_choice,
)
from .world import validated_frozen_socket_pose


@dataclass(frozen=True)
class XYRetryAssessment:
    status: str
    endpoint_screen: dict | None
    overlays: XYOverlayBatch | None
    vlm_observations: tuple[VLMObservation, ...]
    decision: ChoiceDecision | None
    reason: str
    frame_binding: dict | None = None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_xy_retry_assessment_v1",
            "status": self.status,
            "reason": self.reason,
            "endpoint_screen": self.endpoint_screen,
            "overlays": None if self.overlays is None else self.overlays.to_record(),
            "vlm_observations": [item.to_record()
                                 for item in self.vlm_observations],
            "choice_decision": None if self.decision is None
                               else self.decision.to_record(),
            "frame_binding": self.frame_binding,
            "scope": "read_only_retry_assessment_not_robot_motion",
            "robot_ready": False,
        }


def assess_xy_retry(
    *, shared_root: Path, mode: TaskMode, calibration, catalog: dict,
    candidate_key: tuple[str, str, str], tabletop_pose_stem: str,
    current_offset_socket_m: tuple[float, float],
    observed_T_key_hand: np.ndarray,
    held_hand_q_measured: np.ndarray,
    observed_key_hand_source: str,
    max_grasp_translation_drift_m: float,
    max_grasp_rotation_drift_deg: float,
    failed_insertion_observed: bool, guarded_withdrawal_complete: bool,
    grasp_held: bool, hard_abort: bool,
    frames: Sequence[LabeledFrame], intrinsics_full: dict,
    extrinsics_full: dict, frame_timestamp_source: str,
    frame_request_id: int, frame_ids: Mapping[str, int],
    acquisition_metadata: Mapping,
    backend: ImageVLM, max_total_offset_m: float,
    minimum_anchor_separation_px: float, crop_width_px: int,
    decision_timestamp_s: float, max_frame_age_s: float,
    max_capture_skew_s: float,
    screen: Callable = screen_grasp_endpoint,
) -> XYRetryAssessment:
    """Assess one 1 mm correction without accepting a VLM-only robot command."""
    if failed_insertion_observed is not True:
        raise ValueError("XY retry requires an observed insertion failure")
    if frame_timestamp_source != "camera_acquisition":
        raise ValueError("XY voting needs acquisition-time camera frames")
    if hard_abort is not False or grasp_held is not True or (
            guarded_withdrawal_complete is not True):
        return XYRetryAssessment(
            "stop", None, None, (), None,
            "hard_abort_grasp_lost_or_guarded_withdrawal_unconfirmed")
    if not frames or len({frame.camera_id for frame in frames}) != len(frames):
        raise ValueError("retry needs unique synchronized camera frames")
    validate_session_camera_calibration(
        calibration,
        intrinsics_undist={camera_id: params["K_undist"]
                           for camera_id, params in intrinsics_full.items()},
        extrinsics_full=extrinsics_full,
        calibrated_camera_ids=set(intrinsics_full))
    if type(frame_request_id) is not int or frame_request_id <= 0:
        raise ValueError("retry needs a positive camera request ID")
    if (not math.isfinite(decision_timestamp_s) or
            not math.isfinite(max_capture_skew_s) or
            max_capture_skew_s <= 0 or not math.isfinite(max_frame_age_s) or
            max_frame_age_s <= 0):
        raise ValueError("decision and freshness limits must be finite and positive")
    rgb_images = {}
    for frame in frames:
        if frame.phase != "preinsert_hold":
            raise ValueError("XY retry frame must be from preinsert_hold")
        if not isinstance(frame.image, Image.Image) or frame.image.mode != "RGB":
            raise ValueError("XY retry needs unchanged full-frame RGB pixels")
        rgb_images[frame.camera_id] = np.asarray(
            frame.image, dtype=np.uint8)[:, :, ::-1].copy()
    verified_frames = verify_frame_provenance(
        acquisition_metadata, request_id=frame_request_id,
        images_bgr=rgb_images, frame_ids=frame_ids)
    if any(frame.timestamp_s !=
           verified_frames[frame.camera_id]["timestamp_s"]
           for frame in frames):
        raise ValueError("VLM frame timestamp differs from bound acquisition")
    frame_binding = {"request_id": frame_request_id,
                     "frames": verified_frames}
    if (bounded_capture_skew_s(verified_frames) > max_capture_skew_s or
            decision_timestamp_s - min(
                row["timestamp_s"] - row["max_error_s"]
                for row in verified_frames.values()) > max_frame_age_s or
            max(row["timestamp_s"] + row["max_error_s"]
                for row in verified_frames.values()) >
            decision_timestamp_s + max_capture_skew_s):
        return XYRetryAssessment(
            "visual_abstain", None, None, (), None,
            "camera_acquisition_skew_or_age_exceeds_bound", frame_binding)
    root = Path(shared_root).expanduser().resolve()
    if Path(catalog.get("shared_root", "")).expanduser().resolve() != root:
        raise ValueError("catalogue shared root differs from retry root")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if len(candidate_key) != 3:
        raise ValueError("v8 candidate key must have three components")
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=tabletop_pose_stem)
    if selected["status"] != "candidates_available":
        raise ValueError(f"candidate catalogue unavailable: {selected['status']}")
    matches = [row for row in selected["candidates"]
               if tuple(row["key"]) == tuple(candidate_key)]
    if len(matches) != 1:
        raise ValueError("retry grasp is not endpoint eligible for this tabletop")
    if observed_key_hand_source != "multiview_key_pose_plus_live_wrist":
        raise ValueError("retry requires independently observed key/hand relation")
    hand_q = np.asarray(held_hand_q_measured, dtype=np.float64)
    if hand_q.shape != (6,) or not np.all(np.isfinite(hand_q)):
        raise ValueError("retry needs six measured Inspire held joints")
    if (not math.isfinite(max_grasp_translation_drift_m) or
            max_grasp_translation_drift_m <= 0 or
            not math.isfinite(max_grasp_rotation_drift_deg) or
            max_grasp_rotation_drift_deg <= 0):
        raise ValueError("grasp-relation drift limits must be commissioned")
    candidate_dir = Path(matches[0]["candidate_dir"])
    observed_relation = validate_se3(
        observed_T_key_hand, name="observed post-lift T_key_hand")
    nominal_relation = validate_se3(np.load(
        candidate_dir / "wrist_se3.npy", allow_pickle=False),
        name="candidate T_key_hand")
    translation_drift = float(np.linalg.norm(
        observed_relation[:3, 3] - nominal_relation[:3, 3]))
    rotation_drift = pose_angle_deg(observed_relation, nominal_relation)
    if (translation_drift > max_grasp_translation_drift_m or
            rotation_drift > max_grasp_rotation_drift_deg):
        return XYRetryAssessment(
            "stop", None, None, (), None,
            "observed_key_hand_relation_drift_exceeds_commissioned_limit",
            frame_binding)
    screen_report = screen_axis_1mm_endpoint_choices(
        shared_root=root, mode=mode,
        candidate_dir=candidate_dir,
        current_offset_socket_m=current_offset_socket_m,
        max_total_offset_m=max_total_offset_m,
        minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
        T_key_hand_override=observed_relation,
        held_hand_q_measured=hand_q,
        screen=screen)
    screen_report["observed_key_hand_source"] = observed_key_hand_source
    screen_report["held_hand_q_measured"] = hand_q.tolist()
    screen_report["observed_relation_translation_drift_m"] = translation_drift
    screen_report["observed_relation_rotation_drift_deg"] = rotation_drift
    choices = tuple(XYChoice(row["choice_id"], tuple(row["xy_offset_socket_m"]))
                    for row in screen_report["rows"] if row["endpoint_pass"])
    if not choices or all(choice.choice_id == "hold" for choice in choices):
        return XYRetryAssessment(
            "no_safe_direction", screen_report, None, (), None,
            "no alternative 1 mm target passed exact endpoint geometry",
            frame_binding)
    camera_ids = {frame.camera_id for frame in frames}
    if (camera_ids - set(intrinsics_full) or camera_ids - set(extrinsics_full)):
        raise ValueError("retry frame lacks AutoDex camera calibration")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    geometry = json.loads(AssetPaths(root, mode).task_geometry.read_text(
        encoding="utf-8"))
    preinsert = validate_se3(
        geometry["T_socket_key_preinsert"], name="CAD preinsert key pose")
    z = float(preinsert[2, 3])
    projected_frames = []
    for frame in frames:
        params = intrinsics_full[frame.camera_id]
        K = np.asarray(params["K_undist"], dtype=float)
        projected_frames.append(CalibratedXYFrame(
            frame.camera_id, frame.timestamp_s, frame.image,
            camera_socket_transform(
                T_camera_world=extrinsics_full[frame.camera_id],
                T_world_robot=c2r, T_robot_socket=socket), K))
    overlays = build_xy_candidate_overlays(
        choices=choices, frames=projected_frames,
        socket_plane_z_m=z,
        minimum_anchor_separation_px=minimum_anchor_separation_px,
        crop_width_px=crop_width_px)
    if overlays.status != "views_ready":
        return XYRetryAssessment(
            "visual_abstain", screen_report, overlays, (), None,
            "fewer than two calibrated views resolve 1 mm candidates",
            frame_binding)
    votes, observations = observe_xy_views(backend, overlays.views, choices)
    decision = resolve_multiview_choice(
        choices, votes, current_offset_socket_m=current_offset_socket_m,
        grasp_held=grasp_held, hard_abort=hard_abort,
        max_step_m=0.001, max_total_m=max_total_offset_m,
        max_timestamp_skew_s=max_capture_skew_s,
        decision_timestamp_s=decision_timestamp_s,
        max_frame_age_s=max_frame_age_s)
    return XYRetryAssessment(
        "proposal_requires_live_preflight" if decision.status == "propose"
        else "visual_abstain_or_stop",
        screen_report, overlays, tuple(observations), decision,
        decision.reason, frame_binding)
