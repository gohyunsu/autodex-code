"""Fresh camera/robot arrival observation after an observed-key XY retry.

This is the square-key counterpart of the first-transfer preinsert checkpoint.
It reuses the same rendered scene and VLM comparison, but binds the decision
to the saved XY retry rather than pretending it is the original transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image

from .calibration import SessionCalibration
from .config import TaskMode
from .frame_provenance import bounded_capture_skew_s, image_sha256
from .live_robot_state import LiveRobotState
from .observer import HeldSceneView, ImageVLM
from .preinsert_checkpoint import (
    PreinsertCheckpoint, _arrival_label, _canonical_digest, _transfer_log,
    _verify_visual_record, _verify_visual_replay,
    observe_held_preinsert_scene, write_preinsert_checkpoint,
)
from .raw_camera_capture import verify_raw_camera_capture
from .records import AttemptRecord
from .retry_preflight import XYRetryPreflight


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ObservedRetryArrivalCheckpoint(PreinsertCheckpoint):
    def to_record(self) -> dict:
        record = super().to_record()
        record["schema"] = "precision_insertion_observed_xy_retry_arrival_v1"
        record["retry_arrived"] = record.pop("preinsert_reached")
        record["scope"] = "observed_retry_arrival_not_axial_contact_authorization"
        return record


def assess_observed_retry_arrival(
    *, attempt: AttemptRecord, preflight: XYRetryPreflight,
    preflight_report_path: Path, calibration: SessionCalibration,
    shared_root: Path, mode: TaskMode, raw_bundle: Path,
    transfer_execution_path: Path, joint_sample: LiveRobotState,
    backend: ImageVLM, max_capture_skew_s: float,
    max_joint_frame_skew_s: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    renderer_factory: Callable | None = None,
    robot_loader: Callable | None = None,
) -> ObservedRetryArrivalCheckpoint:
    """Label a fresh corrected hold; never reuse the failed stroke's image."""
    limits = (max_capture_skew_s, max_joint_frame_skew_s,
              max_arm_hand_skew_s,
              max_hand_command_error_raw, max_arm_velocity_rad_s)
    if any(not math.isfinite(float(value)) or float(value) <= 0
           for value in limits):
        raise ValueError("retry arrival needs positive observation limits")
    if (not isinstance(attempt, AttemptRecord) or attempt.mode != mode or
            attempt.labels["insertion_success"] is not False or
            not attempt._pending_retry or
            not isinstance(preflight, XYRetryPreflight) or
            not attempt.events or
            attempt.events[-1].get("stage") != "xy_retry" or
            attempt.events[-1].get("value") != preflight.choice_id or
            preflight.status != "sampled_retry_preflight_pass" or
            "/".join(preflight.candidate_key) != attempt.candidate_id or
            preflight.planning.sampled_planning_pass is not True or
            preflight.planning.transfer_trajectory is None or
            tuple(attempt.xy_offset_socket_m) !=
            tuple(preflight.targets.xy_offset_socket_m)):
        raise ValueError("retry arrival lacks this attempt's passing XY plan")
    if not isinstance(calibration, SessionCalibration):
        raise TypeError("retry arrival needs the frozen socket session")
    session_hash = _canonical_digest(calibration.record)
    if attempt.session_calibration_sha256 != session_hash:
        raise ValueError("retry arrival changed the frozen socket session")
    plan_path = Path(preflight_report_path).expanduser().resolve()
    saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (not isinstance(saved_plan, dict) or
            any(saved_plan.get(key) != value for key, value in
                preflight.to_record().items()) or
            saved_plan.get("planned_trajectories") !=
            "planned_trajectories.npz"):
        raise ValueError("retry arrival differs from the saved XY plan")
    archive = plan_path.parent / "planned_trajectories.npz"
    if _sha(archive) != saved_plan.get("planned_trajectories_sha256"):
        raise ValueError("retry arrival trajectory bytes changed")
    transfer_path = Path(transfer_execution_path).expanduser().resolve()
    transfer = _transfer_log(
        transfer_path, attempt,
        schema="precision_insertion_observed_xy_transfer_execution_v1")
    if (transfer.get("preflight_report_path") != str(plan_path) or
            transfer.get("preflight_report_sha256") != _sha(plan_path) or
            transfer.get("trajectory_sha256") != _sha(archive)):
        raise ValueError("retry arrival transfer used another XY plan")
    completed = float(transfer["completed_at_s"])
    bundle = Path(raw_bundle).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="preinsert")
    rows = capture["frame_evidence"]
    if len(rows) < 2 or bounded_capture_skew_s(rows) > max_capture_skew_s:
        raise ValueError("retry arrival cameras are not synchronized")
    first = min(row["timestamp_s"] - row["max_error_s"] for row in rows.values())
    last = max(row["timestamp_s"] + row["max_error_s"] for row in rows.values())
    if completed >= first:
        raise ValueError("retry arrival images do not follow the XY transfer")
    if not isinstance(joint_sample, LiveRobotState):
        raise TypeError("retry arrival needs measured robot feedback")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    if (joint_sample.sample_timestamp_s <= completed or
            abs(joint_sample.sample_timestamp_s - (first + last) / 2) >
            max_joint_frame_skew_s):
        raise ValueError("retry arrival robot feedback is not exposure-aligned")
    position, rotation, comparison, visual = observe_held_preinsert_scene(
        calibration=calibration, shared_root=shared_root, mode=mode,
        raw_bundle=bundle, frame_rows=rows, joint_sample=joint_sample,
        T_key_hand=preflight.observed_T_key_hand,
        T_robot_hand_target=preflight.targets.T_robot_hand_preinsert,
        relation_source="observed_multiview_key_plus_wrist",
        backend=backend, max_capture_skew_s=max_capture_skew_s,
        renderer_factory=renderer_factory, robot_loader=robot_loader)
    label, reason = _arrival_label(
        measurement=transfer["measurement"], visual_status=visual.status)
    return ObservedRetryArrivalCheckpoint(
        attempt.attempt_id, attempt.candidate_id, label, reason,
        position, rotation, visual, comparison, {
            "session_calibration_sha256": session_hash,
            "xy_preflight_report_path": str(plan_path),
            "xy_preflight_report_sha256": _sha(plan_path),
            "transfer_execution_path": str(transfer_path),
            "transfer_execution_sha256": _sha(transfer_path),
            "raw_bundle": str(bundle),
            "raw_manifest_sha256": _sha(bundle / "manifest.json"),
            "transfer_measurement": transfer["measurement"],
            "transfer_started_at_s": transfer["started_at_s"],
            "transfer_completed_at_s": completed,
            "observation_completed_at_s": last,
            "joint_feedback": joint_sample.to_record(),
        })


def write_observed_retry_arrival(
    checkpoint: ObservedRetryArrivalCheckpoint, output_dir: Path,
) -> Path:
    """Store the raw-vs-rendered VLM input using the shared checkpoint writer."""
    if not isinstance(checkpoint, ObservedRetryArrivalCheckpoint):
        raise TypeError("expected an observed XY retry arrival")
    return write_preinsert_checkpoint(checkpoint, output_dir)


def verify_observed_retry_arrival(report_path: Path) -> dict:
    """Replay the saved retry arrival from its plan, camera and VLM bytes."""
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") !=
            "precision_insertion_observed_xy_retry_arrival_v1" or
            report.get("robot_ready") is not False):
        raise ValueError("invalid observed XY retry arrival")
    plan = Path(report["xy_preflight_report_path"])
    transfer = Path(report["transfer_execution_path"])
    raw = Path(report["raw_bundle"])
    if (not all(source.is_absolute() and source.is_file()
                for source in (plan, transfer, raw / "manifest.json")) or
            _sha(plan) != report["xy_preflight_report_sha256"] or
            _sha(transfer) != report["transfer_execution_sha256"] or
            _sha(raw / "manifest.json") != report["raw_manifest_sha256"]):
        raise ValueError("observed XY retry arrival source changed")
    saved_plan = json.loads(plan.read_text(encoding="utf-8"))
    archive = plan.parent / saved_plan["planned_trajectories"]
    if (saved_plan.get("status") != "sampled_retry_preflight_pass" or
            saved_plan.get("candidate_key") != report["candidate_id"].split("/") or
            saved_plan.get("planned_trajectories") !=
                "planned_trajectories.npz" or
            _sha(archive) != saved_plan.get("planned_trajectories_sha256")):
        raise ValueError("observed XY retry arrival plan changed")
    identity = type("AttemptRef", (), {
        "attempt_id": report["attempt_id"],
        "candidate_id": report["candidate_id"],
    })()
    source = _transfer_log(
        transfer, identity,
        schema="precision_insertion_observed_xy_transfer_execution_v1")
    if (source.get("preflight_report_path") != str(plan) or
            source.get("preflight_report_sha256") != _sha(plan) or
            source.get("trajectory_sha256") != _sha(archive) or
            source["measurement"] != report["transfer_measurement"] or
            source["completed_at_s"] != report["transfer_completed_at_s"]):
        raise ValueError("observed XY retry transfer changed")
    capture = verify_raw_camera_capture(raw, phase="preinsert")
    views = report.get("comparison", {}).get("views", {})
    png_hashes = report.get("overlay_png_sha256", {})
    if (len(views) < 2 or set(views) != set(png_hashes) or
            set(views) != set(capture["frame_evidence"])):
        raise ValueError("observed XY retry camera set changed")
    paired = []
    for serial in sorted(views):
        image = cv2.imread(str(raw / "images" / f"{serial}.png"),
                           cv2.IMREAD_COLOR)
        overlay_path = path.parent / "overlays" / f"{serial}.png"
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        stamps = capture["frame_evidence"][serial]
        if (image is None or overlay is None or
                image_sha256(image) != views[serial]["raw_image_sha256"] or
                image_sha256(overlay) != views[serial]["overlay_image_sha256"] or
                _sha(overlay_path) != png_hashes[serial] or
                views[serial]["timestamp_s"] != stamps["timestamp_s"]):
            raise ValueError("observed XY retry arrival pixels changed")
        paired.append(HeldSceneView(
            serial, stamps["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))))
    _verify_visual_record(report["visual"], views)
    _verify_visual_replay(report["visual"], paired)
    label, reason = _arrival_label(
        measurement=report["transfer_measurement"],
        visual_status=report["visual"]["status"])
    if report["retry_arrived"] is not label or report["reason"] != reason:
        raise ValueError("observed XY retry arrival label changed")
    return report
