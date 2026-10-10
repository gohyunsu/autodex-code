"""Bind a saved cylinder VLM diagnostic to a read-only lateral hold plan.

Only the first withdrawn hold from a source-verified physical grasp medoid is
supported. The result is deliberately not a pending retry or an executor
command: a later image, new endpoint screen and guarded insertion are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chi2

from .assets import AssetPaths
from .config import TaskMode
from .frame_provenance import image_sha256
from .geometry import validate_se3
from .lateral_preflight import (
    LateralHoldPreflight, plan_lateral_hold_shift,
    write_lateral_hold_preflight,
)
from .live_robot_state import LiveRobotState
from .retry_session import (
    GroundedXYDiagnostic, RetrySessionLimits,
    _validated_retry_trial_context, _validated_retry_withdrawal,
)
from .uncertainty_margin import SurfaceDeviationBounds
from .world import validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_grounded_frame_artifacts(path: Path, saved: dict) -> None:
    cameras = set(saved.get("frame_binding", {}))
    artifacts = saved.get("artifacts_sha256")
    if (not isinstance(artifacts, dict) or len(cameras) < 2 or
            set(artifacts) != {f"frames/{camera}.png" for camera in cameras} or
            any(not camera or not camera.replace("-", "").replace(
                "_", "").replace(".", "").isalnum()
                for camera in cameras)):
        raise ValueError("grounded report lacks complete original camera frames")
    for camera in cameras:
        relative = f"frames/{camera}.png"
        image_path = path.parent / relative
        if not image_path.is_file() or _sha(image_path) != artifacts[relative]:
            raise ValueError("grounded source image bytes changed")
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if image_sha256(rgb[:, :, ::-1].copy()) != saved[
                "frame_binding"][camera]["image_sha256"]:
            raise ValueError("grounded report image differs from acquired pixels")


def _verified_diagnostic_report(
    diagnostic: GroundedXYDiagnostic, report_path: Path,
) -> tuple[Path, str]:
    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("grounded diagnostic report is missing")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = diagnostic.to_record()
    if not isinstance(saved, dict) or {k: v for k, v in saved.items()
            if k != "artifacts_sha256"} != expected:
        raise ValueError("grounded diagnostic differs from saved report")
    _verify_grounded_frame_artifacts(path, saved)
    return path, _sha(path)


@dataclass(frozen=True)
class GroundedLateralPreflight:
    attempt_id: str
    candidate_id: str
    diagnostic_report_path: Path
    diagnostic_report_sha256: str
    postlift_report_path: Path
    postlift_report_sha256: str
    withdrawal_evidence_path: Path
    withdrawal_evidence_sha256: str
    joint_sample: LiveRobotState
    decision_timestamp_s: float
    predicted_observed_tip_error_m: float
    predicted_observed_axis_error_deg: float
    lateral: LateralHoldPreflight

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_grounded_lateral_preflight_v1",
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "grounded_diagnostic_report_path": str(self.diagnostic_report_path),
            "grounded_diagnostic_report_sha256": self.diagnostic_report_sha256,
            "postlift_report_path": str(self.postlift_report_path),
            "postlift_report_sha256": self.postlift_report_sha256,
            "withdrawal_evidence_path": str(self.withdrawal_evidence_path),
            "withdrawal_evidence_sha256": self.withdrawal_evidence_sha256,
            "joint_sample": self.joint_sample.to_record(),
            "decision_timestamp_s": self.decision_timestamp_s,
            "predicted_observed_tip_error_m": self.predicted_observed_tip_error_m,
            "predicted_observed_axis_error_deg": (
                self.predicted_observed_axis_error_deg),
            "lateral": self.lateral.to_record(),
            "scope": "source_bound_read_only_first_hold_shift_not_insertion_retry",
            "pending_retry": False,
            "robot_ready": False,
        }


def plan_grounded_lateral_from_withdrawal(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    catalog: dict, trial, attempt, diagnostic: GroundedXYDiagnostic,
    diagnostic_report_path: Path, joint_sample: LiveRobotState,
    decision_timestamp_s: float, limits: RetrySessionLimits,
    max_hold_joint_drift_rad: float, max_grounded_tip_error_m: float,
    max_grounded_axis_error_deg: float, max_path_deviation_m: float,
    max_hold_height_deviation_m: float, max_hold_rotation_deg: float,
) -> GroundedLateralPreflight:
    """Recheck sources and preflight only the lateral hold displacement.

    The VLM tip/axis is compared with the medoid prediction at the measured
    wrist. Future-trial key surface bounds plus *separately commissioned*
    visual tip/axis errors limit the allowed discrepancy. This is a veto,
    not proof that the VLM has found the correct physical key feature.
    """
    limits.validate()
    extra_limits = (max_hold_joint_drift_rad, max_grounded_tip_error_m,
                    max_grounded_axis_error_deg, max_path_deviation_m,
                    max_hold_height_deviation_m, max_hold_rotation_deg)
    if not all(math.isfinite(float(v)) and v > 0 for v in extra_limits):
        raise ValueError("grounded lateral needs commissioned positive limits")
    if (mode.family != "cylinder" or not isinstance(
            diagnostic, GroundedXYDiagnostic) or
            diagnostic.status != "diagnostic_metric_xy_correction" or
            diagnostic.alignment.get("schema") !=
            "precision_insertion_grounded_alignment_v2" or
            diagnostic.alignment.get("status") != diagnostic.status):
        raise ValueError("lateral shift needs a confident cylinder diagnostic")
    inliers = diagnostic.alignment.get("inlier_cameras", [])
    required_views = diagnostic.alignment.get(
        "estimator_limits", {}).get("minimum_views")
    if (not isinstance(inliers, list) or
            type(required_views) is not int or required_views < 2 or
            len(inliers) < required_views or len(inliers) != len(set(inliers)) or
            not set(inliers) <= set(diagnostic.frame_binding)):
        raise ValueError("lateral shift lacks independent inlier camera views")
    report_path, report_sha = _verified_diagnostic_report(
        diagnostic, diagnostic_report_path)
    insertion, postlift, postlift_path, root = _validated_retry_trial_context(
        mode=mode, shared_root=shared_root, calibration=calibration,
        catalog=catalog, trial=trial, attempt=attempt,
        postlift_preflight_report_path=diagnostic.postlift_preflight_path)
    if postlift["schema"] != "precision_insertion_bounded_postlift_preflight_v1":
        raise ValueError("grounded lateral requires physical grasp calibration")
    geometry_path = AssetPaths(root, mode).task_geometry
    if (diagnostic.attempt_id != attempt.attempt_id or
            diagnostic.candidate_id != attempt.candidate_id or
            diagnostic.session_calibration_sha256 !=
            attempt.session_calibration_sha256 or
            diagnostic.camera_calibration_sha256 !=
            calibration.record["camera_calibration_sha256"] or
            diagnostic.task_geometry_sha256 != _sha(geometry_path) or
            diagnostic.postlift_preflight_sha256 != _sha(postlift_path) or
            diagnostic.withdrawal_evidence_sha256 != _sha(
                diagnostic.withdrawal_evidence_path) or
            diagnostic.alignment.get("key_object") != mode.key_object or
            diagnostic.alignment.get("socket_object") != mode.socket_object or
            diagnostic.alignment.get("verification_depth_m") !=
            mode.target_depth_m):
        raise ValueError("grounded lateral diagnostic and session sources differ")
    _validated_retry_withdrawal(
        attempt=attempt, insertion=insertion,
        withdrawal_completed_at_s=json.loads(
            diagnostic.withdrawal_evidence_path.read_text(
                encoding="utf-8"))["completed_at_s"],
        withdrawal_evidence_path=diagnostic.withdrawal_evidence_path,
        verified_frames=diagnostic.frame_binding,
        joint_sample=diagnostic.joint_sample)
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("lateral start needs fresh measured robot feedback")
    joint_sample.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    decision = float(decision_timestamp_s)
    if (not math.isfinite(decision) or
            joint_sample.sample_timestamp_s <
            diagnostic.joint_sample.sample_timestamp_s or
            decision < joint_sample.sample_timestamp_s or
            decision - max(row["timestamp_s"] for row in
                           diagnostic.frame_binding.values()) >
            limits.max_frame_age_s or
            np.max(np.abs(joint_sample.full_q -
                          diagnostic.joint_sample.full_q)) >
            max_hold_joint_drift_rad):
        raise ValueError("lateral start joints or camera diagnostic became stale")
    relation_record = postlift["bounded_held_relation"]
    relation = validate_se3(relation_record["T_key_hand"],
                            name="physical grasp medoid")
    bound_record = relation_record["surface_bounds"]
    bounds = SurfaceDeviationBounds(
        bound_record["key_surface_m"], bound_record["hand_surface_m"],
        bound_record["source"])
    bounds.validate()
    alignment = diagnostic.alignment
    increment = np.asarray(alignment["bounded_xy_increment_socket_m"],
                           dtype=np.float64)
    correction = np.asarray(alignment["xy_correction_socket_m"],
                            dtype=np.float64)
    rim = np.asarray(alignment["rim_error_xy_m"], dtype=np.float64)
    depth = np.asarray(alignment["depth_error_xy_m"], dtype=np.float64)
    mean = np.asarray(alignment["mean_error_xy_m"], dtype=np.float64)
    covariance = np.asarray(alignment["lateral_covariance_m2"],
                            dtype=np.float64)
    if (increment.shape != (2,) or correction.shape != (2,) or
            rim.shape != (2,) or depth.shape != (2,) or mean.shape != (2,) or
            covariance.shape != (2, 2) or
            not np.all(np.isfinite(increment)) or
            not np.all(np.isfinite(correction)) or
            not np.all(np.isfinite(rim)) or
            not np.all(np.isfinite(depth)) or
            not np.all(np.isfinite(mean)) or
            not np.all(np.isfinite(covariance)) or
            not np.allclose(covariance, covariance.T, atol=1e-14) or
            np.linalg.eigvalsh(covariance).min() <= 0 or
            not np.allclose(mean, (rim + depth) / 2, atol=1e-10) or
            not np.allclose(correction, -mean, atol=1e-10) or
            np.linalg.norm(correction) <= 0 or
            not 0 < np.linalg.norm(increment) <= .001 + 1e-12 or
            not np.allclose(increment, correction * min(
                1., .001 / np.linalg.norm(correction)), atol=1e-10) or
            np.linalg.norm(np.asarray(attempt.xy_offset_socket_m) +
                           increment) > limits.max_total_offset_m):
        raise ValueError("grounded XY increment is not a valid 1 mm correction")
    lower = (-2 * float(mean @ increment) - float(increment @ increment) -
             2 * math.sqrt(float(chi2.ppf(.95, df=2))) *
             math.sqrt(float(increment @ covariance @ increment)))
    uncertainty = (math.sqrt(float(chi2.ppf(.95, df=2))) *
                   math.sqrt(float(np.linalg.eigvalsh(covariance).max())))
    maximum_uncertainty = float(alignment[
        "estimator_limits"]["max_lateral_uncertainty_95_m"])
    if (lower <= 0 or not math.isclose(
            lower, float(alignment.get(
                "increment_squared_error_improvement_lower_95_m2", -math.inf)),
            abs_tol=1e-12, rel_tol=1e-8) or
            not math.isfinite(maximum_uncertainty) or
            maximum_uncertainty <= 0 or
            uncertainty > maximum_uncertainty or
            not math.isclose(uncertainty, float(alignment.get(
                "lateral_uncertainty_95_m", math.nan)),
                abs_tol=1e-10, rel_tol=1e-8)):
        raise ValueError("grounded XY confidence does not match its covariance")
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    wrist = validate_se3(planner.fk_wrist(joint_sample.full_q),
                         name="live withdrawn wrist")
    key_socket = np.linalg.inv(socket) @ wrist @ np.linalg.inv(relation)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    tip_z = float(geometry["key_frame"]["tip_z_m"])
    axis_local = np.asarray(geometry["key_frame"]["insertion_axis"],
                            dtype=np.float64)
    if (not math.isfinite(tip_z) or tip_z <= 0 or
            not np.allclose(axis_local, [0., 0., 1.], atol=1e-8)):
        raise ValueError("cylinder CAD tip and insertion axis changed")
    predicted_tip = (key_socket @ np.array([0., 0., tip_z, 1.]))[:3]
    predicted_axis = key_socket[:3, :3] @ axis_local
    observed_tip = np.asarray(alignment["tip_socket_m"], dtype=np.float64)
    observed_axis = np.asarray(
        alignment["insertion_axis_socket"], dtype=np.float64)
    if (observed_tip.shape != (3,) or observed_axis.shape != (3,) or
            not np.all(np.isfinite(observed_tip)) or
            not np.all(np.isfinite(observed_axis)) or
            not np.isclose(np.linalg.norm(observed_axis), 1., atol=1e-3)):
        raise ValueError("grounded tip or axis is invalid")
    rim_z = float(geometry["socket_entry_plane_z_m"])
    if (not math.isfinite(rim_z) or observed_axis[2] >= 0 or
            not np.allclose(rim, observed_tip[:2] +
                            (rim_z - observed_tip[2]) *
                            observed_axis[:2] / observed_axis[2],
                            atol=1e-9) or
            not np.allclose(depth, observed_tip[:2] +
                            (rim_z - mode.target_depth_m - observed_tip[2]) *
                            observed_axis[:2] / observed_axis[2],
                            atol=1e-9)):
        raise ValueError("grounded axis contradicts socket insertion geometry")
    tip_error = float(np.linalg.norm(predicted_tip - observed_tip))
    axis_error = math.degrees(math.acos(float(np.clip(
        np.dot(predicted_axis, observed_axis), -1., 1.))))
    surface_axis_allowance = math.degrees(2 * math.asin(min(
        1., bounds.key_surface_m / tip_z)))
    if (tip_error > bounds.key_surface_m + max_grounded_tip_error_m or
            axis_error > surface_axis_allowance + max_grounded_axis_error_deg):
        raise ValueError("visual tip/axis contradicts bounded held relation")
    expected_hold = validate_se3(
        postlift["targets"]["T_robot_hand_preinsert"],
        name="source-verified preinsert wrist")
    lateral = plan_lateral_hold_shift(
        planner=planner, mode=mode, shared_root=root,
        calibration=calibration, trial_scene=trial.trial_scene,
        start_q=joint_sample.full_q,
        expected_hold_pose=expected_hold, T_key_hand=relation,
        increment_socket_xy_m=(float(increment[0]), float(increment[1])),
        bounds=bounds, limits=limits.path_limits,
        max_path_deviation_m=max_path_deviation_m,
        max_hold_height_deviation_m=max_hold_height_deviation_m,
        max_hold_rotation_deg=max_hold_rotation_deg)
    return GroundedLateralPreflight(
        attempt.attempt_id, attempt.candidate_id, report_path, report_sha,
        postlift_path, _sha(postlift_path),
        diagnostic.withdrawal_evidence_path,
        diagnostic.withdrawal_evidence_sha256, joint_sample, decision,
        tip_error, axis_error, lateral)


def write_grounded_lateral_preflight(
    result: GroundedLateralPreflight, output_dir: Path,
) -> Path:
    """Persist provenance binding plus exact path, with no retry event."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    nested = write_lateral_hold_preflight(result.lateral, target / "lateral")
    record = result.to_record()
    record["lateral_report_sha256"] = _sha(nested / "report.json")
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def verify_grounded_lateral_preflight(
    result: GroundedLateralPreflight, report_path: Path,
) -> dict:
    """Recheck the saved plan, trajectory bytes and linked source records.

    This checks file consistency, not authenticity of the physical-calibration
    producer or execution readiness.
    """
    path = Path(report_path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = result.to_record()
    nested_hash = record.pop("lateral_report_sha256", None)
    if (record != expected or
            record.get("schema") !=
            "precision_insertion_grounded_lateral_preflight_v1" or
            record.get("pending_retry") is not False or
            record.get("robot_ready") is not False):
        raise ValueError("saved grounded lateral plan changed")
    nested = path.parent / "lateral" / "report.json"
    if (not nested.is_file() or nested_hash != _sha(nested)):
        raise ValueError("saved lateral path report changed")
    lateral = json.loads(nested.read_text(encoding="utf-8"))
    trajectory_name = lateral.pop("lateral_trajectory", None)
    trajectory_hash = lateral.pop("lateral_trajectory_sha256", None)
    if lateral != result.lateral.to_record():
        raise ValueError("saved lateral audit changed")
    if result.lateral.status == "sampled_lateral_hold_shift_pass":
        if (trajectory_name != "lateral_trajectory.npy" or
                lateral.get("sampled_audit", {}).get("sampled_clear") is not True or
                lateral.get("planner_query", {}).get("success") is not True or
                not isinstance(trajectory_hash, str)):
            raise ValueError("passing lateral plan lacks a checked trajectory")
    if trajectory_name is not None:
        trajectory_file = nested.parent / trajectory_name
        if (trajectory_name != "lateral_trajectory.npy" or
                not trajectory_file.is_file() or
                trajectory_hash != _sha(trajectory_file)):
            raise ValueError("saved lateral trajectory bytes changed")
        path_q = np.load(trajectory_file, allow_pickle=False)
        if (path_q.ndim != 2 or path_q.shape != (
                lateral["trajectory_sample_count"], 13) or
                not np.all(np.isfinite(path_q)) or
                not np.allclose(path_q[0], result.lateral.start_q,
                                atol=1e-4, rtol=0) or
                not np.allclose(path_q[:, 7:],
                                result.lateral.start_q[7:],
                                atol=1e-4, rtol=0) or
                not np.array_equal(path_q, result.lateral.trajectory)):
            raise ValueError("saved lateral trajectory differs from audited path")
    for source, digest in (
            (result.diagnostic_report_path,
             result.diagnostic_report_sha256),
            (result.postlift_report_path, result.postlift_report_sha256),
            (result.withdrawal_evidence_path,
             result.withdrawal_evidence_sha256)):
        if not source.is_file() or _sha(source) != digest:
            raise ValueError("grounded lateral source record changed")
    diagnostic = json.loads(result.diagnostic_report_path.read_text(
        encoding="utf-8"))
    _verify_grounded_frame_artifacts(result.diagnostic_report_path, diagnostic)
    return record
