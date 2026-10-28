"""Observe, plan and execute held-key XY alignment in the *current* trial.

The key stays grasped. Each image estimates the key-face centre in the frozen
socket frame; a cuRobo lateral move is followed by another image. The final
axial path is replanned from the measured corrected hold, never replayed from
the original pre-insertion joint state.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .geometry import validate_se3
from .grounded_alignment import AlignmentLimits
from .live_robot_state import LiveRobotState
from .observer import LabeledFrame
from .path_audit import PathAuditLimits, _audit_held_geometry_samples
from .preflight import _path, plan_held_transfer_and_axial
from .raw_camera_capture import verify_raw_camera_capture
from .retry_session import (
    _compute_grounded_alignment, _compute_sam_half_ellipse_alignment,
)
from .targets import InsertionTargets, build_rigid_insertion_targets
from .transfer_execution import verify_transfer_controller_result
from .visual_views import assessment_cameras
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def choose_xy_move(alignment: dict, *, tolerance_m: float,
                   max_step_m: float = .010) -> dict:
    """Use measured multiview XY; single-view plane guesses are not metric.

    A cylinder's radial CAD gap is the default tolerance. An operator may
    choose a smaller target, but no extra confidence or key-hand error limit
    is introduced here. The step cap is the existing 10 mm motion policy.
    """
    if not (math.isfinite(tolerance_m) and tolerance_m > 0 and
            math.isfinite(max_step_m) and max_step_m > 0):
        raise ValueError("XY target and step must be positive metres")
    inliers = alignment.get("inlier_cameras")
    if (alignment.get("status") != "diagnostic_metric_xy_correction" or
            not isinstance(inliers, list) or len(set(inliers)) < 2):
        return {"status": "not_metric_multiview", "xy_error_m": None,
                "increment_socket_m": None}
    correction = np.asarray(alignment.get("xy_correction_socket_m"), dtype=float)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("metric multiview alignment lacks finite socket XY")
    error = float(np.linalg.norm(correction))
    if error <= tolerance_m:
        return {"status": "aligned", "xy_error_m": error,
                "increment_socket_m": [0., 0.]}
    step = correction * min(1., max_step_m / error)
    return {"status": "shift", "xy_error_m": error,
            "increment_socket_m": step.tolist()}


def measure_preinsert_xy(*, runner, raw_bundle: Path, backend,
                         alignment_limits: AlignmentLimits,
                         index: int, nominal_tip_height_error_m: float,
                         xy_estimator: str = "vlm",
                         sam_segmentor=None) -> dict:
    """Call the existing calibrated multiview tip/axis estimator on raw PNGs."""
    bundle = Path(raw_bundle).expanduser().resolve()
    capture = verify_raw_camera_capture(bundle, phase="preinsert")
    rows = capture["frame_evidence"]
    cameras = assessment_cameras(rows)
    frames = []
    for camera in cameras:
        image = cv2.imread(str(bundle / "images" / f"{camera}.png"),
                           cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("preinsert camera image is missing")
        frames.append(LabeledFrame(
            camera, "preinsert_hold", rows[camera]["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
    calibration = runner.calibration.record["camera_calibration"]
    if xy_estimator == "sam_half_ellipse":
        if sam_segmentor is None:
            raise ValueError("SAM half-ellipse needs its prepared ROI worker")
        geometry_sha, alignment, masks = _compute_sam_half_ellipse_alignment(
            mode=runner.mode, root=Path(runner.shared_root),
            calibration=runner.calibration,
            postlift_file=runner._postlift_report_path, frames=frames,
            frame_rows=rows, backend=backend, sam_segmentor=sam_segmentor,
            intrinsics_full=calibration["intrinsics_full"],
            extrinsics_full=calibration["extrinsics_full"],
            alignment_limits=alignment_limits, search_index=index,
            max_nominal_tip_height_error_m=nominal_tip_height_error_m)
        observations = []
        mask_records = []
        if masks:
            mask_dir = (Path(runner._attempt_dir) / "same_trial_xy" /
                        "observations" / f"{index:03d}_masks")
            mask_dir.mkdir(parents=True, exist_ok=False)
            for camera, png in sorted(masks.items()):
                path = mask_dir / f"{camera}.png"
                path.write_bytes(png)
                mask_records.append({"camera_id": camera, "path": str(path),
                                     "sha256": _sha(path)})
            for row in alignment.get("sam_half_ellipse_views", []):
                match = next((item for item in mask_records
                              if item["camera_id"] == row["camera_id"]), None)
                if match is not None:
                    row["mask_file"] = match["path"]
    elif xy_estimator == "vlm":
        geometry_sha, alignment, observations = _compute_grounded_alignment(
            mode=runner.mode, root=Path(runner.shared_root),
            calibration=runner.calibration,
            postlift_file=runner._postlift_report_path, frames=frames,
            intrinsics_full=calibration["intrinsics_full"],
            extrinsics_full=calibration["extrinsics_full"],
            backend=backend, alignment_limits=alignment_limits,
            axis_reference_key_z_m=None, roi_xyxy_by_camera=None,
            square_translation_only=False, search_index=index,
            max_nominal_tip_height_error_m=nominal_tip_height_error_m)
        mask_records = []
    else:
        raise ValueError("same-trial XY supports vlm or sam_half_ellipse")
    return {
        "schema": "precision_insertion_same_trial_xy_observation_v1",
        "attempt_id": runner.active_attempt.attempt_id,
        "candidate_id": runner.active_attempt.candidate_id,
        "raw_bundle": str(bundle),
        "raw_manifest_sha256": _sha(bundle / "manifest.json"),
        "postlift_report_path": str(runner._postlift_report_path),
        "postlift_report_sha256": _sha(runner._postlift_report_path),
        "task_geometry_sha256": geometry_sha,
        "xy_estimator": xy_estimator,
        "sam_mask_files": mask_records,
        "alignment": alignment,
        "vlm_observations": [item.to_record() for item in observations],
    }


def plan_and_execute_shift(*, runner, planner, adapter, read_state,
                           motion_interlock, limits, path_limits,
                           increment_socket_m, residual_socket_m,
                           index: int) -> Path:
    """Move one held wrist along socket XY using the existing measured follower."""
    increment = np.asarray(increment_socket_m, dtype=float)
    if increment.shape != (2,) or not np.all(np.isfinite(increment)):
        raise ValueError("held XY increment must contain two finite metres")
    start = read_state()
    if not isinstance(start, LiveRobotState):
        raise TypeError("same-trial shift needs measured FR3/Inspire state")
    socket = validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    goal = validate_se3(planner.fk_wrist(start.full_q),
                        name="current measured wrist FK").copy()
    goal[:3, 3] += socket[:3, :2] @ increment
    scene = build_held_scene_from_trial(
        trial_scene=runner._preflight.trial_scene,
        calibration=runner.calibration)
    query = planner.plan_cartesian_pose(
        start.full_q, goal, scene_cfg=scene,
        include_obj_obstacle=False, return_result=True, lock_hand=True,
        timing_phase="precision_insertion_same_trial_xy")
    if not query.success or query.trajectory is None:
        raise RuntimeError("cuRobo could not plan the held XY correction")
    trajectory = _path(query.trajectory, "same-trial held XY",
                       start.full_q, start.full_q[7:])
    residual = np.asarray(residual_socket_m, dtype=float)
    if residual.shape != (2,) or not np.all(np.isfinite(residual)):
        raise ValueError("held key residual must contain finite socket XY")
    key_at_hold = runner._postlift_preflight.targets.T_robot_key_preinsert.copy()
    key_at_hold[:3, 3] += socket[:3, :2] @ residual
    held_relation = np.linalg.inv(key_at_hold) @ validate_se3(
        planner.fk_wrist(start.full_q), name="measured held wrist FK")
    wrist = {"lateral": [validate_se3(planner.fk_wrist(q),
                    name=f"lateral FK[{i}]") for i, q in enumerate(trajectory)]}
    geometry_failures, distances, _ = _audit_held_geometry_samples(
        shared_root=Path(runner.shared_root), mode=runner.mode,
        calibration=runner.calibration, wrist=wrist,
        T_key_hand=held_relation, held_hand_q=start.full_q[7:],
        minimum_hand_clearance_m=path_limits.minimum_hand_clearance_m,
        stop_on_first_failure=True)
    if geometry_failures:
        raise RuntimeError("held XY path intersects the frozen collision world")
    folder = (Path(runner._attempt_dir) / "same_trial_xy" / "shifts" /
              f"{index:03d}")
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "trajectory.npy"
    np.save(path, trajectory, allow_pickle=False)
    started = time.time()
    if motion_interlock() is not True:
        raise PermissionError("same-trial shift needs live motion opt-in")
    try:
        result = adapter.follow_transfer(
            trajectory.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=start.hand_raw_commanded.copy(),
            trajectory_sha256=_sha(path))
        begun, completed, sources = verify_transfer_controller_result(
            result, trajectory_sha256=_sha(path), requested_at_s=started,
            max_execution_duration_s=limits.max_execution_duration_s)
        end = read_state()
        if not isinstance(end, LiveRobotState):
            raise RuntimeError("same-trial shift lacks measured endpoint")
    except Exception as exc:
        stop = adapter.stop_and_acknowledge()
        _write(folder / "failure.json", {
            "schema": "precision_insertion_same_trial_xy_shift_failure_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "stop_acknowledgement": stop,
        })
        raise
    return _write(folder / "report.json", {
        "schema": "precision_insertion_same_trial_xy_shift_v1",
        "attempt_id": runner.active_attempt.attempt_id,
        "candidate_id": runner.active_attempt.candidate_id,
        "increment_socket_m": increment.tolist(),
        "observed_key_xy_socket_m": residual.tolist(),
        "sampled_geometry_clear": True,
        "minimum_surface_distances_m": distances,
        "trajectory_path": str(path),
        "trajectory_sha256": _sha(path),
        "start_state": start.to_record(), "end_state": end.to_record(),
        "started_at_s": begun, "completed_at_s": completed,
        "source_records": sources,
    })


def _shifted_targets(original: InsertionTargets, socket: np.ndarray,
                     total_xy: np.ndarray,
                     residual_xy: np.ndarray) -> InsertionTargets:
    hand_delta = socket[:3, :2] @ total_xy
    key_delta = socket[:3, :2] @ residual_xy

    def moved(pose, delta):
        result = pose.copy()
        result[:3, 3] += delta
        return result

    shifted_key_preinsert = moved(original.T_robot_key_preinsert, key_delta)
    shifted_hand_preinsert = moved(original.T_robot_hand_preinsert, hand_delta)
    effective_relation = (np.linalg.inv(shifted_key_preinsert) @
                          shifted_hand_preinsert)
    return replace(
        original,
        T_key_hand=effective_relation,
        T_robot_key_preinsert=shifted_key_preinsert,
        T_robot_key_entry=moved(original.T_robot_key_entry, key_delta),
        T_robot_key_verification=moved(
            original.T_robot_key_verification, key_delta),
        T_robot_hand_preinsert=shifted_hand_preinsert,
        T_robot_hand_entry=moved(original.T_robot_hand_entry, hand_delta),
        T_robot_hand_verification=moved(
            original.T_robot_hand_verification, hand_delta),
        xy_offset_socket_m=tuple(np.asarray(
            original.xy_offset_socket_m) + residual_xy),
    )


def plan_corrected_axial_handoff(*, runner, planner,
                                path_limits: PathAuditLimits,
                                read_state, total_xy_m,
                                residual_xy_m,
                                observations: list[Path],
                                shifts: list[Path],
                                final_preinsert_bundle: Path) -> Path:
    """Replan the socket-axis Jacobian stroke from the corrected live hold."""
    original = runner._postlift_preflight.targets
    if not isinstance(original, InsertionTargets):
        raise ValueError("same-trial mode needs the selected held-key target")
    total = np.asarray(total_xy_m, dtype=float)
    residual = np.asarray(residual_xy_m, dtype=float)
    if (total.shape != (2,) or residual.shape != (2,) or
            not np.all(np.isfinite(total)) or
            not np.all(np.isfinite(residual))):
        raise ValueError("same-trial XY values must contain finite metres")
    socket = validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    state = read_state()
    provisional = _shifted_targets(original, socket, total, residual)
    measured_wrist = validate_se3(planner.fk_wrist(state.full_q),
                                  name="corrected measured wrist FK")
    relation = np.linalg.inv(provisional.T_robot_key_preinsert) @ measured_wrist
    targets = build_rigid_insertion_targets(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration, T_key_hand=relation,
        xy_offset_socket_m=provisional.xy_offset_socket_m,
        cylinder_yaw_gauge_socket_rad=original.cylinder_yaw_gauge_socket_rad)
    planned = plan_held_transfer_and_axial(
        planner=planner, trial_scene=runner._preflight.trial_scene,
        shared_root=runner.shared_root, calibration=runner.calibration,
        targets=targets, start_q=state.full_q,
        held_hand_q=state.full_q[7:], held_hand_source="measured",
        limits=path_limits, axial_waypoint_step_m=.005,
        start_at_preinsert=True)
    if not planned.sampled_planning_pass or planned.axial_trajectory is None:
        raise RuntimeError(f"corrected 20 mm plan failed: {planned.status}")
    folder = (Path(runner._attempt_dir) / "guarded_axial_handoffs" /
              f"{runner._guarded_axial_index:03d}")
    folder.mkdir(parents=True, exist_ok=False)
    archive = folder / "planned_trajectories.npz"
    np.savez_compressed(archive, axial=planned.axial_trajectory)
    report = _write(folder / "report.json", {
        "schema": "precision_insertion_same_trial_axial_handoff_v1",
        "attempt_id": runner.active_attempt.attempt_id,
        "candidate_id": runner.active_attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "mode": targets.to_record()["mode"],
        "postlift_report_path": str(runner._postlift_report_path),
        "postlift_report_sha256": _sha(runner._postlift_report_path),
        "preinsert_report_path": str(runner._preinsert_report_path),
        "preinsert_report_sha256": _sha(runner._preinsert_report_path),
        "final_preinsert_bundle_path": str(final_preinsert_bundle),
        "final_preinsert_manifest_sha256": _sha(
            final_preinsert_bundle / "manifest.json"),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": _sha(archive),
        "axial_start_q": planned.axial_trajectory[0].tolist(),
        "axial_end_q": planned.axial_trajectory[-1].tolist(),
        "axial_sample_count": len(planned.axial_trajectory),
        "target_depth_m": runner.mode.target_depth_m,
        "measured_start": state.to_record(),
        "decision_timestamp_s": time.time(),
        "total_xy_correction_socket_m": total.tolist(),
        "observed_residual_socket_xy_m": residual.tolist(),
        "held_relation_scope": "observed_key_XY_and_measured_wrist_with_nominal_key_orientation",
        "inserted_key_pose_robot": targets.T_robot_key_verification.tolist(),
        "targets": targets.to_record(),
        "planning": planned.to_record(),
        "observations": [{"path": str(path), "sha256": _sha(path)}
                         for path in observations],
        "shifts": [{"path": str(path), "sha256": _sha(path)}
                   for path in shifts],
        "scope": "same_trial_replanned_axial_path_not_physical_key_depth",
        "robot_ready": False,
    })
    runner._guarded_axial_index += 1
    return report


def verify_same_trial_axial_handoff(path: Path) -> dict:
    """Verify the corrected path and its original and intermediate evidence."""
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("schema") != "precision_insertion_same_trial_axial_handoff_v1"
            or record.get("robot_ready") is not False or
            record.get("scope") !=
            "same_trial_replanned_axial_path_not_physical_key_depth"):
        raise ValueError("invalid same-trial axial handoff")
    for name in ("postlift", "preinsert"):
        file = Path(record[f"{name}_report_path"])
        if _sha(file) != record[f"{name}_report_sha256"]:
            raise ValueError(f"same-trial {name} source changed")
    if _sha(Path(record["final_preinsert_bundle_path"]) / "manifest.json") != (
            record["final_preinsert_manifest_sha256"]):
        raise ValueError("same-trial final preinsert capture changed")
    for kind in ("observations", "shifts"):
        for entry in record[kind]:
            if _sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError(f"same-trial {kind} source changed")
    archive = Path(record["trajectory_archive_path"])
    if archive.parent != source.parent or _sha(archive) != record[
            "trajectory_archive_sha256"]:
        raise ValueError("same-trial axial archive changed")
    with np.load(archive, allow_pickle=False) as arrays:
        axial = np.asarray(arrays["axial"], dtype=float)
    if (axial.shape != (record["axial_sample_count"], 13) or
            not np.array_equal(axial[0], record["axial_start_q"]) or
            not np.array_equal(axial[-1], record["axial_end_q"]) or
            not np.allclose(axial[0], record["measured_start"]["full_q"],
                            atol=1e-4, rtol=0) or
            record["planning"]["sampled_planning_pass"] is not True or
            record["planning"]["sampled_held_path_audit"]["sampled_clear"]
            is not True):
        raise ValueError("same-trial axial plan differs from its measured start")
    return record
