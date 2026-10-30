"""Observe, plan and execute held-key XY alignment in the *current* trial.

The key stays grasped. Each image estimates the key-face centre in the frozen
socket frame; a local Jacobian lateral move is followed by another image. The final
axial path is replanned from the measured corrected hold, never replayed from
the original pre-insertion joint state.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .axis_jacobian import (
    plan_jacobian_linear_stroke, plan_jacobian_tip_axis_alignment,
)
from .endpoint import screen_grasp_endpoint
from .geometry import validate_se3
from .grounded_alignment import AlignmentLimits
from .live_robot_state import LiveRobotState
from .observer import LabeledFrame
from .path_audit import PathAuditLimits, _audit_held_geometry_samples
from .preflight import _path, plan_held_transfer_and_axial
from .raw_camera_capture import RawCameraCapture, verify_raw_camera_capture
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


class CorrectedAxialPlanUnavailable(RuntimeError):
    """A saved corrected 20 mm plan failed before insertion motion."""

    def __init__(self, report_path: Path, status: str):
        self.report_path = Path(report_path)
        self.status = status
        super().__init__(f"corrected 20 mm plan failed: {status}")


class HeldXYPlanUnavailable(RuntimeError):
    """A held lateral plan failed before any XY motion was commanded."""

    def __init__(self, report_path: Path, status: str):
        self.report_path = Path(report_path)
        self.status = status
        super().__init__(f"held XY plan failed: {status}")


def choose_xy_move(alignment: dict, *, tolerance_m: float,
                   max_step_m: float = .010) -> dict:
    """Use multiview XY or a calibrated ray on the known tip-height plane.

    A cylinder's radial CAD gap is the default tolerance. An operator may
    choose a smaller target, but no extra confidence or key-hand error limit
    is introduced here. The step cap is the existing 10 mm motion policy.
    """
    if not (math.isfinite(tolerance_m) and tolerance_m > 0 and
            math.isfinite(max_step_m) and max_step_m > 0):
        raise ValueError("XY target and step must be positive metres")
    inliers = alignment.get("inlier_cameras")
    status = alignment.get("status")
    n_views = (len(set(inliers)) if isinstance(inliers, list) else 0)
    if not ((status == "diagnostic_metric_xy_correction" and n_views >= 2) or
            (status == "diagnostic_planar_xy_hypothesis" and n_views == 1)):
        return {"status": "not_metric_multiview", "xy_error_m": None,
                "increment_socket_m": None}
    correction = np.asarray(alignment.get("xy_correction_socket_m"), dtype=float)
    if correction.shape != (2,) or not np.all(np.isfinite(correction)):
        raise ValueError("calibrated alignment lacks finite socket XY")
    error = float(np.linalg.norm(correction))
    source = alignment.get("xy_source", "current_metric_multiview")
    if error <= tolerance_m:
        return {"status": "aligned", "xy_error_m": error,
                "increment_socket_m": [0., 0.], "source": source}
    step = correction * min(1., max_step_m / error)
    return {"status": "shift", "xy_error_m": error,
            "increment_socket_m": step.tolist(), "source": source}


def choose_key_axis_move(alignment: dict, *, tolerance_rad: float,
                         max_step_rad: float) -> dict:
    """Choose the symmetry-reduced cylinder tilt correction.

    Axial yaw is deliberately absent: for the cylindrical key it does not
    change the insertion geometry.  A valid multi-view shaft estimate gives
    the minimal socket-frame rotation that maps the observed key axis onto
    the insertion direction ``-Z``.  The step bound only discretizes the
    closed-loop motion; it is not an admission/failure threshold.
    """
    if (not math.isfinite(tolerance_rad) or tolerance_rad <= 0 or
            not math.isfinite(max_step_rad) or max_step_rad <= 0):
        raise ValueError("axis tolerance and step must be positive radians")
    cameras = alignment.get("inlier_cameras")
    axis = np.asarray(alignment.get("insertion_axis_socket"), dtype=float)
    tip = np.asarray(alignment.get("tip_socket_m"), dtype=float)
    if (not isinstance(cameras, list) or len(set(cameras)) < 2 or
            axis.shape != (3,) or tip.shape != (3,) or
            not np.all(np.isfinite(axis)) or not np.all(np.isfinite(tip)) or
            np.linalg.norm(axis) <= 1e-9):
        return {"status": "axis_unobserved", "axis_error_rad": None,
                "tilt_rotvec_socket_rad": None, "tip_socket_m": None}
    axis /= np.linalg.norm(axis)
    target = np.array([0., 0., -1.])
    cross = np.cross(axis, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(axis @ target, -1., 1.))
    angle = math.atan2(sine, cosine)
    if angle <= tolerance_rad:
        rotvec = np.zeros(3)
        status = "axis_aligned"
    elif sine > 1e-9:
        rotvec = cross / sine * min(angle, max_step_rad)
        status = "tilt"
    else:
        # A 180-degree flip is not expected after the upright pickup, but the
        # minimum rotation is underdetermined. Do not invent an axis here;
        # the ordinary XY route remains usable.
        return {"status": "axis_unobserved", "axis_error_rad": angle,
                "tilt_rotvec_socket_rad": None,
                "tip_socket_m": tip.tolist()}
    return {
        "status": status,
        "axis_error_rad": angle,
        "axis_error_deg": math.degrees(angle),
        "tilt_rotvec_socket_rad": rotvec.tolist(),
        "tip_socket_m": tip.tolist(),
        "observed_axis_socket": axis.tolist(),
        "source": alignment.get("estimator", "multiview_tip_and_axis"),
    }


def _alignment_frame_inputs(
    bundle: Path, capture: RawCameraCapture | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Reuse the just-written live pixels; verify saved pixels for replay."""
    if capture is None:
        report = verify_raw_camera_capture(bundle, phase="preinsert")
        images = {}
        for camera in assessment_cameras(report["frame_evidence"]):
            image = cv2.imread(str(bundle / "images" / f"{camera}.png"),
                               cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("preinsert camera image is missing")
            images[camera] = image
        return report["frame_evidence"], images

    # write_raw_camera_capture has already verified the exposure identity and
    # persisted these exact images. Re-reading eight PNGs and replaying four
    # undistort operations on every same-trial shift adds no new observation.
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("phase") != "preinsert" or
            manifest.get("capture_id") != capture.capture_id or
            manifest.get("request_id") != capture.request_id or
            set(manifest.get("frame_evidence", {})) != set(capture.images_bgr)):
        raise ValueError("live preinsert frames differ from saved capture")
    return manifest["frame_evidence"], dict(capture.images_bgr)


def measure_preinsert_xy(*, runner, raw_bundle: Path, backend,
                         alignment_limits: AlignmentLimits,
                         index: int, nominal_tip_height_error_m: float,
                         nominal_key_xy_shift_socket_m=(0., 0.),
                         nominal_key_axis_raise_socket_m: float = 0.,
                         xy_estimator: str = "vlm",
                         sam_segmentor=None,
                         capture: RawCameraCapture | None = None) -> dict:
    """Call the existing calibrated multiview tip/axis estimator on raw PNGs."""
    bundle = Path(raw_bundle).expanduser().resolve()
    rows, images = _alignment_frame_inputs(bundle, capture)
    cameras = assessment_cameras(rows)
    frames = []
    for camera in cameras:
        image = images[camera]
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
            max_nominal_tip_height_error_m=nominal_tip_height_error_m,
            nominal_key_xy_shift_socket_m=nominal_key_xy_shift_socket_m,
            nominal_key_axis_raise_socket_m=nominal_key_axis_raise_socket_m)
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
            max_nominal_tip_height_error_m=nominal_tip_height_error_m,
            nominal_key_xy_shift_socket_m=nominal_key_xy_shift_socket_m,
            nominal_key_axis_raise_socket_m=nominal_key_axis_raise_socket_m)
        mask_records = []
    else:
        raise ValueError("same-trial XY supports vlm or sam_half_ellipse")
    return {
        "schema": "precision_insertion_same_trial_xy_observation_v1",
        "attempt_id": runner.active_attempt.attempt_id,
        "candidate_id": runner.active_attempt.candidate_id,
        "raw_bundle": str(bundle),
        "frame_source": ("just_saved_live_capture" if capture is not None
                         else "verified_saved_png"),
        "raw_manifest_sha256": _sha(bundle / "manifest.json"),
        "postlift_report_path": str(runner._postlift_report_path),
        "postlift_report_sha256": _sha(runner._postlift_report_path),
        "task_geometry_sha256": geometry_sha,
        "xy_estimator": xy_estimator,
        "nominal_key_xy_shift_socket_m": list(
            map(float, nominal_key_xy_shift_socket_m)),
        "nominal_key_axis_raise_socket_m": float(
            nominal_key_axis_raise_socket_m),
        "sam_mask_files": mask_records,
        "alignment": alignment,
        "vlm_observations": [item.to_record() for item in observations],
    }


def _held_wrist_goal(start_wrist: np.ndarray, key_origin_robot: np.ndarray,
                     socket: np.ndarray, increment_socket_m: np.ndarray,
                     axis_increment_socket_m: float,
                     yaw_increment_socket_rad: float,
                     tilt_rotvec_socket_rad=(0., 0., 0.)) -> np.ndarray:
    """Rotate the fixed grasp about the observed key tip, then translate it."""
    tilt = np.asarray(tilt_rotvec_socket_rad, dtype=float)
    if tilt.shape != (3,) or not np.all(np.isfinite(tilt)):
        raise ValueError("held-key tilt needs a finite socket-frame rotvec")
    rotvec_robot = (socket[:3, :3] @ tilt +
                    socket[:3, 2] * yaw_increment_socket_rad)
    turn, _ = cv2.Rodrigues(rotvec_robot)
    goal = start_wrist.copy()
    goal[:3, :3] = turn @ start_wrist[:3, :3]
    goal[:3, 3] = (
        key_origin_robot + turn @ (start_wrist[:3, 3] - key_origin_robot) +
        socket[:3, :2] @ increment_socket_m +
        socket[:3, 2] * axis_increment_socket_m)
    return goal


def rank_lateral_observation_offsets(*, runner, current_xy_socket_m,
                                     axis_raise_m: float,
                                     component_m: float) -> list[dict]:
    """Rank viewpoints by key/socket separation and camera-pair parallax.

    The pair score is minimum landmark separation times sine of the angle
    between viewing rays. It downweights nearly duplicate cameras without
    adding a pass/fail threshold. These are projected nominal tip positions,
    not measured key-tip pixels.
    A candidate still needs the normal held-path plan before any motion.
    """
    if not math.isfinite(component_m) or component_m <= 0:
        raise ValueError("observation lateral component must be positive")
    xy = np.asarray(current_xy_socket_m, float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("current nominal socket XY must be finite")
    socket = validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    geometry = json.loads(AssetPaths(
        Path(runner.shared_root), runner.mode).task_geometry.read_text())
    key = runner._postlift_preflight.targets.T_robot_key_preinsert
    tip = key @ np.array([0., 0.,
                          float(geometry["key_frame"]["tip_z_m"]), 1.])
    tip[:3] += socket[:3, :2] @ xy + socket[:3, 2] * axis_raise_m
    rim = socket @ np.array([0., 0.,
                             float(geometry["socket_rim_z_m"]), 1.])
    camera_calibration = runner.calibration.record["camera_calibration"]
    c2r = np.asarray(runner.calibration.record["c2r"], float)
    projections = []
    for camera, intrinsics in camera_calibration["intrinsics_full"].items():
        extrinsic = camera_calibration["extrinsics_full"].get(camera)
        if extrinsic is None:
            continue
        camera_robot = np.asarray(extrinsic, float) @ c2r
        K = np.asarray(intrinsics["K_undist"], float)

        def project(point, camera_robot=camera_robot, K=K,
                    intrinsics=intrinsics):
            camera_point = camera_robot @ point
            if camera_point[2] <= 0:
                return None
            pixel = K @ camera_point[:3]
            pixel = pixel[:2] / pixel[2]
            if (pixel[0] < 0 or pixel[0] >= intrinsics["width"] or
                    pixel[1] < 0 or pixel[1] >= intrinsics["height"]):
                return None
            return pixel

        rim_pixel = project(rim)
        if rim_pixel is not None:
            camera_center = np.linalg.inv(camera_robot)[:3, 3]
            projections.append((camera, project, rim_pixel, camera_center))
    d = component_m
    offsets = ((d, -d), (-d, d), (d, d), (-d, -d),
               (d, 0.), (-d, 0.), (0., d), (0., -d))
    ranked = []
    for offset in offsets:
        moved = tip.copy()
        moved[:3] += socket[:3, :2] @ np.asarray(offset)
        separation = {}
        directions = {}
        for camera, project, rim_pixel, camera_center in projections:
            pixel = project(moved)
            if pixel is not None:
                separation[camera] = float(np.linalg.norm(pixel - rim_pixel))
                toward_camera = camera_center - moved[:3]
                directions[camera] = toward_camera / np.linalg.norm(
                    toward_camera)
        if len(separation) >= 2:
            pair_scores = [
                (min(separation[first], separation[second]) *
                 float(np.linalg.norm(np.cross(
                     directions[first], directions[second]))),
                 first, second)
                for first, second in combinations(sorted(separation), 2)]
            best_pair = max(pair_scores)
            ranked.append({
                "offset_socket_m": list(offset),
                "best_parallax_pair": [best_pair[1], best_pair[2]],
                "pair_separation_parallax_score_px": best_pair[0],
                "second_smallest_separation_px": sorted(
                    separation.values())[1],
                "projected_separation_px_by_camera": separation,
            })
    return sorted(ranked, key=lambda row: row[
        "pair_separation_parallax_score_px"], reverse=True)


def plan_and_execute_shift(*, runner, planner, adapter, read_state,
                           motion_interlock, limits, path_limits,
                           increment_socket_m, residual_socket_m,
                           residual_source: str,
                           index: int,
                           axis_increment_socket_m: float = 0.,
                           current_axis_raise_socket_m: float = 0.,
                           yaw_increment_socket_rad: float = 0.,
                           tilt_rotvec_socket_rad=(0., 0., 0.),
                           observed_tip_socket_m=None,
                           observed_axis_socket=None,
                           action_tag: str = "") -> Path:
    """Move the rigidly held key while leaving all Inspire joints fixed."""
    increment = np.asarray(increment_socket_m, dtype=float)
    if increment.shape != (2,) or not np.all(np.isfinite(increment)):
        raise ValueError("held XY increment must contain two finite metres")
    if not (math.isfinite(axis_increment_socket_m) and
            math.isfinite(current_axis_raise_socket_m) and
            math.isfinite(yaw_increment_socket_rad)):
        raise ValueError("held axis translation and yaw must be finite")
    tilt = np.asarray(tilt_rotvec_socket_rad, dtype=float)
    if tilt.shape != (3,) or not np.all(np.isfinite(tilt)):
        raise ValueError("held key tilt must contain three finite radians")
    observed_axis = (None if observed_axis_socket is None else
                     np.asarray(observed_axis_socket, dtype=float))
    if (observed_axis is not None and
            (observed_axis.shape != (3,) or
             not np.all(np.isfinite(observed_axis)) or
             np.linalg.norm(observed_axis) <= 1e-9)):
        raise ValueError("observed key axis needs three finite components")
    start = read_state()
    if not isinstance(start, LiveRobotState):
        raise TypeError("same-trial shift needs measured FR3/Inspire state")
    socket = validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    start_wrist = validate_se3(planner.fk_wrist(start.full_q),
                               name="current measured wrist FK")
    residual = np.asarray(residual_socket_m, dtype=float)
    if residual.shape != (2,) or not np.all(np.isfinite(residual)):
        raise ValueError("held key residual must contain finite socket XY")
    key_at_hold = runner._postlift_preflight.targets.T_robot_key_preinsert.copy()
    key_at_hold[:3, 3] += (socket[:3, :2] @ residual +
                           socket[:3, 2] * current_axis_raise_socket_m)
    def unavailable(status: str) -> None:
        action = ("_tilt" if np.linalg.norm(tilt) else
                  "_yaw" if yaw_increment_socket_rad else
                  "_axis" if axis_increment_socket_m else "")
        report = _write(
            Path(runner._attempt_dir) / "same_trial_xy" / "plan_failures" /
            f"{index:03d}{action}{'_' + action_tag if action_tag else ''}.json", {
                "schema": "precision_insertion_same_trial_xy_plan_failure_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "candidate_id": runner.active_attempt.candidate_id,
                "status": status,
                "measured_start": start.to_record(),
                "targets": {"T_robot_key_preinsert": key_at_hold.tolist()},
                "increment_socket_m": increment.tolist(),
                "axis_increment_socket_m": axis_increment_socket_m,
                "yaw_increment_socket_rad": yaw_increment_socket_rad,
                "tilt_rotvec_socket_rad": tilt.tolist(),
                "observed_tip_socket_m": observed_tip_socket_m,
                "observed_axis_socket": observed_axis_socket,
                "current_axis_raise_socket_m": current_axis_raise_socket_m,
                "residual_socket_xy_m": residual.tolist(),
                "residual_source": residual_source,
                "scope": "failed_plan_no_xy_command",
            })
        raise HeldXYPlanUnavailable(report, status)

    if observed_tip_socket_m is None:
        pivot = key_at_hold[:3, 3]
    else:
        tip_socket = np.asarray(observed_tip_socket_m, dtype=float)
        if tip_socket.shape != (3,) or not np.all(np.isfinite(tip_socket)):
            raise ValueError("observed key tip needs three finite socket metres")
        pivot = (socket @ np.r_[tip_socket, 1.])[:3]
    if observed_axis is not None:
        observed_axis /= np.linalg.norm(observed_axis)
        target_axis = np.array([0., 0., -1.])
        cross = np.cross(observed_axis, target_axis)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(observed_axis @ target_axis, -1., 1.))
        if sine > 1e-9:
            full_rotvec_socket = cross / sine * math.atan2(sine, cosine)
            full_turn_robot, _ = cv2.Rodrigues(
                socket[:3, :3] @ full_rotvec_socket)
            # ``full_turn_robot`` maps the observed shaft to the nominal
            # insertion shaft. Invert it about the observed tip to recover
            # the current key pose used by the rigid held-object path audit.
            key_at_hold[:3, :3] = (
                full_turn_robot.T @ key_at_hold[:3, :3])
            key_at_hold[:3, 3] = (
                pivot + full_turn_robot.T @ (key_at_hold[:3, 3] - pivot))
    held_relation = np.linalg.inv(key_at_hold) @ start_wrist
    goal = _held_wrist_goal(
        start_wrist, pivot, socket, increment,
        axis_increment_socket_m, yaw_increment_socket_rad, tilt)
    scene = build_held_scene_from_trial(
        trial_scene=runner._preflight.trial_scene,
        calibration=runner.calibration)
    pure_xy = (np.linalg.norm(increment) > 0 and
               axis_increment_socket_m == 0. and
               yaw_increment_socket_rad == 0. and
               np.linalg.norm(tilt) == 0.)
    tip_axis_correction = bool(np.linalg.norm(tilt) > 0.)
    linear_plan_timing = None
    planning_backend = "curobo_endpoint"
    if pure_xy:
        delta = goal[:3, 3] - start_wrist[:3, 3]
        distance = float(np.linalg.norm(delta))
        query = plan_jacobian_linear_stroke(
            planner, start.full_q, start_wrist, goal,
            direction_robot=delta / distance, scene_cfg=scene,
            expected_travel_m=distance, step_m=min(.001, distance))
        if not query.success or query.trajectory is None:
            unavailable(query.failure_code or "held_xy_linear_unreachable")
        trajectory = _path(query.trajectory, "same-trial held XY",
                           start.full_q, start.full_q[7:])
        segment_count = len(query.step_records)
        linear_plan_timing = query.timing
        planning_backend = "damped_jacobian_linear"
    elif tip_axis_correction:
        translation = (socket[:3, :2] @ increment +
                       socket[:3, 2] * axis_increment_socket_m)
        query = plan_jacobian_tip_axis_alignment(
            planner, start.full_q, start_wrist,
            pivot_robot=pivot, translation_robot=translation,
            tilt_rotvec_robot=socket[:3, :3] @ tilt,
            scene_cfg=scene)
        if not query.success or query.trajectory is None:
            unavailable(query.failure_code or "held_tip_axis_unreachable")
        trajectory = _path(query.trajectory, "same-trial held tip/axis",
                           start.full_q, start.full_q[7:])
        segment_count = len(query.step_records)
        linear_plan_timing = query.timing
        planning_backend = "damped_jacobian_tip_axis"
    else:
        query = planner.plan_cartesian_pose(
            start.full_q, goal, scene_cfg=scene,
            include_obj_obstacle=False, return_result=True, lock_hand=True,
            timing_phase="precision_insertion_same_trial_xy")
        if not query.success or query.trajectory is None:
            unavailable("held_xy_pose_unreachable")
        trajectory = _path(query.trajectory, "same-trial held XY",
                           start.full_q, start.full_q[7:])
        segment_count = 1
    wrist = {"lateral": [validate_se3(planner.fk_wrist(q),
                    name=f"lateral FK[{i}]") for i, q in enumerate(trajectory)]}
    geometry_failures, distances, _ = _audit_held_geometry_samples(
        shared_root=Path(runner.shared_root), mode=runner.mode,
        calibration=runner.calibration, wrist=wrist,
        T_key_hand=held_relation, held_hand_q=start.full_q[7:],
        minimum_hand_clearance_m=path_limits.minimum_hand_clearance_m,
        stop_on_first_failure=True, collision_only_if_clear=True)
    if geometry_failures:
        unavailable("held_xy_path_intersects_frozen_world")
    folder = (Path(runner._attempt_dir) / "same_trial_xy" / "shifts" /
              f"{index:03d}{'_' + action_tag if action_tag else ''}")
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
    end_wrist = validate_se3(planner.fk_wrist(end.full_q),
                             name="measured lateral end FK")
    pivot_in_hand = np.linalg.inv(start_wrist) @ np.r_[pivot, 1.]
    measured_key_delta = (end_wrist @ pivot_in_hand)[:3] - pivot
    measured_increment = socket[:3, :2].T @ measured_key_delta
    measured_axis_increment = float(socket[:3, 2] @ measured_key_delta)
    measured_turn = socket[:3, :3].T @ (
        end_wrist[:3, :3] @ start_wrist[:3, :3].T) @ socket[:3, :3]
    measured_rotvec_socket = cv2.Rodrigues(measured_turn)[0].reshape(3)
    return _write(folder / "report.json", {
        "schema": "precision_insertion_same_trial_xy_shift_v2",
        "attempt_id": runner.active_attempt.attempt_id,
        "candidate_id": runner.active_attempt.candidate_id,
        "increment_socket_m": increment.tolist(),
        "measured_increment_socket_m": measured_increment.tolist(),
        "axis_increment_socket_m": axis_increment_socket_m,
        "yaw_increment_socket_rad": yaw_increment_socket_rad,
        "tilt_rotvec_socket_rad": tilt.tolist(),
        "measured_rotvec_socket_rad": measured_rotvec_socket.tolist(),
        "observed_tip_socket_m": observed_tip_socket_m,
        "observed_axis_socket": observed_axis_socket,
        "measured_axis_increment_socket_m": measured_axis_increment,
        "current_axis_raise_socket_m": current_axis_raise_socket_m,
        "key_xy_socket_m": residual.tolist(),
        "residual_source": residual_source,
        "cartesian_target_segments": segment_count,
        "planning_backend": planning_backend,
        "linear_plan_timing": linear_plan_timing,
        "sampled_geometry_clear": True,
        "collision_screen_distances_m": distances,
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
                                residual_source: str,
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
    folder = (Path(runner._attempt_dir) / "guarded_axial_handoffs" /
              f"{runner._guarded_axial_index:03d}")
    folder.mkdir(parents=True, exist_ok=False)
    candidate_dir = (AssetPaths(runner.shared_root, runner.mode).candidate_dir /
                     Path(*runner._preflight.selected_candidate_key))
    endpoint = screen_grasp_endpoint(
        shared_root=runner.shared_root, mode=runner.mode,
        candidate_dir=candidate_dir,
        minimum_hand_clearance_m=path_limits.minimum_hand_clearance_m,
        xy_offset_socket_m=targets.xy_offset_socket_m,
        T_key_hand_override=relation,
        hand_poses_override={"measured_hold": state.full_q[7:]},
        override_source="same_trial_corrected_XY_measured_wrist",
        cylinder_yaw_gauge_socket_rad=targets.cylinder_yaw_gauge_socket_rad)
    if not np.allclose(endpoint["T_socket_key_tested"],
                       np.linalg.inv(socket) @ targets.T_robot_key_verification,
                       atol=1e-8, rtol=0):
        raise ValueError("corrected endpoint and axial goal differ")
    if endpoint["endpoint_pass"] is not True:
        failure = _write(folder / "plan_failure.json", {
            "schema": "precision_insertion_same_trial_axial_plan_failure_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "candidate_id": runner.active_attempt.candidate_id,
            "status": "corrected_20mm_endpoint_rejected",
            "measured_start": state.to_record(),
            "total_xy_correction_socket_m": total.tolist(),
            "final_residual_socket_xy_m": residual.tolist(),
            "residual_source": residual_source,
            "targets": targets.to_record(),
            "endpoint": endpoint,
            "planning": None,
            "scope": "failed_plan_no_insertion_command",
        })
        raise CorrectedAxialPlanUnavailable(
            failure, "corrected_20mm_endpoint_rejected")
    planned = plan_held_transfer_and_axial(
        planner=planner, trial_scene=runner._preflight.trial_scene,
        shared_root=runner.shared_root, calibration=runner.calibration,
        targets=targets, start_q=state.full_q,
        held_hand_q=state.full_q[7:], held_hand_source="measured",
        limits=path_limits, axial_waypoint_step_m=.005,
        start_at_preinsert=True, exact_distances=False)
    if not planned.sampled_planning_pass or planned.axial_trajectory is None:
        failure = _write(folder / "plan_failure.json", {
            "schema": "precision_insertion_same_trial_axial_plan_failure_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "candidate_id": runner.active_attempt.candidate_id,
            "status": planned.status,
            "measured_start": state.to_record(),
            "total_xy_correction_socket_m": total.tolist(),
            "final_residual_socket_xy_m": residual.tolist(),
            "residual_source": residual_source,
            "targets": targets.to_record(),
            "endpoint": endpoint,
            "planning": planned.to_record(),
            "scope": "failed_plan_no_insertion_command",
        })
        raise CorrectedAxialPlanUnavailable(failure, planned.status)
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
        "final_residual_socket_xy_m": residual.tolist(),
        "residual_source": residual_source,
        "held_relation_scope": "observed_key_XY_and_measured_wrist_with_nominal_key_orientation",
        "inserted_key_pose_robot": targets.T_robot_key_verification.tolist(),
        "targets": targets.to_record(),
        "endpoint": endpoint,
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
