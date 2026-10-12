"""Demo-local damped-Jacobian continuation along the frozen socket axis.

This plans a kinematic reference only. Contact stop, force control and actual
stroke measurements belong to the commissioned robot-side controller.
"""

from __future__ import annotations

from dataclasses import replace
import math
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner.jacobian_stroke import (
    JacobianStrokeOptions, JacobianStrokeResult, _arm_execution_limits,
    _check_state, _check_states_batch, _default_options_for_arm, _fk_batch,
    _joint_bounds, _pose_error, _retime_c2, _rotation_error,
    _segment_samples, _weighted_dls_step, numerical_wrist_jacobian,
)


def _fixed_world_cfg(scene_cfg: dict) -> dict:
    # The planner module imports cuRobo, so defer that import until an actual
    # planning request. File-only audits and unit tests remain importable.
    from autodex.planner.planner import _to_curobo_world, _without_target_mesh
    return _without_target_mesh(_to_curobo_world(scene_cfg))


def plan_jacobian_axis_stroke(
    planner, start_full_qpos: np.ndarray, start_wrist_pose: np.ndarray,
    target_wrist_pose: np.ndarray, *, axis_robot: np.ndarray,
    scene_cfg: dict, expected_travel_m: float,
    step_m: float, options: JacobianStrokeOptions | None = None,
) -> JacobianStrokeResult:
    """Follow short socket-axis waypoints from the preceding accepted joints.

    The held key is removed from the cuRobo obstacle world and checked by the
    caller's full-key/hand/socket ``path_audit`` on the returned dense path.
    Stock numerical Jacobian, collision rollout, joint limits and retimer are
    reused without changing the stock world-Z stroke.
    """
    started = perf_counter()
    start = np.asarray(start_full_qpos, dtype=np.float64)
    reference = np.asarray(start_wrist_pose, dtype=np.float64)
    target = np.asarray(target_wrist_pose, dtype=np.float64)
    axis = np.asarray(axis_robot, dtype=np.float64)
    n_arm = int(planner._n_arm)
    if n_arm != 7 or start.shape != (13,) or not np.isfinite(start).all():
        raise ValueError("axis stroke needs finite FR3/Inspire joint state")
    if (reference.shape != (4, 4) or target.shape != (4, 4) or
            not np.isfinite(reference).all() or not np.isfinite(target).all() or
            axis.shape != (3,) or not np.isfinite(axis).all() or
            not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-5) or
            not math.isfinite(expected_travel_m) or expected_travel_m <= 0 or
            not math.isfinite(step_m) or not 0 < step_m <= 0.005):
        raise ValueError("axis stroke needs a fixed pose, unit axis and <=5 mm steps")
    opts = replace(options or _default_options_for_arm(n_arm), step_m=step_m)
    if (any(not math.isfinite(float(value)) for value in vars(opts).values()) or
            opts.finite_difference_rad <= 0 or opts.damping <= 0 or
            opts.max_iterations <= 0 or opts.max_joint_step_rad <= 0 or
            opts.position_tolerance_m <= 0 or
            opts.orientation_tolerance_rad <= 0 or
            opts.max_segment_joint_delta_rad <= 0 or
            opts.max_waypoint_delta_rad <= 0 or
            not 0 < opts.held_object_speed_scale <= 1):
        raise ValueError("axis stroke needs valid numerical limits")
    delta = target[:3, 3] - reference[:3, 3]
    axial = float(np.dot(delta, axis))
    lateral_request = float(np.linalg.norm(delta - axial * axis))
    if (abs(axial - expected_travel_m) > 1e-6 or lateral_request > 1e-6 or
            _rotation_error(target, reference) > 1e-6):
        raise ValueError("axis stroke target differs from frozen socket-axis line")
    if (not isinstance(scene_cfg, dict) or
            not isinstance(scene_cfg.get("mesh"), dict) or
            "fixture_socket" not in scene_cfg["mesh"] or
            "target" not in scene_cfg["mesh"]):
        raise ValueError("axis stroke needs the fresh key and frozen socket world")

    def failure(code: str, detail: str | None = None, *,
                geometric=None, checked=None, records=None) -> JacobianStrokeResult:
        return JacobianStrokeResult(
            False, None, None, geometric, checked, "socket_axis",
            expected_travel_m, code, detail, records or [],
            timing={"total_s": perf_counter() - started})

    # Match the exact trial's fixed table/socket world. The attached key is
    # absent as an obstacle; the separate solid/surface audit checks it.
    planner._set_motion_world(_fixed_world_cfg(scene_cfg))
    start_fk = planner.fk_wrist(start.astype(np.float32))
    if (np.linalg.norm(start_fk[:3, 3] - reference[:3, 3]) >
            opts.request_start_position_tolerance_m or
            _rotation_error(start_fk, reference) >
            opts.request_start_orientation_tolerance_rad):
        return failure("axis_start_pose_mismatch")
    valid, status = _check_state(planner, start.astype(np.float32))
    if not valid:
        return failure("axis_start_robot_collision", status)
    lower, upper = _joint_bounds(planner, 13)
    if np.any(start < lower - 1e-6) or np.any(start > upper + 1e-6):
        return failure("axis_start_joint_limit")
    hand = start[n_arm:].copy()
    previous = start.copy()
    geometric = [start.astype(np.float32)]
    checked = [start.astype(np.float32)]
    records = []
    count = int(math.ceil(expected_travel_m / step_m))
    for index in range(1, count + 1):
        distance = min(index * step_m, expected_travel_m)
        goal = reference.copy()
        goal[:3, 3] += distance * axis
        q = previous.copy()
        converged = False
        reason = "axis_jacobian_residual_not_converged"
        joint_limit_hit = False
        for iteration in range(1, opts.max_iterations + 1):
            current = planner.fk_wrist(q.astype(np.float32))
            error = _pose_error(goal, current)
            if (np.linalg.norm(error[:3]) <= opts.position_tolerance_m and
                    np.linalg.norm(error[3:]) <= opts.orientation_tolerance_rad):
                converged = True
                break
            jacobian = numerical_wrist_jacobian(
                planner.fk_wrist, q, n_arm, opts.finite_difference_rad)
            try:
                update, _, _ = _weighted_dls_step(
                    jacobian, error, opts.damping,
                    opts.rotation_weight_m_per_rad)
            except np.linalg.LinAlgError:
                reason = "axis_jacobian_dls_failed"
                break
            update = np.clip(update, -opts.max_joint_step_rad,
                             opts.max_joint_step_rad)
            if not np.isfinite(update).all():
                reason = "axis_jacobian_nonfinite_update"
                break
            proposed = q[:n_arm] + update
            q[:n_arm] = np.clip(proposed, lower[:n_arm], upper[:n_arm])
            joint_limit_hit |= bool(np.any(abs(proposed - q[:n_arm]) > 1e-10))
            q[n_arm:] = hand
        else:
            iteration = opts.max_iterations
        if not converged and joint_limit_hit:
            reason = "axis_jacobian_joint_limit"
        if converged and np.linalg.norm(q[:n_arm] - previous[:n_arm]) > opts.max_waypoint_delta_rad:
            converged, reason = False, "axis_jacobian_branch_jump"
        segment = None
        if converged:
            segment = _segment_samples(
                previous, q, n_arm=n_arm, hand=hand,
                max_joint_delta=opts.max_segment_joint_delta_rad)
            valid, status, _ = _check_states_batch(planner, segment[1:])
            if not np.all(valid):
                converged, reason = False, "axis_segment_robot_collision"
            else:
                poses = _fk_batch(planner, segment[1:])
                metrics = _axis_metrics(poses, reference, axis)
                if (np.max(metrics["lateral_m"]) > opts.position_tolerance_m or
                        np.max(metrics["rotation_rad"]) > opts.orientation_tolerance_rad or
                        np.min(metrics["progress_m"]) < -opts.monotonic_tolerance_m or
                        np.max(metrics["progress_m"]) >
                        distance + opts.position_tolerance_m):
                    converged, reason = False, "axis_segment_geometric_deviation"
        records.append({
            "stage": "axial_waypoint", "index": index,
            "target_progress_m": distance, "T_robot_hand_goal": goal.tolist(),
            "success": converged, "failure_code": None if converged else reason,
            "iterations": iteration, "joint_limit_hit": joint_limit_hit,
            "segment_samples": 0 if segment is None else len(segment) - 1,
        })
        if not converged:
            return failure(reason, status if reason == "axis_segment_robot_collision"
                           else None, geometric=np.stack(geometric),
                           checked=np.stack(checked), records=records)
        checked.extend(segment[1:])
        geometric.append(q.astype(np.float32))
        previous = q

    geometric_array = np.stack(geometric).astype(np.float32)
    checked_array = np.stack(checked).astype(np.float32)
    velocity, acceleration = _arm_execution_limits(n_arm)
    velocity *= opts.held_object_speed_scale
    acceleration *= opts.held_object_speed_scale
    try:
        arm, times, retime = _retime_c2(
            checked_array[:, :n_arm], velocity_limit=velocity,
            acceleration_limit=acceleration, options=opts)
    except RuntimeError as exc:
        return failure("axis_execution_retime_failed", str(exc),
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    trajectory = np.concatenate([
        arm, np.broadcast_to(hand.astype(np.float32), (len(arm), len(hand)))],
        axis=1).astype(np.float32)
    if np.any(trajectory < lower - 1e-6) or np.any(trajectory > upper + 1e-6):
        return failure("axis_execution_joint_limit", geometric=geometric_array,
                       checked=checked_array, records=records)
    valid, status, collision = _check_states_batch(planner, trajectory)
    if not np.all(valid):
        return failure("axis_execution_robot_collision", status,
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    metrics = _axis_metrics(_fk_batch(planner, trajectory), reference, axis)
    progress = metrics["progress_m"]
    if (np.max(metrics["lateral_m"]) > opts.position_tolerance_m or
            np.max(metrics["rotation_rad"]) > opts.orientation_tolerance_rad or
            np.any(np.diff(progress) < -opts.monotonic_tolerance_m) or
            abs(progress[-1] - expected_travel_m) > opts.position_tolerance_m):
        return failure("axis_execution_geometric_deviation",
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    return JacobianStrokeResult(
        True, trajectory, times, geometric_array, checked_array,
        "socket_axis", expected_travel_m, None, None, records,
        validation={
            "collision_batch": collision, "retime": retime,
            "max_lateral_error_m": float(np.max(metrics["lateral_m"])),
            "max_orientation_error_rad": float(np.max(metrics["rotation_rad"])),
            "final_travel_m": float(progress[-1]),
            "axis_robot": axis.tolist(), "hand_fixed": bool(np.allclose(
                trajectory[:, n_arm:], hand, atol=1e-7)),
        }, timing={"total_s": perf_counter() - started})


def _axis_metrics(poses: np.ndarray, reference: np.ndarray,
                  axis: np.ndarray) -> dict[str, np.ndarray]:
    delta = poses[:, :3, 3] - reference[None, :3, 3]
    progress = delta @ axis
    lateral = np.linalg.norm(delta - progress[:, None] * axis[None, :], axis=1)
    relative = poses[:, :3, :3] @ reference[None, :3, :3].transpose(0, 2, 1)
    rotation = np.linalg.norm(
        Rotation.from_matrix(relative).as_rotvec(), axis=1)
    return {"progress_m": progress, "lateral_m": lateral,
            "rotation_rad": rotation}
