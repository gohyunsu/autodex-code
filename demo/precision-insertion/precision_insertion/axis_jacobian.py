"""Demo-local damped-Jacobian continuation along a straight Cartesian line.

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


# The smoothness term uses the same numerical scale as the stock DLS damping.
# It is an optimization preference, not an execution/failure threshold: setting
# it to zero reproduces the stock minimum-norm DLS update exactly.
DEFAULT_JOINT_INCREMENT_SMOOTHNESS = 2.0e-2


def _smooth_weighted_dls_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
    rotation_weight: float,
    preferred_update: np.ndarray,
    smoothness: float,
) -> tuple[np.ndarray, float, float]:
    """Solve one DLS step while preferring the preceding joint increment.

    The solved local objective is

    ``||W(J dq - e)||^2 + damping^2 ||dq||^2
       + smoothness^2 ||dq - dq_pref||^2``.

    ``dq_pref`` points from the current iterate to the constant-increment
    extrapolation of the two preceding accepted Cartesian waypoints.  Thus the
    extra term removes joint-branch chatter without changing the requested
    Cartesian waypoint or introducing another success/failure condition.
    """
    if smoothness == 0:
        return _weighted_dls_step(
            jacobian, error, damping, rotation_weight)
    weights = np.array([
        1.0, 1.0, 1.0,
        rotation_weight, rotation_weight, rotation_weight,
    ])
    weighted_jacobian = weights[:, None] * jacobian
    weighted_error = weights * error
    regularizer = damping**2 + smoothness**2
    hessian = (weighted_jacobian.T @ weighted_jacobian
               + regularizer * np.eye(jacobian.shape[1]))
    rhs = (weighted_jacobian.T @ weighted_error
           + smoothness**2 * np.asarray(preferred_update, dtype=np.float64))
    delta_q = np.linalg.solve(hessian, rhs)
    singular = np.linalg.svd(weighted_jacobian, compute_uv=False)
    minimum = float(singular[-1]) if len(singular) else 0.0
    condition = (float(singular[0] / minimum)
                 if minimum > 1.0e-12 else float("inf"))
    return delta_q, minimum, condition


def _fixed_world_cfg(scene_cfg: dict) -> dict:
    # The planner module imports cuRobo, so defer that import until an actual
    # planning request. File-only audits and unit tests remain importable.
    from autodex.planner.planner import _to_curobo_world, _without_target_mesh
    return _without_target_mesh(_to_curobo_world(scene_cfg))


def _plan_jacobian_linear_stroke(
    planner, start_full_qpos: np.ndarray, start_wrist_pose: np.ndarray,
    target_wrist_pose: np.ndarray, *, direction_robot: np.ndarray,
    scene_cfg: dict, expected_travel_m: float,
    step_m: float, options: JacobianStrokeOptions | None,
    failure_prefix: str, result_direction: str, waypoint_stage: str,
    request_description: str,
    joint_increment_smoothness: float,
) -> JacobianStrokeResult:
    """Follow collinear waypoints, seeding each from the preceding joints.

    The held key is removed from the cuRobo obstacle world and checked by the
    caller's full-key/hand/socket ``path_audit`` on the returned dense path.
    Stock numerical Jacobian, collision rollout, joint limits and retimer are
    reused without changing the stock world-Z stroke.
    """
    started = perf_counter()
    start = np.asarray(start_full_qpos, dtype=np.float64)
    reference = np.asarray(start_wrist_pose, dtype=np.float64)
    target = np.asarray(target_wrist_pose, dtype=np.float64)
    direction = np.asarray(direction_robot, dtype=np.float64)
    n_arm = int(planner._n_arm)
    if n_arm != 7 or start.shape != (13,) or not np.isfinite(start).all():
        raise ValueError("axis stroke needs finite FR3/Inspire joint state")
    if (reference.shape != (4, 4) or target.shape != (4, 4) or
            not np.isfinite(reference).all() or not np.isfinite(target).all() or
            direction.shape != (3,) or not np.isfinite(direction).all() or
            not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5) or
            not math.isfinite(expected_travel_m) or expected_travel_m <= 0 or
            not math.isfinite(step_m) or not 0 < step_m <= 0.005 or
            not math.isfinite(joint_increment_smoothness) or
            joint_increment_smoothness < 0):
        raise ValueError("linear stroke needs a fixed pose, unit direction and <=5 mm steps")
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
    axial = float(np.dot(delta, direction))
    lateral_request = float(np.linalg.norm(delta - axial * direction))
    if (abs(axial - expected_travel_m) > 1e-6 or lateral_request > 1e-6 or
            _rotation_error(target, reference) > 1e-6):
        raise ValueError(
            f"linear stroke target differs from {request_description}")
    if (not isinstance(scene_cfg, dict) or
            not isinstance(scene_cfg.get("mesh"), dict) or
            "fixture_socket" not in scene_cfg["mesh"] or
            "target" not in scene_cfg["mesh"]):
        raise ValueError("linear stroke needs the fresh key and frozen socket world")

    def code(suffix: str) -> str:
        return f"{failure_prefix}_{suffix}"

    def failure(code: str, detail: str | None = None, *,
                geometric=None, checked=None, records=None) -> JacobianStrokeResult:
        return JacobianStrokeResult(
            False, None, None, geometric, checked, result_direction,
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
        return failure(code("start_pose_mismatch"))
    valid, status = _check_state(planner, start.astype(np.float32))
    if not valid:
        return failure(code("start_robot_collision"), status)
    lower, upper = _joint_bounds(planner, 13)
    if np.any(start < lower - 1e-6) or np.any(start > upper + 1e-6):
        return failure(code("start_joint_limit"))
    hand = start[n_arm:].copy()
    previous = start.copy()
    geometric = [start.astype(np.float32)]
    checked = [start.astype(np.float32)]
    records = []
    count = int(math.ceil(expected_travel_m / step_m))
    for index in range(1, count + 1):
        distance = min(index * step_m, expected_travel_m)
        goal = reference.copy()
        goal[:3, 3] += distance * direction
        q = previous.copy()
        if len(geometric) >= 2:
            prior_increment = (
                previous[:n_arm]
                - np.asarray(geometric[-2], dtype=np.float64)[:n_arm])
        else:
            prior_increment = np.zeros(n_arm, dtype=np.float64)
        predicted_arm = np.clip(
            previous[:n_arm] + prior_increment,
            lower[:n_arm], upper[:n_arm])
        converged = False
        reason = code("jacobian_residual_not_converged")
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
                # Penalize deviation of the *next waypoint node* from a
                # constant-increment extrapolation.  Recomputing this vector
                # after every nonlinear iteration avoids repeatedly adding the
                # entire preceding increment.
                preferred_update = predicted_arm - q[:n_arm]
                update, _, _ = _smooth_weighted_dls_step(
                    jacobian, error, opts.damping,
                    opts.rotation_weight_m_per_rad,
                    preferred_update, joint_increment_smoothness)
            except np.linalg.LinAlgError:
                reason = code("jacobian_dls_failed")
                break
            update = np.clip(update, -opts.max_joint_step_rad,
                             opts.max_joint_step_rad)
            if not np.isfinite(update).all():
                reason = code("jacobian_nonfinite_update")
                break
            proposed = q[:n_arm] + update
            q[:n_arm] = np.clip(proposed, lower[:n_arm], upper[:n_arm])
            joint_limit_hit |= bool(np.any(abs(proposed - q[:n_arm]) > 1e-10))
            q[n_arm:] = hand
        else:
            iteration = opts.max_iterations
        if not converged and joint_limit_hit:
            reason = code("jacobian_joint_limit")
        if converged and np.linalg.norm(q[:n_arm] - previous[:n_arm]) > opts.max_waypoint_delta_rad:
            converged, reason = False, code("jacobian_branch_jump")
        segment = None
        collision_detail = None
        if converged:
            segment = _segment_samples(
                previous, q, n_arm=n_arm, hand=hand,
                max_joint_delta=opts.max_segment_joint_delta_rad)
            valid, status, collision_batch = _check_states_batch(
                planner, segment[1:])
            if not np.all(valid):
                converged, reason = False, code("segment_robot_collision")
                first = int(np.flatnonzero(~valid)[0])
                collision_detail = {
                    "curobo_status": status,
                    "first_invalid_segment_sample": first + 1,
                    "first_invalid_full_q": segment[first + 1].tolist(),
                    "collision_batch": collision_batch,
                }
            else:
                poses = _fk_batch(planner, segment[1:])
                metrics = _axis_metrics(poses, reference, direction)
                if (np.max(metrics["lateral_m"]) > opts.position_tolerance_m or
                        np.max(metrics["rotation_rad"]) > opts.orientation_tolerance_rad or
                        np.min(metrics["progress_m"]) < -opts.monotonic_tolerance_m or
                        np.max(metrics["progress_m"]) >
                        distance + opts.position_tolerance_m):
                    converged, reason = False, code("segment_geometric_deviation")
        record = {
            "stage": waypoint_stage, "index": index,
            "target_progress_m": distance, "T_robot_hand_goal": goal.tolist(),
            "success": converged, "failure_code": None if converged else reason,
            "iterations": iteration, "joint_limit_hit": joint_limit_hit,
            "segment_samples": 0 if segment is None else len(segment) - 1,
            "joint_increment_smoothness": joint_increment_smoothness,
            "prior_increment_norm_rad": float(np.linalg.norm(prior_increment)),
            "accepted_increment_norm_rad": (
                float(np.linalg.norm(q[:n_arm] - previous[:n_arm]))
                if converged else None),
            "increment_change_norm_rad": (
                float(np.linalg.norm(
                    (q[:n_arm] - previous[:n_arm]) - prior_increment))
                if converged else None),
        }
        if collision_detail is not None:
            record["collision_detail"] = collision_detail
        records.append(record)
        if not converged:
            return failure(reason, status if reason == code("segment_robot_collision")
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
        return failure(code("execution_retime_failed"), str(exc),
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    trajectory = np.concatenate([
        arm, np.broadcast_to(hand.astype(np.float32), (len(arm), len(hand)))],
        axis=1).astype(np.float32)
    if np.any(trajectory < lower - 1e-6) or np.any(trajectory > upper + 1e-6):
        return failure(code("execution_joint_limit"), geometric=geometric_array,
                       checked=checked_array, records=records)
    valid, status, collision = _check_states_batch(planner, trajectory)
    if not np.all(valid):
        return failure(code("execution_robot_collision"), status,
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    metrics = _axis_metrics(
        _fk_batch(planner, trajectory), reference, direction)
    progress = metrics["progress_m"]
    if (np.max(metrics["lateral_m"]) > opts.position_tolerance_m or
            np.max(metrics["rotation_rad"]) > opts.orientation_tolerance_rad or
            np.any(np.diff(progress) < -opts.monotonic_tolerance_m) or
            abs(progress[-1] - expected_travel_m) > opts.position_tolerance_m):
        return failure(code("execution_geometric_deviation"),
                       geometric=geometric_array, checked=checked_array,
                       records=records)
    return JacobianStrokeResult(
        True, trajectory, times, geometric_array, checked_array,
        result_direction, expected_travel_m, None, None, records,
        validation={
            "collision_batch": collision, "retime": retime,
            "max_lateral_error_m": float(np.max(metrics["lateral_m"])),
            "max_orientation_error_rad": float(np.max(metrics["rotation_rad"])),
            "final_travel_m": float(progress[-1]),
            "direction_robot": direction.tolist(),
            "axis_robot": direction.tolist(), "hand_fixed": bool(np.allclose(
                trajectory[:, n_arm:], hand, atol=1e-7)),
        }, timing={"total_s": perf_counter() - started})


def plan_jacobian_axis_stroke(
    planner, start_full_qpos: np.ndarray, start_wrist_pose: np.ndarray,
    target_wrist_pose: np.ndarray, *, axis_robot: np.ndarray,
    scene_cfg: dict, expected_travel_m: float,
    step_m: float, options: JacobianStrokeOptions | None = None,
    joint_increment_smoothness: float = DEFAULT_JOINT_INCREMENT_SMOOTHNESS,
) -> JacobianStrokeResult:
    """Follow the frozen socket axis from the preceding accepted joints."""
    return _plan_jacobian_linear_stroke(
        planner, start_full_qpos, start_wrist_pose, target_wrist_pose,
        direction_robot=axis_robot, scene_cfg=scene_cfg,
        expected_travel_m=expected_travel_m, step_m=step_m,
        options=options, failure_prefix="axis",
        result_direction="socket_axis", waypoint_stage="axial_waypoint",
        request_description="frozen socket-axis line",
        joint_increment_smoothness=joint_increment_smoothness)


def plan_jacobian_linear_stroke(
    planner, start_full_qpos: np.ndarray, start_wrist_pose: np.ndarray,
    target_wrist_pose: np.ndarray, *, direction_robot: np.ndarray,
    scene_cfg: dict, expected_travel_m: float,
    step_m: float = .001, options: JacobianStrokeOptions | None = None,
    joint_increment_smoothness: float = DEFAULT_JOINT_INCREMENT_SMOOTHNESS,
) -> JacobianStrokeResult:
    """Generate one short, direct Cartesian correction without MotionGen.

    Every waypoint is initialized from the preceding accepted joint state.
    The returned trajectory is retimed once and collision checked as a whole.
    """
    if options is None:
        base = _default_options_for_arm(int(planner._n_arm))
        # A waypoint must cause a meaningful update. The stock 1.5 mm axial
        # tolerance is larger than a 1 mm visual-correction waypoint.
        options = replace(
            base, position_tolerance_m=min(
                base.position_tolerance_m, step_m / 4.))
    return _plan_jacobian_linear_stroke(
        planner, start_full_qpos, start_wrist_pose, target_wrist_pose,
        direction_robot=direction_robot, scene_cfg=scene_cfg,
        expected_travel_m=expected_travel_m, step_m=step_m,
        options=options, failure_prefix="linear",
        result_direction="linear", waypoint_stage="linear_waypoint",
        request_description="requested Cartesian line",
        joint_increment_smoothness=joint_increment_smoothness)


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
