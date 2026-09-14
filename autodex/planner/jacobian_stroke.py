"""Continuous differential-IK planning for a pure world-Z wrist stroke.

The solver deliberately follows the local joint branch from the supplied
start state.  It creates 5 mm Cartesian targets, solves each target with a
damped least-squares Jacobian update seeded only by the preceding accepted
configuration, collision-checks the joint chord between targets, then turns
the accepted path into a dense C2 reference and validates that exact output.

This module is robot-neutral.  It only relies on the kinematics and collision
rollout owned by :class:`autodex.planner.GraspPlanner`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class JacobianStrokeOptions:
    """Numerical, geometric, and execution limits for one vertical stroke."""

    step_m: float = 0.005
    finite_difference_rad: float = 1.0e-4
    damping: float = 2.0e-2
    max_iterations: int = 24
    max_joint_step_rad: float = 0.10
    position_tolerance_m: float = 0.0015
    orientation_tolerance_rad: float = np.deg2rad(2.0)
    request_start_position_tolerance_m: float = 0.002
    request_start_orientation_tolerance_rad: float = np.deg2rad(2.0)
    rotation_weight_m_per_rad: float = 0.10
    max_waypoint_delta_rad: float = 0.45
    max_segment_joint_delta_rad: float = 0.02
    support_clearance_tolerance_m: float = 0.001
    monotonic_tolerance_m: float = 2.0e-4
    sample_dt_s: float = 0.01
    held_object_speed_scale: float = 0.40
    max_retime_iterations: int = 12


@dataclass
class JacobianStrokeResult:
    """Complete result of one signed world-Z continuation request."""

    success: bool
    trajectory: np.ndarray | None
    time_s: np.ndarray | None
    geometric_qpos: np.ndarray | None
    collision_checked_qpos: np.ndarray | None
    direction: str
    distance_m: float
    failure_code: str | None
    failure_detail: str | None
    step_records: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


def options_as_dict(options: JacobianStrokeOptions) -> dict[str, Any]:
    return asdict(options)


def _pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    dp = target[:3, 3] - current[:3, 3]
    dr = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return np.concatenate([dp, dr])


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(
        Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()))


def numerical_wrist_jacobian(
    fk_wrist: Callable[[np.ndarray], np.ndarray],
    q_full: np.ndarray,
    n_arm: int,
    eps: float,
) -> np.ndarray:
    """Central finite-difference 6 x arm-DOF world-frame Jacobian."""
    q = np.asarray(q_full, dtype=np.float64).copy()
    jacobian = np.empty((6, n_arm), dtype=np.float64)
    for joint in range(n_arm):
        positive = q.copy()
        negative = q.copy()
        positive[joint] += eps
        negative[joint] -= eps
        pose_positive = fk_wrist(positive)
        pose_negative = fk_wrist(negative)
        jacobian[:3, joint] = (
            pose_positive[:3, 3] - pose_negative[:3, 3]) / (2.0 * eps)
        jacobian[3:, joint] = Rotation.from_matrix(
            pose_positive[:3, :3] @ pose_negative[:3, :3].T
        ).as_rotvec() / (2.0 * eps)
    return jacobian


def _joint_bounds(planner, dof: int) -> tuple[np.ndarray, np.ndarray]:
    limits = planner._motion_gen.kinematics.get_joint_limits().position
    limits = limits.detach().cpu().numpy()
    return (np.asarray(limits[0, :dof], dtype=np.float64),
            np.asarray(limits[1, :dof], dtype=np.float64))


def _weighted_dls_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
    rotation_weight: float,
) -> tuple[np.ndarray, float, float]:
    weights = np.array([1.0, 1.0, 1.0,
                        rotation_weight, rotation_weight, rotation_weight])
    weighted_jacobian = weights[:, None] * jacobian
    weighted_error = weights * error
    hessian = (weighted_jacobian.T @ weighted_jacobian
               + damping**2 * np.eye(jacobian.shape[1]))
    delta_q = np.linalg.solve(
        hessian, weighted_jacobian.T @ weighted_error)
    singular = np.linalg.svd(weighted_jacobian, compute_uv=False)
    minimum = float(singular[-1]) if len(singular) else 0.0
    condition = (float(singular[0] / minimum)
                 if minimum > 1.0e-12 else float("inf"))
    return delta_q, minimum, condition


def _collision_constraints(planner) -> list[Any]:
    rollout = planner._motion_gen.rollout_fn
    return [rollout.primitive_collision_constraint,
            rollout.robot_self_collision_constraint]


def _enable_collision_constraints(planner) -> None:
    for constraint in _collision_constraints(planner):
        constraint.enable_cost()


def _check_state(planner, q_full: np.ndarray) -> tuple[bool, str | None]:
    import torch
    from curobo.types.robot import JointState

    position = torch.as_tensor(
        np.ascontiguousarray(np.asarray(q_full, dtype=np.float32).reshape(1, -1)),
        dtype=torch.float32, device=planner._tensor_args.device)
    _enable_collision_constraints(planner)
    try:
        valid, status = planner._motion_gen.check_start_state(
            JointState.from_position(position))
        return bool(valid), None if status is None else str(status)
    finally:
        # Some cuRobo diagnostic branches leave self collision disabled.
        _enable_collision_constraints(planner)


def _check_states_batch(
    planner,
    q_full: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, str | None, dict[str, Any]]:
    import torch

    qpos = np.asarray(q_full, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or not np.isfinite(qpos).all():
        raise ValueError("collision qpos must be a finite 2D array")
    rollout = planner._motion_gen.rollout_fn
    chunks: list[np.ndarray] = []
    _enable_collision_constraints(planner)
    try:
        with torch.inference_mode():
            for start in range(0, len(qpos), batch_size):
                host = np.ascontiguousarray(qpos[start:start + batch_size])
                gpu = torch.as_tensor(
                    host, dtype=torch.float32, device=planner._tensor_args.device)
                metrics = rollout.rollout_constraint(
                    gpu.unsqueeze(1), use_batch_env=False)
                feasible = metrics.feasible.reshape(len(host), -1).all(dim=1)
                chunks.append(feasible.detach().cpu().numpy().astype(bool, copy=False))
    finally:
        _enable_collision_constraints(planner)
    valid = np.concatenate(chunks) if chunks else np.empty(0, dtype=bool)
    invalid = np.flatnonzero(~valid)
    first = None if len(invalid) == 0 else int(invalid[0])
    status = None
    if first is not None:
        scalar_valid, status = _check_state(planner, qpos[first])
        if scalar_valid:
            status = "batch_scalar_feasibility_mismatch"
    return valid, status, {
        "backend": "curobo_rollout_batch",
        "batch_size": int(batch_size),
        "sample_count": int(len(qpos)),
        "chunk_count": int(len(chunks)),
        "first_invalid_index": first,
    }


def _fk_batch(planner, q_full: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    import torch

    qpos = np.asarray(q_full, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(qpos), batch_size):
            host = np.ascontiguousarray(qpos[start:start + batch_size])
            gpu = torch.as_tensor(
                host, dtype=torch.float32, device=planner._tensor_args.device)
            state = planner._motion_gen.kinematics.get_state(gpu)
            positions.append(state.ee_position.detach().cpu().numpy())
            quaternions.append(state.ee_quaternion.detach().cpu().numpy())
    position = np.concatenate(positions).astype(np.float64, copy=False)
    quat_wxyz = np.concatenate(quaternions).astype(np.float64, copy=False)
    transforms = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(qpos), 4, 4)).copy()
    transforms[:, :3, :3] = Rotation.from_quat(
        quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    transforms[:, :3, 3] = position
    return transforms


def _object_bottom_z(
    mesh_vertices: np.ndarray,
    object_transforms: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    transforms = np.asarray(object_transforms, dtype=np.float64)
    if transforms.ndim == 2:
        transforms = transforms[None, ...]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("attached-object mesh vertices must have shape (N, 3)")
    return np.min(
        np.einsum("si,vi->sv", transforms[:, 2, :3], vertices)
        + transforms[:, 2, 3, None],
        axis=1,
    )


def _segment_samples(
    q_start: np.ndarray,
    q_end: np.ndarray,
    *,
    n_arm: int,
    hand: np.ndarray,
    max_joint_delta: float,
) -> np.ndarray:
    maximum = float(np.max(np.abs(q_end[:n_arm] - q_start[:n_arm])))
    count = max(1, int(np.ceil(maximum / max_joint_delta)))
    alpha = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    samples = ((1.0 - alpha[:, None]) * q_start[None, :]
               + alpha[:, None] * q_end[None, :])
    samples[:, n_arm:] = hand
    return samples.astype(np.float32)


def _arm_execution_limits(n_arm: int) -> tuple[float, float]:
    if n_arm == 7:  # FR3
        return 1.2, 4.0
    if n_arm == 6:  # XArm6
        return 0.50, 1.50
    raise ValueError(f"no vertical-stroke execution limits for {n_arm}-DOF arm")


def _sample_times(duration_s: float, dt_s: float) -> np.ndarray:
    intervals = int(round(duration_s / dt_s))
    if intervals < 1 or not np.isclose(intervals * dt_s, duration_s,
                                        atol=1.0e-9, rtol=0.0):
        raise ValueError("retimed duration must align with the sample period")
    return np.arange(intervals + 1, dtype=np.float64) * dt_s


def _derivative_peaks(spline: CubicSpline, knot_t: np.ndarray) -> tuple[float, float]:
    c0, c1, c2, _ = np.asarray(spline.c, dtype=np.float64)
    width = np.diff(knot_t)[:, None]
    velocity_start = c2
    velocity_end = 3.0 * c0 * width**2 + 2.0 * c1 * width + c2
    max_velocity = float(max(np.max(np.abs(velocity_start)),
                             np.max(np.abs(velocity_end))))
    acceleration_start = 2.0 * c1
    acceleration_end = 6.0 * c0 * width + 2.0 * c1
    max_acceleration = float(max(np.max(np.abs(acceleration_start)),
                                 np.max(np.abs(acceleration_end))))
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = -c1 / (3.0 * c0)
    valid = np.isfinite(stationary) & (stationary > 0.0) & (stationary < width)
    if np.any(valid):
        values = 3.0 * c0 * stationary**2 + 2.0 * c1 * stationary + c2
        max_velocity = max(max_velocity, float(np.max(np.abs(values[valid]))))
    return max_velocity, max_acceleration


def _retime_c2(
    arm_nodes: np.ndarray,
    *,
    velocity_limit: float,
    acceleration_limit: float,
    options: JacobianStrokeOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    nodes = np.asarray(arm_nodes, dtype=np.float64)
    delta = np.max(np.abs(np.diff(nodes, axis=0)), axis=1)
    segment_dt = np.maximum.reduce((
        1.75 * delta / velocity_limit,
        np.sqrt(6.0 * delta / acceleration_limit),
        np.full(len(delta), 1.0e-3),
    ))
    knot_t = np.r_[0.0, np.cumsum(segment_dt)]
    for iteration in range(1, options.max_retime_iterations + 1):
        segment_dt = np.ceil(np.diff(knot_t) / options.sample_dt_s) * options.sample_dt_s
        knot_t = np.r_[0.0, np.cumsum(segment_dt)]
        spline = CubicSpline(
            knot_t, nodes, axis=0,
            bc_type=((1, np.zeros(nodes.shape[1])),
                     (1, np.zeros(nodes.shape[1]))),
        )
        sample_t = _sample_times(float(knot_t[-1]), options.sample_dt_s)
        arm = np.asarray(spline(sample_t), dtype=np.float64)
        max_velocity, max_acceleration = _derivative_peaks(spline, knot_t)
        stretch = max(1.0, max_velocity / velocity_limit,
                      np.sqrt(max_acceleration / acceleration_limit))
        if stretch <= 1.0005:
            return arm.astype(np.float32), sample_t, {
                "method": "clamped_c2_cubic_spline",
                "retime_iterations": iteration,
                "knot_count": int(len(nodes)),
                "sample_count": int(len(arm)),
                "duration_s": float(sample_t[-1]),
                "max_joint_velocity_rad_s": max_velocity,
                "max_joint_acceleration_rad_s2": max_acceleration,
                "velocity_limit_rad_s": velocity_limit,
                "acceleration_limit_rad_s2": acceleration_limit,
            }
        knot_t *= stretch * 1.01
    raise RuntimeError("vertical_stroke_retime_limit_not_reached")


def _failure(
    *,
    code: str,
    detail: str | None,
    direction: str,
    distance_m: float,
    records: list[dict[str, Any]],
    started: float,
    geometric: np.ndarray | None = None,
    checked: np.ndarray | None = None,
    timing: dict[str, float] | None = None,
) -> JacobianStrokeResult:
    measured = dict(timing or {})
    measured["total_s"] = perf_counter() - started
    return JacobianStrokeResult(
        success=False, trajectory=None, time_s=None,
        geometric_qpos=geometric, collision_checked_qpos=checked,
        direction=direction, distance_m=distance_m,
        failure_code=code, failure_detail=detail,
        step_records=records, timing=measured,
    )


def plan_jacobian_vertical_stroke(
    planner,
    start_full_qpos: np.ndarray,
    target_wrist_pose: np.ndarray,
    *,
    options: JacobianStrokeOptions | None = None,
    attached_object_vertices: np.ndarray | None = None,
    attached_object_pose_at_start: np.ndarray | None = None,
    support_surface_z_m: float | None = None,
    expected_travel_m: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JacobianStrokeResult:
    """Plan and fully validate one pure +Z or -Z wrist stroke.

    The active MotionGen world must already match the requested collision
    semantics.  When an object is attached, the target mesh itself should be
    absent from that world; its rigid transform and table clearance are
    checked separately here.
    """
    started = perf_counter()
    options = options or JacobianStrokeOptions()
    start = np.asarray(start_full_qpos, dtype=np.float64).reshape(-1)
    target = np.asarray(target_wrist_pose, dtype=np.float64)
    n_arm = int(planner._n_arm)
    if (start.shape != np.asarray(planner._init_state).shape
            or not np.isfinite(start).all()):
        raise ValueError("vertical stroke start q must be finite and match planner DOF")
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("vertical stroke target must be a finite 4x4 pose")
    if (options.step_m <= 0.0 or options.finite_difference_rad <= 0.0
            or options.max_iterations <= 0 or options.max_segment_joint_delta_rad <= 0.0
            or options.sample_dt_s <= 0.0):
        raise ValueError("vertical stroke numerical and sampling options must be positive")
    hand = start[n_arm:].copy()
    payload_enabled = attached_object_pose_at_start is not None
    if payload_enabled != (attached_object_vertices is not None):
        raise ValueError("attached object pose and mesh vertices must be supplied together")
    if payload_enabled and support_surface_z_m is None:
        raise ValueError("attached-object validation requires a support surface z")

    start_fk = planner.fk_wrist(start.astype(np.float32))
    lateral_request_error = float(np.linalg.norm(
        target[:2, 3] - start_fk[:2, 3]))
    rotation_request_error = _rotation_error(target, start_fk)
    if lateral_request_error > options.request_start_position_tolerance_m:
        return _failure(
            code="jacobian_request_start_lateral_mismatch",
            detail=f"requested x/y differs from start FK by {lateral_request_error:.6f}m",
            direction="unknown", distance_m=0.0, records=[], started=started)
    if rotation_request_error > options.request_start_orientation_tolerance_rad:
        return _failure(
            code="jacobian_request_start_orientation_mismatch",
            detail=("requested orientation differs from start FK by "
                    f"{rotation_request_error:.6f}rad"),
            direction="unknown", distance_m=0.0, records=[], started=started)
    signed_distance = float(target[2, 3] - start_fk[2, 3])
    distance = abs(signed_distance)
    if (expected_travel_m is not None
            and abs(distance - float(expected_travel_m))
            > options.request_start_position_tolerance_m):
        return _failure(
            code="jacobian_request_start_z_mismatch",
            detail=(f"FK-relative travel={distance:.6f}m, "
                    f"requested={float(expected_travel_m):.6f}m"),
            direction="unknown", distance_m=distance, records=[], started=started)
    if distance <= options.position_tolerance_m:
        return _failure(
            code="jacobian_zero_vertical_travel",
            detail=f"vertical travel {distance:.6f}m is within tolerance",
            direction="none", distance_m=distance, records=[], started=started)
    sign = 1.0 if signed_distance > 0.0 else -1.0
    direction = "+Z" if sign > 0.0 else "-Z"

    start_valid, start_status = _check_state(planner, start.astype(np.float32))
    if not start_valid:
        return _failure(
            code="jacobian_start_robot_collision",
            detail=start_status, direction=direction, distance_m=distance,
            records=[], started=started)

    object_in_wrist = None
    if payload_enabled:
        object_start = np.asarray(attached_object_pose_at_start, dtype=np.float64)
        if object_start.shape != (4, 4) or not np.isfinite(object_start).all():
            raise ValueError("attached object start pose must be a finite 4x4 matrix")
        object_in_wrist = np.linalg.inv(start_fk) @ object_start
        bottom = float(_object_bottom_z(attached_object_vertices, object_start)[0])
        if bottom < float(support_surface_z_m) - options.support_clearance_tolerance_m:
            return _failure(
                code="jacobian_start_object_below_support",
                detail=f"object bottom z={bottom:.6f}m",
                direction=direction, distance_m=distance, records=[], started=started)

    lower, upper = _joint_bounds(planner, len(start))
    step_count = int(np.ceil(distance / options.step_m))
    q_previous = start.copy()
    geometric = [start.astype(np.float32)]
    checked = [start.astype(np.float32)]
    records: list[dict[str, Any]] = []
    solve_s = 0.0
    chord_validation_s = 0.0

    for step in range(1, step_count + 1):
        step_started = perf_counter()
        waypoint_target = start_fk.copy()
        waypoint_target[2, 3] += sign * min(step * options.step_m, distance)
        q = q_previous.copy()
        success = False
        reason = None
        minimum_singular = float("nan")
        condition = float("nan")
        position_error = float("inf")
        orientation_error = float("inf")
        joint_limit_hit = False
        iterations = 0
        fk_error_s = 0.0
        jacobian_s = 0.0
        dls_solve_s = 0.0
        segment_validation_s = 0.0
        segment_collision_valid: bool | None = None
        segment_object_clearance_valid: bool | None = None
        local_solve_started = perf_counter()
        for iteration in range(1, options.max_iterations + 1):
            iterations = iteration
            operation_started = perf_counter()
            current = planner.fk_wrist(q.astype(np.float32))
            error = _pose_error(waypoint_target, current)
            position_error = float(np.linalg.norm(error[:3]))
            orientation_error = float(np.linalg.norm(error[3:]))
            fk_error_s += perf_counter() - operation_started
            if (position_error <= options.position_tolerance_m
                    and orientation_error <= options.orientation_tolerance_rad):
                success = True
                break
            operation_started = perf_counter()
            jacobian = numerical_wrist_jacobian(
                planner.fk_wrist, q, n_arm, options.finite_difference_rad)
            jacobian_s += perf_counter() - operation_started
            try:
                operation_started = perf_counter()
                delta_q, minimum_singular, condition = _weighted_dls_step(
                    jacobian, error, options.damping,
                    options.rotation_weight_m_per_rad)
                dls_solve_s += perf_counter() - operation_started
            except np.linalg.LinAlgError:
                dls_solve_s += perf_counter() - operation_started
                reason = "jacobian_dls_solve_failed"
                break
            delta_q = np.clip(
                delta_q, -options.max_joint_step_rad, options.max_joint_step_rad)
            if not np.isfinite(delta_q).all():
                reason = "jacobian_non_finite_joint_update"
                break
            proposed = q[:n_arm] + delta_q
            q[:n_arm] = np.clip(proposed, lower[:n_arm], upper[:n_arm])
            joint_limit_hit = bool(joint_limit_hit or np.any(
                np.abs(proposed - q[:n_arm]) > 1.0e-10))
            q[n_arm:] = hand
        local_solve_s = perf_counter() - local_solve_started
        solve_s += local_solve_s

        delta_norm = float(np.linalg.norm(q[:n_arm] - q_previous[:n_arm]))
        segment_count = 0
        failure_alpha = None
        if not success:
            reason = reason or (
                "jacobian_joint_limit" if joint_limit_hit
                else "jacobian_residual_not_converged")
        elif delta_norm > options.max_waypoint_delta_rad:
            success = False
            reason = "jacobian_branch_jump"
        else:
            segment = _segment_samples(
                q_previous, q, n_arm=n_arm, hand=hand,
                max_joint_delta=options.max_segment_joint_delta_rad)
            segment_count = len(segment) - 1
            validation_started = perf_counter()
            valid, collision_status, _ = _check_states_batch(planner, segment[1:])
            invalid = np.flatnonzero(~valid)
            segment_collision_valid = len(invalid) == 0
            if len(invalid):
                success = False
                first = int(invalid[0])
                failure_alpha = float((first + 1) / max(segment_count, 1))
                reason = "jacobian_segment_robot_collision"
                if collision_status:
                    reason += f":{collision_status}"
            if success and payload_enabled:
                wrist = _fk_batch(planner, segment[1:])
                object_poses = wrist @ object_in_wrist
                bottoms = _object_bottom_z(attached_object_vertices, object_poses)
                below = np.flatnonzero(
                    bottoms < float(support_surface_z_m)
                    - options.support_clearance_tolerance_m)
                segment_object_clearance_valid = len(below) == 0
                if len(below):
                    success = False
                    first = int(below[0])
                    failure_alpha = float((first + 1) / max(segment_count, 1))
                    reason = "jacobian_segment_object_below_support"
            segment_validation_s = perf_counter() - validation_started
            chord_validation_s += segment_validation_s
            if success:
                checked.extend(segment[1:])

        record = {
            "step": step,
            "target_z_m": float(waypoint_target[2, 3]),
            "direction": direction,
            "success": success,
            "failure_code": None if success else str(reason).split(":", 1)[0],
            "failure_detail": None if success else reason,
            "iterations": iterations,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "delta_q_norm": delta_norm,
            "min_singular_value": minimum_singular,
            "condition_number": condition,
            "joint_limit_hit": joint_limit_hit,
            "segment_samples": segment_count,
            "segment_failure_alpha": failure_alpha,
            "segment_collision_valid": segment_collision_valid,
            "segment_object_clearance_valid": segment_object_clearance_valid,
            "fk_error_s": fk_error_s,
            "jacobian_s": jacobian_s,
            "dls_solve_s": dls_solve_s,
            "correction_s": local_solve_s,
            # Endpoint feasibility is included in the same GPU batch as all
            # other segment samples; there is deliberately no duplicate
            # scalar endpoint check on the successful path.
            "endpoint_validation_s": 0.0,
            "segment_validation_s": segment_validation_s,
            "total_s": perf_counter() - step_started,
        }
        records.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if not success:
            return _failure(
                code=record["failure_code"], detail=record["failure_detail"],
                direction=direction, distance_m=distance, records=records,
                started=started, geometric=np.stack(geometric),
                checked=np.stack(checked),
                timing={"continuation_s": solve_s,
                        "chord_validation_s": chord_validation_s})
        q_previous = q
        geometric.append(q.astype(np.float32))

    geometric_qpos = np.stack(geometric).astype(np.float32)
    checked_qpos = np.stack(checked).astype(np.float32)
    velocity_limit, acceleration_limit = _arm_execution_limits(n_arm)
    if payload_enabled:
        velocity_limit *= options.held_object_speed_scale
        acceleration_limit *= options.held_object_speed_scale
    retime_started = perf_counter()
    try:
        arm_trajectory, time_s, retime = _retime_c2(
            checked_qpos[:, :n_arm], velocity_limit=velocity_limit,
            acceleration_limit=acceleration_limit, options=options)
    except RuntimeError as exc:
        return _failure(
            code="jacobian_execution_retime_failed", detail=str(exc),
            direction=direction, distance_m=distance, records=records,
            started=started, geometric=geometric_qpos, checked=checked_qpos,
            timing={"continuation_s": solve_s,
                    "chord_validation_s": chord_validation_s,
                    "retime_s": perf_counter() - retime_started})
    trajectory = np.concatenate([
        arm_trajectory,
        np.broadcast_to(hand.astype(np.float32),
                        (len(arm_trajectory), len(hand))),
    ], axis=1).astype(np.float32)
    retime_s = perf_counter() - retime_started

    final_validation_started = perf_counter()
    outside_limits = np.argwhere(
        (trajectory < lower[None, :] - 1.0e-6)
        | (trajectory > upper[None, :] + 1.0e-6))
    if len(outside_limits):
        sample_index, joint_index = (int(value) for value in outside_limits[0])
        return _failure(
            code="jacobian_execution_joint_limit",
            detail=f"sample={sample_index}, joint={joint_index}",
            direction=direction, distance_m=distance, records=records,
            started=started, geometric=geometric_qpos, checked=checked_qpos,
            timing={"continuation_s": solve_s,
                    "chord_validation_s": chord_validation_s,
                    "retime_s": retime_s,
                    "execution_validation_s": perf_counter() - final_validation_started})
    valid, collision_status, collision_batch = _check_states_batch(planner, trajectory)
    invalid = np.flatnonzero(~valid)
    if len(invalid):
        index = int(invalid[0])
        return _failure(
            code="jacobian_execution_robot_collision",
            detail=f"sample={index}, status={collision_status}",
            direction=direction, distance_m=distance, records=records,
            started=started, geometric=geometric_qpos, checked=checked_qpos,
            timing={"continuation_s": solve_s,
                    "chord_validation_s": chord_validation_s,
                    "retime_s": retime_s,
                    "execution_validation_s": perf_counter() - final_validation_started})
    wrist = _fk_batch(planner, trajectory)
    lateral = np.linalg.norm(wrist[:, :2, 3] - start_fk[None, :2, 3], axis=1)
    relative_rotation = (
        wrist[:, :3, :3] @ start_fk[None, :3, :3].transpose(0, 2, 1))
    rotation = np.linalg.norm(
        Rotation.from_matrix(relative_rotation).as_rotvec(), axis=1)
    progress = sign * (wrist[:, 2, 3] - start_fk[2, 3])
    failure_code = None
    failure_detail = None
    if np.max(lateral) > options.position_tolerance_m:
        failure_code = "jacobian_execution_lateral_deviation"
        failure_detail = f"max={float(np.max(lateral)):.6f}m"
    elif np.max(rotation) > options.orientation_tolerance_rad:
        failure_code = "jacobian_execution_orientation_deviation"
        failure_detail = f"max={float(np.max(rotation)):.6f}rad"
    elif np.any(np.diff(progress) < -options.monotonic_tolerance_m):
        failure_code = "jacobian_execution_nonmonotonic_z"
    elif abs(float(progress[-1]) - distance) > options.position_tolerance_m:
        failure_code = "jacobian_execution_final_height_deviation"
        failure_detail = f"error={abs(float(progress[-1]) - distance):.6f}m"
    object_bottom_min = None
    if failure_code is None and payload_enabled:
        object_poses = wrist @ object_in_wrist
        bottoms = _object_bottom_z(attached_object_vertices, object_poses)
        object_bottom_min = float(np.min(bottoms))
        if object_bottom_min < (float(support_surface_z_m)
                                - options.support_clearance_tolerance_m):
            failure_code = "jacobian_execution_object_below_support"
            failure_detail = f"minimum bottom z={object_bottom_min:.6f}m"
    validation_s = perf_counter() - final_validation_started
    if failure_code is not None:
        return _failure(
            code=failure_code, detail=failure_detail,
            direction=direction, distance_m=distance, records=records,
            started=started, geometric=geometric_qpos, checked=checked_qpos,
            timing={"continuation_s": solve_s,
                    "chord_validation_s": chord_validation_s,
                    "retime_s": retime_s,
                    "execution_validation_s": validation_s})

    timing = {
        "continuation_s": solve_s,
        "chord_validation_s": chord_validation_s,
        "retime_s": retime_s,
        "execution_validation_s": validation_s,
        "total_s": perf_counter() - started,
    }
    return JacobianStrokeResult(
        success=True, trajectory=trajectory, time_s=time_s,
        geometric_qpos=geometric_qpos,
        collision_checked_qpos=checked_qpos,
        direction=direction, distance_m=distance,
        failure_code=None, failure_detail=None, step_records=records,
        validation={
            "collision_batch": collision_batch,
            "retime": retime,
            "max_lateral_error_m": float(np.max(lateral)),
            "max_orientation_error_rad": float(np.max(rotation)),
            "final_travel_m": float(progress[-1]),
            "object_bottom_min_z_m": object_bottom_min,
            "hand_fixed": bool(np.allclose(
                trajectory[:, n_arm:], hand[None, :], atol=1.0e-7)),
        },
        timing=timing,
    )
