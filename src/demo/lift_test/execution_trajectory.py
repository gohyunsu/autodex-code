"""Robot-neutral time parameterization for a validated pickup path.

The Jacobian solver emits geometric continuation nodes.  This module turns
them into a dense, timestamped reference trajectory without calling a second
motion planner (which could choose another IK branch).  The approach and lift
paths are clamped C2 splines through their existing joint nodes; callers must
collision/FK-validate the resulting samples in their actual planning world
before treating them as executable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class ExecutionProfile:
    """Conservative reference limits before a robot adapter applies its caps."""

    arm: str
    max_joint_velocity_rad_s: float
    max_joint_acceleration_rad_s2: float
    held_object_speed_scale: float = 0.4
    sample_dt_s: float = 0.01
    squeeze_duration_s: float = 0.50
    max_retime_iterations: int = 12

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_for_arm(arm: str, *, sample_dt_s: float = 0.01,
                    squeeze_duration_s: float = 0.50) -> ExecutionProfile:
    """Return the conservative common trajectory contract for one adapter.

    Franka's values match its streaming follower caps.  XArm's values are
    intentionally lower than the low-level servo's per-tick limit; this gives
    the same carried-object 0.4x policy a meaningful, conservative reference
    speed before the hardware adapter performs its own final limiting.
    """
    if sample_dt_s <= 0.0 or squeeze_duration_s <= 0.0:
        raise ValueError("execution sample dt and squeeze duration must be positive")
    if arm == "franka":
        return ExecutionProfile(
            arm=arm, max_joint_velocity_rad_s=1.2,
            max_joint_acceleration_rad_s2=4.0,
            sample_dt_s=sample_dt_s, squeeze_duration_s=squeeze_duration_s)
    if arm == "xarm":
        return ExecutionProfile(
            arm=arm, max_joint_velocity_rad_s=0.50,
            max_joint_acceleration_rad_s2=1.50,
            sample_dt_s=sample_dt_s, squeeze_duration_s=squeeze_duration_s)
    raise ValueError(f"unsupported execution arm profile: {arm!r}")


def _strictly_increasing(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector with at least two samples")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _sample_times(duration_s: float, dt_s: float) -> np.ndarray:
    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("duration and sample dt must be positive")
    # ``duration_s`` is quantized by ``_retime_c2_path``.  Keeping this
    # grid uniform is intentional: a timestamped hardware follower can then
    # stream every saved sample with one fixed control period, rather than
    # silently dropping a short last interval.
    n_intervals = int(round(duration_s / dt_s))
    if n_intervals < 1 or not np.isclose(n_intervals * dt_s, duration_s,
                                         atol=1.0e-9, rtol=0.0):
        raise ValueError("retimed duration must be an integer number of samples")
    return np.arange(n_intervals + 1, dtype=np.float64) * dt_s


def _initial_knot_times(nodes: np.ndarray, *, velocity_limit: float,
                        acceleration_limit: float) -> np.ndarray:
    delta = np.max(np.abs(np.diff(nodes, axis=0)), axis=1)
    # A cubic spline can exceed the chord's average velocity.  This intentionally
    # starts conservative; the derivative loop below tightens it exactly.
    by_velocity = 1.75 * delta / velocity_limit
    by_acceleration = np.sqrt(6.0 * delta / acceleration_limit)
    segment_dt = np.maximum.reduce((by_velocity, by_acceleration,
                                    np.full(len(delta), 1.0e-3)))
    return np.r_[0.0, np.cumsum(segment_dt)]


def _cubic_derivative_peaks(spline: CubicSpline, knot_t: np.ndarray) -> tuple[float, float]:
    """Return exact continuous qdot/qddot maxima for a cubic spline.

    Checking only 10 ms output samples can miss a velocity extremum inside a
    cubic interval.  Knot times are on the output grid, so evaluating the
    derivative endpoints and the analytic qdot extrema gives a true bound for
    the whole continuous reference before we write any executor samples.
    """
    c0, c1, c2, _c3 = np.asarray(spline.c, dtype=np.float64)
    h = np.diff(np.asarray(knot_t, dtype=np.float64))[:, None]
    qd_start = c2
    qd_end = 3.0 * c0 * h**2 + 2.0 * c1 * h + c2
    max_velocity = float(max(np.max(np.abs(qd_start)), np.max(np.abs(qd_end))))
    # qdd is affine in each cubic segment, so its extrema are its endpoints.
    qdd_start = 2.0 * c1
    qdd_end = 6.0 * c0 * h + 2.0 * c1
    max_acceleration = float(max(np.max(np.abs(qdd_start)), np.max(np.abs(qdd_end))))
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = -c1 / (3.0 * c0)
    valid = np.isfinite(stationary) & (stationary > 0.0) & (stationary < h)
    if np.any(valid):
        qd_stationary = 3.0 * c0 * stationary**2 + 2.0 * c1 * stationary + c2
        max_velocity = max(max_velocity, float(np.max(np.abs(qd_stationary[valid]))))
    return max_velocity, max_acceleration


def _retime_c2_path(nodes: np.ndarray, *, profile: ExecutionProfile,
                    speed_scale: float, label: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Spline and globally stretch a joint path until qdot/qddot are bounded.

    ``speed_scale`` belongs to the motion phase: free-hand approach uses 1.0
    while a lifted object uses the conservative held-object factor.  The same
    routine is intentionally used for both phases so the persisted path has a
    real timestamp contract throughout, rather than an untimed approach glued
    to a timed lift.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or len(nodes) < 2 or not np.isfinite(nodes).all():
        raise ValueError("execution nodes must be a finite (N>=2, dof) array")
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError("execution phase speed scale must be in (0, 1]")
    velocity_limit = profile.max_joint_velocity_rad_s * speed_scale
    acceleration_limit = profile.max_joint_acceleration_rad_s2 * speed_scale
    if velocity_limit <= 0.0 or acceleration_limit <= 0.0:
        raise ValueError("execution limits must be positive")
    knot_t = _initial_knot_times(
        nodes, velocity_limit=velocity_limit, acceleration_limit=acceleration_limit)
    for iteration in range(1, profile.max_retime_iterations + 1):
        # Quantize *each* chord duration, not only the overall duration.  Every
        # input node consequently lands on a saved executor sample.  For lift,
        # these are precisely the collision-checked q samples recovered from
        # the 5 mm chords; they remain explicit q(t) samples after retiming.
        segment_t = np.diff(knot_t)
        segment_t = np.ceil(segment_t / profile.sample_dt_s) * profile.sample_dt_s
        knot_t = np.r_[0.0, np.cumsum(segment_t)]
        # Rest at squeeze→lift and at the final height.  Interior derivatives
        # are C2-continuous, unlike per-5-mm stop-and-go interpolation.
        spline = CubicSpline(
            knot_t, nodes, axis=0,
            bc_type=((1, np.zeros(nodes.shape[1])), (1, np.zeros(nodes.shape[1]))),
        )
        sample_t = _sample_times(float(knot_t[-1]), profile.sample_dt_s)
        q = np.asarray(spline(sample_t), dtype=np.float64)
        max_velocity, max_acceleration = _cubic_derivative_peaks(spline, knot_t)
        velocity_ratio = max_velocity / velocity_limit
        acceleration_ratio = np.sqrt(max_acceleration / acceleration_limit)
        stretch = max(1.0, velocity_ratio, acceleration_ratio)
        if stretch <= 1.0005:
            return q.astype(np.float32), sample_t.astype(np.float64), {
                "label": label,
                "method": "clamped_c2_cubic_spline",
                "retime_iterations": iteration,
                "knot_count": int(len(nodes)),
                "duration_s": float(knot_t[-1]),
                "max_joint_velocity_rad_s": max_velocity,
                "max_joint_acceleration_rad_s2": max_acceleration,
                "velocity_limit_rad_s": velocity_limit,
                "acceleration_limit_rad_s2": acceleration_limit,
            }
        knot_t *= stretch * 1.01
    raise RuntimeError("execution_retime_limit_not_reached")


def build_execution_trajectory(*, approach_qpos: np.ndarray,
                               squeeze_qpos: np.ndarray,
                               lift_execution_qpos: np.ndarray,
                               arm_dof: int,
                               profile: ExecutionProfile) -> dict[str, Any]:
    """Create one timestamped approach→squeeze→lift reference ``q(t)``.

    ``lift_execution_qpos`` must be the collision-checked joint-chord samples
    emitted by :func:`continue_vertical_lift`, not only its 5 mm nodes.
    """
    approach = np.asarray(approach_qpos, dtype=np.float64)
    squeeze = np.asarray(squeeze_qpos, dtype=np.float64).reshape(-1)
    lift = np.asarray(lift_execution_qpos, dtype=np.float64)
    if approach.ndim != 2 or lift.ndim != 2 or len(approach) < 2 or len(lift) < 2:
        raise ValueError("approach and lift paths must each contain at least 2 q samples")
    if approach.shape[1] != len(squeeze) or lift.shape[1] != len(squeeze):
        raise ValueError("approach, squeeze, and lift DOF dimensions must match")
    if not 0 < arm_dof < len(squeeze):
        raise ValueError("arm_dof must split a full robot configuration")
    if not (np.isfinite(approach).all() and np.isfinite(squeeze).all()
            and np.isfinite(lift).all()):
        raise ValueError("execution path contains non-finite q values")
    if not np.allclose(lift[:, arm_dof:], squeeze[arm_dof:], atol=1.0e-6):
        raise ValueError("lift hand q must be fixed to squeeze q")
    if not np.allclose(approach[-1, :arm_dof], squeeze[:arm_dof], atol=1.0e-5):
        raise ValueError("squeeze must start at the approach arm endpoint")
    if not np.allclose(lift[0], squeeze, atol=1.0e-5):
        raise ValueError("lift must begin at squeeze endpoint")

    # Retiming the original MotionGen samples as well is important.  They are
    # dense geometric samples, but their interpolation period is a planner
    # detail, not yet the selected arm adapter's qdot/qddot contract.
    approach_path, approach_t, approach_summary = _retime_c2_path(
        approach, profile=profile, speed_scale=1.0, label="approach")
    squeeze_count = max(1, int(np.ceil(profile.squeeze_duration_s / profile.sample_dt_s)))
    # The requested duration is rounded *up* to a sample boundary.  This is a
    # conservative timing change (< one control tick) which keeps the entire
    # stitched artifact uniformly timestamped.
    squeeze_duration_s = squeeze_count * profile.sample_dt_s
    squeeze_local_t = (np.arange(1, squeeze_count + 1, dtype=np.float64)
                       * profile.sample_dt_s)
    alpha = (squeeze_local_t / squeeze_duration_s)[:, None]
    squeeze_path = approach_path[-1][None, :] * (1.0 - alpha) + squeeze[None, :] * alpha

    lift_arm, lift_local_t, lift_summary = _retime_c2_path(
        lift[:, :arm_dof], profile=profile,
        speed_scale=profile.held_object_speed_scale, label="lift")
    lift_path = np.concatenate([
        lift_arm,
        np.broadcast_to(squeeze[arm_dof:], (len(lift_arm), len(squeeze) - arm_dof)),
    ], axis=1)

    squeeze_t = approach_t[-1] + squeeze_local_t
    lift_t = squeeze_t[-1] + lift_local_t[1:]
    qpos = np.concatenate([approach_path, squeeze_path, lift_path[1:]], axis=0).astype(np.float32)
    time_s = np.concatenate([approach_t, squeeze_t, lift_t]).astype(np.float64)
    phase = np.concatenate([
        np.full(len(approach_path), "approach", dtype="<U12"),
        np.full(len(squeeze_path), "squeeze", dtype="<U12"),
        np.full(len(lift_path) - 1, "lift", dtype="<U12"),
    ])
    _strictly_increasing(time_s, name="execution time_s")
    return {
        "schema_version": 1,
        "qpos": qpos,
        "time_s": time_s,
        "phase": phase,
        "profile": profile.as_dict(),
        "segments": {
            "approach_sample_count": int(len(approach_path)),
            "squeeze_sample_count": int(len(squeeze_path)),
            "lift_sample_count": int(len(lift_path)),
            "total_sample_count": int(len(qpos)),
            "approach_duration_s": float(approach_t[-1]),
            "squeeze_duration_s": float(squeeze_duration_s),
            "lift_duration_s": float(lift_local_t[-1]),
            "total_duration_s": float(time_s[-1]),
            "approach_retime": approach_summary,
            "lift_retime": lift_summary,
        },
    }
