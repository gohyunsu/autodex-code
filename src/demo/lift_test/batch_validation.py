"""Batched validation helpers for externally generated robot trajectories.

The lift demo deliberately generates Jacobian and C2 paths outside cuRobo's
trajectory optimizer.  cuRobo is still the authority for joint-bound,
self-collision, and robot-vs-world feasibility.  This module evaluates the
same independent q samples in GPU batches instead of invoking the scalar
``MotionGen.check_start_state`` diagnostic once per sample.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_COLLISION_BATCH_SIZE = 512


def _collision_constraints(planner) -> list[Any]:
    rollout = planner._motion_gen.rollout_fn
    return [
        rollout.primitive_collision_constraint,
        rollout.robot_self_collision_constraint,
    ]


def _enable_collision_constraints(planner) -> None:
    """Leave every configured collision constraint enabled after diagnostics.

    cuRobo's scalar start-state diagnostic temporarily toggles these terms to
    classify a failure.  Bulk validation must never inherit that temporary
    state.  ``enable_cost`` remains a no-op for a term configured with zero
    weight, so this preserves the robot configuration's intended contract.
    """
    for constraint in _collision_constraints(planner):
        constraint.enable_cost()


def check_robot_state(planner, q_full: np.ndarray) -> tuple[bool, str | None]:
    """Run cuRobo's scalar diagnostic and restore its collision constraints."""
    import torch
    from curobo.types.state import JointState

    q = torch.as_tensor(
        np.ascontiguousarray(np.asarray(q_full, dtype=np.float32).reshape(1, -1)),
        dtype=torch.float32,
        device=planner._tensor_args.device,
    )
    _enable_collision_constraints(planner)
    try:
        valid, status = planner._motion_gen.check_start_state(
            JointState.from_position(q))
        return bool(valid), (None if status is None else str(status))
    finally:
        # In the installed cuRobo version the world-collision classification
        # branch can return before re-enabling self-collision.  Do not let a
        # diagnostic call weaken any following candidate validation.
        _enable_collision_constraints(planner)


def check_robot_states_batch(
    planner,
    q_full: np.ndarray,
    *,
    batch_size: int = DEFAULT_COLLISION_BATCH_SIZE,
    diagnose_first_invalid: bool = True,
) -> tuple[np.ndarray, str | None, dict[str, Any]]:
    """Return a feasibility mask for independent q samples in the active world.

    The exact rollout used by ``MotionGen.check_start_state`` is retained, but
    its scalar ``.item()`` and per-sample failure-classification passes are
    avoided on the successful path.  Input rows keep their original ordering.
    """
    import torch

    qpos = np.asarray(q_full, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[1] <= 0:
        raise ValueError("batch qpos must have shape (samples, dof)")
    if not np.isfinite(qpos).all():
        raise ValueError("batch qpos contains non-finite values")
    if batch_size <= 0:
        raise ValueError("collision batch size must be positive")
    if len(qpos) == 0:
        return np.empty(0, dtype=bool), None, {
            "backend": "curobo_rollout_batch",
            "batch_size": int(batch_size),
            "chunk_count": 0,
            "sample_count": 0,
            "first_invalid_index": None,
        }

    rollout = planner._motion_gen.rollout_fn
    _enable_collision_constraints(planner)
    chunks: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(qpos), batch_size):
                host_chunk = np.ascontiguousarray(qpos[start:start + batch_size])
                q_gpu = torch.as_tensor(
                    host_chunk, dtype=torch.float32,
                    device=planner._tensor_args.device)
                metrics = rollout.rollout_constraint(
                    q_gpu.unsqueeze(1), use_batch_env=False)
                feasible = metrics.feasible.reshape(len(host_chunk), -1).all(dim=1)
                chunks.append(feasible.detach().cpu().numpy().astype(bool, copy=False))
    finally:
        _enable_collision_constraints(planner)

    valid = np.concatenate(chunks)
    invalid = np.flatnonzero(~valid)
    first_invalid = None if len(invalid) == 0 else int(invalid[0])
    status = None
    if first_invalid is not None and diagnose_first_invalid:
        _valid, status = check_robot_state(planner, qpos[first_invalid])
        if _valid:
            # Fail closed if a boundary case changes classification between a
            # batch kernel and the diagnostic kernel.
            status = "batch_scalar_feasibility_mismatch"
    return valid, status, {
        "backend": "curobo_rollout_batch",
        "batch_size": int(batch_size),
        "chunk_count": int(len(chunks)),
        "sample_count": int(len(qpos)),
        "first_invalid_index": first_invalid,
    }


def fk_wrist_batch(
    planner,
    q_full: np.ndarray,
    *,
    batch_size: int = DEFAULT_COLLISION_BATCH_SIZE,
) -> np.ndarray:
    """Compute wrist transforms for full-q rows with batched cuRobo FK."""
    import torch

    qpos = np.asarray(q_full, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[1] <= 0:
        raise ValueError("FK qpos must have shape (samples, dof)")
    if not np.isfinite(qpos).all():
        raise ValueError("FK qpos contains non-finite values")
    if batch_size <= 0:
        raise ValueError("FK batch size must be positive")
    if len(qpos) == 0:
        return np.empty((0, 4, 4), dtype=np.float64)

    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(qpos), batch_size):
            host_chunk = np.ascontiguousarray(qpos[start:start + batch_size])
            q_gpu = torch.as_tensor(
                host_chunk, dtype=torch.float32,
                device=planner._tensor_args.device)
            state = planner._motion_gen.kinematics.get_state(q_gpu)
            positions.append(state.ee_position.detach().cpu().numpy())
            quaternions.append(state.ee_quaternion.detach().cpu().numpy())

    position = np.concatenate(positions).astype(np.float64, copy=False)
    quat_wxyz = np.concatenate(quaternions).astype(np.float64, copy=False)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    transforms = np.broadcast_to(np.eye(4, dtype=np.float64),
                                 (len(qpos), 4, 4)).copy()
    transforms[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    transforms[:, :3, 3] = position
    return transforms


def object_bottom_z_batch(
    mesh_vertices: np.ndarray,
    object_transforms: np.ndarray,
    *,
    transform_batch_size: int = 128,
) -> np.ndarray:
    """Compute the exact minimum transformed vertex z for every object pose."""
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    transforms = np.asarray(object_transforms, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh vertices must have shape (vertices, 3)")
    if transforms.ndim == 2:
        transforms = transforms[None, ...]
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("object transforms must have shape (samples, 4, 4)")
    if transform_batch_size <= 0:
        raise ValueError("object transform batch size must be positive")
    bottoms = np.empty(len(transforms), dtype=np.float64)
    for start in range(0, len(transforms), transform_batch_size):
        chunk = transforms[start:start + transform_batch_size]
        # Only the third transform row contributes to world z.  This avoids a
        # full (sample, vertex, xyz) transformed-mesh allocation.
        vertex_z = np.einsum("si,vi->sv", chunk[:, 2, :3], vertices)
        vertex_z += chunk[:, 2, 3, None]
        bottoms[start:start + len(chunk)] = np.min(vertex_z, axis=1)
    return bottoms
