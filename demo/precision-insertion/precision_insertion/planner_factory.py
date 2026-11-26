"""Construct offline FR3 planners with the demo's selected mount geometry."""

from __future__ import annotations

from pathlib import Path

from .assets import AssetPaths
from .config import TaskMode
from .end_effector import end_effector_spec


def new_offline_fr3_planner(
    *, shared_root: Path, mode: TaskMode,
    use_cuda_graph: bool | None = None,
):
    """Use the same URDF choice as live planning and held-path geometry."""
    from autodex.planner import GraspPlanner

    options = {
        "hand": end_effector_spec().planner_hand,
        "robot_urdf_path": str(AssetPaths(
            Path(shared_root).expanduser().resolve(), mode).robot_urdf),
    }
    if use_cuda_graph is not None:
        options["use_cuda_graph"] = use_cuda_graph
    return GraspPlanner(**options)
