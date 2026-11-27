"""Construct offline FR3 planners with the demo's selected mount geometry."""

from __future__ import annotations

from pathlib import Path

from .assets import AssetPaths
from .config import TaskMode
from .end_effector import EndEffectorSpec, end_effector_spec


_CUROBO_CONTENT = (
    Path(__file__).resolve().parents[3] / "autodex" / "planner" / "src" /
    "curobo" / "content")


def selected_fr3_planner_config_options(
    spec: EndEffectorSpec | None = None,
) -> dict[str, object]:
    """Bind FR3 planner YAML, spheres and meshes to this checkout.

    The original AutoDex planner defaults intentionally remain rooted at
    ``~/shared_data/AutoDex``.  ZeroDex is a separate checkout/runtime, so an
    explicit planner construction must not silently fall back to that tree.
    """
    selected = end_effector_spec() if spec is None else spec
    names = {
        "inspire": ("fr3_inspire.yml", "inspire_floating.yml"),
        "franka_gripper": (
            "fr3_gripper.yml", "franka_gripper_floating.yml"),
        "robotiq_2f85": (
            "fr3_robotiq_2f85.yml", "robotiq_2f85_floating.yml"),
    }
    robot_name, hand_name = names[selected.name]
    config_root = _CUROBO_CONTENT / "configs" / "robot"
    return {
        "robot_cfg_path": str(config_root / robot_name),
        "hand_cfg_path": str(config_root / hand_name),
        "external_robot_configs_path": str(config_root),
        "external_asset_path": str(_CUROBO_CONTENT / "assets"),
        "use_hand_planner_defaults": True,
        # Process the small commissioned Robotiq catalog one candidate at a
        # time. All candidates are still visited through the existing chunk
        # loop, while each 32-seed IK graph avoids unused pose padding and
        # leaves VRAM for the subsequent motion graph.
        **({"candidate_batch_size": 1}
           if selected.name == "robotiq_2f85" else {}),
    }


def new_offline_fr3_planner(
    *, shared_root: Path, mode: TaskMode,
    use_cuda_graph: bool | None = None,
):
    """Use the same URDF choice as live planning and held-path geometry."""
    from autodex.planner import GraspPlanner

    spec = end_effector_spec()
    options = {
        **selected_fr3_planner_config_options(spec),
        "hand": spec.planner_hand,
        "robot_urdf_path": str(AssetPaths(
            Path(shared_root).expanduser().resolve(), mode).robot_urdf),
    }
    if use_cuda_graph is not None:
        options["use_cuda_graph"] = use_cuda_graph
    return GraspPlanner(**options)
