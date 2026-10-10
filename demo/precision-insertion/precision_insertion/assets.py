"""Read-only v8 asset readiness audit against one explicit shared-data root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import TaskMode


@dataclass(frozen=True)
class AssetPaths:
    shared_root: Path
    mode: TaskMode

    @property
    def object_root(self) -> Path:
        return self.shared_root / "object_processing"

    def object_dir(self, name: str) -> Path:
        return self.object_root / name

    def raw_mesh(self, name: str) -> Path:
        return self.object_dir(name) / "raw_mesh" / f"{name}.obj"

    def foundpose_repre(self, name: str) -> Path:
        return (self.shared_root / "AutoDex" / "foundpose_assets" / name /
                "object_repre" / "v1" / name / "1" / "repre.pth")

    @property
    def key_planning_mesh(self) -> Path:
        return (self.object_dir(self.mode.key_object) / "processed_data" /
                "mesh" / "simplified.obj")

    @property
    def key_tabletop_dir(self) -> Path:
        return (self.object_dir(self.mode.key_object) / "processed_data" /
                "info" / "tabletop")

    @property
    def socket_collision_mesh(self) -> Path:
        return (self.object_dir(self.mode.socket_object) / "processed_data" /
                "mesh" / "static_collision.obj")

    @property
    def task_geometry(self) -> Path:
        fixture = ("unified_socket" if self.mode.family == "square"
                   else self.mode.socket_object)
        return (self.shared_root / "AutoDex" / "precision_insertion" /
                "fixtures" / fixture / "task_geometry.json")

    @property
    def robot_urdf(self) -> Path:
        return (self.shared_root / "AutoDex" / "content" / "assets" /
                "robot" / "fr3_inspire_description" / "fr3_inspire.urdf")

    @property
    def candidate_dir(self) -> Path:
        return (self.shared_root / "AutoDex" / "candidates" / "inspire" /
                "v8" / self.mode.key_object)

    @property
    def scene_dir(self) -> Path:
        return (self.shared_root / "AutoDex" / "scene" / "inspire" /
                self.mode.key_object / "table")

    def reset_dir(self, height_cm: int) -> Path:
        return (self.shared_root / "AutoDex" / "candidates" / "inspire" /
                f"reset_{height_cm}" / self.mode.key_object /
                f"reorient_{height_cm}")


def _complete_grasp_dirs(root: Path) -> list[Path]:
    """Count actual v8 grasp files, not a generation-required marker."""
    if not root.is_dir():
        return []
    needed = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy")
    return sorted(path.parent for path in root.rglob("wrist_se3.npy")
                  if all((path.parent / name).is_file() for name in needed))


def audit_assets(shared_root: Path, mode: TaskMode) -> dict:
    """Report missing inputs without touching cameras, robot, or NAS state.

    Passing this audit is only file availability, never full-task validation.
    Paths remain tied to the supplied root; no legacy global object path is
    consulted.
    """
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    key = mode.key_object
    socket = mode.socket_object
    files = {
        "key_raw_mesh": paths.raw_mesh(key),
        "key_planning_mesh": paths.key_planning_mesh,
        "socket_raw_mesh": paths.raw_mesh(socket),
        "socket_exact_collision_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "franka_inspire_urdf": paths.robot_urdf,
        "key_foundpose_repre": paths.foundpose_repre(key),
        "socket_foundpose_repre": paths.foundpose_repre(socket),
    }
    checks = {name: {"path": str(path), "present": path.is_file()}
              for name, path in files.items()}
    tabletop = sorted(paths.key_tabletop_dir.glob("*.npy"))
    scenes = sorted(paths.scene_dir.glob("*.json"))
    grasps = _complete_grasp_dirs(paths.candidate_dir)
    resets = {
        str(height): len(_complete_grasp_dirs(paths.reset_dir(height)))
        for height in (0, 4, 8, 12)
    }
    missing = [name for name, result in checks.items()
               if not result["present"]]
    if not tabletop:
        missing.append("key_tabletop_poses")
    if not scenes:
        missing.append("inspire_v8_table_scenes")
    if not grasps:
        missing.append("inspire_v8_grasp_candidates")
    return {
        "schema": "precision_insertion_asset_audit_v1",
        "shared_root": str(root),
        "mode": {
            "family": mode.family,
            "gap_mm": mode.gap_mm,
            "key_object": key,
            "socket_object": socket,
            "target_depth_m": mode.target_depth_m,
        },
        "checks": checks,
        "counts": {
            "key_tabletop_poses": len(tabletop),
            "inspire_v8_table_scenes": len(scenes),
            "inspire_v8_grasp_candidates": len(grasps),
            "inspire_v8_reset_grasps_by_height_cm": resets,
        },
        "missing": missing,
        "file_inputs_present": not missing,
        "robot_ready": False,
        "robot_ready_reason": (
            "File availability does not prove offline 20mm grasp-endpoint "
            "eligibility, live-path planning, guarded contact control, "
            "calibration, or physical validation."
        ),
    }
