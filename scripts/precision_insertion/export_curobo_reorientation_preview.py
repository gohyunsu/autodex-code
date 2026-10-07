#!/usr/bin/env python3
"""Export an honest FR3/Inspire cuRobo reorientation-plan preview.

This utility consumes BODex reset candidates from an explicit staging root. It
does not promote those candidates into the runtime pool.  Each candidate is
tested with the existing AutoDex reset planner as one complete chain:

    approach -> close -> lift -> rotate -> place -> release -> depart -> retract

The exported trajectory is suitable for the original-mesh Blender renderer.
Passing this script establishes a continuous, collision-planned robot-motion
preview only.  It does not establish grasp stability, release dynamics,
tabletop-pose classification, or physical success.  In particular, candidates
from a contact-policy ablation remain diagnostic even when motion planning
passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.utils.conversion import se32cart
from autodex.utils.path import get_obj_root, project_dir
from src.experiment.reset.reorient import _plan_reorient_full_chain
from src.grasp_generation.reorient.plan_reset import (
    load_fk_urdf,
    load_tabletop_pose,
    make_obj_pose,
)


JOINT_NAMES = np.asarray([
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint", "right_little_1_joint",
])

PHASES = (
    "approach", "grasp_close", "lift", "reorient", "preplace",
    "vertical_place", "release", "drop_preview", "post_release_lift",
    "retract",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2) + "\n", encoding="utf-8"
    )


def _candidate_dirs(root: Path, cells: list[str]):
    for cell in cells:
        cell_dir = root / cell
        if not cell_dir.is_dir():
            continue
        candidates = sorted(
            (path for path in cell_dir.iterdir() if path.is_dir()),
            key=lambda path: int(path.name),
        )
        for candidate in candidates:
            yield cell, candidate


def _load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")
    return (
        np.load(path / "wrist_se3.npy").astype(np.float64),
        np.load(path / "pregrasp_pose.npy").astype(np.float32),
        np.load(path / "grasp_pose.npy").astype(np.float32),
    )


def _load_cell_seeds(
    root: Path,
    cell: str,
    object_start: np.ndarray,
    maximum: int,
) -> tuple[dict[str, Any], list[Path]] | None:
    cell_dir = root / cell
    if not cell_dir.is_dir():
        return None
    paths = sorted(
        (path for path in cell_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )[:maximum]
    if not paths:
        return None
    loaded = [_load_candidate(path) for path in paths]
    wrist_object = np.stack([item[0] for item in loaded])
    return ({
        "wrist_se3": object_start[None] @ wrist_object,
        "pregrasp": np.stack([item[1] for item in loaded]),
        "grasp": np.stack([item[2] for item in loaded]),
        "openpose_start": [None] * len(paths),
        "openpose_target": [None] * len(paths),
        "scene_info": [{
            "grasp_idx": path.name,
            "cell": cell,
            "candidate_contract": "explicit_diagnostic_staging_root",
            "source": str(path),
        } for path in paths],
        "n_total": len(paths),
    }, paths)


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, maximum)).astype(np.int64))


def _fk_wrist(urdf, trajectory: np.ndarray, ee_link: str) -> np.ndarray:
    transforms = np.tile(np.eye(4), (len(trajectory), 1, 1))
    base = urdf.base_link
    for index, qpos in enumerate(trajectory):
        urdf.update_cfg(qpos)
        transforms[index] = urdf.get_transform(ee_link, base)
    return transforms


def _export(
    output: Path,
    *,
    trajectories: dict[str, np.ndarray],
    wrist_se3_object: np.ndarray,
    object_start: np.ndarray,
    object_end: np.ndarray,
    urdf,
    ee_link: str,
    object_mesh: Path,
    socket_mesh: Path,
    socket_pose: np.ndarray,
    robot_urdf: Path,
    max_frames_per_phase: int,
    release_object_pose: np.ndarray,
    landed_object_pose: np.ndarray,
) -> dict[str, Any]:
    qpos_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    object_parts: list[np.ndarray] = []
    original_counts: dict[str, int] = {}
    rendered_counts: dict[str, int] = {}

    for phase in PHASES:
        full = np.asarray(trajectories[phase], dtype=np.float32)
        original_counts[phase] = len(full)
        indices = _sample_indices(len(full), max_frames_per_phase)
        sampled = full[indices]
        rendered_counts[phase] = len(sampled)
        qpos_parts.append(sampled)
        phase_parts.append(np.full(len(sampled), phase))
        if phase in {"approach", "grasp_close"}:
            poses = np.repeat(object_start[None], len(sampled), axis=0)
        elif phase == "release":
            poses = np.repeat(release_object_pose[None], len(sampled), axis=0)
        elif phase == "drop_preview":
            alpha = np.linspace(0.0, 1.0, len(sampled))[:, None, None]
            poses = np.repeat(release_object_pose[None], len(sampled), axis=0)
            poses[:, :3, 3] = (
                (1.0 - alpha[:, 0]) * release_object_pose[:3, 3]
                + alpha[:, 0] * landed_object_pose[:3, 3]
            )
        elif phase in {"post_release_lift", "retract"}:
            poses = np.repeat(landed_object_pose[None], len(sampled), axis=0)
        else:
            wrist_world = _fk_wrist(urdf, sampled, ee_link)
            poses = wrist_world @ np.linalg.inv(wrist_se3_object)
        object_parts.append(poses)

    qpos = np.concatenate(qpos_parts).astype(np.float32)
    phases = np.concatenate(phase_parts)
    object_poses = np.concatenate(object_parts).astype(np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos,
        phase=phases,
        object_pose=object_poses,
        socket_pose=socket_pose,
        joint_names=JOINT_NAMES,
        object_mesh_path=np.asarray(str(object_mesh)),
        socket_mesh_path=np.asarray(str(socket_mesh)),
        robot_urdf_path=np.asarray(str(robot_urdf)),
        preview_status=np.asarray(
            "curobo_continuous_reorientation_plan_not_physical_validation"
        ),
        preview_kind=np.asarray("curobo_reorientation_diagnostic"),
        **{
            f"trajectory_{phase}_qpos": np.asarray(trajectories[phase], dtype=np.float32)
            for phase in PHASES
        },
    )
    return {
        "original_waypoints": original_counts,
        "rendered_waypoints": rendered_counts,
        "rendered_total_frames": len(qpos),
        "rigid_object_attachment_phases": [
            "lift", "reorient", "preplace", "vertical_place"
        ],
        "composed_not_dynamics_validated_phases": ["drop_preview"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--object", default="precision_key_1p5mm")
    parser.add_argument("--version", default="v8")
    parser.add_argument("--cells", nargs="+", default=["0_4", "2_4", "3_4"])
    parser.add_argument("--pickup-x", type=float, default=0.40)
    parser.add_argument("--pickup-y", type=float, default=0.18)
    parser.add_argument("--pickup-yaw-deg", type=float, default=180.0)
    parser.add_argument("--release-height-m", type=float, default=0.12)
    parser.add_argument("--socket-x", type=float, default=0.45)
    parser.add_argument("--socket-y", type=float, default=-0.10)
    parser.add_argument("--socket-z", type=float, default=0.04)
    parser.add_argument("--max-candidates", type=int, default=37)
    parser.add_argument("--max-frames-per-phase", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("reorientation uses the v8 object-processing contract")

    candidate_root = args.candidate_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    object_root = get_obj_root(args.version)
    object_mesh = (
        Path(object_root) / args.object / "processed_data" / "mesh" /
        "simplified.obj"
    )
    socket_mesh = (
        Path(object_root) / "precision_socket_unified" / "processed_data" /
        "mesh" / "static_collision.obj"
    )
    robot_urdf = (
        Path(project_dir) / "content" / "assets" / "robot" /
        "fr3_inspire_description" / "fr3_inspire.urdf"
    )
    for path in (object_mesh, socket_mesh, robot_urdf):
        if not path.is_file():
            raise FileNotFoundError(path)

    planner = GraspPlanner(hand="fr3_inspire")
    urdf, ee_link = load_fk_urdf("fr3_inspire")
    failures: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    remaining = args.max_candidates
    for cell in args.cells:
        source_text, target_text = cell.split("_", maxsplit=1)
        source_id, target_id = int(source_text), int(target_text)
        source_canonical = load_tabletop_pose(
            args.object, source_id, object_root
        )
        target_canonical = load_tabletop_pose(
            args.object, target_id, object_root
        )
        object_start = make_obj_pose(
            source_canonical,
            np.asarray([
                args.pickup_x, args.pickup_y,
                source_canonical[2, 3] + 0.04,
            ]),
            args.pickup_yaw_deg,
        )
        loaded = _load_cell_seeds(
            candidate_root, cell, object_start, remaining
        )
        if loaded is None:
            failures.append({"cell": cell, "failure": "no_candidates"})
            continue
        seeds, candidate_paths = loaded
        remaining -= len(candidate_paths)
        scene_cfg = {
            "mesh": {"target": {
                "pose": se32cart(object_start).tolist(),
                "file_path": str(object_mesh),
            }},
            "cuboid": {"table": dict(TABLE_CUBOID)},
        }
        plan = _plan_reorient_full_chain(
            planner=planner,
            scene_cfg=scene_cfg,
            obj=args.object,
            pose_robot_before=object_start,
            target_tabletop_robot=target_canonical,
            seeds=seeds,
            planner_robot="fr3_inspire",
            release_height_m=args.release_height_m,
            tabletop_geometry=None,
        )
        print(f"[runtime] {cell}: {plan.get('success')} {plan.get('reason', 'ok')}")
        if not plan.get("success", False):
            failures.append({
                "cell": cell,
                "failure": plan.get("reason", "unknown"),
                "counts": plan.get("counts"),
            })
            continue
        result = plan["result"]
        selected_index = int(result.timing["candidate_idx"])
        grasp = np.asarray(result.grasp_pose, dtype=np.float32)
        pregrasp = np.asarray(result.pregrasp_pose, dtype=np.float32)
        arm_dof = planner._n_arm
        close_count = 20
        close_alpha = np.linspace(0.0, 1.0, close_count)[:, None]
        close = np.repeat(result.traj[-1][None], close_count, axis=0)
        close[:, arm_dof:] = (
            (1.0 - close_alpha) * pregrasp[None]
            + close_alpha * grasp[None]
        )
        placement = plan["placement_preflight"]
        trajectories = {
            "approach": np.asarray(result.traj),
            "grasp_close": close,
            "lift": np.asarray(plan["lift_traj"]),
            "reorient": np.asarray(plan["reorient_traj"]),
            "preplace": np.asarray(placement["preplace_traj"]),
            "vertical_place": np.asarray(placement["descent_traj"]),
        }
        for phase in ("lift", "reorient", "preplace", "vertical_place"):
            trajectories[phase][:, arm_dof:] = grasp[None]
        release_count = 20
        release_alpha = np.linspace(0.0, 1.0, release_count)[:, None]
        release = np.repeat(trajectories["vertical_place"][-1][None], release_count, axis=0)
        release[:, arm_dof:] = (
            (1.0 - release_alpha) * grasp[None]
            + release_alpha * pregrasp[None]
        )
        trajectories["release"] = release
        drop_count = 16
        trajectories["drop_preview"] = np.repeat(release[-1][None], drop_count, axis=0)
        trajectories["post_release_lift"] = np.asarray(placement["post_lift_traj"])
        trajectories["retract"] = np.asarray(placement["retract_traj"])
        release_object = (
            _fk_wrist(urdf, trajectories["vertical_place"][-1:], ee_link)[0]
            @ plan["obj_in_wrist"]
        )
        landed_object = release_object.copy()
        landed_object[2, 3] = target_canonical[2, 3] + 0.04
        selected = {
            "cell": cell,
            "candidate": candidate_paths[selected_index].name,
            "candidate_dir": str(candidate_paths[selected_index]),
            "source_pose": source_id,
            "target_pose": target_id,
            "object_start": object_start,
            "landed_object": landed_object,
            "release_object": release_object,
            "wrist_se3_object": np.linalg.inv(object_start) @ result.wrist_se3,
            "trajectories": trajectories,
            "runtime_counts": plan["counts"],
            "release_height_m": args.release_height_m,
        }
        break

    report_path = output.with_suffix(".json")
    if selected is None:
        _write_json(report_path, {
            "schema_version": 1,
            "status": "no_continuous_reorientation_plan_pass",
            "candidate_root": str(candidate_root),
            "cells": args.cells,
            "failures": failures,
            "non_claims": ["not physical execution", "not grasp stability"],
        })
        print(report_path)
        return 2

    socket_pose = np.eye(4)
    socket_pose[:3, 3] = [args.socket_x, args.socket_y, args.socket_z]
    trajectory_record = _export(
        output,
        trajectories=selected.pop("trajectories"),
        wrist_se3_object=selected["wrist_se3_object"],
        object_start=selected["object_start"],
        object_end=selected["landed_object"],
        urdf=urdf,
        ee_link=ee_link,
        object_mesh=object_mesh,
        socket_mesh=socket_mesh,
        socket_pose=socket_pose,
        robot_urdf=robot_urdf,
        max_frames_per_phase=args.max_frames_per_phase,
        release_object_pose=selected["release_object"],
        landed_object_pose=selected["landed_object"],
    )
    _write_json(report_path, {
        "schema_version": 1,
        "status": "curobo_continuous_reorientation_plan_not_physical_validation",
        "planner": "existing AutoDex runtime _plan_reorient_full_chain with fr3_inspire",
        "selected": selected,
        "trajectory": trajectory_record,
        "attempted_before_pass": failures,
        "contact_policy": {
            "mode": "diagnostic_only",
            "reason": (
                "the explicit staging root is evaluated for motion feasibility; "
                "contact-policy acceptance remains a separate source report"
            ),
        },
        "validated": [
            "continuous cuRobo approach",
            "continuous cuRobo lift/reorient/preplace/vertical-place",
            "continuous cuRobo post-release lift and retract",
            "table and robot collision model used by the runtime reset planner",
            "rigid T_object_wrist during lift/reorient/place",
        ],
        "not_validated": [
            "physical grasp stability",
            "release/drop dynamics; the rendered drop segment is composed",
            "post-release target tabletop-pose classification",
            "real-robot execution",
        ],
    })
    print(output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
