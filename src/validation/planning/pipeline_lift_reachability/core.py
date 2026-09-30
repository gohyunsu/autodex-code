"""Production-path evaluator for per-grasp and full-pool lift reachability."""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np


CandidateKey = tuple[str, str, str]


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def planner_robot_for(arm: str, hand: str) -> str:
    """Mirror ``src.execution.run_auto._planner_robot`` without hardware imports."""
    if arm == "xarm":
        if hand not in {"allegro", "inspire", "inspire_left"}:
            raise ValueError(f"unsupported xarm hand: {hand!r}")
        return hand
    if arm == "franka" and hand == "inspire":
        return "fr3_inspire"
    raise ValueError(f"unsupported arm/hand combination: {arm}/{hand}")


def available_objects(hand: str, version: str) -> list[str]:
    """Return every directory/archive object in the hand's candidate pool."""
    from autodex.utils.path import get_candidate_path

    root = Path(get_candidate_path(hand)) / version
    if not root.is_dir():
        return []
    names: set[str] = set()
    for path in root.iterdir():
        if path.is_dir():
            names.add(path.name)
        elif path.name.endswith(".tar.gz"):
            names.add(path.name[:-7])
        elif path.suffix in {".tgz", ".zip"}:
            names.add(path.stem)
    return sorted(names)


def tabletop_files(obj: str, version: str) -> list[Path]:
    from autodex.utils.path import get_obj_root

    root = Path(get_obj_root(version)) / obj / "processed_data" / "info" / "tabletop"
    return sorted(root.glob("*.npy")) if root.is_dir() else []


def load_tabletop_transform(path: Path, r_m: float, theta_deg: float,
                            table_surface_z_m: float = 0.040) -> np.ndarray:
    """Place one tabletop asset at a production-style polar table location."""
    raw = np.asarray(np.load(path), dtype=np.float64)
    if raw.shape == (3, 3):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = raw
    elif raw.shape == (4, 4):
        transform = raw.copy()
    else:
        raise ValueError(f"unsupported tabletop pose shape {raw.shape}: {path}")
    transform[:3, 3] += np.array([float(r_m), 0.0, float(table_surface_z_m)])
    theta = math.radians(float(theta_deg))
    rotation_z = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    # As in reachability_set.py, orbit the position while keeping the object
    # orientation fixed in the robot/world frame.
    transform[:3, 3] = rotation_z @ transform[:3, 3]
    return transform


def build_scene(obj: str, version: str, pose_robot: np.ndarray,
                table_surface_z_m: float = 0.040) -> tuple[dict, np.ndarray]:
    """Use the production scene builder and return its final robot-frame pose."""
    from autodex.utils.conversion import cart2se3
    from autodex.utils.path import get_obj_root
    from src.execution.scene_cfg import pose_world_to_scene_cfg

    tabletop_geometry = {"table_surface_z_m": float(table_surface_z_m)}
    scene = pose_world_to_scene_cfg(
        np.asarray(pose_robot, dtype=np.float64), np.eye(4), obj,
        get_obj_root(version), tabletop_geometry=tabletop_geometry)
    final_pose = cart2se3(np.asarray(scene["mesh"]["target"]["pose"], dtype=np.float64))
    return scene, final_pose


def polar_cells(radii_m: Iterable[float], thetas_deg: Iterable[float]) -> list[dict]:
    rows = []
    for r_index, radius in enumerate(radii_m):
        for theta_index, theta in enumerate(thetas_deg):
            angle = math.radians(float(theta))
            rows.append({
                "cell_id": f"r{r_index:03d}_t{theta_index:03d}",
                "r_index": int(r_index),
                "theta_index": int(theta_index),
                "r_m": float(radius),
                "theta_deg": float(theta),
                "nominal_x_m": float(radius) * math.cos(angle),
                "nominal_y_m": float(radius) * math.sin(angle),
            })
    return rows


def _candidate_result(obj_root: Path, key: CandidateKey) -> dict[str, Any] | None:
    scene_type, scene_id, grasp_id = key
    directory = (obj_root / scene_type / scene_id / grasp_id
                 if scene_type else obj_root / scene_id / grasp_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        return None
    try:
        with result_path.open() as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


@dataclass
class CandidateCatalogue:
    obj: str
    version: str
    hand: str
    pose_stem: str
    pool: str
    wrist_object: np.ndarray
    pregrasp: np.ndarray
    grasp: np.ndarray
    openpose: list[np.ndarray | None]
    scene_info: list[CandidateKey]
    groups: list[dict[str, Any]]
    candidate_order: list[CandidateKey] | None
    load_contract: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "object": self.obj,
            "version": self.version,
            "hand": self.hand,
            "tabletop_pose_stem": self.pose_stem,
            "candidate_pool": self.pool,
            "load_contract": self.load_contract,
            "base_grasp_count": len(self.groups),
            "symmetry_expanded_count": len(self.wrist_object),
            "candidate_order": (
                None if self.candidate_order is None
                else [list(key) for key in self.candidate_order]),
            "groups": self.groups,
        }


def load_candidate_catalogue(*, obj: str, version: str, hand: str,
                             pose_stem: str, pool: str,
                             clean_state_root: Path) -> CandidateCatalogue:
    """Load object-frame candidates using the same path/coverage utilities as production."""
    from autodex.planner.planner import _expand_candidates_cyl
    from autodex.utils.coverage import load_coverage_map
    from autodex.utils.path import (get_candidate_path, load_candidate,
                                    load_openpose_for_candidates,
                                    resolve_candidate_object_path)
    from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid

    if pool not in {"all", "training-remaining", "verified-only"}:
        raise ValueError(f"unknown candidate pool: {pool}")

    skip_done = pool == "training-remaining"
    skip_scenes = pool == "training-remaining"
    success_only = pool == "verified-only"
    candidate_order: list[CandidateKey] | None = None
    remaining_map = load_coverage_map(
        obj, tabletop_pose_stem=pose_stem, hand=hand, version=version)
    immutable_map = load_coverage_map(
        obj, tabletop_pose_stem=pose_stem, hand=hand, version=version,
        success_root=str(clean_state_root))
    if pool == "training-remaining":
        if remaining_map is None:
            raise RuntimeError(
                f"production coverage JSON missing for {obj}/{version}; "
                "build it with src/dataset/compute_v8_coverage.py")
        useful = {tuple(map(str, key)): int(value)
                  for key, value in remaining_map.items() if int(value) > 0}
        candidate_order = sorted(useful, key=lambda key: -useful[key])
    elif pool == "verified-only" and immutable_map is not None:
        useful = {tuple(map(str, key)): int(value)
                  for key, value in immutable_map.items() if int(value) > 0}
        candidate_order = sorted(useful, key=lambda key: -useful[key])

    wrist, pregrasp, grasp, scene_info_raw = load_candidate(
        obj, np.eye(4), version, shuffle=False,
        skip_done=skip_done, success_only=success_only, hand=hand,
        scene_id=None, scene_type_filter=None,
        skip_scenes_with_success=skip_scenes,
        tabletop_pose_stem=pose_stem, candidate_order=candidate_order)
    scene_info_base = [tuple(map(str, key)) for key in scene_info_raw]
    openpose = (load_openpose_for_candidates(
        obj, scene_info_base, hand, version, pose_stem)
                if len(scene_info_base) else [])

    axis = get_cyl_axis_local(obj)
    yaw_grid = get_cyl_yaw_grid(obj)
    wrist, pregrasp, grasp, openpose, expanded_info = _expand_candidates_cyl(
        wrist, pregrasp, grasp, openpose, scene_info_base, np.eye(4), axis, yaw_grid)
    scene_info = [tuple(map(str, key)) for key in expanded_info]

    grouped: OrderedDict[CandidateKey, list[int]] = OrderedDict()
    for index, key in enumerate(scene_info):
        grouped.setdefault(key, []).append(index)

    candidate_root = resolve_candidate_object_path(
        get_candidate_path(hand), version, obj)
    root_path = Path(candidate_root) if candidate_root is not None else Path("/")
    groups = []
    for rank, (key, indices) in enumerate(grouped.items(), start=1):
        prior = _candidate_result(root_path, key)
        groups.append({
            "candidate_key": list(key),
            "catalogue_rank": rank,
            "variant_indices": [int(index) for index in indices],
            "variant_count": len(indices),
            "symmetry_offsets_rad": (
                [float(value) for value in np.asarray(yaw_grid, dtype=float)[:len(indices)]]
                if yaw_grid is not None and len(yaw_grid) > 1
                else [0.0]),
            "immutable_coverage": (
                None if immutable_map is None else int(immutable_map.get(key, 0))),
            "remaining_coverage": (
                None if remaining_map is None else int(remaining_map.get(key, 0))),
            "prior_result_present": prior is not None,
            "prior_success": bool((prior or {}).get("success", False)),
            "prior_result_arm": (prior or {}).get("arm"),
        })

    return CandidateCatalogue(
        obj=obj, version=version, hand=hand, pose_stem=pose_stem, pool=pool,
        wrist_object=np.asarray(wrist, dtype=np.float64),
        pregrasp=np.asarray(pregrasp, dtype=np.float32),
        grasp=np.asarray(grasp, dtype=np.float32),
        openpose=list(openpose), scene_info=scene_info, groups=groups,
        candidate_order=candidate_order,
        load_contract={
            "skip_done": skip_done,
            "skip_scenes_with_success": skip_scenes,
            "success_only": success_only,
            "tabletop_pose_stem": pose_stem,
            "coverage_ordered": candidate_order is not None,
            "symmetry_axis_local": None if axis is None else np.asarray(axis).tolist(),
            "symmetry_yaw_grid_rad": (
                None if yaw_grid is None else np.asarray(yaw_grid).tolist()),
        },
    )


def _padded_endpoint_solve(planner, poses: np.ndarray, *, retract: np.ndarray,
                           seed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the planner's own IKSolver using its fixed CUDA batch shape."""
    import torch
    from autodex.planner.planner import _to_curobo_pose

    transforms = np.asarray(poses, dtype=np.float64)
    count = len(transforms)
    success = np.zeros(count, dtype=bool)
    solutions = np.full((count, len(planner._init_state)), np.nan, dtype=np.float32)
    if count == 0:
        return success, solutions
    retract_rows = np.asarray(retract, dtype=np.float32)
    if retract_rows.ndim == 1:
        retract_rows = np.repeat(retract_rows[None, :], count, axis=0)
    seed_rows = None if seed is None else np.asarray(seed, dtype=np.float32)
    if seed_rows is not None and seed_rows.ndim == 2:
        seed_rows = seed_rows[:, None, :]

    for begin in range(0, count, planner.BATCH_SIZE):
        actual = min(planner.BATCH_SIZE, count - begin)
        pose_chunk = transforms[begin:begin + actual]
        retract_chunk = retract_rows[begin:begin + actual]
        seed_chunk = None if seed_rows is None else seed_rows[begin:begin + actual]
        if actual < planner.BATCH_SIZE:
            pad = planner.BATCH_SIZE - actual
            pose_chunk = np.concatenate(
                [pose_chunk, np.repeat(pose_chunk[:1], pad, axis=0)])
            retract_chunk = np.concatenate(
                [retract_chunk, np.repeat(retract_chunk[:1], pad, axis=0)])
            if seed_chunk is not None:
                seed_chunk = np.concatenate(
                    [seed_chunk, np.repeat(seed_chunk[:1], pad, axis=0)])
        kwargs: dict[str, Any] = {
            "retract_config": torch.as_tensor(
                np.ascontiguousarray(retract_chunk), dtype=torch.float32,
                device=planner._tensor_args.device),
        }
        if seed_chunk is not None:
            kwargs["seed_config"] = torch.as_tensor(
                np.ascontiguousarray(seed_chunk), dtype=torch.float32,
                device=planner._tensor_args.device)
        result = planner._ik_solver.solve_batch(
            _to_curobo_pose(pose_chunk, planner._tensor_args.device), **kwargs)
        raw_success = result.success.detach().cpu().numpy()[:actual]
        raw_success = raw_success.reshape(actual, -1).any(axis=1)
        raw_solution = result.solution.detach().cpu().numpy()[:actual]
        if raw_solution.ndim == 3:
            raw_solution = raw_solution[:, 0, :]
        for local in range(actual):
            if not raw_success[local]:
                continue
            arm = raw_solution[local, :planner._n_arm].copy()
            planner._snap_arm(arm, retract_chunk[local])
            # This is deliberately the same current production policy, even
            # for an arm whose URDF admits a wider non-wrapping joint.
            if np.any(np.abs(arm) > np.pi):
                continue
            success[begin + local] = True
            solutions[begin + local] = raw_solution[local]
            solutions[begin + local, :planner._n_arm] = arm
    return success, solutions


def endpoint_diagnostics(planner, scene_cfg: dict, wrist_world: np.ndarray,
                         grasp: np.ndarray, lift_height_m: float = 0.10) -> dict[str, Any]:
    """Probe both endpoints without changing the authoritative plan outcome."""
    from autodex.planner.planner import _to_curobo_world, _without_target_mesh

    world = _without_target_mesh(_to_curobo_world(scene_cfg))
    planner._set_ik_world(world)
    wrist = np.asarray(wrist_world, dtype=np.float64)
    top = wrist.copy()
    top[:, 2, 3] += float(lift_height_m)
    retract = np.repeat(
        np.asarray(planner._init_state, dtype=np.float32)[None, :], len(wrist), axis=0)

    started = perf_counter()
    bottom_ok, bottom_q = _padded_endpoint_solve(
        planner, wrist, retract=retract)
    bottom_s = perf_counter() - started
    started = perf_counter()
    top_ok, _top_q = _padded_endpoint_solve(planner, top, retract=retract)
    top_s = perf_counter() - started

    local_ok = np.zeros(len(wrist), dtype=bool)
    local_s = 0.0
    local_indices = np.flatnonzero(bottom_ok)
    if len(local_indices):
        local_seed = bottom_q[local_indices].copy()
        local_seed[:, planner._n_arm:] = np.asarray(grasp, dtype=np.float32)[local_indices]
        started = perf_counter()
        solved, _local_q = _padded_endpoint_solve(
            planner, top[local_indices], retract=local_seed, seed=local_seed)
        local_s = perf_counter() - started
        local_ok[local_indices] = solved

    both = bottom_ok & top_ok
    return {
        "lift_height_m": float(lift_height_m),
        "bottom_ik_by_variant": bottom_ok.tolist(),
        "top_ik_independent_by_variant": top_ok.tolist(),
        "top_ik_local_by_variant": local_ok.tolist(),
        "bottom_ik_success": bool(bottom_ok.any()),
        "top_ik_independent_success": bool(top_ok.any()),
        "both_endpoint_same_variant_success": bool(both.any()),
        "top_ik_local_success": bool(local_ok.any()),
        "timing": {
            "bottom_ik_s": bottom_s,
            "top_ik_independent_s": top_s,
            "top_ik_local_s": local_s,
            "total_s": bottom_s + top_s + local_s,
        },
    }


def _plan_failure(timing: Mapping[str, Any]) -> tuple[str, str]:
    if int(timing.get("n_total", 0)) == 0:
        return "candidate", "candidate_catalogue_empty"
    if int(timing.get("n_valid", 0)) == 0:
        return "candidate_filter", "candidate_all_filtered"
    if int(timing.get("n_ik_success", 0)) == 0:
        return "endpoint_ik", "candidate_all_ik_failed"
    if int(timing.get("n_lift_preflight_attempts", 0)) == 0:
        return "approach", "approach_all_failed"
    failures = timing.get("jacobian_lift_failures") or {}
    if failures:
        code = max(failures, key=lambda key: int(failures[key]))
        return "jacobian_lift", str(code)
    return "planning", "planning_failed_unknown"


def _stroke_summary(result) -> dict[str, Any] | None:
    preflight = getattr(result, "lift_preflight", None)
    stroke = None if preflight is None else preflight.vertical_stroke
    if stroke is None:
        return None
    conditions = [float(row.get("condition_number", np.nan))
                  for row in stroke.step_records]
    singular = [float(row.get("min_singular_value", np.nan))
                for row in stroke.step_records]
    return {
        "direction": stroke.direction,
        "distance_m": float(stroke.distance_m),
        "trajectory_samples": (
            None if stroke.trajectory is None else int(len(stroke.trajectory))),
        "trajectory_duration_s": (
            None if stroke.time_s is None or len(stroke.time_s) == 0
            else float(stroke.time_s[-1])),
        "max_condition_number": (
            None if not conditions else float(np.nanmax(conditions))),
        "min_singular_value": (
            None if not singular else float(np.nanmin(singular))),
        "validation": stroke.validation,
        "timing": stroke.timing,
    }


def evaluate_candidate_group(planner, *, catalogue: CandidateCatalogue,
                             group: Mapping[str, Any], scene_cfg: dict,
                             object_pose: np.ndarray, cell: Mapping[str, Any],
                             trial: int, seed: int,
                             trajectory_dir: Path | None = None) -> dict[str, Any]:
    indices = np.asarray(group["variant_indices"], dtype=int)
    wrist_world = np.matmul(object_pose, catalogue.wrist_object[indices])
    pregrasp = catalogue.pregrasp[indices]
    grasp = catalogue.grasp[indices]
    openpose = [catalogue.openpose[index] for index in indices]
    key = tuple(map(str, group["candidate_key"]))
    scene_info = [key] * len(indices)

    started = perf_counter()
    result = planner.plan(
        scene_cfg, catalogue.obj, catalogue.version, seed=seed,
        skip_done=False, success_only=False, hand=catalogue.hand,
        openpose_pose_stem=catalogue.pose_stem,
        tabletop_pose_stem=catalogue.pose_stem,
        # A non-None order prevents random reordering of symmetry hypotheses.
        candidate_order=[key],
        candidate_override=(wrist_world, pregrasp, grasp, scene_info, openpose),
    )
    elapsed = perf_counter() - started
    timing = dict(result.timing or {})
    approach_success = int(timing.get("n_lift_preflight_attempts", 0)) > 0
    lift_success = int(timing.get("n_lift_preflight_success", 0)) > 0
    if result.success:
        failure_stage = failure_code = None
    else:
        failure_stage, failure_code = _plan_failure(timing)

    diagnostic = endpoint_diagnostics(
        planner, scene_cfg, wrist_world, grasp, lift_height_m=0.10)
    selected_variant = None
    if result.success and result.wrist_se3 is not None:
        distances = [float(np.linalg.norm(np.asarray(result.wrist_se3) - pose))
                     for pose in wrist_world]
        selected_variant = int(np.argmin(distances))

    record = {
        **dict(cell),
        "trial": int(trial),
        "seed": int(seed),
        "candidate_key": list(key),
        "candidate_key_str": "/".join(key),
        "catalogue_rank": int(group["catalogue_rank"]),
        "variant_count": int(group["variant_count"]),
        "selected_variant_ordinal": selected_variant,
        "prior_success": bool(group.get("prior_success", False)),
        **diagnostic,
        "approach_success": bool(approach_success),
        "jacobian_lift_success": bool(lift_success),
        "pipeline_success": bool(result.success),
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "planner_timing": timing,
        "planner_wall_s": elapsed,
        "stroke": _stroke_summary(result),
    }
    if trajectory_dir is not None and result.success:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{cell['cell_id']}__{key[0]}_{key[1]}_{key[2]}__trial{trial:02d}"
        np.savez_compressed(
            trajectory_dir / f"{stem}.npz",
            approach_qpos=np.asarray(result.traj, dtype=np.float32),
            lift_qpos=np.asarray(result.lift_preflight.traj, dtype=np.float32),
            lift_time_s=np.asarray(result.lift_preflight.time_s, dtype=np.float64),
            wrist_world=wrist_world,
        )
        record["trajectory_file"] = str(Path("trajectories") / f"{stem}.npz")
    return record


def evaluate_pipeline_pool(planner, *, catalogue: CandidateCatalogue,
                           scene_cfg: dict, cell: Mapping[str, Any],
                           trial: int, seed: int) -> dict[str, Any]:
    """Call the public planner once with the complete production-style pool."""
    from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid

    contract = catalogue.load_contract
    started = perf_counter()
    result = planner.plan(
        scene_cfg, catalogue.obj, catalogue.version, seed=seed,
        skip_done=bool(contract["skip_done"]),
        success_only=bool(contract["success_only"]), hand=catalogue.hand,
        openpose_pose_stem=catalogue.pose_stem,
        cyl_axis_local=get_cyl_axis_local(catalogue.obj),
        cyl_yaw_grid=get_cyl_yaw_grid(catalogue.obj),
        skip_scenes_with_success=bool(contract["skip_scenes_with_success"]),
        tabletop_pose_stem=catalogue.pose_stem,
        candidate_order=catalogue.candidate_order,
    )
    elapsed = perf_counter() - started
    timing = dict(result.timing or {})
    if result.success:
        failure_stage = failure_code = None
    else:
        failure_stage, failure_code = _plan_failure(timing)
    return {
        **dict(cell), "trial": int(trial), "seed": int(seed),
        "pipeline_success": bool(result.success),
        "selected_candidate_key": (
            None if not result.success else list(map(str, result.scene_info))),
        "failure_stage": failure_stage, "failure_code": failure_code,
        "planner_timing": timing, "planner_wall_s": elapsed,
        "stroke": _stroke_summary(result),
    }


def evaluate_pose(*, planner, catalogue: CandidateCatalogue, pose_file: Path,
                  cells: list[dict[str, Any]], output_dir: Path,
                  evaluation: str, n_trials: int, base_seed: int,
                  max_grasps: int | None, save_trajectories: bool,
                  table_surface_z_m: float = 0.040) -> dict[str, Any]:
    """Evaluate one object/tabletop pose with resumable JSONL checkpoints."""
    if evaluation not in {"per-grasp", "pipeline-replay", "both"}:
        raise ValueError(f"unsupported evaluation mode: {evaluation}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_path = output_dir / "per_grasp.jsonl"
    replay_path = output_dir / "pipeline_replay.jsonl"
    completed_per = {
        (row["cell_id"], row["candidate_key_str"], int(row["trial"]))
        for row in read_jsonl(per_path)
    }
    completed_replay = {
        (row["cell_id"], int(row["trial"])) for row in read_jsonl(replay_path)
    }
    groups = catalogue.groups[:max_grasps] if max_grasps else catalogue.groups
    total_work = len(cells) * n_trials * (
        (len(groups) if evaluation in {"per-grasp", "both"} else 0)
        + (1 if evaluation in {"pipeline-replay", "both"} else 0))
    done = len(completed_per) + len(completed_replay)
    started = perf_counter()

    for cell_index, cell_base in enumerate(cells):
        raw_pose = load_tabletop_transform(
            pose_file, cell_base["r_m"], cell_base["theta_deg"], table_surface_z_m)
        scene_cfg, object_pose = build_scene(
            catalogue.obj, catalogue.version, raw_pose, table_surface_z_m)
        cell = {
            **cell_base,
            "object_x_m": float(object_pose[0, 3]),
            "object_y_m": float(object_pose[1, 3]),
            "object_z_m": float(object_pose[2, 3]),
        }
        for trial in range(n_trials):
            seed = int(base_seed + cell_index * 100003 + trial)
            if evaluation in {"pipeline-replay", "both"}:
                replay_id = (cell["cell_id"], trial)
                if replay_id not in completed_replay:
                    row = evaluate_pipeline_pool(
                        planner, catalogue=catalogue, scene_cfg=scene_cfg,
                        cell=cell, trial=trial, seed=seed)
                    append_jsonl(replay_path, row)
                    completed_replay.add(replay_id)
                    done += 1
                    print(
                        f"[pipeline] {done}/{total_work} {catalogue.obj}/{catalogue.pose_stem} "
                        f"cell={cell['cell_id']} trial={trial} "
                        f"{'ok' if row['pipeline_success'] else row['failure_code']} "
                        f"{row['planner_wall_s']:.2f}s", flush=True)

            if evaluation not in {"per-grasp", "both"}:
                continue
            for group in groups:
                key_str = "/".join(group["candidate_key"])
                work_id = (cell["cell_id"], key_str, trial)
                if work_id in completed_per:
                    continue
                row = evaluate_candidate_group(
                    planner, catalogue=catalogue, group=group,
                    scene_cfg=scene_cfg, object_pose=object_pose, cell=cell,
                    trial=trial, seed=seed,
                    trajectory_dir=(output_dir / "trajectories"
                                    if save_trajectories else None))
                append_jsonl(per_path, row)
                completed_per.add(work_id)
                done += 1
                eta = ((perf_counter() - started) / max(done, 1)
                       * max(total_work - done, 0))
                print(
                    f"[grasp] {done}/{total_work} {catalogue.obj}/{catalogue.pose_stem} "
                    f"cell={cell['cell_id']} grasp={key_str} "
                    f"endpoints={int(row['both_endpoint_same_variant_success'])} "
                    f"lift={int(row['jacobian_lift_success'])} "
                    f"full={int(row['pipeline_success'])} "
                    f"{row['planner_wall_s']:.2f}s ETA={eta / 60.0:.1f}m",
                    flush=True)
    return {
        "per_grasp_records": len(completed_per),
        "pipeline_replay_records": len(completed_replay),
        "wall_s": perf_counter() - started,
    }
