"""Explicit-root v8 insertion grasp catalogue and pose-conditioned selection.

Offline eligibility is MuJoCo-supported grasp stability plus a centered
20 mm whole-hand/socket endpoint. It is not continuous arm planning, contact
control, or physical success. AutoDex's v8 NPY loader is reused only after
this module has resolved exact candidate keys under the selected shared root.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .assets import AssetPaths
from .config import TaskMode, select_mode
from .geometry import validate_se3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _candidate_dirs(root: Path) -> list[Path]:
    """Walk only the v8 ``scene_type/scene_id/grasp_id`` layout."""
    if not root.is_dir():
        return []
    return sorted(
        grasp for scene_type in root.iterdir() if scene_type.is_dir()
        for scene in scene_type.iterdir() if scene.is_dir()
        for grasp in scene.iterdir() if grasp.is_dir()
    )


def _scene_pose(shared_root: Path, key: str, candidate: Path) -> tuple[str, Path]:
    scene = (shared_root / "AutoDex" / "scene" / "inspire" / key /
             candidate.parent.parent.name / f"{candidate.parent.name}.json")
    meta = _json(scene).get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("pose_idx"), str):
        raise ValueError(f"v8 scene pose_idx missing: {scene}")
    stem = meta["pose_idx"]
    tabletop = (shared_root / "object_processing" / key / "processed_data" /
                "info" / "tabletop" / f"{stem}.npy")
    if not tabletop.is_file():
        raise FileNotFoundError(f"scene references missing tabletop pose: {tabletop}")
    return stem, scene


def _grasp_evidence(candidate: Path, key_mesh: Path) -> tuple[bool, str]:
    required = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
                "sim_eval.json", "simulation_validation.json")
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        return False, "missing:" + ",".join(missing)
    try:
        validate_se3(np.load(candidate / "wrist_se3.npy", allow_pickle=False),
                     name="candidate T_key_hand")
        for name in ("pregrasp_pose.npy", "grasp_pose.npy"):
            joints = np.asarray(np.load(candidate / name, allow_pickle=False))
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                return False, f"invalid:{name}"
        simulation = _json(candidate / "sim_eval.json")
        validation = _json(candidate / "simulation_validation.json")
        if simulation.get("success") is not True:
            return False, "mujoco_grasp_not_passed"
        if validation.get("status") != "passed":
            return False, "simulation_validation_not_passed"
        declared = validation.get("full_object_mesh_sha256")
        if declared is not None and declared != _sha256(key_mesh):
            return False, "stale_full_key_mesh"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"invalid_grasp_evidence:{exc}"
    return True, "mujoco_grasp_passed_not_physical"


def build_endpoint_catalog(
    *,
    shared_root: Path,
    mode: TaskMode,
    minimum_hand_clearance_m: float,
    max_candidates: int | None = None,
    screen: Callable | None = None,
) -> dict:
    """Screen the entire current v8 pool, or a clearly marked pilot prefix.

    ``screen`` is injectable for offline tests; normal calls use the exact
    Coal CAD screen. A screen error makes catalogue completeness false and
    cannot silently promote a candidate. Clearance is an explicit physical
    commissioning input, never an inferred default.
    """
    if not math.isfinite(minimum_hand_clearance_m) or minimum_hand_clearance_m <= 0:
        raise ValueError("minimum_hand_clearance_m must be finite and positive")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    required_assets = {
        "key_raw_mesh": paths.raw_mesh(mode.key_object),
        "key_planning_mesh": paths.key_planning_mesh,
        "socket_collision_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
    }
    missing_assets = [f"{name}: {path}" for name, path in required_assets.items()
                      if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError("missing exact endpoint assets: " +
                                ", ".join(missing_assets))
    if screen is None:
        from .endpoint import screen_grasp_endpoint
        screen = screen_grasp_endpoint
    all_dirs = _candidate_dirs(paths.candidate_dir)
    if max_candidates is None:
        dirs = all_dirs
    else:
        dirs = all_dirs[:max_candidates]
    rows = []
    errors = []
    endpoint_attempted = 0
    for candidate in dirs:
        key = list(candidate.relative_to(paths.candidate_dir).parts)
        row = {
            "key": key,
            "candidate_dir": str(candidate),
            "tabletop_pose_stem": None,
            "grasp_stability_pass": False,
            "grasp_stability_reason": None,
            "grasp_input_sha256": {},
            "endpoint_pass": False,
            "endpoint_report": None,
            "eligible": False,
            "error": None,
        }
        try:
            stem, scene = _scene_pose(root, mode.key_object, candidate)
            row["tabletop_pose_stem"] = stem
            row["scene_path"] = str(scene)
            row["scene_sha256"] = _sha256(scene)
            grasp_inputs = {
                "key_planning_mesh": paths.key_planning_mesh,
                "wrist_se3": candidate / "wrist_se3.npy",
                "pregrasp_pose": candidate / "pregrasp_pose.npy",
                "grasp_pose": candidate / "grasp_pose.npy",
                "sim_eval": candidate / "sim_eval.json",
                "simulation_validation": candidate / "simulation_validation.json",
            }
            row["grasp_input_sha256"] = {
                name: _sha256(path) if path.is_file() else None
                for name, path in grasp_inputs.items()
            }
            passed, reason = _grasp_evidence(candidate, paths.key_planning_mesh)
            row["grasp_stability_pass"] = passed
            row["grasp_stability_reason"] = reason
            if passed:
                endpoint_attempted += 1
                endpoint = screen(
                    shared_root=root, mode=mode, candidate_dir=candidate,
                    minimum_hand_clearance_m=minimum_hand_clearance_m,
                )
                if (not isinstance(endpoint, dict) or
                        type(endpoint.get("endpoint_pass")) is not bool or
                        not isinstance(endpoint.get("input_sha256"), dict)):
                    raise ValueError("endpoint screen returned an invalid report")
                row["endpoint_report"] = endpoint
                row["endpoint_pass"] = endpoint.get("endpoint_pass") is True
                row["eligible"] = row["endpoint_pass"]
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"key": key, "error": row["error"]})
        rows.append(row)
    return {
        "schema": "precision_insertion_endpoint_catalog_v1",
        "shared_root": str(root),
        "mode": {
            "family": mode.family, "gap_mm": mode.gap_mm,
            "key_object": mode.key_object, "socket_object": mode.socket_object,
            "target_depth_m": mode.target_depth_m,
        },
        "minimum_hand_clearance_m": float(minimum_hand_clearance_m),
        "candidate_root": str(paths.candidate_dir),
        "total_candidate_directories": len(all_dirs),
        "screened_directories": len(dirs),
        "endpoint_screen_attempted_count": endpoint_attempted,
        "complete_scan": (paths.candidate_dir.is_dir() and bool(all_dirs) and
                          max_candidates is None and not errors),
        "errors": errors,
        "candidates": rows,
        "eligible_count": sum(row["eligible"] for row in rows),
        "not_validated": [
            "live pose-conditioned Franka IK or full transfer/insertion paths",
            "guarded contact, release, reset, or reorientation",
            "physical grasp or insertion success",
        ],
        "robot_ready": False,
    }


def select_pose_candidates(
    catalog: dict,
    *,
    tabletop_pose_stem: str,
    attempted: Iterable[tuple[str, str, str]] = (),
    covered_scenes: Iterable[int] = (),
) -> dict:
    """Select only eligible grasps matching a freshly observed v8 tabletop.

    Missing coverage JSON leaves deterministic candidate order; coverage
    gains are optional ranking metadata, never a substitute for endpoint
    evidence. A successful insertion (not merely a lift) should advance
    task-level covered scenes in the caller's private trial record.
    """
    if not catalog.get("complete_scan"):
        return {"status": "catalog_incomplete", "candidates": [],
                "reason": "full endpoint screen or source pool is missing"}
    root = Path(catalog["shared_root"])
    identity = catalog["mode"]
    mode = select_mode(identity["family"], identity["gap_mm"])
    if (identity["key_object"] != mode.key_object or
            identity["socket_object"] != mode.socket_object or
            identity["target_depth_m"] != mode.target_depth_m):
        raise ValueError("catalog mode identity does not match configured v8 objects")
    paths = AssetPaths(root, mode)
    if Path(catalog["candidate_root"]).resolve() != paths.candidate_dir.resolve():
        raise ValueError("catalog candidate root does not match shared root")
    key_name = mode.key_object
    current_keys = [list(path.relative_to(paths.candidate_dir).parts)
                    for path in _candidate_dirs(paths.candidate_dir)]
    if current_keys != [row["key"] for row in catalog["candidates"]]:
        return {"status": "catalog_stale", "candidates": [],
                "reason": "v8 candidate directory set changed"}
    pose_file = paths.key_tabletop_dir / f"{tabletop_pose_stem}.npy"
    if not pose_file.is_file():
        raise FileNotFoundError(f"unknown v8 tabletop pose: {pose_file}")
    excluded = {tuple(str(x) for x in key) for key in attempted}
    if any(len(key) != 3 for key in excluded):
        raise ValueError("attempted keys must be scene_type/scene_id/grasp_id")
    covered = set(int(i) for i in covered_scenes)
    coverage_path = (root / "AutoDex" / "experiment" / "v8" / "coverage" /
                     f"cov_v8_cand_{key_name}.json")
    gains = {}
    if coverage_path.is_file():
        for entry in _json(coverage_path).get("grasps", []):
            if not isinstance(entry, dict):
                continue
            candidate_key = tuple(str(entry.get(name, ""))
                                  for name in ("type", "sid", "gid"))
            gains[candidate_key] = len(set(entry.get("covers", [])) - covered)

    matching = []
    for row in catalog["candidates"]:
        candidate_key = tuple(row["key"])
        if row["tabletop_pose_stem"] != tabletop_pose_stem:
            continue
        scene = Path(row["scene_path"])
        if not scene.is_file() or _sha256(scene) != row["scene_sha256"]:
            return {"status": "catalog_stale", "candidates": [],
                    "reason": f"v8 scene changed: {scene}"}
        candidate = Path(row["candidate_dir"])
        sources = {
            "key_planning_mesh": paths.key_planning_mesh,
            "wrist_se3": candidate / "wrist_se3.npy",
            "pregrasp_pose": candidate / "pregrasp_pose.npy",
            "grasp_pose": candidate / "grasp_pose.npy",
            "sim_eval": candidate / "sim_eval.json",
            "simulation_validation": candidate / "simulation_validation.json",
        }
        for name, saved_hash in row["grasp_input_sha256"].items():
            source = sources[name]
            current_hash = _sha256(source) if source.is_file() else None
            if current_hash != saved_hash:
                return {"status": "catalog_stale", "candidates": [],
                        "reason": f"grasp evidence changed: {source}"}
        if not row["eligible"] or candidate_key in excluded:
            continue
        report = row["endpoint_report"]
        endpoint_sources = {
            "key_mesh": paths.raw_mesh(key_name),
            "socket_mesh": paths.socket_collision_mesh,
            "task_geometry": paths.task_geometry,
            "robot_urdf": paths.robot_urdf,
            "wrist_se3": candidate / "wrist_se3.npy",
            "grasp_pose": candidate / "grasp_pose.npy",
        }
        for name, saved_hash in report["input_sha256"].items():
            source = endpoint_sources.get(name)
            if source is None or not source.is_file() or _sha256(source) != saved_hash:
                return {"status": "catalog_stale", "candidates": [],
                        "reason": f"endpoint input changed: {name}"}
        matching.append({
            "key": row["key"], "candidate_dir": row["candidate_dir"],
            "uncovered_scene_gain": gains.get(candidate_key),
            "endpoint_report": row["endpoint_report"],
        })
    matching.sort(key=lambda row: (
        -(row["uncovered_scene_gain"] or 0), tuple(row["key"])))
    return {
        "status": ("candidates_available" if matching else
                   "no_eligible_in_screened_pool"),
        "tabletop_pose_stem": tabletop_pose_stem,
        "candidates": matching,
        "coverage_path": str(coverage_path) if coverage_path.is_file() else None,
        "reason": ("offline eligibility only; online full-motion preflight required"
                   if matching else
                   "no eligible grasp in this finite v8 pool; this is not impossibility proof"),
    }


def planner_candidate_override(
    *, catalog: dict, selected: list[dict],
    pose_robot_key: np.ndarray, tabletop_pose_stem: str,
) -> tuple:
    """Reuse AutoDex's v8 transform/NPY loader with an explicit whitelist.

    This is input to unchanged ``GraspPlanner.plan(candidate_override=...)``;
    that planner still stops at pickup/lift and needs demo-local transfer and
    insertion preflight before the candidate can be executed.
    """
    if not selected:
        raise ValueError("at least one selected candidate is required")
    if not catalog.get("complete_scan"):
        raise ValueError("cannot load an incomplete endpoint catalog")
    pose = validate_se3(pose_robot_key, name="T_robot_key")
    current = select_pose_candidates(
        catalog, tabletop_pose_stem=tabletop_pose_stem)
    if current["status"] != "candidates_available":
        raise ValueError(f"catalog cannot supply live candidates: {current['status']}")
    root = Path(catalog["shared_root"])
    name = catalog["mode"]["key_object"]
    order = [tuple(row["key"]) for row in selected]
    if len(order) != len(set(order)):
        raise ValueError("selected candidate keys must be unique")
    approved = {
        tuple(row["key"]): row for row in current["candidates"]
    }
    if any(key not in approved or
           selected_row.get("candidate_dir") != approved[key]["candidate_dir"]
           for key, selected_row in zip(order, selected)):
        raise ValueError("selected candidates do not match eligible pose rows")
    from autodex.utils.path import load_candidate

    wrist, pregrasp, grasp, info = load_candidate(
        name, pose, "v8", hand="inspire", shuffle=False,
        skip_done=False, candidate_order=order,
        candidates_root=root / "AutoDex" / "candidates" / "inspire",
    )
    if [tuple(entry) for entry in info] != order:
        raise RuntimeError("v8 loader returned a different candidate order")
    openposes = []
    for row in selected:
        candidate = Path(row["candidate_dir"])
        path = candidate / f"openpose_{tabletop_pose_stem}.npy"
        value = np.load(path, allow_pickle=False) if path.is_file() else None
        if value is not None and (value.shape != (6,) or
                                  not np.all(np.isfinite(value))):
            raise ValueError(f"invalid candidate openpose: {path}")
        openposes.append(value)
    return wrist, pregrasp, grasp, info, openposes
