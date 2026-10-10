"""Audit and prepare v8 reorientation *proposal* scenes for the new keys.

The stock AutoDex reset runner maps v8 tabletop stems to legacy ``paradex``
stems before loading reset grasps. Precision keys have no legacy tabletop
tree, so its old cell loader cannot be used directly. Further, its carried
scene omits this demo's frozen socket fixture. Scenes prepared here are only
BODex proposal inputs, never a physical reorientation authorization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from autodex.utils.path import RESET_RELEASE_HEIGHTS_CM

from .config import TaskMode
from .geometry import validate_se3


def _paths(shared_root: Path, mode: TaskMode) -> tuple[Path, Path]:
    root = Path(shared_root).expanduser().resolve()
    object_dir = root / "object_processing" / mode.key_object
    tabletop = object_dir / "processed_data" / "info" / "tabletop"
    simplified = object_dir / "processed_data" / "mesh" / "simplified.obj"
    raw = object_dir / "raw_mesh" / f"{mode.key_object}.obj"
    if not tabletop.is_dir() or not simplified.is_file() or not raw.is_file():
        raise FileNotFoundError("v8 key mesh or tabletop assets are missing")
    return root, object_dir


def _tabletop_ids(object_dir: Path) -> tuple[int, ...]:
    folder = object_dir / "processed_data" / "info" / "tabletop"
    files = sorted(folder.glob("*.npy"))
    if len(files) < 2:
        raise ValueError("reorientation needs at least two v8 tabletop poses")
    ids = []
    for file in files:
        if not file.stem.isdigit():
            raise ValueError(f"nonnumeric v8 tabletop pose: {file.name}")
        validate_se3(np.load(file, allow_pickle=False),
                     name=f"v8 tabletop {file.stem}")
        ids.append(int(file.stem))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate numeric v8 tabletop stems")
    return tuple(sorted(ids))


def _scene_path(object_dir: Path, h_cm: int, i: int, j: int) -> Path:
    return object_dir / "scene" / f"reorient_{h_cm}" / f"{i}_{j}.json"


def _validate_scene(scene: dict, *, mode: TaskMode, object_dir: Path,
                    h_cm: int, i: int, j: int) -> None:
    if not isinstance(scene, dict) or scene.get("meta") != {
            "scene_type": f"reorient_{h_cm}",
            "pose_i": f"{i:03d}", "pose_j": f"{j:03d}",
            "h": h_cm / 100.0, "thickness": 0.01, "version": "v8"}:
        raise ValueError(f"reorient scene metadata differs from v8 cell {i}_{j}")
    try:
        target = scene["scene"]["mesh"]["target"]
        cuboids = scene["scene"]["cuboid"]
    except (KeyError, TypeError) as exc:
        raise ValueError("reorient scene lacks target or cuboid geometry") from exc
    expected_mesh = (object_dir / "processed_data" / "mesh" /
                     "simplified.obj").resolve()
    if (Path(target["file_path"]).resolve() != expected_mesh or
            not Path(target["urdf_path"]).is_file() or
            "table_i" not in cuboids or "table_j" not in cuboids):
        raise ValueError("reorient scene uses wrong key mesh or lacks tables")
    if mode.key_object not in str(expected_mesh):
        raise ValueError("reorient scene key object differs from selected mode")


def prepare_v8_reorient_scenes(
    *, shared_root: Path, mode: TaskMode, manifest_path: Path,
    heights_cm: tuple[int, ...] = RESET_RELEASE_HEIGHTS_CM,
) -> dict:
    """Create missing BODex scenes by reusing AutoDex's v8 scene generator.

    Existing scenes are checked, never overwritten. This creates no grasps,
    release trajectory, socket-aware collision world, or physical reset plan.
    """
    root, object_dir = _paths(shared_root, mode)
    heights = tuple(heights_cm)
    if not heights or len(set(heights)) != len(heights) or any(
            type(h) is not int or h not in RESET_RELEASE_HEIGHTS_CM
            for h in heights):
        raise ValueError("heights must be unique AutoDex release heights")
    output = Path(manifest_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"scene manifest already exists: {output}")
    ids = _tabletop_ids(object_dir)
    from src.grasp_generation.reorient.gen_scene import gen_reorient_scene

    rows = []
    pending: list[tuple[Path, dict]] = []
    for h_cm in heights:
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                path = _scene_path(object_dir, h_cm, i, j)
                if path.exists():
                    scene = json.loads(path.read_text(encoding="utf-8"))
                    status = "existing_verified"
                else:
                    scene = gen_reorient_scene(
                        mode.key_object, i, j, h_cm / 100.0,
                        obj_root=str(root / "object_processing"))
                    scene["meta"]["scene_type"] = f"reorient_{h_cm}"
                    pending.append((path, scene))
                    status = "new_v8_proposal_scene"
                _validate_scene(
                    scene, mode=mode, object_dir=object_dir,
                    h_cm=h_cm, i=i, j=j)
                rows.append({
                    "h_cm": h_cm, "from_v8_pose": i, "to_v8_pose": j,
                    "scene": str(path), "status": status,
                })
    for path, scene in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(scene, stream, indent=2, allow_nan=False)
            stream.write("\n")
    for row in rows:
        row["sha256"] = hashlib.sha256(Path(row["scene"]).read_bytes()).hexdigest()
    report = {
        "schema": "precision_insertion_v8_reorient_scene_manifest_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "shared_root": str(root), "v8_pose_stems": list(ids),
        "directed_scene_count": len(rows),
        "new_scene_count": len(pending), "scenes": rows,
        "scope": "BODex_reorient_proposal_scenes_only",
        "missing_next_stages": [
            "BODex plus original grasp-stability filter for each directed cell",
            "demo-local v8 reset grasp loader without legacy pose-index mapping",
            "socket-aware full Franka lift/reorient/place/retract preflight",
            "physical release and post-repose pose verification",
        ],
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def audit_v8_reorient_assets(*, shared_root: Path, mode: TaskMode) -> dict:
    """Separate BODex scene availability from truly staged reset grasps."""
    root, object_dir = _paths(shared_root, mode)
    ids = _tabletop_ids(object_dir)
    rows = []
    for i in ids:
        for j in ids:
            if i == j:
                continue
            scenes = []
            candidate_counts = {}
            for h_cm in RESET_RELEASE_HEIGHTS_CM:
                scene = _scene_path(object_dir, h_cm, i, j)
                if scene.is_file():
                    data = json.loads(scene.read_text(encoding="utf-8"))
                    _validate_scene(data, mode=mode, object_dir=object_dir,
                                    h_cm=h_cm, i=i, j=j)
                    scenes.append(h_cm)
                cell = (root / "AutoDex" / "candidates" / "inspire" /
                        f"reset_{h_cm}" / mode.key_object /
                        f"reorient_{h_cm}" / f"{i}_{j}")
                candidates = []
                if cell.is_dir():
                    for seed in cell.iterdir():
                        if not seed.is_dir():
                            continue
                        required = ("wrist_se3.npy", "pregrasp_pose.npy",
                                    "grasp_pose.npy", "sim_eval.json")
                        if not all((seed / name).is_file() for name in required):
                            continue
                        try:
                            stable = json.loads((seed / "sim_eval.json").read_text(
                                encoding="utf-8")).get("success") is True
                            validate_se3(np.load(seed / "wrist_se3.npy",
                                                 allow_pickle=False),
                                         name="reset candidate T_key_hand")
                        except (ValueError, TypeError, json.JSONDecodeError):
                            continue
                        if stable:
                            candidates.append(seed.name)
                candidate_counts[str(h_cm)] = len(candidates)
            rows.append({
                "from_v8_pose": i, "to_v8_pose": j,
                "scene_heights_cm": scenes,
                "stable_reset_seed_counts_by_height_cm": candidate_counts,
                "has_any_stable_seed": any(candidate_counts.values()),
            })
    legacy = (root / "AutoDex" / "object" / "paradex" /
              mode.key_object / "processed_data" / "info" / "tabletop")
    staging = []
    for manifest in sorted((root / "AutoDex" / "precision_insertion").glob(
            "reorient_candidates*/inspire/reset_*/" + mode.key_object +
            "/reorient_*/manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            staging.append({
                "manifest": str(manifest),
                "sampled_passes": data.get("total_sampled_passes"),
                "status": data.get("status"),
            })
        except (ValueError, OSError):
            staging.append({"manifest": str(manifest),
                            "status": "unreadable_staging_manifest"})
    return {
        "schema": "precision_insertion_v8_reorient_asset_audit_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "v8_pose_stems": list(ids), "directed_pairs": rows,
        "legacy_tabletop_tree_present": legacy.is_dir(),
        "stock_reset_runner_compatible": False,
        "stock_reset_runner_reason": (
            "new key has no verified legacy pose-index map and stock carried "
            "scene omits the fixed socket fixture"),
        "staging_manifests_not_runtime_candidates": staging,
        "robot_ready": False,
    }
