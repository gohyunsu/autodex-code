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
import math
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


def _sim_filter_scene_path(root: Path, mode: TaskMode,
                           h_cm: int, i: int, j: int) -> Path:
    # Mirror get_scene_dir("inspire", ...) while respecting the explicitly
    # selected shared root rather than AutoDex's process-global home path.
    return (root / "AutoDex" / "scene" / "inspire" / mode.key_object /
            f"reorient_{h_cm}" / f"{i}_{j}.json")


def _validate_scene(scene: dict, *, mode: TaskMode, object_dir: Path,
                    h_cm: int, i: int, j: int) -> None:
    expected_meta = {
            "scene_type": f"reorient_{h_cm}",
            "pose_i": f"{i:03d}", "pose_j": f"{j:03d}",
            "h": h_cm / 100.0, "thickness": 0.01, "version": "v8"}
    meta = scene.get("meta") if isinstance(scene, dict) else None
    if (not isinstance(meta, dict) or
            any(meta.get(key) != value for key, value in expected_meta.items()) or
            set(meta) - set(expected_meta) - {
                "geometry_object", "grasp_target_object"} or
            any(meta.get(key, mode.key_object) != mode.key_object
                for key in ("geometry_object", "grasp_target_object"))):
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


def _same_scene_geometry(a: dict, b: dict) -> bool:
    """Optional descriptive metadata may differ; collision worlds may not."""
    return a["scene"] == b["scene"]


def prepare_v8_reorient_scenes(
    *, shared_root: Path, mode: TaskMode, manifest_path: Path,
    heights_cm: tuple[int, ...] = RESET_RELEASE_HEIGHTS_CM,
) -> dict:
    """Create both BODex and sim-filter scenes using AutoDex's v8 generator.

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
    new_bodex = 0
    new_sim_filter = 0
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
                sim_path = _sim_filter_scene_path(root, mode, h_cm, i, j)
                if sim_path.exists():
                    sim_scene = json.loads(sim_path.read_text(encoding="utf-8"))
                    _validate_scene(
                        sim_scene, mode=mode, object_dir=object_dir,
                        h_cm=h_cm, i=i, j=j)
                    if not _same_scene_geometry(sim_scene, scene):
                        raise ValueError(
                            f"BODex and sim-filter reorient scenes disagree: {i}_{j}")
                    sim_status = "existing_verified"
                else:
                    pending.append((sim_path, scene))
                    sim_status = "new_sim_filter_scene"
                    new_sim_filter += 1
                if status == "new_v8_proposal_scene":
                    new_bodex += 1
                rows.append({
                    "h_cm": h_cm, "from_v8_pose": i, "to_v8_pose": j,
                    "bodex_scene": str(path), "bodex_status": status,
                    "sim_filter_scene": str(sim_path),
                    "sim_filter_status": sim_status,
                })
    for path, scene in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(scene, stream, indent=2, allow_nan=False)
            stream.write("\n")
    for row in rows:
        row["bodex_sha256"] = hashlib.sha256(
            Path(row["bodex_scene"]).read_bytes()).hexdigest()
        row["sim_filter_sha256"] = hashlib.sha256(
            Path(row["sim_filter_scene"]).read_bytes()).hexdigest()
    report = {
        "schema": "precision_insertion_v8_reorient_scene_manifest_v2",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "shared_root": str(root), "v8_pose_stems": list(ids),
        "directed_scene_count": len(rows),
        "new_scene_count": sum(
            row["bodex_status"].startswith("new_") or
            row["sim_filter_status"].startswith("new_") for row in rows),
        "new_bodex_scene_files": new_bodex,
        "new_sim_filter_scene_files": new_sim_filter,
        "scenes": rows,
        "scope": "BODex_and_sim_filter_reorient_proposal_scenes_only",
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


def audit_v8_reorient_assets(
    *, shared_root: Path, mode: TaskMode,
    candidate_root: Path | None = None,
    max_center_in_hand_drift_m: float | None = None,
    max_symmetry_axis_tilt_deg: float | None = None,
) -> dict:
    """Separate proposal, MuJoCo pass, and optional pose-fidelity eligibility."""
    # Import here to avoid a module cycle: the direct-v8 loader reuses this
    # module's scene contract, while this audit reuses its seed validator.
    from .reset_candidates import _candidate_arrays, fidelity_within_limits

    if (max_center_in_hand_drift_m is None) != (
            max_symmetry_axis_tilt_deg is None):
        raise ValueError("reset fidelity audit needs both drift and tilt limits")
    check_fidelity = max_center_in_hand_drift_m is not None
    if check_fidelity:
        drift = float(max_center_in_hand_drift_m)
        tilt = float(max_symmetry_axis_tilt_deg)
        if (not math.isfinite(drift) or drift <= 0 or
                not math.isfinite(tilt) or tilt <= 0):
            raise ValueError("positive finite reset fidelity limits are required")

    root, object_dir = _paths(shared_root, mode)
    candidate_base = (root / "AutoDex" / "candidates" / "inspire"
                      if candidate_root is None else
                      Path(candidate_root).expanduser().resolve())
    ids = _tabletop_ids(object_dir)
    rows = []
    for i in ids:
        for j in ids:
            if i == j:
                continue
            scenes = []
            missing_sim_scenes = []
            candidate_counts = {}
            reported_counts = {}
            rejected_ids = {}
            fidelity_eligible_counts = {}
            fidelity_rejected_ids = {}
            fidelity_rows = {}
            for h_cm in RESET_RELEASE_HEIGHTS_CM:
                scene = _scene_path(object_dir, h_cm, i, j)
                sim_scene = _sim_filter_scene_path(root, mode, h_cm, i, j)
                if scene.is_file():
                    data = json.loads(scene.read_text(encoding="utf-8"))
                    _validate_scene(data, mode=mode, object_dir=object_dir,
                                    h_cm=h_cm, i=i, j=j)
                    if sim_scene.is_file():
                        sim_data = json.loads(sim_scene.read_text(encoding="utf-8"))
                        _validate_scene(sim_data, mode=mode, object_dir=object_dir,
                                        h_cm=h_cm, i=i, j=j)
                        if not _same_scene_geometry(sim_data, data):
                            raise ValueError(
                                f"BODex and sim-filter scenes differ: {i}_{j}")
                        scenes.append(h_cm)
                    else:
                        missing_sim_scenes.append(h_cm)
                cell = (candidate_base / f"reset_{h_cm}" / mode.key_object /
                        f"reorient_{h_cm}" / f"{i}_{j}")
                candidates = []
                reported = []
                rejected = []
                eligible = []
                fidelity_rejected = []
                metrics = []
                if cell.is_dir():
                    for seed in cell.iterdir():
                        if not seed.is_dir() or not seed.name.isdigit():
                            continue
                        result_file = seed / "sim_eval.json"
                        if not result_file.is_file():
                            continue
                        try:
                            stable = json.loads(result_file.read_text(
                                encoding="utf-8")).get("success") is True
                        except (ValueError, TypeError, OSError):
                            continue
                        if not stable:
                            continue
                        reported.append(seed.name)
                        try:
                            arrays = _candidate_arrays(
                                seed, mode=mode, cell=f"{i}_{j}",
                                h_cm=h_cm, scenes=(scene, sim_scene))
                        except (FileNotFoundError, ValueError, TypeError,
                                KeyError, OSError, EOFError):
                            rejected.append(seed.name)
                            continue
                        candidates.append(seed.name)
                        if check_fidelity:
                            fidelity = arrays[5]
                            accepted = fidelity_within_limits(
                                fidelity,
                                max_center_in_hand_drift_m=drift,
                                max_symmetry_axis_tilt_deg=tilt)
                            (eligible if accepted else fidelity_rejected).append(
                                seed.name)
                            metrics.append({
                                "seed_id": seed.name,
                                "end_squeeze": fidelity["end_squeeze"],
                                "end_gravity": fidelity["end_gravity"],
                                "fidelity_gate_passed": accepted,
                            })
                candidate_counts[str(h_cm)] = len(candidates)
                reported_counts[str(h_cm)] = len(reported)
                rejected_ids[str(h_cm)] = sorted(rejected, key=int)
                if check_fidelity:
                    fidelity_eligible_counts[str(h_cm)] = len(eligible)
                    fidelity_rejected_ids[str(h_cm)] = sorted(
                        fidelity_rejected, key=int)
                    fidelity_rows[str(h_cm)] = sorted(
                        metrics, key=lambda row: int(row["seed_id"]))
            row = {
                "from_v8_pose": i, "to_v8_pose": j,
                "scene_heights_cm": scenes,
                "bodex_scene_missing_sim_filter_mirror_heights_cm":
                    missing_sim_scenes,
                "stable_reset_seed_counts_by_height_cm": candidate_counts,
                "reported_mujoco_pass_counts_by_height_cm": reported_counts,
                "reported_passes_rejected_by_loader_by_height_cm": rejected_ids,
                "has_any_stable_seed": any(candidate_counts.values()),
            }
            if check_fidelity:
                row.update({
                    "fidelity_eligible_seed_counts_by_height_cm":
                        fidelity_eligible_counts,
                    "fidelity_rejected_seed_ids_by_height_cm":
                        fidelity_rejected_ids,
                    "seed_fidelity_by_height_cm": fidelity_rows,
                    "has_any_fidelity_eligible_seed": any(
                        fidelity_eligible_counts.values()),
                })
            rows.append(row)
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
    report = {
        "schema": "precision_insertion_v8_reorient_asset_audit_v4",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "v8_pose_stems": list(ids), "directed_pairs": rows,
        "candidate_root": str(candidate_base),
        "candidate_root_is_canonical": candidate_root is None,
        "legacy_tabletop_tree_present": legacy.is_dir(),
        "stock_reset_runner_compatible": False,
        "stock_reset_runner_reason": (
            "new key has no verified legacy pose-index map and stock carried "
            "scene omits the fixed socket fixture"),
        "staging_manifests_not_runtime_candidates": staging,
        "robot_ready": False,
    }
    if check_fidelity:
        report["fidelity_limits"] = {
            "max_center_in_hand_drift_m": drift,
            "max_symmetry_axis_tilt_deg": tilt,
            "source": "caller_supplied_not_automatically_commissioned",
        }
    return report
