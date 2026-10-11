"""Add endpoint-screened stock v8 square grasps without replacing old ones.

This is an offline, append-only install into AutoDex's ordinary v8 candidate
layout. The untouched stock sim filter supplies grasp stability; exact-CAD
screen evidence supplies only the nominal aligned 20 mm endpoint. A large
numeric ID offset keeps the source seed and any historical v8 seed distinct.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .stock_tabletop import audit_tabletop_cell


SCHEMA = "precision_insertion_square_stock_tabletop_1000_v1"
SOURCE_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
                "bodex_info.npy", "coll_valid.npy", "sim_eval.json",
                "sim_traj.json")
SCENES = tuple(str(index) for index in range(5))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_source(raw: Path, passed: Path) -> None:
    if not all((raw / name).is_file() for name in SOURCE_FILES):
        raise FileNotFoundError(f"incomplete full-key v8 seed: {raw}")
    if any(_sha(raw / name) != _sha(passed / name)
           for name in SOURCE_FILES[:4]):
        raise ValueError(f"stock-passing arrays differ from BODex: {raw}")
    collision = np.load(raw / "coll_valid.npy", allow_pickle=False)
    evaluation = _json(raw / "sim_eval.json")
    trajectory = _json(raw / "sim_traj.json")
    if (collision.shape != () or collision.dtype != np.dtype(bool) or
            not bool(collision) or evaluation != {
                "success": True, "hand": "inspire", "version": "v8"}):
        raise ValueError(f"not a stock v8 MuJoCo pass: {raw}")
    phases = trajectory.get("phase")
    if (not isinstance(phases, list) or "squeeze" not in phases or
            "force_gravity" not in phases or
            len(phases) != len(trajectory.get("object_pose", [])) or
            len(phases) != len(trajectory.get("robot_qpos", []))):
        raise ValueError(f"incomplete MuJoCo squeeze/gravity trace: {raw}")


def _validate_endpoint(report: dict, *, mode: TaskMode, passed: Path,
                       paths: AssetPaths, clearance_m: float) -> None:
    if (report.get("schema") != "precision_insertion_endpoint_screen_v1" or
            report.get("endpoint_pass") is not True or
            report.get("hand_socket_clear_at_20mm") is not True or
            report.get("key_socket_fit", {}).get("colliding") is not False or
            Path(report.get("candidate_dir", "")).resolve() != passed.resolve() or
            report.get("mode") != {
                "family": "square", "gap_mm": mode.gap_mm,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object} or
            report.get("verification_depth_m") != mode.target_depth_m or
            report.get("xy_offset_socket_m") != [0.0, 0.0] or
            report.get("minimum_required_hand_clearance_m") != clearance_m):
        raise ValueError(f"endpoint report is not this centered task: {passed}")
    inputs = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": passed / "wrist_se3.npy",
        "pregrasp_pose": passed / "pregrasp_pose.npy",
        "grasp_pose": passed / "grasp_pose.npy",
    }
    hashes = report.get("input_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(inputs) or any(
            not path.is_file() or hashes[name] != _sha(path)
            for name, path in inputs.items()):
        raise ValueError(f"stale endpoint input: {passed}")


def promote_screened_batch(
    *, shared_root: Path, mode: TaskMode, screen_root: Path,
    candidate_id_offset: int, output_manifest: Path,
    install: bool = False,
) -> dict:
    """Preflight a full 1,000×5 scan; optionally append eligible seeds."""
    if mode.family != "square":
        raise ValueError("square key mode required")
    if (type(candidate_id_offset) is not int or
            candidate_id_offset < 1000 or candidate_id_offset % 1000):
        raise ValueError("candidate_id_offset must be a multiple of 1000")
    root = Path(shared_root).expanduser().resolve()
    screen = Path(screen_root).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace promotion manifest: {output}")
    paths = AssetPaths(root, mode)
    summary = _json(screen / "summary.json")
    mode_record = summary.get("modes", {}).get(mode.key_object)
    if (summary.get("schema") !=
            "precision_insertion_square_tabletop_1000_screen_v1" or
            summary.get("expected_proposals_per_scene") != 1000 or
            not isinstance(mode_record, dict) or
            mode_record.get("gap_mm") != mode.gap_mm or
            mode_record.get("socket_object") != mode.socket_object or
            mode_record.get("errors") or
            summary.get("robot_ready") is not False):
        raise ValueError("screen is not a complete error-free square v8 batch")
    raw_root = Path(summary["raw_root"]).resolve()
    pass_root = Path(summary["original_autodex_sim_candidate_root"]).resolve()
    clearance = float(summary["minimum_hand_clearance_m"])
    if clearance <= 0:
        raise ValueError("invalid endpoint clearance")
    eligible = set(mode_record.get("eligible_offline_grasp_ids", []))
    planned = []
    all_report_ids = set()
    for scene in SCENES:
        counts, passed = audit_tabletop_cell(
            raw_root=raw_root, pass_root=pass_root,
            key_object=mode.key_object, scene_id=scene,
            expected_seed_count=1000)
        if counts != mode_record.get("scene_counts", {}).get(scene):
            raise ValueError(f"stock filter changed since screening: {scene}")
        for seed_id, passed_seed in passed:
            source_id = f"table/{scene}/{seed_id}"
            all_report_ids.add(source_id)
            report_path = (screen / mode.key_object / "table" / scene /
                           f"{seed_id}.json")
            report = _json(report_path)
            if (report.get("endpoint_pass") is True) != (source_id in eligible):
                raise ValueError(f"screen summary disagrees with report: {source_id}")
            if source_id not in eligible:
                continue
            raw_seed = raw_root / mode.key_object / "table" / scene / seed_id
            _validate_source(raw_seed, passed_seed)
            _validate_endpoint(report, mode=mode, passed=passed_seed,
                               paths=paths, clearance_m=clearance)
            target_id = str(candidate_id_offset + int(seed_id))
            target = paths.candidate_dir / "table" / scene / target_id
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing to replace v8 grasp: {target}")
            planned.append((source_id, raw_seed, passed_seed, report_path,
                            target))
    if not eligible.issubset(all_report_ids) or len(planned) != len(eligible):
        raise ValueError("screen selected a grasp outside the complete pass pool")
    rows = []
    for source_id, raw_seed, passed_seed, report_path, target in planned:
        scene_file = paths.scene_dir / f"{raw_seed.parent.name}.json"
        scene_record = _json(scene_file)
        target_mesh = Path(scene_record["scene"]["mesh"]["target"]["file_path"])
        if target_mesh.resolve() != paths.key_planning_mesh.resolve():
            raise ValueError(f"v8 scene does not use the full key: {scene_file}")
        row = {"source_id": source_id, "target_id": target.name,
               "target": str(target), "source": str(raw_seed),
               "screen_report": str(report_path)}
        rows.append(row)
        if not install:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".precision-square-",
                                         dir=target.parent) as temp_dir:
            staged = Path(temp_dir) / "candidate"
            shutil.copytree(raw_seed, staged)
            shutil.copy2(report_path, staged / "nominal_endpoint_screen.json")
            validation = {
                "schema": SCHEMA, "status": "passed",
                "source_candidate": str(raw_seed),
                "source_sim_filter_pass": str(passed_seed),
                "source_candidate_id": source_id,
                "full_object_mesh_sha256": _sha(paths.key_planning_mesh),
                "raw_key_mesh_sha256": _sha(paths.raw_mesh(mode.key_object)),
                "source_scene_sha256": _sha(scene_file),
                "source_file_sha256": {
                    name: _sha(staged / name) for name in SOURCE_FILES},
                "nominal_endpoint_screen_sha256": _sha(
                    staged / "nominal_endpoint_screen.json"),
                "contact_policy_mode": "unrestricted_full_key",
                "physical_validation": False, "robot_ready": False,
            }
            with (staged / "simulation_validation.json").open(
                    "x", encoding="utf-8") as stream:
                json.dump(validation, stream, indent=2, allow_nan=False)
                stream.write("\n")
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"v8 grasp appeared during install: {target}")
            staged.rename(target)
    manifest = {
        "schema": "precision_insertion_square_stock_promotion_manifest_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "screen_summary": str(screen / "summary.json"),
        "screen_summary_sha256": _sha(screen / "summary.json"),
        "candidate_id_offset": candidate_id_offset,
        "installed": install, "rows": rows,
        "scope": "stock_v8_grasp_filter_plus_nominal_20mm_endpoint_only",
        "robot_ready": False,
    }
    if install:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest
