#!/usr/bin/env python3
"""Stage stock AutoDex square-key reset passes with full-key provenance.

The stock filter copies only four proposal arrays. This script checks every
raw result in one directed v8 cell, binds passing copies to their original
scene and MuJoCo trajectory, and writes a separate *offline* reset pool.
It neither selects a commissioned fidelity limit nor plans Franka motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from precision_insertion.config import select_mode
from precision_insertion.geometry import validate_se3
from precision_insertion.grasp_fidelity import trajectory_rigid_closure_audit
from precision_insertion.reset_candidates import (
    EVIDENCE_SCHEMA, GRASP_FILES, _candidate_arrays, _valid_scene_pair,
)


STOCK_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
               "bodex_info.npy")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_passes(
    *, shared_root: Path, gap_mm: float, raw_root: Path,
    stock_candidate_root: Path, cell: str, expected_seed_count: int,
    output_root: Path,
) -> dict:
    """Verify complete stock filtering before writing a new reset_12 tree."""
    mode = select_mode("square", gap_mm)
    shared = Path(shared_root).expanduser().resolve()
    raw = Path(raw_root).expanduser().resolve()
    stock = Path(stock_candidate_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace reset handoff: {output}")
    if (not isinstance(cell, str) or len(cell.split("_")) != 2 or
            any(not part.isdigit() or str(int(part)) != part
                for part in cell.split("_"))):
        raise ValueError("cell must be a canonical directed v8 pair such as 0_4")
    i, j = (int(part) for part in cell.split("_"))
    if i == j or type(expected_seed_count) is not int or expected_seed_count < 1:
        raise ValueError("different tabletop poses and positive seed count required")
    object_dir = shared / "object_processing" / mode.key_object
    scenes = _valid_scene_pair(shared, object_dir, mode, 12, i, j)
    info_path = object_dir / "processed_data/info/simplified.json"
    mesh_path = object_dir / "processed_data/mesh/simplified.obj"
    center = np.asarray(json.loads(info_path.read_text(encoding="utf-8"))[
        "gravity_center"], dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("full-key local center is invalid")
    raw_cell = raw / mode.key_object / "reorient_12" / cell
    stock_cell = stock / mode.key_object / "reorient_12" / cell
    expected = {str(seed) for seed in range(expected_seed_count)}
    found = {path.name for path in raw_cell.iterdir() if path.is_dir()}
    if found != expected:
        raise ValueError(f"incomplete BODex cell: {len(found)}/{len(expected)}")
    counts = {
        "raw": expected_seed_count, "scene_clear": 0,
        "squeeze_contact": 0, "mujoco_stable": 0,
        "scene_collision": 0, "no_object_contact": 0,
        "mujoco_unstable": 0,
    }
    stable = []
    for seed_id in range(expected_seed_count):
        seed = raw_cell / str(seed_id)
        collision = np.load(seed / "coll_valid.npy", allow_pickle=False)
        if collision.shape != () or collision.dtype != np.dtype(bool):
            raise ValueError(f"invalid scene collision flag: {seed}")
        result = json.loads((seed / "sim_eval.json").read_text(
            encoding="utf-8"))
        if (result.get("hand") != "inspire" or
                result.get("version") != "v8" or
                type(result.get("success")) is not bool):
            raise ValueError(f"wrong stock filter contract: {seed}")
        reason = result.get("reason")
        if not bool(collision):
            if result["success"] or reason != "scene_collision":
                raise ValueError(f"inconsistent scene collision: {seed}")
            counts["scene_collision"] += 1
            continue
        counts["scene_clear"] += 1
        if reason == "no_object_contact":
            if result["success"]:
                raise ValueError(f"contact-failed seed marked stable: {seed}")
            counts["no_object_contact"] += 1
            continue
        if reason is not None:
            raise ValueError(f"unknown filter reason: {seed}")
        counts["squeeze_contact"] += 1
        if not result["success"]:
            counts["mujoco_unstable"] += 1
            continue
        counts["mujoco_stable"] += 1
        copied = stock_cell / str(seed_id)
        for name in STOCK_FILES:
            if _sha(seed / name) != _sha(copied / name):
                raise ValueError(f"stock candidate differs from raw: {copied / name}")
        validate_se3(np.load(seed / "wrist_se3.npy", allow_pickle=False),
                     name=f"reset {cell}/{seed_id} T_key_hand")
        for name in ("pregrasp_pose.npy", "grasp_pose.npy"):
            q = np.load(seed / name, allow_pickle=False)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                raise ValueError(f"invalid Inspire joints: {seed / name}")
        fidelity = trajectory_rigid_closure_audit(
            json.loads((seed / "sim_traj.json").read_text(encoding="utf-8")),
            key_center_local_m=center)
        stable.append((seed_id, seed, copied, fidelity))
    stock_found = ({path.name for path in stock_cell.iterdir() if path.is_dir()}
                   if stock_cell.is_dir() else set())
    if stock_found != {str(seed_id) for seed_id, *_ in stable}:
        raise ValueError("stock passing pool disagrees with all raw MuJoCo results")

    scene_hashes = {"bodex": _sha(scenes[0]),
                    "sim_filter": _sha(scenes[1])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    rows = []
    for seed_id, seed, copied, fidelity in stable:
        target = (output / "reset_12" / mode.key_object / "reorient_12" /
                  cell / str(seed_id))
        target.mkdir(parents=True, exist_ok=False)
        for name in STOCK_FILES:
            shutil.copy2(copied / name, target / name)
        for name in ("sim_eval.json", "sim_traj.json"):
            shutil.copy2(seed / name, target / name)
        evidence = {
            "schema": EVIDENCE_SCHEMA, "full_key_object": mode.key_object,
            "height_cm": 12, "cell": cell, "seed_id": str(seed_id),
            "candidate_sha256": {name: _sha(target / name)
                                 for name in GRASP_FILES},
            "scene_sha256": scene_hashes,
            "key_mesh_sha256": _sha(mesh_path),
            "key_info_sha256": _sha(info_path),
            "key_center_local_m": center.tolist(),
            "post_squeeze_fidelity": fidelity,
            "raw_proposal": str(seed), "stock_pass": str(copied),
            "robot_ready": False,
        }
        with (target / "source_evidence.json").open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write("\n")
        _candidate_arrays(target, mode=mode, cell=cell, h_cm=12,
                          scenes=scenes)
        rows.append({"seed_id": seed_id, "candidate_dir": str(target),
                     "post_squeeze_fidelity": fidelity})
    report = {
        "schema": "precision_insertion_square_reorient_stock_handoff_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "height_cm": 12, "cell": cell, "expected_seed_count": expected_seed_count,
        "raw_root": str(raw), "stock_candidate_root": str(stock),
        "candidate_root": str(output / "reset_12"),
        "scene_sha256": scene_hashes, "key_mesh_sha256": _sha(mesh_path),
        "key_info_sha256": _sha(info_path), "counts": counts,
        "stable_seeds": rows,
        "fidelity_threshold_commissioned": False,
        "scope": "stock_full_key_grasp_filter_only_not_repose_plan",
        "robot_ready": False,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stock-candidate-root", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--expected-seed-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = stage_passes(**vars(args))
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError, OSError) as exc:
        parser.exit(2, f"square reset staging rejected: {exc}\n")
    print(json.dumps({"counts": report["counts"],
                      "candidate_root": report["candidate_root"],
                      "robot_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
