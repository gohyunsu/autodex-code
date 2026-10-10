#!/usr/bin/env python3
"""Bind stock AutoDex full-key reset passes to v8 candidate evidence.

The stock sim filter copies only grasp arrays into its output directory. This
script verifies that those copies match the evaluated raw proposals, then
adds the original MuJoCo result and immutable hashes to the v8 reset tree.
It does not establish a robot-executable reorientation trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from precision_insertion.config import select_mode
from precision_insertion.grasp_fidelity import trajectory_closure_audit
from precision_insertion.reset_candidates import (
    EVIDENCE_SCHEMA, GRASP_FILES, _sha256, _valid_scene_pair,
)
from stage_cylinder_reorient_proposals import KEY, PAIRS


STOCK_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
               "bodex_info.npy")


def promote(
    *, shared_root: Path, stage_root: Path, stock_candidate_root: Path,
    audit_path: Path, output_manifest: Path,
    output_candidate_root: Path | None = None,
) -> dict:
    """Copy only fully validated stock passes; refuse overwrite everywhere."""
    shared = Path(shared_root).expanduser().resolve()
    stage = Path(stage_root).expanduser().resolve()
    stock = Path(stock_candidate_root).expanduser().resolve()
    report_path = Path(audit_path).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"reset promotion manifest exists: {output}")
    audit_bytes = report_path.read_bytes()
    audit = json.loads(audit_bytes)
    stage_manifest_path = stage / "stage_manifest.json"
    stage_manifest_bytes = stage_manifest_path.read_bytes()
    stage_manifest = json.loads(stage_manifest_bytes)
    if (audit.get("schema") != "precision_insertion_cylinder_reorient_pilot_audit_v1" or
            audit.get("stage_root") != str(stage) or
            audit.get("source_stage_manifest_sha256") !=
            hashlib.sha256(stage_manifest_bytes).hexdigest() or
            stage_manifest.get("full_key_object") != KEY or
            stage_manifest.get("stage_root") != str(stage)):
        raise ValueError("reset audit and proposal stage are not bound")
    mode = select_mode("cylinder", 20)
    object_dir = shared / "object_processing" / KEY
    key_mesh = object_dir / "processed_data/mesh/simplified.obj"
    key_info = object_dir / "processed_data/info/simplified.json"
    key_height_m = float(json.loads(key_info.read_text(encoding="utf-8"))["obb"][2])
    if not math.isfinite(key_height_m) or key_height_m <= 0:
        raise ValueError("invalid full-key OBB height")
    key_mesh_sha = _sha256(key_mesh)
    key_info_sha = _sha256(key_info)
    canonical_base = shared / "AutoDex/candidates/inspire/reset_12"
    base = (canonical_base if output_candidate_root is None else
            Path(output_candidate_root).expanduser().resolve())
    destination = base / KEY / "reorient_12"
    by_cell = {row["cell"]: row for row in audit["cells"]}
    staged_scenes = {row["cell"]: row for row in stage_manifest["scenes"]}
    if set(by_cell) != set(PAIRS) or set(staged_scenes) != set(PAIRS):
        raise ValueError("reset audit does not cover both directed v8 cells")

    selected = []
    for cell in PAIRS:
        i, j = map(int, cell.split("_"))
        scenes = _valid_scene_pair(shared, object_dir, mode, 12, i, j)
        scene_hashes = {"bodex": _sha256(scenes[0]),
                        "sim_filter": _sha256(scenes[1])}
        staged_scene = staged_scenes[cell]
        proxy_scene = Path(staged_scene["proxy_scene"]).resolve()
        if (Path(staged_scene["full_key_scene"]).resolve() != scenes[1] or
                staged_scene["full_key_sha256"] != scene_hashes["sim_filter"] or
                staged_scene["proxy_sha256"] != _sha256(proxy_scene)):
            raise ValueError(f"reset proposal scene changed since staging: {cell}")
        row = by_cell[cell]
        claimed = {str(value) for value in row["mujoco_stable_seed_ids"]}
        expected = stage_manifest["expected_per_cell"]
        if type(expected) is not int or expected < 1 or len(claimed) != len(
                row["mujoco_stable_seed_ids"]):
            raise ValueError(f"invalid audited stable IDs: {cell}")
        actual = set()
        for seed_id in range(expected):
            seed = stage / KEY / "reorient_12" / cell / str(seed_id)
            result = json.loads((seed / "sim_eval.json").read_text())
            collision = np.load(seed / "coll_valid.npy", allow_pickle=False)
            if collision.shape != () or collision.dtype != np.dtype(bool):
                raise ValueError(f"invalid collision flag: {seed}")
            if result.get("success") is True:
                if (not bool(collision) or result.get("hand") != "inspire" or
                        result.get("version") != "v8" or
                        result.get("reason") is not None):
                    raise ValueError(f"inconsistent MuJoCo pass: {seed}")
                actual.add(str(seed_id))
        if actual != claimed:
            raise ValueError(f"reset audit is stale for cell {cell}")
        stock_cell = stock / KEY / "reorient_12" / cell
        found_stock = ({path.name for path in stock_cell.iterdir() if path.is_dir()}
                       if stock_cell.is_dir() else set())
        if found_stock != actual:
            raise ValueError(f"stock output does not match full-key passes: {cell}")
        for seed_id in sorted(actual, key=int):
            raw_seed = stage / KEY / "reorient_12" / cell / seed_id
            stock_seed = stock_cell / seed_id
            target = destination / cell / seed_id
            if target.exists():
                raise FileExistsError(f"refusing to overwrite reset candidate: {target}")
            for name in STOCK_FILES:
                if _sha256(raw_seed / name) != _sha256(stock_seed / name):
                    raise ValueError(f"stock/raw reset proposal mismatch: {stock_seed / name}")
            fidelity = trajectory_closure_audit(
                json.loads((raw_seed / "sim_traj.json").read_text(encoding="utf-8")),
                key_height_m=key_height_m)
            selected.append((cell, seed_id, raw_seed, stock_seed, target,
                             scene_hashes, fidelity))

    for cell, seed_id, raw_seed, stock_seed, target, scene_hashes, fidelity in selected:
        target.mkdir(parents=True, exist_ok=False)
        for name in STOCK_FILES:
            shutil.copy2(stock_seed / name, target / name)
        shutil.copy2(raw_seed / "sim_eval.json", target / "sim_eval.json")
        shutil.copy2(raw_seed / "sim_traj.json", target / "sim_traj.json")
        evidence = {
            "schema": EVIDENCE_SCHEMA, "full_key_object": KEY,
            "height_cm": 12, "cell": cell, "seed_id": seed_id,
            "candidate_sha256": {name: _sha256(target / name)
                                 for name in GRASP_FILES},
            "scene_sha256": scene_hashes,
            "key_mesh_sha256": key_mesh_sha,
            "key_info_sha256": key_info_sha,
            "key_height_m": key_height_m,
            "post_squeeze_fidelity": fidelity,
            "raw_proposal": str(raw_seed), "stock_pass": str(stock_seed),
            "source_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "robot_ready": False,
        }
        with (target / "source_evidence.json").open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
    manifest = {
        "schema": "precision_insertion_v8_reset_promotion_v1",
        "full_key_object": KEY, "height_cm": 12,
        "source_audit": str(report_path),
        "source_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "candidate_output_root": str(destination),
        "canonical_candidate_root": str(canonical_base / KEY / "reorient_12"),
        "installed_in_canonical_reset_tree": (
            output_candidate_root is None and bool(selected)),
        "promoted_count": len(selected),
        "promoted_seed_ids_by_cell": {
            cell: sorted(by_cell[cell]["mujoco_stable_seed_ids"])
            for cell in PAIRS},
        "scope": "full_key_MuJoCo_reset_grasps_not_robot_paths",
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--stock-candidate-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-candidate-root", type=Path,
        help="optional local reset_12 root for handoff when canonical NAS is read-only",
    )
    args = parser.parse_args()
    try:
        result = promote(
            shared_root=args.shared_root, stage_root=args.stage_root,
            stock_candidate_root=args.stock_candidate_root,
            audit_path=args.audit, output_manifest=args.manifest,
            output_candidate_root=args.output_candidate_root)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError, OSError) as exc:
        parser.exit(2, f"reset promotion rejected: {exc}\n")
    print(json.dumps({"promoted_count": result["promoted_count"],
                      "robot_ready": result["robot_ready"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
