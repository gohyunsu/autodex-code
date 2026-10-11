#!/usr/bin/env python3
"""Append full-length-proposal grasps that passed stock v8 and 20 mm CAD.

Each socket gap is screened independently. A candidate passing any selected
gap is installed once under a new numeric v8 ID; the live per-gap catalog
will screen that combined pool again. No achieved-grasp fidelity threshold,
cross-gap intersection, arm path or physical success is inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.stock_tabletop import audit_tabletop_cell


KEY = "precision_key_cylinder_r15_h80"
PROPOSAL = f"{KEY}_full_length_proposal"
FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
         "bodex_info.npy", "sim_eval.json", "sim_traj.json", "coll_valid.npy")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def prepare(*, shared_root: Path, screen_root: Path,
            candidate_id_offset: int, manifest_path: Path) -> tuple[dict, list[dict]]:
    """Verify complete immutable source lineage before any runtime write."""
    if (type(candidate_id_offset) is not int or
            candidate_id_offset < 1000 or candidate_id_offset % 1000):
        raise ValueError("candidate ID offset must be a positive multiple of 1000")
    root = Path(shared_root).expanduser().resolve()
    screen = Path(screen_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists():
        raise FileExistsError(f"promotion manifest already exists: {manifest}")
    summary = _load(screen / "summary.json")
    stage = Path(summary["stage_root"]).resolve()
    stable_root = Path(summary["original_autodex_sim_candidate_root"]).resolve()
    stage_manifest = _load(stage / "stage_manifest.json")
    expected_gaps = {f"{int(gap):02d}mm" for gap in CYLINDER_RADIAL_GAPS_MM}
    if (summary.get("schema") !=
            "precision_insertion_cylinder_1000_per_scene_screen_v1" or
            summary.get("physical_key_object") != KEY or
            summary.get("total_raw_proposals") != 2000 or
            summary.get("minimum_hand_clearance_m") != 0 or
            set(summary.get("socket_gaps", {})) != expected_gaps or
            stage_manifest.get("schema") !=
            "precision_insertion_proxy_to_full_key_stage_v1" or
            stage_manifest.get("proxy_object") != PROPOSAL or
            stage_manifest.get("proposal_contact_z_range_m") != [0.0, 0.08] or
            stage_manifest.get("proposal_contact_region_restricted") is not False or
            stage_manifest.get("seed_count_per_scene") != 1000 or
            stage_manifest.get("seed_count_total") != 2000 or
            Path(stage_manifest.get("stage_root", "")).resolve() != stage or
            Path(summary.get("stage_root", "")).resolve() != stage or
            Path(summary.get("raw_proxy_object", "")) != Path(PROPOSAL)):
        raise ValueError("screen is not the complete unrestricted 1000/scene run")
    if any(row.get("errors") for row in summary["socket_gaps"].values()):
        raise ValueError("endpoint screen contains an error")
    if set(summary.get("scene_counts", {})) != {"0", "1"}:
        raise ValueError("both cylinder tabletop scenes must be present")
    paths = AssetPaths(root, select_mode("cylinder", 20))
    key_mesh_hash = _sha(paths.key_planning_mesh)
    rows = []
    for scene_id in ("0", "1"):
        scene = (root / "AutoDex" / "scene" / "inspire" / KEY /
                 "table" / f"{scene_id}.json")
        matched = [item for item in stage_manifest["scenes"]
                   if item["scene_id"] == scene_id]
        if (len(matched) != 1 or
                matched[0]["full_key_scene_sha256"] != _sha(scene)):
            raise ValueError("physical-key tabletop scene changed")
        counts, stable = audit_tabletop_cell(
            raw_root=stage, pass_root=stable_root,
            key_object=KEY, scene_id=scene_id, expected_seed_count=1000)
        if counts != summary["scene_counts"][scene_id]:
            raise ValueError("stock v8 results differ from screened summary")
        for seed_id, stock in stable:
            source_id = f"table/{scene_id}/{seed_id}"
            source = stage / KEY / "table" / scene_id / seed_id
            eligible = []
            report_hashes = {}
            for gap in CYLINDER_RADIAL_GAPS_MM:
                label = f"{int(gap):02d}mm"
                mode = select_mode("cylinder", gap)
                report_path = (screen / f"gap_{label}" / "table" /
                               scene_id / f"{seed_id}.json")
                report = _load(report_path)
                accepted = source_id in summary["socket_gaps"][label][
                    "eligible_offline_grasp_ids"]
                if (report.get("endpoint_pass") is True) != accepted:
                    raise ValueError(f"endpoint summary/report disagreement: {source_id}/{label}")
                if not accepted:
                    continue
                assets = AssetPaths(root, mode)
                inputs = {
                    "key_mesh": assets.raw_mesh(KEY),
                    "socket_mesh": assets.socket_collision_mesh,
                    "task_geometry": assets.task_geometry,
                    "robot_urdf": assets.robot_urdf,
                    "wrist_se3": stock / "wrist_se3.npy",
                    "pregrasp_pose": stock / "pregrasp_pose.npy",
                    "grasp_pose": stock / "grasp_pose.npy",
                }
                if (report.get("schema") !=
                        "precision_insertion_endpoint_screen_v1" or
                        report.get("candidate_dir") != str(stock) or
                        report.get("mode") != {
                            "family": "cylinder", "gap_mm": gap,
                            "key_object": KEY,
                            "socket_object": mode.socket_object} or
                        report.get("verification_depth_m") != .02 or
                        report.get("xy_offset_socket_m") != [0.0, 0.0] or
                        report.get("minimum_required_hand_clearance_m") != 0 or
                        report.get("key_socket_fit", {}).get("colliding") is not False or
                        report.get("hand_socket_clear_at_20mm") is not True or
                        report.get("input_sha256") != {
                            name: _sha(path) for name, path in inputs.items()}):
                    raise ValueError(f"stale 20 mm endpoint evidence: {source_id}/{label}")
                eligible.append(label)
                report_hashes[label] = _sha(report_path)
            if not eligible:
                continue
            destination = (paths.candidate_dir / "table" / scene_id /
                           str(candidate_id_offset + int(seed_id)))
            if destination.exists() or destination.with_name(
                    destination.name + ".incomplete").exists():
                raise FileExistsError(f"new v8 ID is occupied: {destination}")
            if any(_sha(source / name) != _sha(stock / name)
                   for name in FILES[:4]):
                raise ValueError(f"stock candidate differs from BODex source: {source_id}")
            collision = np.load(source / "coll_valid.npy", allow_pickle=False)
            evaluation = _load(source / "sim_eval.json")
            if (collision.shape != () or not bool(collision) or
                    evaluation != {"success": True, "hand": "inspire",
                                   "version": "v8"}):
                raise ValueError(f"not a full-key stock MuJoCo pass: {source_id}")
            rows.append({
                "source_id": source_id, "source": source,
                "destination": destination, "eligible_gaps": eligible,
                "endpoint_report_sha256": report_hashes,
                "file_sha256": {name: _sha(source / name) for name in FILES},
            })
    report = {
        "schema": "precision_insertion_cylinder_full_length_promotion_v1",
        "status": "offline_candidates_not_physical",
        "shared_root": str(root), "screen_root": str(screen),
        "screen_summary_sha256": _sha(screen / "summary.json"),
        "stage_manifest_sha256": _sha(stage / "stage_manifest.json"),
        "candidate_id_offset": candidate_id_offset,
        "selected_count": len(rows),
        "selected": [{
            "source_id": row["source_id"],
            "installed_id": "/".join(
                row["destination"].relative_to(paths.candidate_dir).parts),
            "eligible_socket_gaps": row["eligible_gaps"],
        } for row in rows],
        "full_key_planning_mesh_sha256": key_mesh_hash,
        "scope": "stock_v8_full_key_grasp_plus_per_socket_nominal_20mm_noncollision",
        "robot_ready": False,
    }
    return report, rows


def install(report: dict, rows: list[dict], manifest_path: Path) -> dict:
    """Append only source-verified grasps; never replace historical IDs."""
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists():
        raise FileExistsError(f"promotion manifest already exists: {manifest}")
    for row in rows:
        target = row["destination"]
        if target.exists() or target.with_name(target.name + ".incomplete").exists():
            raise FileExistsError(f"new v8 ID is occupied: {target}")
        if (set(row["file_sha256"]) != set(FILES) or
                any(not (row["source"] / name).is_file() or
                    _sha(row["source"] / name) != row["file_sha256"][name]
                    for name in FILES)):
            raise ValueError(f"verified proposal source changed: {row['source']}")
    for row in rows:
        target = row["destination"]
        working = target.with_name(target.name + ".incomplete")
        working.mkdir(parents=True, exist_ok=False)
        for name in FILES:
            shutil.copy2(row["source"] / name, working / name)
            if _sha(working / name) != row["file_sha256"][name]:
                raise ValueError(f"copied proposal differs from source: {working / name}")
        validation = {
            "schema": "precision_insertion_cylinder_tabletop_simulation_v1",
            "status": "passed", "physical_validation": False,
            "full_object_mesh_sha256": report["full_key_planning_mesh_sha256"],
            "source_file_sha256": row["file_sha256"],
            "source_candidate": str(row["source"]),
            "source_screen_summary_sha256": report["screen_summary_sha256"],
            "nominal_20mm_endpoint_sha256_by_gap": row[
                "endpoint_report_sha256"],
            "eligible_socket_gaps": row["eligible_gaps"],
            "fidelity_acceptance_threshold_commissioned": False,
            "warning": "Nominal BODex key/hand relation is not a physical observation",
            "robot_ready": False,
        }
        with (working / "simulation_validation.json").open(
                "x", encoding="utf-8") as stream:
            json.dump(validation, stream, indent=2, sort_keys=True)
            stream.write("\n")
        working.rename(target)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--candidate-id-offset", type=int, default=20000)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="append passing candidates after source-only preflight")
    args = parser.parse_args()
    report, rows = prepare(
        shared_root=args.shared_root, screen_root=args.screen_root,
        candidate_id_offset=args.candidate_id_offset,
        manifest_path=args.manifest)
    if args.install:
        install(report, rows, args.manifest)
    print(json.dumps({"installed": args.install, "selected_count": len(rows),
                      "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
