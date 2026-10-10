#!/usr/bin/env python3
"""Install only stock-filtered, 20 mm-screened cylinder grasps into v8.

This is offline candidate staging, not physical grasp validation. The current
grasp relation is the *nominal* BODex relation; MuJoCo squeeze may move the key
relative to the hand. No measured grasp, Franka path or insertion is claimed.
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
from precision_insertion.grasp_fidelity import trajectory_closure_audit
from screen_cylinder_achieved_endpoints import _load_audited_rows


KEY = "precision_key_cylinder_r15_h80"
STOCK_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
               "bodex_info.npy")
RAW_FILES = ("sim_eval.json", "sim_traj.json", "coll_valid.npy")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def select_ids(nominal: dict, achieved: dict) -> list[str]:
    """Require stock stability plus nominal and achieved endpoint clearance.

    The same key candidate pool serves all six sockets, so installation uses
    their intersection. This is intentionally stricter than a per-gap pool.
    """
    if (nominal.get("schema") !=
            "precision_insertion_cylinder_1000_per_scene_screen_v1" or
            nominal.get("total_raw_proposals") != 2000 or
            nominal.get("physical_key_object") != KEY or
            achieved.get("schema") !=
            "precision_insertion_cylinder_achieved_endpoint_comparison_v1" or
            achieved.get("socket_family_complete") is not True or
            achieved.get("candidate_count") != sum(
                row["mujoco_stable"] for row in nominal["scene_counts"].values())):
        raise ValueError("incomplete or mismatched cylinder offline screens")
    expected = {f"{int(gap):02d}mm" for gap in CYLINDER_RADIAL_GAPS_MM}
    if (set(nominal.get("socket_gaps", {})) != expected or
            set(achieved.get("socket_gaps", {})) != expected):
        raise ValueError("offline screens must cover every socket gap")
    per_gap = []
    for label in sorted(expected):
        nominal_ids = nominal["socket_gaps"][label]["eligible_offline_grasp_ids"]
        achieved_ids = achieved["socket_gaps"][label]["both_achieved_states_clear_ids"]
        if (len(nominal_ids) != len(set(nominal_ids)) or
                len(achieved_ids) != len(set(achieved_ids)) or
                achieved["socket_gaps"][label]["nominal_initial_pose_endpoint_pass_count"]
                != len(nominal_ids)):
            raise ValueError(f"duplicate/inconsistent endpoint IDs for {label}")
        per_gap.append(set(nominal_ids) & set(achieved_ids))
    return sorted(set.intersection(*per_gap),
                  key=lambda value: tuple(int(part) if part.isdigit() else part
                                          for part in value.split("/")))


def prepare(*, shared_root: Path, achieved_summary_path: Path) -> tuple[dict, list[dict]]:
    """Validate full source lineage before considering any destination write."""
    root = Path(shared_root).expanduser().resolve()
    source_path = Path(achieved_summary_path).expanduser().resolve()
    achieved = _json(source_path)
    audit_path = Path(achieved["source_fidelity_audit"]).resolve()
    nominal_path = Path(achieved["source_nominal_summary"]).resolve()
    if (_hash(audit_path) != achieved["source_fidelity_audit_sha256"] or
            _hash(nominal_path) != achieved["source_nominal_summary_sha256"]):
        raise ValueError("achieved endpoint summary has changed sources")
    audit, nominal, rows = _load_audited_rows(audit_path)
    if (Path(audit["source_summary"]).resolve() != nominal_path or
            nominal != _json(nominal_path)):
        raise ValueError("fidelity and achieved summaries disagree")
    stage = Path(nominal["stage_root"]).resolve()
    pool = Path(nominal["original_autodex_sim_candidate_root"]).resolve()
    stage_manifest_path = stage / "stage_manifest.json"
    stage_manifest = _json(stage_manifest_path)
    if (stage_manifest.get("schema") !=
            "precision_insertion_proxy_to_full_key_stage_v1" or
            stage_manifest.get("full_key_object") != KEY or
            stage_manifest.get("seed_count_per_scene") != 1000 or
            stage_manifest.get("seed_count_total") != 2000 or
            Path(stage_manifest.get("stage_root", "")).resolve() != stage):
        raise ValueError("not the complete 1,000-per-scene full-key stage")
    scenes = {str(row["scene_id"]): row for row in stage_manifest["scenes"]}
    if set(scenes) != {"0", "1"}:
        raise ValueError("expected both cylinder tabletop scenes")
    for scene_id, item in scenes.items():
        scene = root / "AutoDex/scene/inspire" / KEY / "table" / f"{scene_id}.json"
        if (Path(item["full_key_scene"]).resolve() != scene or
                _hash(scene) != item["full_key_scene_sha256"]):
            raise ValueError(f"v8 scene changed since proposal stage: {scene}")
    mode = select_mode("cylinder", 20)
    paths = AssetPaths(root, mode)
    if (_hash(paths.raw_mesh(KEY)) != audit["key_mesh_sha256"] or
            _hash(paths.robot_urdf) != audit["robot_urdf_sha256"]):
        raise ValueError("key or robot mesh changed since fidelity audit")
    selected = select_ids(nominal, achieved)
    if not selected:
        raise ValueError("no candidates pass the complete socket family")
    by_id = {row["id"]: row for row in rows}
    if not set(selected) <= set(by_id):
        raise ValueError("selected IDs missing from complete MuJoCo audit")
    prepared = []
    for identifier in selected:
        _, scene_id, seed = identifier.split("/")
        raw = stage / KEY / "table" / scene_id / seed
        stock = pool / KEY / "table" / scene_id / seed
        target = paths.candidate_dir / "table" / scene_id / seed
        if target.exists():
            raise FileExistsError(f"refusing to overwrite v8 grasp: {target}")
        for name in STOCK_FILES:
            if _hash(raw / name) != _hash(stock / name):
                raise ValueError(f"stock/raw grasp files differ: {identifier}/{name}")
        collision = np.load(raw / "coll_valid.npy", allow_pickle=False)
        evaluation = _json(raw / "sim_eval.json")
        if (collision.shape != () or collision.dtype != np.dtype(bool) or
                not bool(collision) or evaluation.get("success") is not True or
                evaluation.get("hand") != "inspire" or
                evaluation.get("version") != "v8"):
            raise ValueError(f"stock full-key simulation not passed: {identifier}")
        trajectory = _json(raw / "sim_traj.json")
        fidelity = trajectory_closure_audit(trajectory, key_height_m=0.08)
        if fidelity != by_id[identifier]["fidelity"]["closure"]:
            raise ValueError(f"MuJoCo grasp relation changed: {identifier}")
        endpoint_hashes = {}
        for gap in CYLINDER_RADIAL_GAPS_MM:
            label = f"{int(gap):02d}mm"
            report_path = (source_path.parent / f"gap_{label}" / "table" /
                           scene_id / f"{seed}.json")
            report = _json(report_path)
            if (report.get("id") != identifier or
                    report.get("nominal_initial_pose_endpoint_pass") is not True or
                    report.get("both_achieved_states_endpoint_pass") is not True or
                    any(report["achieved_states"][state]["endpoint_pass"] is not True
                        for state in ("end_squeeze", "end_gravity"))):
                raise ValueError(f"achieved endpoint evidence failed: {identifier}/{label}")
            endpoint_hashes[label] = _hash(report_path)
        prepared.append({"id": identifier, "raw": raw, "stock": stock,
                         "target": target, "fidelity": fidelity,
                         "endpoint_report_sha256": endpoint_hashes})
    report = {
        "schema": "precision_insertion_cylinder_tabletop_v8_promotion_v1",
        "status": "offline_simulated_candidates_not_physical",
        "shared_root": str(root), "candidate_root": str(paths.candidate_dir),
        "selected_ids": selected, "count": len(selected),
        "source_achieved_summary": str(source_path),
        "source_achieved_summary_sha256": _hash(source_path),
        "stage_manifest_sha256": _hash(stage_manifest_path),
        "key_planning_mesh_sha256": _hash(paths.key_planning_mesh),
        "scope": "stock_v8_full_key_grasp_plus_nominal_and_achieved_20mm_endpoints",
        "not_validated": ["physical key-in-hand relation/repeatability",
                          "Franka transfer and axial path",
                          "guarded contact insertion and physical task success"],
        "robot_ready": False,
    }
    return report, prepared


def install(report: dict, prepared: list[dict], manifest_path: Path) -> dict:
    """Install each grasp atomically; never replace an existing candidate."""
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists():
        raise FileExistsError(f"promotion manifest exists: {manifest}")
    root = Path(report["candidate_root"])
    planning_hash = report["key_planning_mesh_sha256"]
    for row in prepared:
        target = row["target"]
        if target.exists() or target.with_name(target.name + ".incomplete").exists():
            raise FileExistsError(f"v8 candidate or partial staging exists: {target}")
    for row in prepared:
        target = row["target"]
        work = target.with_name(target.name + ".incomplete")
        work.mkdir(parents=True, exist_ok=False)
        for name in STOCK_FILES:
            shutil.copy2(row["stock"] / name, work / name)
        for name in RAW_FILES:
            shutil.copy2(row["raw"] / name, work / name)
        validation = {
            "schema": "precision_insertion_cylinder_tabletop_simulation_v1",
            "status": "passed", "physical_validation": False,
            "full_object_mesh_sha256": planning_hash,
            "source_achieved_summary_sha256": report["source_achieved_summary_sha256"],
            "source_candidate": str(row["stock"]),
            "source_file_sha256": {name: _hash(row["raw"] / name)
                                   for name in STOCK_FILES + RAW_FILES},
            "achieved_endpoint_report_sha256": row["endpoint_report_sha256"],
            "post_squeeze_fidelity_diagnostic": row["fidelity"],
            "fidelity_acceptance_threshold_commissioned": False,
            "warning": "Nominal BODex key-in-hand pose is not an achieved or physical pose",
            "robot_ready": False,
        }
        _write_new(work / "simulation_validation.json", validation)
        work.rename(target)
    _write_new(manifest, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--achieved-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="exclusive output JSON for --install")
    parser.add_argument("--install", action="store_true",
                        help="write validated candidates into the canonical v8 tree")
    args = parser.parse_args()
    report, prepared = prepare(shared_root=args.shared_root,
                               achieved_summary_path=args.achieved_summary)
    if args.install:
        if args.manifest is None:
            parser.error("--install requires --manifest")
        install(report, prepared, args.manifest)
    print(json.dumps({"installed": args.install, "manifest": str(args.manifest)
                      if args.install else None, **report}, indent=2))


if __name__ == "__main__":
    main()
