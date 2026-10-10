#!/usr/bin/env python3
"""Inventory insertion evidence for every key tabletop pose and reset cell.

This is an offline, read-only scan of input assets.  It writes two audit JSONs,
never promotes a grasp or reset seed into an executable runtime candidate pool,
and never interprets a missing/failed finite sample as proof of impossibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from autodex.tasks.precision_insertion import mode_config  # noqa: E402


SQUARE_GAPS = (0.3, 0.5, 1.0, 1.5)
CYLINDER_GAPS = (1, 3, 5, 10, 15, 20)
RESET_HEIGHTS_CM = (0, 4, 8, 12)
GRASP_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy")
LEVELS = (
    "geometry_only", "sampled_insertion_geometry", "grasp_sim_pass",
    "full_task_sim_pass", "hardware_validated",
)


def _pose_files(shared_root: Path, key: str) -> list[Path]:
    folder = (shared_root / "object_processing" / key / "processed_data" /
              "info" / "tabletop")
    if not folder.is_dir():
        return []
    poses = sorted((p for p in folder.glob("*.npy") if p.stem.isdigit()),
                   key=lambda p: int(p.stem))
    for path in poses:
        pose = np.load(path, allow_pickle=False)
        if pose.shape not in ((3, 3), (4, 4)) or not np.isfinite(pose).all():
            raise ValueError(f"invalid tabletop pose: {path}")
    return poses


def _grasp_dirs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_dir() and
                   all((p / name).is_file() for name in GRASP_FILES)),
                  key=lambda p: (not p.name.isdigit(),
                                 int(p.name) if p.name.isdigit() else p.name))


def _scenario_evidence(scenario: dict[str, Any]) -> dict[str, Any]:
    level = scenario.get("validation_level")
    if level not in LEVELS:
        raise ValueError(f"invalid scenario validation level: {level!r}")
    full_pass = scenario.get("full_task_sim_pass") is True
    hardware_ready = scenario.get("hardware_ready") is True
    if full_pass != (LEVELS.index(level) >= LEVELS.index("full_task_sim_pass")):
        raise ValueError(f"inconsistent full-task evidence: {scenario.get('id')}")
    if hardware_ready != (level == "hardware_validated"):
        raise ValueError(f"inconsistent hardware evidence: {scenario.get('id')}")
    return {
        "id": scenario.get("id"), "candidate_dir": scenario.get("candidate_dir"),
        "validation_level": level, "full_task_sim_pass": full_pass,
        "hardware_ready": hardware_ready,
        "evidence": scenario.get("evidence", {}),
    }


def _scene_files(shared_root: Path, key: str, cell: str) -> dict[str, str]:
    scenes: dict[str, str] = {}
    for height in RESET_HEIGHTS_CM:
        name = f"reorient_{height}"
        for candidate in (
            shared_root / "AutoDex" / "scene" / "inspire" / key / name / f"{cell}.json",
            shared_root / "object_processing" / key / "scene" / name / f"{cell}.json",
        ):
            if candidate.is_file():
                scenes[str(height)] = str(candidate)
                break
    return scenes


def _reset_candidates(shared_root: Path, key: str, cell: str) -> list[dict[str, Any]]:
    found = []
    for height in RESET_HEIGHTS_CM:
        folder = (shared_root / "AutoDex" / "candidates" / "inspire" /
                  f"reset_{height}" / key / f"reorient_{height}" / cell)
        for path in _grasp_dirs(folder):
            found.append({"id": path.name, "height_cm": height,
                          "path": str(path), "source": "runtime_reset_pool"})
    return found


def _bodex_raw_runs(shared_root: Path, key: str, cell: str) -> list[dict[str, Any]]:
    """Locate native BODex proposal runs without promoting their saved seeds.

    A BODex output directory contains all optimized seeds, including failed
    ones. Presence alone says nothing about numerical quality or safety.
    """
    root = shared_root / "AutoDex" / "bodex_raw" / "inspire"
    # Square handle proxies use AutoDex's older naming contract; the cylinder
    # uses a key-specific grip proxy. Neither changes the runtime key ID.
    if key == "precision_key_1p5mm":
        proxy_names = (key, "precision_key_handle_contact_proxy")
    elif key.startswith("precision_key_") and key.endswith("mm"):
        proxy_names = (key, f"{key}_handle_contact_proxy")
    else:
        proxy_names = (key, f"{key}_grip_proxy")
    runs = []
    for proxy in proxy_names:
        for height in RESET_HEIGHTS_CM:
            for path in sorted(root.glob(f"*/{proxy}/reorient_{height}/{cell}")):
                if not path.is_dir():
                    continue
                samples = sorted((p for p in path.iterdir() if p.is_dir() and
                                  (p / "bodex_info.npy").is_file()),
                                 key=lambda p: int(p.name) if p.name.isdigit() else 10**12)
                runs.append({
                    "run": path.parents[2].name,
                    "proposal_object": proxy,
                    "height_cm": height,
                    "path": str(path),
                    "saved_optimization_seeds": len(samples),
                    "evidence_level": "raw_bodex_output_not_a_screened_grasp",
                })
    return runs


def _diagnostic_seeds(shared_root: Path, key: str, cell: str) -> list[dict[str, Any]]:
    # This archived experiment contains a 1.5 mm key candidate only.  The
    # same wrist pose must never be copied into another key or socket mode.
    if key != "precision_key_1p5mm":
        return []
    folder = (shared_root / "AutoDex" / "precision_insertion" / "experiments" /
              "reorientation_from_stable_grasp" / cell)
    return [{"id": p.name, "path": str(p), "source": "diagnostic_staging_only",
             "grasp_sim_evidence": str(p / "sim_eval.json") if
             (p / "sim_eval.json").is_file() else None}
            for p in _grasp_dirs(folder)]


def _motion_preview(shared_root: Path, key: str, cell: str) -> dict[str, Any] | None:
    if key != "precision_key_1p5mm":
        return None
    folder = (shared_root / "AutoDex" / "precision_insertion" /
              "presentation_assets" / "06_reorientation")
    # Only the known stable-grasp diagnostic path is inspected.  Other
    # presentation animations are not motion-planning evidence.
    path = folder / "stable_grasp_104" / "reorientation_pose_004_to_000.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("selected", {}).get("cell") != cell:
        return None
    return {"path": str(path), "status": report.get("status"),
            "selected_candidate": report.get("selected", {}).get("candidate"),
            "physical_validation": False}


def build_catalogs(shared_root: Path, scenario_catalog: dict[str, Any],
                   *, variants: tuple[tuple[str, float], ...] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exhaustive *pose inventory*, not exhaustive grasp search."""
    if variants is None:
        variants = (tuple(("square", gap) for gap in SQUARE_GAPS) +
                    tuple(("cylinder", gap) for gap in CYLINDER_GAPS))
    scenarios = scenario_catalog.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario catalog must contain a scenarios list")
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    missing_assets: list[dict[str, Any]] = []
    for mode_name, gap in variants:
        mode = mode_config(mode_name, gap_mm=gap)
        pose_files = _pose_files(shared_root, mode.key)
        if not pose_files:
            missing_assets.append({"mode": mode.name, "gap_mm": gap,
                                   "key": mode.key, "reason": "tabletop_pose_assets_missing"})
            continue
        for source in pose_files:
            pose_id = int(source.stem)
            pool_dir = (shared_root / "AutoDex" / "candidates" / "inspire" /
                        "v8" / mode.key / "table" / str(pose_id))
            pool = [{"id": path.name, "path": str(path),
                     "evidence_level": "candidate_files_only"}
                    for path in _grasp_dirs(pool_dir)]
            matches = [s for s in scenarios if s.get("mode") == mode.name and
                       s.get("key") == mode.key and s.get("socket") == mode.socket and
                       float(s.get("gap_mm", -1)) == gap and
                       s.get("tabletop_pose") == pose_id]
            evidence = [_scenario_evidence(s) for s in matches]
            full_pass = [s["id"] for s in evidence if s["full_task_sim_pass"]]
            hardware_pass = [s["id"] for s in evidence if s["hardware_ready"]]
            # An object-relative pickup grasp can be proposed as a reset
            # hypothesis for any target pose, but it has NOT passed the paired
            # BODex scene or the reorientation motion/contact chain.
            hypotheses = [
                {"id": item["id"], "path": item["path"],
                 "source": "v8_source_pose_grasp_not_pair_validated"}
                for item in pool
            ]
            hypotheses.extend(
                {"id": item["id"], "path": item["candidate_dir"],
                 "source": "scenario_grasp_not_pair_validated"}
                for item in evidence if item["candidate_dir"] and
                item["validation_level"] in LEVELS[2:]
            )
            if hardware_pass:
                state = "hardware_evidence_present"
            elif full_pass:
                state = "full_task_sim_evidence_present"
            elif pool or any(item["validation_level"] in LEVELS[2:]
                             for item in evidence):
                state = "grasp_candidates_below_full_task_gate"
            elif evidence:
                state = "geometry_evidence_only"
            else:
                state = "no_candidate_evidence_yet"
            rows.append({
                "mode": mode.name, "gap_mm": gap, "key": mode.key,
                "socket": mode.socket, "tabletop_pose": pose_id,
                "tabletop_pose_file": str(source), "runtime_pool_candidates": pool,
                "scenario_evidence": evidence, "evidence_state": state,
                "full_task_sim_pass_ids": full_pass,
                "hardware_validated_ids": hardware_pass,
                "candidate_pool_closed": False,
                "all_candidates_full_task_tested": False,
                "absence_verdict": "unknown_not_proven",
                "may_trigger_reorient_for_no_feasible_scenario": False,
            })
            for target in pose_files:
                if target == source:
                    continue
                cell = f"{pose_id}_{int(target.stem)}"
                scenes = _scene_files(shared_root, mode.key, cell)
                proxy_scenes = _scene_files(
                    shared_root, f"{mode.key}_grip_proxy", cell
                )
                raw_runs = _bodex_raw_runs(shared_root, mode.key, cell)
                runtime = _reset_candidates(shared_root, mode.key, cell)
                diagnostic = _diagnostic_seeds(shared_root, mode.key, cell)
                preview = _motion_preview(shared_root, mode.key, cell)
                if runtime:
                    state = "runtime_seed_files_unvalidated_for_this_task"
                elif diagnostic and preview:
                    state = "diagnostic_curobo_motion_preview_only"
                elif diagnostic:
                    state = "diagnostic_grasp_seed_only"
                elif raw_runs:
                    state = "raw_bodex_proposals_not_screened"
                elif scenes:
                    state = "bodex_scene_only_no_grasp_seed"
                else:
                    state = "scene_and_grasp_seed_missing"
                transitions.append({
                    "mode": mode.name, "gap_mm": gap, "key": mode.key,
                    "source_pose": pose_id, "target_pose": int(target.stem),
                    "cell": cell, "source_pose_file": str(source),
                    "target_pose_file": str(target),
                    "scene_by_release_height_cm": scenes,
                    "bodex_proxy_scene_by_release_height_cm": proxy_scenes,
                    "bodex_raw_runs": raw_runs,
                    "source_grasp_hypotheses": hypotheses,
                    "runtime_reset_candidates": runtime,
                    "diagnostic_seeds": diagnostic,
                    "motion_preview": preview, "evidence_state": state,
                    "target_pose_verified_insertable": False,
                    "physical_transition_validated": False,
                    "ready_for_automatic_reorient": False,
                })
    pose_catalog = {
        "schema": "precision_insertion_pose_task_catalog_v1",
        "definition": "Evidence inventory, not a list of insertion successes or impossible poses.",
        "shared_root": str(shared_root), "rows": rows,
        "missing_variant_assets": missing_assets,
        "absence_rule": ("No full-task pass is UNKNOWN, not impossible. A finite pool may "
                         "only be called exhausted after its generation scope is frozen "
                         "and every member has a complete negative full-task result."),
    }
    seed_catalog = {
        "schema": "precision_insertion_reorient_seed_catalog_v1",
        "definition": "Directed source-to-target scene and grasp-seed inventory; never a robot command.",
        "shared_root": str(shared_root), "transitions": transitions,
        "activation_rule": ("Automatic reorient needs a validated source transition, "
                            "a verified insertable target pose, and fresh post-placement perception."),
    }
    return pose_catalog, seed_catalog


def _write_new(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing catalog: {path}; use --force")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--scenario-catalog", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="replace only these two generated catalog JSONs")
    args = parser.parse_args()
    root = args.shared_root.expanduser().resolve()
    scenario_path = args.scenario_catalog or (
        root / "AutoDex" / "precision_insertion" / "scenario_catalog.json")
    out_dir = args.out_dir or (root / "AutoDex" / "precision_insertion" /
                               "pose_task_catalog")
    scenario_catalog = json.loads(scenario_path.read_text(encoding="utf-8"))
    poses, seeds = build_catalogs(root, scenario_catalog)
    pose_path = out_dir / "full_task_catalog.json"
    seed_path = out_dir / "reorient_seed_catalog.json"
    if not args.force and (pose_path.exists() or seed_path.exists()):
        raise FileExistsError("generated catalog already exists; use --force")
    _write_new(pose_path, poses, force=args.force)
    _write_new(seed_path, seeds, force=args.force)
    print(json.dumps({"full_task_catalog": str(pose_path), "pose_rows": len(poses["rows"]),
                      "reorient_seed_catalog": str(seed_path),
                      "directed_transitions": len(seeds["transitions"]),
                      "full_task_sim_pass_rows": sum(bool(row["full_task_sim_pass_ids"])
                                                     for row in poses["rows"]),
                      "automatic_reorient_ready": sum(row["ready_for_automatic_reorient"]
                                                      for row in seeds["transitions"])}))


if __name__ == "__main__":
    main()
