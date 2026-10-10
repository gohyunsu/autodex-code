#!/usr/bin/env python3
"""Audit every stock-filtered cylinder grasp for post-squeeze pose fidelity.

Read-only with respect to AutoDex inputs and the robot. Results are a separate
diagnostic layer; no uncommissioned displacement cutoff silently changes the
original 1,000-per-scene candidate counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.grasp_fidelity import (
    nominal_visual_penetration_audit, simulated_visual_penetration_audit,
    trajectory_closure_audit,
)


KEY = "precision_key_cylinder_r15_h80"


def run(*, summary_path: Path, shared_root: Path, output_root: Path,
        points_per_link: int = 300) -> dict:
    source = Path(summary_path).resolve()
    root = Path(shared_root).resolve()
    target = Path(output_root).resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"refusing to overwrite audit or partial run: {target}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (summary.get("schema") != "precision_insertion_cylinder_1000_per_scene_screen_v1"
            or summary.get("physical_key_object") != KEY):
        raise ValueError("expected complete full-key cylinder screen")
    stage = Path(summary["stage_root"]).resolve()
    pool = Path(summary["original_autodex_sim_candidate_root"]).resolve()
    mode = select_mode("cylinder", 20.0)
    paths = AssetPaths(root, mode)
    candidates = []
    for scene_id in ("0", "1"):
        directory = pool / KEY / "table" / scene_id
        rows = sorted((p for p in directory.iterdir() if p.is_dir()),
                      key=lambda p: int(p.name))
        if len(rows) != summary["scene_counts"][scene_id]["mujoco_stable"]:
            raise ValueError(f"candidate count mismatch for table/{scene_id}")
        candidates.extend((scene_id, path) for path in rows)
    work.mkdir(parents=True)
    reports = []
    for scene_id, candidate in candidates:
        identifier = f"table/{scene_id}/{candidate.name}"
        raw = stage / KEY / "table" / scene_id / candidate.name
        sim_eval = json.loads((raw / "sim_eval.json").read_text(encoding="utf-8"))
        if sim_eval.get("success") is not True:
            raise ValueError(f"candidate lacks original AutoDex success: {identifier}")
        trajectory = json.loads((raw / "sim_traj.json").read_text(encoding="utf-8"))
        report = {
            "id": identifier,
            "candidate_dir": str(candidate),
            "sim_traj": str(raw / "sim_traj.json"),
            "closure": trajectory_closure_audit(trajectory, key_height_m=0.08),
            "nominal_fixed_key_visual_penetration": nominal_visual_penetration_audit(
                candidate_dir=candidate, key_mesh_path=paths.raw_mesh(KEY),
                robot_urdf=paths.robot_urdf, points_per_link=points_per_link),
            "achieved_mujoco_visual_penetration": simulated_visual_penetration_audit(
                trajectory=trajectory, key_mesh_path=paths.raw_mesh(KEY),
                robot_urdf=paths.robot_urdf, points_per_link=points_per_link),
        }
        reports.append(report)
        file = work / "table" / scene_id / f"{candidate.name}.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{identifier}: center-in-hand drift "
              f"{report['closure']['end_squeeze']['center_in_hand_displacement_m']*1000:.1f} mm",
              flush=True)
    audit = {
        "schema": "precision_insertion_cylinder_grasp_fidelity_audit_v2",
        "source_summary": str(source),
        "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "key_mesh": str(paths.raw_mesh(KEY)),
        "key_mesh_sha256": hashlib.sha256(paths.raw_mesh(KEY).read_bytes()).hexdigest(),
        "robot_urdf": str(paths.robot_urdf),
        "robot_urdf_sha256": hashlib.sha256(paths.robot_urdf.read_bytes()).hexdigest(),
        "status": "diagnostic_only_no_robot_authorization",
        "candidate_count": len(reports),
        "key_height_m": 0.08,
        "symmetry": "axial yaw and end-for-end flip quotient; center and axis are measured",
        "visual_sample_points_per_link": points_per_link,
        "visual_depth_threshold_m": 0.0002,
        "not_a_filter": "No displacement or penetration acceptance threshold is commissioned.",
        "candidates_with_nominal_visual_penetration": sum(
            any(hold["sample_points_over_threshold"] > 0 for hold in
                row["nominal_fixed_key_visual_penetration"].values())
            for row in reports),
        "candidates_with_achieved_mujoco_visual_penetration": sum(
            any(hold["sample_points_over_threshold"] > 0 for hold in
                row["achieved_mujoco_visual_penetration"].values())
            for row in reports),
        "candidates": [row["id"] for row in reports],
    }
    (work / "summary.json").write_text(json.dumps(audit, indent=2) + "\n",
                                       encoding="utf-8")
    work.rename(target)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points-per-link", type=int, default=300)
    args = parser.parse_args()
    result = run(summary_path=args.summary, shared_root=args.shared_root,
                 output_root=args.output_root, points_per_link=args.points_per_link)
    print(json.dumps({"audit": str(args.output_root / "summary.json"),
                      "candidate_count": result["candidate_count"],
                      "nominal_visual_penetration":
                          result["candidates_with_nominal_visual_penetration"]},
                     indent=2))


if __name__ == "__main__":
    main()
