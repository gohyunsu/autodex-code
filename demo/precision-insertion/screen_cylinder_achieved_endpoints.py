#!/usr/bin/env python3
"""Rescreen 20 mm endpoints using *achieved* MuJoCo hand/key states.

The original nominal socket screens remain untouched. This separate offline
comparison tests both the first closure's end-squeeze state and the second
closure's end-gravity state. It does not establish that either state can be
replayed on hardware, that the key stayed fixed during a real lift, or that
Franka can reach/insert along a collision-free continuous trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.endpoint import screen_grasp_endpoint
from precision_insertion.grasp_fidelity import (
    achieved_hand_state, trajectory_closure_audit,
)


KEY = "precision_key_cylinder_r15_h80"
STATES = ("end_squeeze", "end_gravity")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_audited_rows(audit_path: Path) -> tuple[dict, dict, list[dict]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "precision_insertion_cylinder_grasp_fidelity_audit_v2"
            or audit.get("status") != "diagnostic_only_no_robot_authorization"
            or audit.get("candidate_count") != len(audit.get("candidates", []))):
        raise ValueError("expected complete dynamic cylinder fidelity audit")
    source = Path(audit["source_summary"]).resolve()
    if _hash(source) != audit["source_summary_sha256"]:
        raise ValueError("source nominal endpoint summary changed")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if (summary.get("schema") != "precision_insertion_cylinder_1000_per_scene_screen_v1"
            or summary.get("physical_key_object") != KEY):
        raise ValueError("source is not the full-key cylinder run")
    if (_hash(Path(audit["key_mesh"])) != audit["key_mesh_sha256"] or
            _hash(Path(audit["robot_urdf"])) != audit["robot_urdf_sha256"]):
        raise ValueError("key mesh or Inspire URDF changed since fidelity audit")
    pool = Path(summary["original_autodex_sim_candidate_root"]).resolve()
    stage = Path(summary["stage_root"]).resolve()
    rows = []
    for identifier in audit["candidates"]:
        parts = identifier.split("/")
        if len(parts) != 3 or parts[0] != "table" or parts[1] not in ("0", "1") or not parts[2].isdigit():
            raise ValueError(f"invalid candidate ID: {identifier}")
        scene_id, seed = parts[1:]
        report_path = audit_path.parent / "table" / scene_id / f"{seed}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = pool / KEY / "table" / scene_id / seed
        trajectory_path = stage / KEY / "table" / scene_id / seed / "sim_traj.json"
        if (report.get("id") != identifier or
                report.get("candidate_dir") != str(candidate) or
                report.get("sim_traj") != str(trajectory_path)):
            raise ValueError(f"fidelity row source mismatch: {identifier}")
        if json.loads((trajectory_path.parent / "sim_eval.json").read_text()).get("success") is not True:
            raise ValueError(f"stock MuJoCo did not pass: {identifier}")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        current_closure = trajectory_closure_audit(trajectory, key_height_m=0.08)
        for state in STATES:
            old = report["closure"][state]
            new = current_closure[state]
            if old["trajectory_index"] != new["trajectory_index"] or any(
                not math.isclose(old[name], new[name], abs_tol=1e-10)
                for name in ("center_world_displacement_m",
                             "center_in_hand_displacement_m",
                             "symmetry_reduced_axis_tilt_deg")):
                raise ValueError(f"trajectory changed since fidelity audit: {identifier}")
        rows.append({"id": identifier, "candidate": candidate,
                     "trajectory": trajectory, "fidelity": report})
    if len(rows) != sum(v["mujoco_stable"] for v in summary["scene_counts"].values()):
        raise ValueError("fidelity audit is not the complete MuJoCo pass set")
    return audit, summary, rows


def run(*, fidelity_audit: Path, shared_root: Path, output_root: Path,
        minimum_hand_clearance_m: float,
        gaps_mm: tuple[float, ...] = CYLINDER_RADIAL_GAPS_MM) -> dict:
    audit_path = Path(fidelity_audit).resolve()
    root = Path(shared_root).resolve()
    target = Path(output_root).resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"refusing to overwrite endpoint result: {target}")
    audit, original, rows = _load_audited_rows(audit_path)
    paths = AssetPaths(root, select_mode("cylinder", 20.0))
    if (Path(audit["key_mesh"]).resolve() != paths.raw_mesh(KEY).resolve() or
            Path(audit["robot_urdf"]).resolve() != paths.robot_urdf.resolve()):
        raise ValueError("shared root differs from audited key/robot assets")
    if (not math.isfinite(minimum_hand_clearance_m) or
            minimum_hand_clearance_m != original["minimum_hand_clearance_m"]):
        raise ValueError("use the original nominal-screen clearance for comparison")
    if not gaps_mm or len(set(gaps_mm)) != len(gaps_mm):
        raise ValueError("provide a nonempty unique socket-gap selection")
    gaps = [select_mode("cylinder", gap) for gap in gaps_mm]
    work.mkdir(parents=True)
    by_gap = {}
    for mode in gaps:
        label = f"{int(mode.gap_mm):02d}mm"
        nominal_ids = set(original["socket_gaps"][label]["eligible_offline_grasp_ids"])
        state_passes = {state: [] for state in STATES}
        both_pass = []
        for row in rows:
            identifier = row["id"]
            scene_id, seed = identifier.split("/")[1:]
            result = {"id": identifier, "socket_object": mode.socket_object,
                      "nominal_initial_pose_endpoint_pass": identifier in nominal_ids,
                      "fidelity_closure": row["fidelity"]["closure"],
                      "achieved_states": {}}
            for state in STATES:
                T_key_hand, hand_q, index = achieved_hand_state(row["trajectory"], state)
                if index != row["fidelity"]["closure"][state]["trajectory_index"]:
                    raise ValueError(f"state index mismatch: {identifier}/{state}")
                if not np.allclose(
                    hand_q,
                    row["fidelity"]["achieved_mujoco_visual_penetration"][state]["achieved_hand_q"],
                    atol=1e-10):
                    raise ValueError(f"achieved joint mismatch: {identifier}/{state}")
                screen = screen_grasp_endpoint(
                    shared_root=root, mode=mode, candidate_dir=row["candidate"],
                    minimum_hand_clearance_m=minimum_hand_clearance_m,
                    T_key_hand_override=T_key_hand,
                    hand_poses_override={"achieved_mujoco_hand": hand_q},
                    override_source=f"simulated_{state}",
                )
                result["achieved_states"][state] = screen
                if screen["endpoint_pass"]:
                    state_passes[state].append(identifier)
            result["both_achieved_states_endpoint_pass"] = all(
                result["achieved_states"][state]["endpoint_pass"] for state in STATES)
            if result["both_achieved_states_endpoint_pass"]:
                both_pass.append(identifier)
            file = work / f"gap_{label}" / "table" / scene_id / f"{seed}.json"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        by_gap[label] = {
            "socket_object": mode.socket_object,
            "nominal_initial_pose_endpoint_pass_count": len(nominal_ids),
            "achieved_state_clear_ids": state_passes,
            "both_achieved_states_clear_ids": both_pass,
        }
        print(f"{label}: {len(both_pass)}/{len(rows)} clear in both achieved states", flush=True)
    summary = {
        "schema": "precision_insertion_cylinder_achieved_endpoint_comparison_v1",
        "status": "offline_achieved_pose_endpoint_only_not_robot_ready",
        "source_fidelity_audit": str(audit_path),
        "source_fidelity_audit_sha256": _hash(audit_path),
        "source_nominal_summary": audit["source_summary"],
        "source_nominal_summary_sha256": audit["source_summary_sha256"],
        "candidate_count": len(rows),
        "minimum_hand_clearance_m": minimum_hand_clearance_m,
        "socket_family_complete": len(gaps) == len(CYLINDER_RADIAL_GAPS_MM),
        "socket_gaps": by_gap,
        "not_validated": [
            "post-lift physical hand/key pose or controller tracking",
            "Franka IK, arm/world collision, continuous transfer and insertion path",
            "continuous contact dynamics, guarded insertion or physical success",
            "axial-yaw alternatives and robot reachability",
            "post-squeeze drift acceptance thresholds",
        ],
        "robot_ready": False,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                       encoding="utf-8")
    work.rename(target)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity-audit", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-hand-clearance-m", type=float, required=True)
    parser.add_argument("--gap-mm", type=float, action="append",
                        help="optional socket-gap subset; otherwise all six")
    args = parser.parse_args()
    report = run(
        fidelity_audit=args.fidelity_audit, shared_root=args.shared_root,
        output_root=args.output_root,
        minimum_hand_clearance_m=args.minimum_hand_clearance_m,
        gaps_mm=tuple(args.gap_mm) if args.gap_mm else CYLINDER_RADIAL_GAPS_MM,
    )
    print(json.dumps({"summary": str(args.output_root / "summary.json"),
                      "counts": {gap: len(row["both_achieved_states_clear_ids"])
                                 for gap, row in report["socket_gaps"].items()},
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
