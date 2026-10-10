#!/usr/bin/env python3
"""Read-only MuJoCo square-key closure audit for a complete v8 catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.candidates import select_pose_candidates
from precision_insertion.config import select_mode
from precision_insertion.grasp_fidelity import trajectory_rigid_closure_audit
from precision_insertion.square_promotion import SCHEMA


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--pose-stem", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new diagnostic JSON path; refuses overwrite")
    args = parser.parse_args()
    try:
        mode = select_mode("square", args.gap_mm)
        catalog_path = args.catalog.expanduser().resolve()
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selected = select_pose_candidates(
            catalog, expected_mode=mode, tabletop_pose_stem=args.pose_stem)
        if selected["status"] != "candidates_available":
            raise ValueError(f"catalog cannot select this pose: {selected['status']}")
        root = Path(catalog["shared_root"])
        info_path = (root / "object_processing" / mode.key_object /
                     "processed_data/info/simplified.json")
        center = np.asarray(json.loads(info_path.read_text(
            encoding="utf-8"))["gravity_center"], dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("invalid full-key gravity center")
        rows = []
        for row in selected["candidates"]:
            candidate = Path(row["candidate_dir"])
            validation = json.loads((candidate / "simulation_validation.json")
                                    .read_text(encoding="utf-8"))
            if validation.get("schema") != SCHEMA:
                continue
            trajectory_path = candidate / "sim_traj.json"
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            closure = trajectory_rigid_closure_audit(
                trajectory, key_center_local_m=center)
            rows.append({
                "key": row["key"], "candidate_dir": str(candidate),
                "sim_traj_sha256": _sha(trajectory_path),
                "post_squeeze_fidelity_diagnostic": closure,
                "declared_contacts_obey_policy": json.loads(
                    (candidate / "contact_screen.json").read_text(
                        encoding="utf-8")).get("declared_contacts_obey_policy"),
            })
        if not rows:
            raise ValueError("selected pose has no promoted square pilot candidates")
        report = {
            "schema": "precision_insertion_square_grasp_fidelity_audit_v1",
            "status": "simulated_diagnostic_only_not_accepted_relation",
            "catalog": str(catalog_path), "catalog_sha256": _sha(catalog_path),
            "key_info": str(info_path), "key_info_sha256": _sha(info_path),
            "pose_stem": args.pose_stem,
            "key_center_local_m": center.tolist(),
            "candidate_count": len(rows), "candidates": rows,
            "fidelity_threshold_commissioned": False,
            "not_validated": [
                "physical post-lift key-in-hand relation",
                "Franka transfer, guarded insertion, or physical success",
            ],
            "robot_ready": False,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "report": str(output), "candidate_count": len(rows),
        "fidelity_threshold_commissioned": False, "robot_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
