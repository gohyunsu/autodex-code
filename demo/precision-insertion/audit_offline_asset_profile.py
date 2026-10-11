#!/usr/bin/env python3
"""Write one explicit, non-promoting inventory for the current demo modes.

This checks file-backed CAD, FoundPose, v8 endpoint catalogs and directed
reset grasp seeds. It does not generate a MuJoCo pass, accept a pending NAS
representation, or infer a successful robot trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from precision_insertion.assets import audit_assets
from precision_insertion.config import select_mode
from precision_insertion.live_session_runner import preflight_endpoint_catalog_file
from precision_insertion.reorient_assets import audit_v8_reorient_assets


SQUARE_GAPS = (0.3, 0.5, 1.0, 1.5)
CYLINDER_GAPS = (1, 3, 5, 10, 15, 20)


def _catalog_path(root: Path, family: str, gap: float) -> Path:
    base = root / "AutoDex/precision_insertion"
    if family == "cylinder":
        return (base / "cylindrical" /
                f"endpoint_catalog_current_gap_{int(gap)}mm_20261011_stock_schema3.json")
    name = str(gap).replace(".", "p")
    return (base / "endpoint_catalogs" /
            f"square_{name}mm_20261011_fullkey1000_schema3.json")


def _reset_root(root: Path, family: str, gap: float) -> Path | None:
    base = root / "AutoDex/precision_insertion"
    if family == "square" and gap == 0.5:
        return base / "square/reorient_0p5_curated_5000_20261011"
    if family == "square" and gap == 1.5:
        return base / "square/reorient_curated_handoff_20261011"
    if family == "cylinder":
        # All radial gaps use the same 80 mm key and directed reset scenes.
        return base / "cylindrical/reorient_curated_handoff_20261011"
    return None


def build_profile(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    modes = []
    for family, gaps in (("square", SQUARE_GAPS),
                         ("cylinder", CYLINDER_GAPS)):
        for gap in gaps:
            mode = select_mode(family, gap)
            files = audit_assets(root, mode)
            catalog_path = _catalog_path(root, family, gap)
            try:
                catalog = preflight_endpoint_catalog_file(
                    mode=mode, shared_root=root, catalog_path=catalog_path)
                catalog_status = "source_verified_complete_scan"
                eligible = sum(row.get("eligible") is True
                               for row in catalog["candidates"])
                poses = sorted({row["tabletop_pose_stem"]
                                for row in catalog["candidates"]
                                if row.get("eligible") is True})
                catalog_error = None
            except (FileNotFoundError, ValueError, KeyError, TypeError,
                    OSError) as exc:
                catalog_status = "missing_or_not_current"
                eligible = 0
                poses = []
                catalog_error = str(exc)
            reset_root = _reset_root(root, family, gap)
            modes.append({
                "family": family, "gap_mm": gap,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object,
                "object_file_audit": files,
                "endpoint_catalog": {
                    "path": str(catalog_path), "status": catalog_status,
                    "eligible_count": eligible,
                    "eligible_pose_stems": poses,
                    "error": catalog_error,
                },
                "reorient_seed_root": (
                    str(reset_root / "reset_12") if reset_root is not None
                    else None),
            })
    reset_audits = {}
    for family, gap in (("square", 0.5), ("square", 1.5),
                        ("cylinder", 20)):
        reset_root = _reset_root(root, family, gap)
        assert reset_root is not None
        audit = audit_v8_reorient_assets(
            shared_root=root, mode=select_mode(family, gap),
            candidate_root=reset_root,
            max_center_in_hand_drift_m=.005,
            max_symmetry_axis_tilt_deg=10.)
        key = "square_0p5" if family == "square" and gap == 0.5 else family
        reset_audits[key] = {
            "candidate_root": str(reset_root / "reset_12"),
            "diagnostic_fidelity_only": {"drift_mm": 5, "rotation_deg": 10},
            "directed_cells": [
                {"from_pose": row["from_v8_pose"],
                 "to_pose": row["to_v8_pose"],
                 "mujoco_stable_12cm": row[
                     "stable_reset_seed_counts_by_height_cm"]["12"],
                 "fidelity_eligible_12cm": row[
                     "fidelity_eligible_seed_counts_by_height_cm"]["12"]}
                for row in audit["directed_pairs"]],
        }
    return {
        "schema": "precision_insertion_offline_asset_profile_v1",
        "shared_root": str(root), "modes": modes,
        "reorient": reset_audits,
        "scope": "offline_files_and_verified_source_seeds_only",
        "physical_or_robot_success": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    profile = build_profile(args.shared_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "profile": str(output),
        "modes": len(profile["modes"]),
        "complete_catalogs": sum(
            row["endpoint_catalog"]["status"] == "source_verified_complete_scan"
            for row in profile["modes"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
