#!/usr/bin/env python3
"""Summarize a complete, isolated stock AutoDex cylinder-reset filter run.

This does not convert a simulated grasp into a socket-aware robot reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stage_cylinder_reorient_proposals import KEY, PAIRS


def audit(*, stage_root: Path, output: Path) -> dict:
    root = Path(stage_root).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to replace reset audit: {target}")
    manifest_path = root / "stage_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != "precision_insertion_cylinder_reset_proxy_stage_v1" or
            manifest.get("full_key_object") != KEY or
            Path(manifest.get("stage_root", "")).resolve() != root):
        raise ValueError("staging manifest does not describe this full-key run")
    expected = manifest["expected_per_cell"]
    if type(expected) is not int or expected < 1:
        raise ValueError("invalid expected reset seed count")

    rows = []
    for cell in PAIRS:
        scene = root / KEY / "reorient_12" / cell
        found = {path.name for path in scene.iterdir() if path.is_dir()}
        if found != {str(seed) for seed in range(expected)}:
            raise ValueError(f"incomplete reset cell {cell}: {len(found)}/{expected}")
        counts = {
            "raw": expected, "scene_clear": 0, "squeeze_contact": 0,
            "mujoco_stable": 0, "scene_collision": 0,
            "no_object_contact": 0, "mujoco_unstable": 0,
        }
        stable_ids = []
        for seed in range(expected):
            seed_dir = scene / str(seed)
            collision = np.load(seed_dir / "coll_valid.npy", allow_pickle=False)
            if collision.shape != () or collision.dtype != np.dtype(bool):
                raise ValueError(f"invalid collision result: {seed_dir}")
            scene_clear = bool(collision)
            result = json.loads((seed_dir / "sim_eval.json").read_text())
            if (result.get("hand") != "inspire" or
                    result.get("version") != "v8" or
                    type(result.get("success")) is not bool):
                raise ValueError(f"invalid AutoDex sim result: {seed_dir}")
            reason = result.get("reason")
            if not scene_clear:
                if reason != "scene_collision" or result["success"]:
                    raise ValueError(f"inconsistent collision result: {seed_dir}")
                counts["scene_collision"] += 1
                continue
            counts["scene_clear"] += 1
            if reason == "no_object_contact":
                if result["success"]:
                    raise ValueError(f"inconsistent contact result: {seed_dir}")
                counts["no_object_contact"] += 1
                continue
            if reason is not None:
                raise ValueError(f"unknown sim failure reason {reason}: {seed_dir}")
            counts["squeeze_contact"] += 1
            if result["success"]:
                counts["mujoco_stable"] += 1
                stable_ids.append(seed)
            else:
                counts["mujoco_unstable"] += 1
        rows.append({"cell": cell, "counts": counts, "mujoco_stable_seed_ids": stable_ids})

    totals = {name: sum(row["counts"][name] for row in rows)
              for name in rows[0]["counts"]}
    report = {
        "schema": "precision_insertion_cylinder_reorient_pilot_audit_v1",
        "source_stage_manifest": str(manifest_path),
        "source_stage_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "stage_root": str(root), "cells": rows, "totals": totals,
        "status": "offline_autoDex_reset_filter_only",
        "robot_ready": False,
        "not_validated": [
            "socket-aware Franka pickup/lift/transfer/repose/retract preflight",
            "release and observed tabletop pose after placement",
            "physical reset or insertion success",
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(stage_root=args.stage_root, output=args.output)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError, OSError) as exc:
        parser.exit(2, f"reset audit rejected: {exc}\n")
    print(json.dumps({"totals": report["totals"],
                      "robot_ready": report["robot_ready"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
