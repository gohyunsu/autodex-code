#!/usr/bin/env python3
"""Replay the current main-PC AutoDex multi-view key refinement offline.

Input FoundPose candidates, source PNGs and SAM masks must describe the same
saved capture. This runs local CUDA rendering, but never opens a camera or
robot and cannot establish independent metric pose accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_foundpose_stock_refinement import _replay_stock
from probe_zerodex_saved_foundpose import _load_inputs


def _bind_views(views: list[dict], source: dict, object_name: str) -> dict:
    if (source.get("schema") != "precision_insertion_zerodex_saved_foundpose_probe_v1"
            or source.get("object_name") != object_name):
        raise ValueError("FoundPose report is for another object or replay")
    rows = {row["serial"]: row for row in source["views"]}
    if len(rows) != len(views):
        raise ValueError("FoundPose report does not cover all saved views")
    bound = {}
    for view in views:
        serial = view["serial"]
        row = rows.get(serial)
        if (row is None or row.get("status") != "pose_estimated_not_validated"
                or row.get("image_sha256") != view["image_sha256"]
                or row.get("mask_sha256") != view["mask_sha256"]):
            raise ValueError(f"FoundPose frame/mask differs from {serial}")
        bound[serial] = {**view, "pose_world": np.asarray(
            row["pose_world_m"], dtype=np.float64)}
    return bound


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--foundpose-report", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    views = _load_inputs(
        args.capture_dir.expanduser().resolve(),
        args.mask_dir.expanduser().resolve(),
        args.calib_dir.expanduser().resolve(),
        args.probe_report.expanduser().resolve(), args.object_name)
    source_path = args.foundpose_report.expanduser().resolve()
    source_raw = source_path.read_bytes()
    bound = _bind_views(views, json.loads(source_raw), args.object_name)
    mesh = args.mesh.expanduser().resolve()
    diagnostics, pose = _replay_stock(
        mesh, bound, iterations=100, loss_threshold=0.003)
    record = {
        "schema": "precision_insertion_zerodex_current_stock_refine_v1",
        "scope": "saved_mask_fit_not_independent_metric_pose_or_robot_validation",
        "object_name": args.object_name,
        "foundpose_report": str(source_path),
        "foundpose_report_sha256": hashlib.sha256(source_raw).hexdigest(),
        "camera_ids": list(bound),
        "mesh": str(mesh),
        "stock_refinement": diagnostics,
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "accepted": pose is not None,
                      "selected": diagnostics["stock_selection"].get("best_serial")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
