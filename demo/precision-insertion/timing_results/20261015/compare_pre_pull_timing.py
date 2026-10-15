#!/usr/bin/env python3
"""Compare saved-data CPU/NAS calls across two precision-insertion revisions.

This never constructs a camera, planner, robot or VLM backend. Output ledger
records are written under /tmp only.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source-tree", type=Path, required=True)
parser.add_argument("--label", choices=("before", "after"), required=True)
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()
tree = args.source_tree.resolve()
sys.path.insert(0, str(tree / "demo/precision-insertion"))
sys.path.insert(1, str(tree))

import numpy as np  # noqa: E402
from precision_insertion.calibration import load_session_calibration  # noqa: E402
from precision_insertion.candidates import (  # noqa: E402
    planner_candidate_override, select_pose_candidates,
)
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.live_session_runner import (  # noqa: E402
    open_verified_session, preflight_endpoint_catalog_file,
)
from precision_insertion.session_bootstrap import (  # noqa: E402
    verify_session_evidence_bundle,
)

root = Path("/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392")
catalog_path = root / "AutoDex/precision_insertion/cylindrical/recipient_gap20_endpoint_catalog_20261013.json"
evidence = root / "AutoDex/precision_insertion/cylindrical/live_board9_gap20_20261013/session_evidence_001"
mode = select_mode("cylinder", 20.0)
args.output_root.mkdir(parents=True, exist_ok=True)


def measure(fn):
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


runs = []
for iteration in range(4):
    output_dir = args.output_root / f"{args.label}_ledger_{iteration:02d}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    result = {"iteration": iteration, "seconds": {}}
    catalog, result["seconds"]["catalog_preflight"] = measure(
        lambda: preflight_endpoint_catalog_file(
            mode=mode, shared_root=root, catalog_path=catalog_path))
    manifest, result["seconds"]["producer_bundle_verify"] = measure(
        lambda: verify_session_evidence_bundle(evidence))
    calibration, result["seconds"]["producer_calibration_reload"] = measure(
        lambda: load_session_calibration(
            evidence / "session_calibration.json", mode=mode,
            shared_root=root))
    opening = dict(mode=mode, shared_root=root, evidence_dir=evidence,
                   catalog_path=catalog_path, output_dir=output_dir,
                   max_xy_retries=0)
    if args.label == "after":
        opening.update(catalog_snapshot=catalog, producer_manifest=manifest,
                       calibration_snapshot=calibration)
    opened, result["seconds"]["ledger_open"] = measure(
        lambda: open_verified_session(**opening))
    selection, result["seconds"]["candidate_selection_000"] = measure(
        lambda: select_pose_candidates(
            catalog, expected_mode=mode, tabletop_pose_stem="000"))
    if selection["status"] != "candidates_available":
        raise ValueError(selection["status"])
    selected = selection["candidates"][:1]
    override, result["seconds"]["one_candidate_override"] = measure(
        lambda: planner_candidate_override(
            catalog=catalog, selected=selected, mode=mode,
            session_record=calibration.record,
            pose_robot_key=np.eye(4), tabletop_pose_stem="000"))
    if len(override[3]) != 1 or opened.next_decision.action != "capture_fresh_key":
        raise ValueError("saved-data comparison diverged")
    result["selected_key"] = selected[0]["key"]
    result["seconds"]["known_subset_total"] = sum(result["seconds"].values())
    runs.append(result)
    print(json.dumps({"label": args.label, **result}), flush=True)

summary = {"label": args.label, "source_tree": str(tree),
           "catalog": str(catalog_path), "evidence": str(evidence),
           "scope": "saved_data_cpu_nas_subset_no_camera_robot_planner_vlm",
           "runs": runs,
           "warm_median_s": {name: statistics.median(
               run["seconds"][name] for run in runs[1:])
               for name in runs[0]["seconds"]}}
(args.output_root / f"{args.label}_comparison.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
