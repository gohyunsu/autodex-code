#!/usr/bin/env python3
"""Prepare a reproducible 5x5 actual-mesh grasp-policy comparison.

The labels come directly from one rigid-insertion screen report.  The script
does not infer success from a render: it selects up to 12 report passes, fills
the remaining 25 cells with report failures, exports the corresponding
original Inspire visual meshes in the key frame, and records every source path
and failure reason in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_contact_policy_presentation import _export_hand


SHARED = Path.home() / "shared_data"
DEFAULT_REPORT = (
    SHARED / "AutoDex/precision_insertion/"
    "rigid_insertion_grasp_screen_table_004_frame_fixed_51000.json"
)
DEFAULT_ROBOT = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/precision_insertion/presentation_assets/"
    "03_contact_policy/grid"
)
DEFAULT_CANDIDATE_SET = (
    Path(__file__).resolve().parents[2]
    / "assets/precision_insertion/presentation_candidate_set.json"
)


def _spread(items: list[dict], count: int) -> list[dict]:
    """Choose deterministic samples across the complete report ordering."""
    if len(items) < count:
        raise ValueError(f"need {count} items, found {len(items)}")
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def _failure_reason(candidate: dict) -> str:
    policy = candidate["contact_policy"]
    if policy["status"] != "sampled_pass":
        links = policy.get("permitted_contact_links", [])
        if not links:
            return "no permitted handle contact"
        if not any("thumb" in name for name in links):
            return "no thumb opposition"
        if not any("thumb" not in name for name in links):
            return "thumb-only contact"
        if policy.get("forbidden_penetrating_samples", 0):
            return "forbidden-region penetration"
        return "contact policy rejected"
    return "passed sampled whole-hand contact-policy gate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate-set", type=Path, default=DEFAULT_CANDIDATE_SET)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_path = args.report.expanduser().resolve()
    candidate_set_path = args.candidate_set.expanduser().resolve()
    robot = args.robot_urdf.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (report_path, candidate_set_path, robot):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = json.loads(candidate_set_path.read_text(encoding="utf-8"))[
        "screening_evidence"
    ]
    candidates = report["candidates"]
    # This figure is intentionally a *contact-policy* comparison.  The task
    # screen's row-level ``passed`` also contains table/socket gates and must
    # not be presented as a BODex contact-policy label.
    passed = [
        item for item in candidates
        if item["contact_policy"]["status"] == "sampled_pass"
    ]
    failed = [
        item for item in candidates
        if item["contact_policy"]["status"] != "sampled_pass"
    ]

    # Interleave labels so the image communicates a comparison rather than
    # presenting two disconnected blocks.
    pass_cells = min(12, len(passed))
    fail_cells = 25 - pass_cells
    passed_sample = _spread(passed, pass_cells)
    failed_sample = _spread(failed, fail_cells)
    selected: list[dict] = []
    pass_index = 0
    fail_index = 0
    while len(selected) < 25:
        if fail_index < len(failed_sample):
            selected.append(failed_sample[fail_index])
            fail_index += 1
        if pass_index < len(passed_sample) and len(selected) < 25:
            selected.append(passed_sample[pass_index])
            pass_index += 1

    source_scene = Path(report["scene"])
    meshes = output / "source_meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    cells = []
    for item in selected:
        candidate_id = str(item["candidate"])
        candidate_dir = source_scene / candidate_id
        required = (
            candidate_dir / "wrist_se3.npy",
            candidate_dir / "grasp_pose.npy",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        hand_mesh = meshes / f"candidate_{candidate_id}.ply"
        if not hand_mesh.is_file():
            _export_hand(candidate_dir, robot, "grasp_pose.npy", hand_mesh)
        cells.append({
            "candidate": candidate_id,
            "passed": item["contact_policy"]["status"] == "sampled_pass",
            "reason": _failure_reason(item),
            "candidate_dir": str(candidate_dir),
            "hand_mesh": str(hand_mesh),
            "contact_policy": item["contact_policy"],
            "environment_passed": bool(item["environment"]["passed"]),
        })

    manifest = {
        "schema_version": 1,
        "kind": "bodex_contact_policy_5x5_sample",
        "labels_are_from": str(report_path),
        "selection": {
            "cell_count": 25,
            "passed_cells": pass_cells,
            "failed_cells": fail_cells,
            "method": "evenly-spaced deterministic sample, interleaved fail/pass",
        },
        "screening_evidence": {
            "raw_bodex_proposals": int(evidence["raw_bodex_proposals"]),
            "screened_candidates": int(report["candidate_count"]),
            "passed": len(passed),
            "failed": len(failed),
            "screen_scope": (
                "sampled whole-hand handle-contact policy only; not task "
                "trajectory or physical success"
            ),
        },
        "cells": cells,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
