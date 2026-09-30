#!/usr/bin/env python3
"""Regenerate matrices and plots from a completed or interrupted run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validation.planning.pipeline_lift_reachability.analysis import analyze_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary = analyze_run(args.run_dir)
    print(f"[analysis] {Path(args.run_dir) / 'summary.json'}")
    print(f"[analysis] grasps={summary['base_grasp_count']} cells={summary['cell_count']} "
          f"endpoint-pass/lift-fail={summary['both_endpoint_pass_lift_fail_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
