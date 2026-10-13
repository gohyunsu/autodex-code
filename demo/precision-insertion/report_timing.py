#!/usr/bin/env python3
"""Summarize one precision-insertion live run's stock AutoDex trace."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for module_root in (str(REPO_ROOT), str(DEMO_DIR)):
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

from precision_insertion.timing_profile import summarize_events  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_dir", type=Path,
                        help="the run's <output-dir>.timing directory")
    args = parser.parse_args(argv)
    events = args.timing_dir.expanduser() / "events.jsonl"
    if not events.is_file():
        parser.error(f"missing trace: {events}")
    summary = summarize_events(events)
    spans = summary["spans"]
    print(f"trace: {events}")
    print(f"completed spans: {len(spans)}; unfinished: "
          f"{summary['unfinished_span_count']}")
    print("\nOperation                         count  inclusive_s     self_s")
    for name, group in sorted(summary["groups"].items(),
                              key=lambda item: item[1]["self_s"], reverse=True):
        print(f"{name[:33]:33} {group['count']:5d} "
              f"{group['inclusive_s']:12.3f} {group['self_s']:10.3f}")
    by_stage = defaultdict(lambda: [0, 0.0])
    for row in spans:
        if row["name"] == "vlm.infer":
            attributes = row["attributes"]
            key = (attributes.get("stage", "unknown"),
                   attributes.get("camera_id", "unknown"))
            by_stage[key][0] += 1
            by_stage[key][1] += row["duration_s"]
    if by_stage:
        print("\nVLM stage / camera                 calls  total_s")
        for (stage, camera), (count, seconds) in sorted(
                by_stage.items(), key=lambda item: item[1][1], reverse=True):
            print(f"{stage[:18]:18} {str(camera)[:14]:14} "
                  f"{count:5d} {seconds:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
