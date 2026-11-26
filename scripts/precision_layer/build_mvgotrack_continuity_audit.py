#!/usr/bin/env python3
"""Build an offline pickup-to-precontact MV-GoTrack continuity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
DEMO = REPOSITORY / "demo" / "precision-insertion"
sys.path.insert(0, str(DEMO))

from precision_insertion.undergraduate_evidence import (  # noqa: E402
    compile_continuity_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("pose_stream", type=Path)
    parser.add_argument("timing_events", type=Path)
    parser.add_argument("--attempt-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compile_continuity_audit(
        run_id=args.run_id, pose_stream=args.pose_stream,
        timing_events=args.timing_events, attempt_id=args.attempt_id)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output or (
        args.pose_stream.parent / "mvgotrack_continuity_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
