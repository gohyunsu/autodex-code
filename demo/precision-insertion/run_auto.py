#!/usr/bin/env python3
"""Open a frozen AutoDex precision-insertion session; never command motors.

Acquire board and socket images first with ``start_precision_session`` and
commissioned camera-time adapters.  This CLI validates that saved evidence,
the canonical FoundPose assets and a complete v8 endpoint catalog, then
opens a new evidence-only SessionRunner for subsequent trial integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.live_session_runner import open_verified_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--session-evidence", type=Path, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-xy-retries", type=int, required=True)
    args = parser.parse_args(argv)
    opened = open_verified_session(
        mode=select_mode(args.mode, args.gap_mm),
        shared_root=args.shared_root, evidence_dir=args.session_evidence,
        catalog_path=args.endpoint_catalog, output_dir=args.output_dir,
        max_xy_retries=args.max_xy_retries)
    print(json.dumps({
        "session_run_dir": str(opened.runner.output_dir),
        "source_binding": str(opened.source_binding_path),
        "next_decision": opened.next_decision.to_record(),
        "robot_ready": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
