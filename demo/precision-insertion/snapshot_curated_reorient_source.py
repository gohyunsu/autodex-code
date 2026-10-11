#!/usr/bin/env python3
"""Save the original v8 key and scenes beside a curated reset handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reorient_source_snapshot import (  # noqa: E402
    snapshot_curated_reorient_source,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True,
                        help="curated parent of reset_12")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new source_shared_data directory; no overwrite")
    args = parser.parse_args(argv)
    result = snapshot_curated_reorient_source(
        shared_root=args.shared_root, handoff_root=args.handoff_root,
        snapshot_root=args.output_root,
        mode=select_mode(args.mode, args.gap_mm),
    )
    print(json.dumps({"snapshot_root": result["snapshot_root"],
                      "verified_seed_count": result["verified_seed_count"],
                      "robot_ready": result["robot_ready"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
