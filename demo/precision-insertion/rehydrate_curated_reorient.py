#!/usr/bin/env python3
"""Rebind curated square/cylinder reset seeds to a new AutoDex shared root."""

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
from precision_insertion.curated_reorient_relocation import (  # noqa: E402
    rehydrate_curated_reorient,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--source-shared-root", type=Path, required=True,
                        help="source or extracted handoff shared_data tree")
    parser.add_argument("--source-handoff-root", type=Path, required=True,
                        help="curated handoff parent of reset_12")
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--target-handoff-root", type=Path, required=True,
                        help="new noncanonical staging root on target PC")
    parser.add_argument("--archive-source-bound-scenes", action="store_true",
                        help="only archive target scene files byte-identical "
                             "to source-bound scenes, then regenerate paths")
    args = parser.parse_args(argv)
    report = rehydrate_curated_reorient(
        source_shared_root=args.source_shared_root,
        source_handoff_root=args.source_handoff_root,
        target_shared_root=args.target_shared_root,
        target_handoff_root=args.target_handoff_root,
        mode=select_mode(args.mode, args.gap_mm),
        archive_source_bound_scenes=args.archive_source_bound_scenes,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
