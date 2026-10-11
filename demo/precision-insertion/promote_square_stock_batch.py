#!/usr/bin/env python3
"""Preview or append 20 mm endpoint-screened full-key square v8 grasps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from precision_insertion.config import select_mode
from precision_insertion.square_stock_promotion import promote_screened_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True,
                        choices=(0.3, 0.5, 1.0, 1.5))
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--candidate-id-offset", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="append candidates after complete preflight; default is read-only")
    args = parser.parse_args()
    report = promote_screened_batch(
        shared_root=args.shared_root, mode=select_mode("square", args.gap_mm),
        screen_root=args.screen_root,
        candidate_id_offset=args.candidate_id_offset,
        output_manifest=args.output_manifest, install=args.install)
    print(json.dumps({
        "eligible_candidates": len(report["rows"]),
        "installed": report["installed"],
        "manifest": str(args.output_manifest) if report["installed"] else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
