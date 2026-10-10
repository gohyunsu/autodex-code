#!/usr/bin/env python3
"""Non-overwriting installation of one fully screened square v8 scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.config import select_mode
from precision_insertion.square_promotion import promote_square_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--pass-scene", type=Path, required=True)
    parser.add_argument("--highres-report", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--simulation-version", required=True)
    parser.add_argument("--min-hand-clearance-mm", type=float, required=True)
    args = parser.parse_args()
    try:
        result = promote_square_scene(
            shared_root=args.shared_root, mode=select_mode("square", args.gap_mm),
            source_scene=args.source_scene, pass_scene=args.pass_scene,
            highres_report=args.highres_report, output_scene=args.output_scene,
            simulation_version=args.simulation_version,
            minimum_hand_clearance_m=args.min_hand_clearance_mm / 1000.0)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
