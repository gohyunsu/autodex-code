#!/usr/bin/env python3
"""Summarize independent physical pickups of one eligible insertion grasp."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.insertion_grasp_calibration import (  # noqa: E402
    build_insertion_grasp_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--family", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True,
                        help="exact v8 scene_type/scene_id/grasp_id")
    parser.add_argument("--sample", type=Path, action="append", required=True,
                        help="repeat for independent physical pickup JSON files")
    parser.add_argument("--minimum-independent-trials", type=int, default=5)
    parser.add_argument("--max-nominal-translation-drift-mm", type=float,
                        required=True, help="SE(3) medoid selection scale")
    parser.add_argument("--max-nominal-rotation-drift-deg", type=float,
                        required=True, help="SE(3) medoid selection scale")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="mode/grasp-specific directories are created here")
    args = parser.parse_args(argv)
    result = build_insertion_grasp_calibration(
        shared_root=args.shared_root,
        mode=select_mode(args.family, args.gap_mm),
        catalog_path=args.endpoint_catalog, candidate_id=args.candidate_id,
        sample_paths=args.sample, output_root=args.output_root,
        minimum_independent_trials=args.minimum_independent_trials,
        max_nominal_translation_drift_m=(
            args.max_nominal_translation_drift_mm / 1000.),
        max_nominal_rotation_drift_deg=args.max_nominal_rotation_drift_deg)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
