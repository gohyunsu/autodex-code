#!/usr/bin/env python3
"""Render final extrinsic or hand-eye reprojection debug artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration.reprojection_debug import (
    render_extrinsic_debug,
    render_handeye_debug,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("extrinsic", "handeye"))
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-calibration-dir", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--c2r", type=Path)
    parser.add_argument("--ee-board", type=Path)
    parser.add_argument(
        "--grid-only", action="store_true",
        help="save grids/GIF/report without duplicate full-resolution views")
    args = parser.parse_args(argv)
    if args.kind == "extrinsic":
        if args.intrinsics is None or args.extrinsics is None:
            parser.error("extrinsic rendering needs --intrinsics and --extrinsics")
        report = render_extrinsic_debug(
            session_dir=args.session_dir,
            board_config=args.board_config,
            intrinsics_path=args.intrinsics,
            extrinsics_path=args.extrinsics,
            output_dir=args.output_dir,
            save_per_view=not args.grid_only)
    else:
        if (args.camera_calibration_dir is None or args.c2r is None or
                args.ee_board is None):
            parser.error(
                "handeye rendering needs --camera-calibration-dir, --c2r, "
                "and --ee-board")
        report = render_handeye_debug(
            session_dir=args.session_dir,
            board_config=args.board_config,
            camera_calibration_dir=args.camera_calibration_dir,
            c2r_path=args.c2r,
            ee_board_path=args.ee_board,
            output_dir=args.output_dir,
            save_per_view=not args.grid_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
