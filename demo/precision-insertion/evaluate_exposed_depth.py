#!/usr/bin/env python3
"""Compare saved VLM/CAD depth with independent synchronized measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.calibration import load_session_calibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.exposed_depth_eval import (  # noqa: E402
    evaluate_exposed_depth_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--session", type=Path, required=True,
                        help="saved ChArUco/socket calibration for every sample")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report; refuses to overwrite")
    args = parser.parse_args(argv)
    mode = select_mode(args.mode, args.gap_mm)
    root = args.shared_root.expanduser().resolve()
    session = load_session_calibration(args.session, mode=mode, shared_root=root)
    report = evaluate_exposed_depth_manifest(
        args.manifest, mode=mode, shared_root=root, calibration=session)
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
