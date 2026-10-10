#!/usr/bin/env python3
"""Read-only fit audit of one camera's chunk ticks against external UTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.camera_chunk_clock import (  # noqa: E402
    evaluate_chunk_utc_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_chunk_utc_calibration(args.calibration),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
