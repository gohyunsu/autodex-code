#!/usr/bin/env python3
"""Read-only integrity summary of one stopped camera chunk-timestamp journal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.camera_chunk_tap import verify_chunk_journal  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--camera-serial", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_chunk_journal(
        args.journal, camera_serial=args.camera_serial),
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
