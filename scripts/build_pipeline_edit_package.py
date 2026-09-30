#!/usr/bin/env python3
"""Rebuild editor markers/layout after optional external-video sync."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autodex.pipeline_edit import build_edit_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Pipeline run directory")
    parser.add_argument("--sync", default=None,
                        help="Optional external_video_sync.json override")
    args = parser.parse_args()
    package = build_edit_package(args.run, sync_path=args.sync)
    print(f"edit package: {args.run}/edit "
          f"({len(package['segments'])} timeline segments)")


if __name__ == "__main__":
    main()
