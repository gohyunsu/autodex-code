#!/usr/bin/env python3
"""Audit saved ZeroDex runs against the thesis U1 evidence contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
DEMO = REPOSITORY / "demo" / "precision-insertion"
sys.path.insert(0, str(DEMO))

from precision_insertion.undergraduate_evidence import build_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--extra-episode-root", action="append", type=Path,
                        default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.run_root, run_ids=args.run_ids,
        extra_episode_roots=args.extra_episode_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
