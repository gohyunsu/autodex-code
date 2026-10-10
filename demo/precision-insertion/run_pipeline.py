#!/usr/bin/env python3
"""Independent precision-insertion entry point; audit is read-only.

No command in this module connects to a robot. Robot execution will be added
only after the complete task preflight and guarded controller are commissioned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from precision_insertion.assets import audit_assets
from precision_insertion.config import select_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    audit = command.add_parser("audit", help="read-only v8 asset readiness report")
    audit.add_argument("--shared-root", type=Path, required=True)
    audit.add_argument("--mode", choices=("square", "cylinder"), required=True)
    audit.add_argument("--gap-mm", type=float, required=True)
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            mode = select_mode(args.mode, args.gap_mm)
        except ValueError as exc:
            parser.error(str(exc))
        report = audit_assets(args.shared_root, mode)
        print(json.dumps(report, indent=2))
        return 0 if report["file_inputs_present"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
