#!/usr/bin/env python3
"""Independent precision-insertion entry point; no robot connection.

Robot execution will be added only after the live-path preflight and guarded
controller are commissioned. The endpoint screen excludes arm trajectories.
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
    endpoint = command.add_parser(
        "screen-endpoint",
        help="offline exact-mesh grasp-only 20 mm endpoint screen",
    )
    endpoint.add_argument("--shared-root", type=Path, required=True)
    endpoint.add_argument("--mode", choices=("square", "cylinder"), required=True)
    endpoint.add_argument("--gap-mm", type=float, required=True)
    endpoint.add_argument("--candidate-dir", type=Path, required=True)
    endpoint.add_argument("--min-hand-clearance-mm", type=float, required=True)
    endpoint.add_argument(
        "--output", type=Path,
        help="optional JSON report path; refuses to overwrite an existing file",
    )
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            mode = select_mode(args.mode, args.gap_mm)
        except ValueError as exc:
            parser.error(str(exc))
        report = audit_assets(args.shared_root, mode)
        print(json.dumps(report, indent=2))
        return 0 if report["file_inputs_present"] else 2
    if args.command == "screen-endpoint":
        from precision_insertion.endpoint import screen_grasp_endpoint

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = screen_grasp_endpoint(
                shared_root=args.shared_root, mode=mode,
                candidate_dir=args.candidate_dir,
                minimum_hand_clearance_m=args.min_hand_clearance_mm / 1000.0,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        payload = json.dumps(report, indent=2) + "\n"
        if args.output is not None:
            target = args.output.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0 if report["endpoint_pass"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
