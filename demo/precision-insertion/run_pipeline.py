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
    xy = command.add_parser(
        "screen-xy-endpoint",
        help="read-only exact-mesh 1 mm XY retry endpoint pre-filter",
    )
    xy.add_argument("--shared-root", type=Path, required=True)
    xy.add_argument("--mode", choices=("square", "cylinder"), required=True)
    xy.add_argument("--gap-mm", type=float, required=True)
    xy.add_argument("--candidate-dir", type=Path, required=True)
    xy.add_argument("--current-x-mm", type=float, required=True)
    xy.add_argument("--current-y-mm", type=float, required=True)
    xy.add_argument("--max-total-mm", type=float, required=True)
    xy.add_argument("--min-hand-clearance-mm", type=float, required=True)
    xy.add_argument("--output", type=Path,
                    help="optional new JSON path; refuses to overwrite")
    catalog = command.add_parser(
        "screen-catalog", help="screen all current pose-indexed v8 grasp endpoints",
    )
    catalog.add_argument("--shared-root", type=Path, required=True)
    catalog.add_argument("--mode", choices=("square", "cylinder"), required=True)
    catalog.add_argument("--gap-mm", type=float, required=True)
    catalog.add_argument("--min-hand-clearance-mm", type=float, required=True)
    catalog.add_argument("--max-candidates", type=int,
                         help="pilot prefix only; output will be marked incomplete")
    catalog.add_argument("--output", type=Path, required=True,
                         help="new JSON path outside candidate geometry; no overwrite")
    select = command.add_parser(
        "select-catalog", help="read-only pose-conditioned offline candidate list",
    )
    select.add_argument("--catalog", type=Path, required=True)
    select.add_argument("--mode", choices=("square", "cylinder"), required=True)
    select.add_argument("--gap-mm", type=float, required=True)
    select.add_argument("--pose-stem", required=True)
    select.add_argument("--attempted", action="append", default=[],
                        metavar="TYPE/SID/GID")
    select.add_argument("--covered-scene", type=int, action="append", default=[])
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
    if args.command == "screen-xy-endpoint":
        from precision_insertion.xy_endpoint import (
            screen_axis_1mm_endpoint_choices,
        )

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = screen_axis_1mm_endpoint_choices(
                shared_root=args.shared_root, mode=mode,
                candidate_dir=args.candidate_dir,
                current_offset_socket_m=(
                    args.current_x_mm / 1000.0,
                    args.current_y_mm / 1000.0,
                ),
                max_total_offset_m=args.max_total_mm / 1000.0,
                minimum_hand_clearance_m=(
                    args.min_hand_clearance_mm / 1000.0),
            )
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        payload = json.dumps(report, indent=2) + "\n"
        if args.output is not None:
            target = args.output.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(payload)
            print(json.dumps({
                "report": str(target),
                "endpoint_clear_choice_ids": report["endpoint_clear_choice_ids"],
                "robot_ready": False,
            }, indent=2))
        else:
            print(payload, end="")
        return 0 if report["endpoint_clear_choice_ids"] else 2
    if args.command == "screen-catalog":
        from precision_insertion.candidates import build_endpoint_catalog

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = build_endpoint_catalog(
                shared_root=args.shared_root, mode=mode,
                minimum_hand_clearance_m=args.min_hand_clearance_mm / 1000.0,
                max_candidates=args.max_candidates,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({
            "catalog": str(target),
            "complete_scan": report["complete_scan"],
            "screened_directories": report["screened_directories"],
            "eligible_count": report["eligible_count"],
            "errors": report["errors"],
            "robot_ready": False,
        }, indent=2))
        return 0 if report["complete_scan"] else 2
    if args.command == "select-catalog":
        from precision_insertion.candidates import select_pose_candidates

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = json.loads(args.catalog.read_text(encoding="utf-8"))
            attempted = [tuple(value.split("/")) for value in args.attempted]
            result = select_pose_candidates(
                report, expected_mode=mode, tabletop_pose_stem=args.pose_stem,
                attempted=attempted, covered_scenes=args.covered_scene,
            )
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "candidates_available" else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
