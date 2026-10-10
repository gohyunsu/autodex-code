#!/usr/bin/env python3
"""Independent precision-insertion entry point; no robot connection.

Robot execution will be added only after the live-path preflight and guarded
controller are commissioned. The saved-trial planner never sends commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.assets import audit_assets
from precision_insertion.config import select_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    audit = command.add_parser("audit", help="read-only v8 asset readiness report")
    audit.add_argument("--shared-root", type=Path, required=True)
    audit.add_argument("--mode", choices=("square", "cylinder"), required=True)
    audit.add_argument("--gap-mm", type=float, required=True)
    reorient_audit = command.add_parser(
        "audit-reorient", help="read-only v8 reset scene/seed readiness report",
    )
    reorient_audit.add_argument("--shared-root", type=Path, required=True)
    reorient_audit.add_argument("--mode", choices=("square", "cylinder"), required=True)
    reorient_audit.add_argument("--gap-mm", type=float, required=True)
    reorient_audit.add_argument(
        "--candidate-root", type=Path,
        help="optional local handoff root containing reset_<h>/; not canonical NAS",
    )
    reorient_audit.add_argument("--output", type=Path,
                                help="optional new JSON report; no overwrite")
    reorient_scenes = command.add_parser(
        "prepare-reorient-scenes",
        help="generate missing v8 BODex proposal scenes, not executable reset paths",
    )
    reorient_scenes.add_argument("--shared-root", type=Path, required=True)
    reorient_scenes.add_argument("--mode", choices=("square", "cylinder"),
                                 required=True)
    reorient_scenes.add_argument("--gap-mm", type=float, required=True)
    reorient_scenes.add_argument("--manifest", type=Path, required=True,
                                 help="new output manifest path; no overwrite")
    reorient_scenes.add_argument("--height-cm", type=int, action="append",
                                 help="release-height subset; default 0,4,8,12")
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
    trial = command.add_parser(
        "preflight-trial",
        help="offline replay of one saved key observation through v8 planning",
    )
    trial.add_argument("--shared-root", type=Path, required=True)
    trial.add_argument("--mode", choices=("square", "cylinder"), required=True)
    trial.add_argument("--gap-mm", type=float, required=True)
    trial.add_argument("--session", type=Path, required=True,
                       help="saved session calibration with frozen scene snapshot")
    trial.add_argument("--catalog", type=Path, required=True,
                       help="complete endpoint catalogue for this socket")
    trial.add_argument("--key-pose-world-npy", type=Path, required=True,
                       help="fresh FoundPose 4x4 pose in calibrated world frame")
    trial.add_argument("--key-observation-id", required=True)
    trial.add_argument("--key-capture-time-s", type=float, required=True,
                       help="actual image-acquisition timestamp, same clock as session")
    trial.add_argument("--live-start-q-npy", type=Path, required=True,
                       help="saved measured 13-DOF FR3/Inspire start joints")
    trial.add_argument("--start-q-time-s", type=float, required=True,
                       help="joint sample timestamp on the key exposure clock")
    trial.add_argument("--max-key-state-skew-s", type=float, required=True)
    trial.add_argument("--limits-json", type=Path, required=True,
                       help="commissioned PathAuditLimits fields")
    trial.add_argument("--max-pose-error-deg", type=float, required=True)
    trial.add_argument("--axial-waypoint-step-mm", type=float, required=True)
    trial.add_argument("--attempted", action="append", default=[],
                       metavar="TYPE/SID/GID")
    trial.add_argument("--covered-scene", type=int, action="append", default=[])
    trial.add_argument("--max-candidate-attempts", type=int,
                       help="pilot prefix; never report pose exhaustion")
    trial.add_argument("--max-reset-drift-mm", type=float,
                       help="commissioned key-in-hand reset drift limit")
    trial.add_argument("--max-reset-axis-tilt-deg", type=float,
                       help="commissioned symmetry-reduced reset tilt limit")
    trial.add_argument("--reset-candidate-root", type=Path,
                       help="optional handoff parent of reset_<h> directories")
    trial.add_argument("--attempted-reset", action="append", default=[],
                       metavar="HEIGHT/TARGET_STEM/SEED_ID")
    trial.add_argument("--output-dir", type=Path, required=True,
                       help="new report directory; refuses to overwrite")
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            mode = select_mode(args.mode, args.gap_mm)
        except ValueError as exc:
            parser.error(str(exc))
        report = audit_assets(args.shared_root, mode)
        print(json.dumps(report, indent=2))
        return 0 if report["file_inputs_present"] else 2
    if args.command == "audit-reorient":
        from precision_insertion.reorient_assets import audit_v8_reorient_assets

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = audit_v8_reorient_assets(
                shared_root=args.shared_root, mode=mode,
                candidate_root=args.candidate_root)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        payload = json.dumps(report, indent=2) + "\n"
        if args.output is not None:
            target = args.output.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    if args.command == "prepare-reorient-scenes":
        from precision_insertion.reorient_assets import prepare_v8_reorient_scenes
        from autodex.utils.path import RESET_RELEASE_HEIGHTS_CM

        try:
            mode = select_mode(args.mode, args.gap_mm)
            report = prepare_v8_reorient_scenes(
                shared_root=args.shared_root, mode=mode,
                manifest_path=args.manifest,
                heights_cm=tuple(args.height_cm) if args.height_cm
                else RESET_RELEASE_HEIGHTS_CM)
        except (FileExistsError, FileNotFoundError, KeyError, TypeError,
                ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            "manifest": str(args.manifest.expanduser().resolve()),
            "new_scene_count": report["new_scene_count"],
            "directed_scene_count": report["directed_scene_count"],
            "scope": report["scope"], "robot_ready": False,
        }, indent=2))
        return 0
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
    if args.command == "preflight-trial":
        if args.output_dir.expanduser().resolve().exists():
            parser.error("preflight output directory already exists")
        from precision_insertion.calibration import load_session_calibration
        from precision_insertion.path_audit import PathAuditLimits
        from precision_insertion.trial_preflight import (
            plan_fresh_key_trial, write_trial_preflight_artifacts,
        )
        import numpy as np

        try:
            mode = select_mode(args.mode, args.gap_mm)
            session = load_session_calibration(
                args.session, mode=mode, shared_root=args.shared_root)
            catalog_data = json.loads(args.catalog.read_text(encoding="utf-8"))
            limits_data = json.loads(args.limits_json.read_text(encoding="utf-8"))
            if not isinstance(catalog_data, dict) or not isinstance(
                    limits_data, dict):
                raise ValueError("catalog and limits files must be JSON objects")
            limits = PathAuditLimits(**limits_data)
            limits.validate()
            key_pose = np.load(args.key_pose_world_npy, allow_pickle=False)
            start_q = np.load(args.live_start_q_npy, allow_pickle=False)
            attempted = tuple(tuple(item.split("/")) for item in args.attempted)
            if any(len(item) != 3 or not all(item) for item in attempted):
                raise ValueError("attempted keys must be TYPE/SID/GID")
            attempted_reset = tuple(tuple(item.split("/"))
                                    for item in args.attempted_reset)
            if any(len(item) != 3 or not all(item) for item in attempted_reset):
                raise ValueError(
                    "attempted reset keys must be HEIGHT/TARGET_STEM/SEED_ID")
            from autodex.planner import GraspPlanner

            planner = GraspPlanner(hand="fr3_inspire")
            result = plan_fresh_key_trial(
                planner=planner, mode=mode, shared_root=args.shared_root,
                calibration=session, catalog=catalog_data,
                key_pose_world=key_pose,
                key_observation_id=args.key_observation_id,
                key_capture_timestamp_s=args.key_capture_time_s,
                live_start_q=start_q,
                start_q_acquisition_timestamp_s=args.start_q_time_s,
                max_key_state_skew_s=args.max_key_state_skew_s,
                limits=limits,
                max_pose_error_deg=args.max_pose_error_deg,
                axial_waypoint_step_m=args.axial_waypoint_step_mm / 1000.0,
                attempted=attempted,
                covered_scenes=tuple(args.covered_scene),
                max_candidate_attempts=args.max_candidate_attempts,
                max_reset_center_drift_m=(
                    None if args.max_reset_drift_mm is None
                    else args.max_reset_drift_mm / 1000.0),
                max_reset_axis_tilt_deg=args.max_reset_axis_tilt_deg,
                reset_candidate_root=args.reset_candidate_root,
                attempted_reset=attempted_reset,
            )
            output = write_trial_preflight_artifacts(result, args.output_dir)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({
            "status": result.status,
            "attempted_candidates": len(result.attempted_candidates),
            "repose_target_stems": result.repose_target_stems,
            "repose_assessment_status": (
                None if result.repose_assessment is None else
                result.repose_assessment["status"]),
            "report": str(output / "report.json"),
            "robot_ready": False,
        }, indent=2))
        return 0 if result.status == "sampled_planning_pass" else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
