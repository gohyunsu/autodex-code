#!/usr/bin/env python3
"""Collect a ranked, experiment-private library of table-only lift grasps.

The object is always placed at the measured Charuco-proxy centre.  Existing
v8 candidate geometry and coverage metadata are read-only; outcomes are stored
under ``experiment/<exp_name>`` exactly like an isolated AutoDex campaign.
No robot executor is created by this program.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
import sys
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from autodex.timing import TimingRecorder
from src.demo.lift_test import run_session as session
from src.demo.lift_test.board import load_proxy
from src.demo.lift_test.campaign_state import (
    TERMINAL_STATUSES,
    apply_candidate_outcome,
    campaign_paths,
    candidate_key,
    choose_next_candidate,
    create_or_resume_progress,
    determine_status,
    key_string,
    load_coverage_records,
    write_candidate_outcome,
    write_json_atomic,
)
from src.demo.lift_test.candidate_policy import make_candidate_policy
from src.demo.lift_test.execution_trajectory import profile_for_arm
from src.demo.lift_test.jacobian_lift import LiftOptions, options_as_dict, save_step_csv


def _resolve_object(value: str | None, *, hand: str, version: str) -> str:
    if value is None:
        return session._prompt_object(hand, version, None)
    choices = session._candidate_objects(hand, version)
    if value not in choices:
        raise ValueError(f"unknown object {value!r}; available: {', '.join(choices)}")
    return value


def _resolve_tabletop(value: str | None, *, obj: str, version: str) -> tuple[int, Path]:
    if value is None:
        return session._prompt_tabletop_pose(obj, version, None)
    stem = str(value).removesuffix(".npy")
    for index, path in enumerate(session._tabletop_files(obj, version)):
        if path.stem == stem:
            return index, path
    available = ", ".join(path.stem for path in session._tabletop_files(obj, version))
    raise ValueError(f"unknown tabletop pose {value!r}; available: {available}")


def _span_sum(trace: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    wanted = set(names)
    out = {name: 0.0 for name in names}
    for span in trace.get("spans", []):
        name = str(span.get("name"))
        if name in wanted:
            out[name] += float(span.get("duration_s", 0.0))
    return {name: round(value, 6) for name, value in out.items() if value > 0.0}


def _group_catalogue(catalogue: session.PreparedCandidateCatalogue,
                     key: Sequence[Any], marginal_gain: int) -> tuple[session.PreparedCandidateCatalogue, list[int]]:
    wanted = candidate_key(key)
    indices = [index for index, info in enumerate(catalogue.scene_info)
               if candidate_key(info) == wanted]
    source = {
        **catalogue.source_info,
        "ordered_keys": [list(wanted)],
        "n_coverage_candidates": 1,
        "n_coverage_useful": 1,
        "n_coverage_zero": 0,
        "remaining_coverage_by_key": {key_string(wanted): int(marginal_gain)},
        "selection_mode": "training_private_coverage_greedy",
        "n_candidate_records_after_symmetry_expansion": len(indices),
    }
    return session.PreparedCandidateCatalogue(
        obj=catalogue.obj, version=catalogue.version, hand=catalogue.hand,
        pose_stem=catalogue.pose_stem, source_info=source,
        wrist_object=np.asarray(catalogue.wrist_object[indices]),
        pregrasp=np.asarray(catalogue.pregrasp[indices]),
        grasp=np.asarray(catalogue.grasp[indices]),
        openpose=[catalogue.openpose[index] for index in indices],
        scene_info=[catalogue.scene_info[index] for index in indices],
    ), indices


def _save_verified_candidate(path: Path, *, catalogue: session.PreparedCandidateCatalogue,
                             global_index: int, variant_ordinal: int) -> None:
    approach_hand = catalogue.openpose[global_index]
    if approach_hand is None:
        approach_hand = catalogue.pregrasp[global_index]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int32),
        candidate_key=np.asarray(candidate_key(catalogue.scene_info[global_index])),
        variant_ordinal=np.array(int(variant_ordinal), dtype=np.int32),
        wrist_object=np.asarray(catalogue.wrist_object[global_index], dtype=np.float64),
        pregrasp_qpos=np.asarray(catalogue.pregrasp[global_index], dtype=np.float32),
        approach_hand_qpos=np.asarray(approach_hand, dtype=np.float32),
        grasp_hand_qpos=np.asarray(catalogue.grasp[global_index], dtype=np.float32),
    )


def _candidate_state_payload(*, progress: Mapping[str, Any], attempt: Mapping[str, Any],
                             selected: Mapping[str, Any]) -> dict[str, Any]:
    success = bool(attempt["success"])
    return {
        "schema_version": 1,
        "success": success,
        "status": "planning_feasible" if success else "terminal_planning_failure",
        "verification_level": "planning_feasible",
        "arm": progress["contract"]["arm"],
        "hand": progress["contract"]["hand"],
        "grasp_version": progress["contract"]["grasp_version"],
        "object": progress["contract"]["object"],
        "tabletop_pose_stem": progress["contract"]["tabletop_pose_stem"],
        "candidate_key": list(candidate_key(selected["key"])),
        "covers": list(selected.get("covers", [])),
        "success_rank": attempt.get("success_rank"),
        "failure_code": attempt.get("failure_code"),
        "all_symmetry_variants_failed": not success,
        "trial": attempt["trial"],
        "planning_wall_s": attempt["planning_wall_s"],
        "updated_at": attempt["completed_at"],
    }


def _sync_private_state(progress: Mapping[str, Any], state_root: Path) -> None:
    """Repair candidate mirrors from the atomic coverage-progress index."""
    verified = {key_string(item["candidate_key"]): item
                for item in progress.get("verified_grasps", [])}
    attempts = {int(item["attempt_index"]): item for item in progress.get("attempts", [])}
    for key_text, item in verified.items():
        attempt = next((row for row in attempts.values()
                        if row.get("success_rank") == item.get("success_rank")), None)
        if attempt is None:
            continue
        selected = {"key": item["candidate_key"], "covers": item.get("covers", [])}
        write_candidate_outcome(
            state_root=state_root, key=key_text.split("/"),
            payload=_candidate_state_payload(progress=progress, attempt=attempt, selected=selected))
    for key_text, item in progress.get("terminal_failures", {}).items():
        attempt = next((row for row in reversed(list(attempts.values()))
                        if key_string(row["candidate_key"]) == key_text and not row["success"]), None)
        if attempt is None:
            continue
        write_candidate_outcome(
            state_root=state_root, key=key_text.split("/"),
            payload=_candidate_state_payload(
                progress=progress, attempt=attempt,
                selected={"key": item["candidate_key"],
                          "covers": item.get("covers", [])}))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", "--exp_name", default=None,
                        help="private experiment name (default: lift_training_<arm>)")
    parser.add_argument("--object", help="omit to select interactively")
    parser.add_argument("--tabletop-pose", help="tabletop filename stem; omit to select")
    parser.add_argument("--arm", choices=sorted(session.ARM_TO_PLANNER_ROBOT), default="franka")
    parser.add_argument("--hand", choices=["inspire"], default="inspire")
    parser.add_argument("--grasp-version", "--grasp_version", choices=["v8"], default="v8")
    parser.add_argument("--board-source", choices=["live-charuco", "file"], default="live-charuco")
    parser.add_argument("--board-json", type=Path)
    parser.add_argument("--board-tolerance-m", type=float, default=0.005)
    parser.add_argument("--max-consecutive-failures", type=int, default=20,
                        help="0 disables the stalled-training heuristic")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="stop this process after N new candidates; 0 runs to campaign stop")
    parser.add_argument("--lift-height-m", type=float, default=0.10)
    parser.add_argument("--lift-step-m", type=float, default=0.005)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=24)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.10)
    parser.add_argument("--max-segment-joint-delta-rad", type=float, default=0.02)
    parser.add_argument("--execution-dt-s", type=float, default=0.01)
    parser.add_argument("--squeeze-duration-s", type=float, default=0.50)
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    parser.add_argument("--pc-list", nargs="+", default=None)
    parser.add_argument("--calib-dir")
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    parser.add_argument("--port-snap", type=int, default=5009)
    parser.add_argument("--port-snap-cmd", type=int, default=6894)
    parser.add_argument("--snapshot-timeout-s", type=float, default=5.0)
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--stream-warmup-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.board_source == "file" and args.board_json is None:
        raise SystemExit("--board-json is required with --board-source file")
    if args.max_consecutive_failures < 0 or args.max_attempts < 0:
        raise SystemExit("failure and attempt limits must be non-negative")
    if args.board_tolerance_m < 0.0:
        raise SystemExit("--board-tolerance-m must be non-negative")
    args.exp_name = args.exp_name or f"lift_training_{args.arm}"
    session._load_runtime_dependencies()
    try:
        obj = _resolve_object(args.object, hand=args.hand, version=args.grasp_version)
        pose_index, pose_file = _resolve_tabletop(
            args.tabletop_pose, obj=obj, version=args.grasp_version)
    except session.UserQuit:
        return 0

    paths = campaign_paths(
        project_dir=session.project_dir, exp_name=args.exp_name, hand=args.hand,
        version=args.grasp_version, obj=obj)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = paths.session_root / stamp
    session_dir.mkdir(parents=True, exist_ok=False)
    session_trace = TimingRecorder()
    capture = None
    interrupted = False
    new_attempts = 0
    options = LiftOptions(
        height_m=args.lift_height_m, step_m=args.lift_step_m, damping=args.damping,
        max_iterations=args.max_iterations, max_joint_step_rad=args.max_joint_step_rad,
        max_segment_joint_delta_rad=args.max_segment_joint_delta_rad)
    profile = profile_for_arm(
        args.arm, sample_dt_s=args.execution_dt_s,
        squeeze_duration_s=args.squeeze_duration_s)

    try:
        board_span = session_trace.begin(
            phase="preparation", kind="check", name="board_measurement",
            source=args.board_source)
        if args.board_source == "live-charuco":
            capture = session.CaptureContext(args)
            print("[board] Clear Charuco board and press Enter to measure (q to quit).")
            session._input("  ready")
            measurement, proxy = capture.measure_board(session_dir)
        else:
            proxy = load_proxy(args.board_json)
            measurement = {"source": "loaded_board_proxy",
                           "table_surface_z_m": proxy["table_surface_z_m"]}
            session._write_json(session_dir / "board" / "board_proxy.json", proxy)
        session_trace.end(board_span, center_xy_m=proxy["center_xy_m"],
                          table_surface_z_m=proxy["table_surface_z_m"])

        records_span = session_trace.begin(
            phase="preparation", kind="io", name="coverage_asset_load")
        records = load_coverage_records(
            project_dir=session.project_dir, obj=obj, version=args.grasp_version,
            pose_stem=pose_file.stem)
        session_trace.end(records_span, candidate_count=len(records))
        progress = create_or_resume_progress(
            path=paths.progress_path, exp_name=args.exp_name, arm=args.arm,
            hand=args.hand, version=args.grasp_version, obj=obj,
            pose_stem=pose_file.stem, board_proxy=proxy,
            lift_options=options_as_dict(options), execution_profile=profile.as_dict(),
            max_consecutive_failures=args.max_consecutive_failures, records=records,
            board_tolerance_m=args.board_tolerance_m)
        _sync_private_state(progress, paths.candidate_state_root)
        if progress.get("status") in TERMINAL_STATUSES:
            session._write_json(session_dir / "session.json", {
                "schema_version": 1, "exp_name": args.exp_name,
                "session_stamp": stamp, "arm": args.arm, "hand": args.hand,
                "grasp_version": args.grasp_version, "object": obj,
                "tabletop_pose_stem": pose_file.stem, "scene": "table",
                "campaign_status": progress["status"],
                "stop_reason": progress.get("stop_reason"),
                "coverage_progress": str(paths.progress_path),
            })
            print(f"[training] campaign already stopped: {progress['status']} "
                  f"({progress.get('stop_reason')})")
            return 0

        scenario_span = session_trace.begin(
            phase="preparation", kind="setup", name="center_table_scenario")
        center_xy = np.asarray(proxy["center_xy_m"], dtype=np.float64)
        scenario = session._constructed_scenario(
            obj, args.grasp_version, center_xy, pose_index, pose_file, measurement)
        session_trace.end(scenario_span, object_xy_m=center_xy.tolist(), scene="table")
        session._write_json(session_dir / "scene_cfg.json", scenario["scene_cfg"])

        planner_span = session_trace.begin(
            phase="preparation", kind="setup", name="planner_initialization",
            arm=args.arm, cuda_graph=args.cuda_graph)
        planner_robot = session._planner_robot_for_arm(args.arm)
        print(f"[planner] initializing {planner_robot} (cuda_graph={args.cuda_graph})...", flush=True)
        planner = session.GraspPlanner(
            hand=planner_robot, use_cuda_graph=(args.cuda_graph == "on"))
        session_trace.end(planner_span)

        catalogue_span = session_trace.begin(
            phase="preparation", kind="io", name="candidate_catalogue_load")
        clean_policy = make_candidate_policy(
            "clean-state", clean_state_root=session_dir / "candidate_state_clean")
        catalogue = session.prepare_candidate_catalogue(
            obj=obj, version=args.grasp_version, candidate_hand=args.hand,
            pose_stem=pose_file.stem, candidate_policy=clean_policy)
        session_trace.end(catalogue_span, expanded_variant_count=len(catalogue.scene_info))
        session._write_json(session_dir / "session.json", {
            "schema_version": 1, "exp_name": args.exp_name, "session_stamp": stamp,
            "arm": args.arm, "planner_robot": planner_robot, "hand": args.hand,
            "grasp_version": args.grasp_version, "object": obj,
            "tabletop_pose_stem": pose_file.stem, "scene": "table",
            "object_xy_m": center_xy.tolist(), "board_proxy": proxy,
            "max_consecutive_failures": args.max_consecutive_failures,
            "lift_options": options_as_dict(options), "execution_profile": profile.as_dict(),
            "candidate_state_root": str(paths.candidate_state_root),
            "coverage_progress": str(paths.progress_path),
        })

        while progress.get("status") not in TERMINAL_STATUSES:
            if args.max_attempts and new_attempts >= args.max_attempts:
                print(f"[training] process attempt limit reached ({args.max_attempts}); "
                      "campaign remains active")
                break
            trace = TimingRecorder()
            selection_span = trace.begin(
                phase="planning", kind="decision", name="candidate_selection",
                covered_scene_count=len(progress["coverage"]["covered_scene_ids"]),
                remaining_scene_count=len(progress["coverage"]["remaining_scene_ids"]))
            selected = choose_next_candidate(records, progress)
            trace.end(selection_span, outcome="success" if selected else "failure",
                      candidate_key=None if selected is None else selected["key"],
                      marginal_gain=None if selected is None else selected["marginal_gain"])
            if selected is None:
                status, reason = determine_status(progress, records)
                progress["status"], progress["stop_reason"] = status, reason
                write_json_atomic(paths.progress_path, progress)
                break

            new_attempts += 1
            attempt_seq = int(progress.get("attempt_count", 0)) + 1
            episode = session._episode_dir(
                paths.experiment_root, args.hand, obj, stamp, attempt_seq)
            trial_rel = str(episode.relative_to(Path(session.project_dir)))
            group, global_indices = _group_catalogue(
                catalogue, selected["key"], selected["marginal_gain"])
            print(
                f"\n[training {attempt_seq}] key={key_string(selected['key'])} "
                f"gain={selected['marginal_gain']} variants={len(global_indices)} "
                f"coverage={len(progress['coverage']['covered_scene_ids'])}/"
                f"{progress['coverage']['scene_count']} "
                f"failure_streak={progress['consecutive_failures']}/"
                f"{args.max_consecutive_failures or 'off'}",
                flush=True)

            preflight_span = trace.begin(
                phase="planning", kind="plan", name="candidate_preflight",
                candidate_key=selected["key"], variant_count=len(global_indices))
            if not global_indices:
                planning = {
                    "success": False, "reason": "candidate_geometry_missing",
                    "failure_code": "candidate_catalogue_key_missing",
                    "candidate_source": group.source_info,
                    "candidate_metadata": [], "candidate_attempts": [],
                    "timing": {"total_s": 0.0},
                }
            else:
                planning = session._approach_and_lift(
                    planner, scene_cfg=scenario["scene_cfg"], obj=obj,
                    version=args.grasp_version, candidate_hand=args.hand,
                    pose_stem=pose_file.stem, options=options,
                    candidate_policy=clean_policy, candidate_budget=None,
                    timing=trace, timing_parent_id=preflight_span,
                    candidate_catalogue=group, execution_profile=profile)
            trace.end(preflight_span,
                      outcome="success" if planning["success"] else "failure",
                      failure_code=planning.get("failure_code"))
            planning_trace = trace.as_dict()
            stage_times = _span_sum(planning_trace, (
                "candidate_collision_filter", "candidate_endpoint_ik",
                "candidate_approach", "candidate_jacobian_lift",
                "candidate_execution_trajectory"))

            success = bool(planning["success"])
            local_variant = int(planning.get("candidate_index", 0)) if success else None
            global_variant = (global_indices[local_variant]
                              if success and local_variant is not None else None)
            verified_rel = (f"{trial_rel}/plan/verified_candidate.npz" if success else None)
            predicted_duration = None
            if success:
                predicted_duration = float(planning["execution"]["segments"]["total_duration_s"])
            attempt = apply_candidate_outcome(
                progress=progress, records=records, selected=selected,
                success=success, trial_relpath=trial_rel,
                failure_code=planning.get("failure_code"),
                variant_ordinal=local_variant, verified_artifact=verified_rel,
                planning_wall_s=float(planning.get("timing", {}).get("total_s", 0.0)),
                stage_time_s=stage_times,
                predicted_execution_duration_s=predicted_duration)

            artifact_span = trace.begin(
                phase="artifacts", kind="io", name="episode_planning_artifacts")
            session._write_json(episode / "request.json", {
                "schema_version": 1, "exp_name": args.exp_name,
                "object": obj, "arm": args.arm, "planner_robot": planner_robot,
                "hand": args.hand, "grasp_version": args.grasp_version,
                "scene": "table", "object_xy_m": center_xy.tolist(),
                "tabletop": {"idx": pose_index, "filename": pose_file.name},
                "candidate_key": selected["key"],
                "marginal_scene_ids": selected["marginal_scene_ids"],
                "lift_options": options_as_dict(options),
                "execution_profile": profile.as_dict(),
            })
            session._write_json(episode / "candidate_source.json", planning.get("candidate_source", {}))
            session._write_json(episode / "candidate_metadata.json", planning.get("candidate_metadata", []))
            session._write_json(episode / "candidate_attempts.json", planning.get("candidate_attempts", []))
            if planning.get("jacobian_steps"):
                save_step_csv(episode / "jacobian_steps.csv", planning["jacobian_steps"])
            if success and global_variant is not None:
                plan_dir = episode / "plan"
                plan_dir.mkdir(parents=True, exist_ok=True)
                np.save(plan_dir / "traj.npy", planning["approach"])
                np.save(plan_dir / "wrist_se3.npy", planning["wrist_se3"])
                session._save_jacobian_lift(plan_dir / "lift_jacobian.npz", planning, options)
                session._save_execution_trajectory(
                    plan_dir / "execution_trajectory.npz", planning)
                _save_verified_candidate(
                    plan_dir / "verified_candidate.npz", catalogue=catalogue,
                    global_index=global_variant, variant_ordinal=int(local_variant))
            trace.end(artifact_span)

            state_span = trace.begin(
                phase="artifacts", kind="io", name="coverage_progress_write")
            write_json_atomic(paths.progress_path, progress)
            trace.end(state_span, status=progress["status"])
            mirror_span = trace.begin(
                phase="artifacts", kind="io", name="candidate_state_write")
            state_path = write_candidate_outcome(
                state_root=paths.candidate_state_root, key=selected["key"],
                payload=_candidate_state_payload(
                    progress=progress, attempt=attempt, selected=selected))
            trace.end(mirror_span, path=str(state_path))
            timing = trace.as_dict()
            result = {
                "schema_version": 1, "success": success,
                "status": "planning_feasible" if success else "terminal_planning_failure",
                "reason": planning.get("reason"),
                "failure_code": planning.get("failure_code"),
                "candidate_key": selected["key"],
                "variant_ordinal": local_variant,
                "success_rank": attempt.get("success_rank"),
                "coverage_added": selected["marginal_scene_ids"] if success else [],
                "coverage": progress["coverage"],
                "consecutive_failures": progress["consecutive_failures"],
                "campaign_status": progress["status"],
                "predicted_execution_duration_s": predicted_duration,
                "timing": timing,
            }
            session._write_json(episode / "result.json", result)
            print(
                f"[training] success={success} rank={attempt.get('success_rank')} "
                f"planning={attempt['planning_wall_s']:.2f}s "
                f"coverage={len(progress['coverage']['covered_scene_ids'])}/"
                f"{progress['coverage']['scene_count']} "
                f"streak={progress['consecutive_failures']} status={progress['status']} "
                f"-> {episode}", flush=True)

        print(
            f"[training] finished status={progress['status']} "
            f"verified={progress['success_count']} "
            f"coverage={len(progress['coverage']['covered_scene_ids'])}/"
            f"{progress['coverage']['scene_count']} -> {paths.progress_path}", flush=True)
    except (session.UserQuit, KeyboardInterrupt):
        interrupted = True
        print("\n[training] interrupted; saved progress remains resumable", flush=True)
    finally:
        if capture is not None:
            capture.close()
        timing = session_trace.as_dict()
        session._write_json(session_dir / "timing.json", timing)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
