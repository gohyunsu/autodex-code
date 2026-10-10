#!/usr/bin/env python3
"""Exercise the real v8 reset planner in a deliberately hypothetical scene.

This is an offline code/geometry diagnostic, never a measured session, a
verified reset seed, a physical release, or a robot motion permit.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnose_synthetic_full_chain import _sha, synthetic_calibration  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.planner_mode import require_declared_cartesian_mode  # noqa: E402
from precision_insertion.repose_artifacts import (  # noqa: E402
    write_repose_preflight_artifacts,
)
from precision_insertion.repose_transition import (  # noqa: E402
    preflight_v8_repose_transition,
)
from precision_insertion.world import build_trial_scene_from_session  # noqa: E402


def _finite(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("coordinate must be finite")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reset-candidate-dir", type=Path, required=True,
                        help="non-runtime handoff reset_<height> directory")
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=_finite, required=True)
    parser.add_argument("--from-pose-stem", required=True)
    parser.add_argument("--to-pose-stem", required=True)
    parser.add_argument("--height-cm", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--table-z-m", type=_finite, required=True)
    parser.add_argument("--key-x-m", type=_finite, required=True)
    parser.add_argument("--key-y-m", type=_finite, required=True)
    parser.add_argument("--socket-x-m", type=_finite, required=True)
    parser.add_argument("--socket-y-m", type=_finite, required=True)
    parser.add_argument("--release-x-m", type=_finite, required=True)
    parser.add_argument("--release-y-m", type=_finite, required=True)
    parser.add_argument("--board-x-min-m", type=_finite, required=True)
    parser.add_argument("--board-x-max-m", type=_finite, required=True)
    parser.add_argument("--board-y-min-m", type=_finite, required=True)
    parser.add_argument("--board-y-max-m", type=_finite, required=True)
    parser.add_argument("--max-reset-drift-mm", type=_finite, required=True)
    parser.add_argument("--max-reset-axis-tilt-deg", type=_finite, required=True)
    parser.add_argument("--min-rest-socket-clearance-mm", type=_finite,
                        required=True)
    parser.add_argument("--min-board-edge-clearance-mm", type=_finite,
                        required=True)
    parser.add_argument("--max-seed-attempts", type=int, default=1)
    parser.add_argument("--retreat-goal-q-npy", type=Path,
                        help="optional explicit seven-joint post-release arm goal")
    parser.add_argument("--min-release-key-clearance-mm", type=_finite,
                        help="required with --retreat-goal-q-npy")
    parser.add_argument("--planner-mode",
                        choices=("default", "native-locked-experimental"),
                        default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        native_enabled = require_declared_cartesian_mode(args.planner_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.board_x_min_m >= args.board_x_max_m or
            args.board_y_min_m >= args.board_y_max_m or
            args.max_seed_attempts < 1 or
            min(args.max_reset_drift_mm, args.max_reset_axis_tilt_deg,
                args.min_rest_socket_clearance_mm,
                args.min_board_edge_clearance_mm) <= 0):
        parser.error("board bounds, seed budget and tolerances must be positive")
    if ((args.retreat_goal_q_npy is None) !=
            (args.min_release_key_clearance_mm is None)):
        parser.error("release exit needs both retreat goal and key clearance")
    if (args.min_release_key_clearance_mm is not None and
            args.min_release_key_clearance_mm <= 0):
        parser.error("release key clearance must be positive")

    root = args.shared_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    sources = output.with_name(output.name + "_synthetic_inputs")
    if output.exists() or sources.exists():
        raise FileExistsError("output or synthetic input directory already exists")
    mode = select_mode(args.mode, args.gap_mm)
    paths = AssetPaths(root, mode)
    catalog_path = args.catalog.expanduser().resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    reset_root = args.reset_candidate_dir.expanduser().resolve()
    retreat_path = (None if args.retreat_goal_q_npy is None else
                    args.retreat_goal_q_npy.expanduser().resolve())
    retreat_q = (None if retreat_path is None else
                 np.asarray(np.load(retreat_path, allow_pickle=False), dtype=float))
    if retreat_q is not None and (
            retreat_q.shape != (7,) or not np.all(np.isfinite(retreat_q))):
        raise ValueError("retreat arm goal must be seven finite joints")
    tabletop_path = paths.key_tabletop_dir / f"{args.from_pose_stem}.npy"
    key_pose = np.asarray(np.load(tabletop_path, allow_pickle=False), dtype=float)
    if key_pose.shape != (4, 4):
        raise ValueError("source v8 tabletop pose is not a 4x4 transform")
    key_pose = key_pose.copy()
    key_pose[:3, 3] += [args.key_x_m, args.key_y_m, args.table_z_m]
    base = synthetic_calibration(
        root=root, mode=mode, table_z_m=args.table_z_m,
        socket_xy_m=(args.socket_x_m, args.socket_y_m))
    board = dict(base.board)
    board["corners_robot_m"] = [
        [x, y, args.table_z_m]
        for x, y in ((args.board_x_min_m, args.board_y_min_m),
                     (args.board_x_max_m, args.board_y_min_m),
                     (args.board_x_max_m, args.board_y_max_m),
                     (args.board_x_min_m, args.board_y_max_m))
    ]
    calibration = SessionCalibration(
        board, base.socket_pose_robot, base.socket_diagnostics,
        base.collision_scene, base.record)
    trial_scene = build_trial_scene_from_session(
        mode=mode, shared_root=root, calibration=calibration,
        key_pose_world=key_pose)
    limits = PathAuditLimits(
        max_joint_step_rad=.12, max_wrist_step_m=.015,
        max_wrist_rotation_deg=8., goal_position_tolerance_m=.005,
        goal_rotation_tolerance_deg=5., axial_lateral_tolerance_m=.003,
        axial_rotation_tolerance_deg=5., minimum_hand_clearance_m=.0002)

    from autodex.planner.planner import GraspPlanner
    planner = GraspPlanner(hand="fr3_inspire", use_cuda_graph=False)
    if planner._native_pose_constraints_enabled != native_enabled:
        raise RuntimeError("AutoDex planner mode differs from declared mode")
    start = planner._init_state.copy()  # hypothetical stock home, not measured
    sources.mkdir(parents=True, exist_ok=False)
    source_files = {
        "session": sources / "session.json",
        "catalog": catalog_path,
        "key_pose_world": sources / "key_pose_world.npy",
        "live_start_q": sources / "start_q.npy",
        "limits": sources / "limits.json",
    }
    (sources / "session.json").write_text(json.dumps({
        "synthetic": True, "board": board, "record": calibration.record,
        "collision_scene": calibration.collision_scene,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.save(source_files["key_pose_world"], key_pose)
    np.save(source_files["live_start_q"], start)
    source_files["limits"].write_text(
        json.dumps(asdict(limits), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="precision_repose_log.") as temp:
        log = Path(temp) / "planner_stdout_stderr.txt"
        try:
            with log.open("w", encoding="utf-8") as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = preflight_v8_repose_transition(
                        planner=planner, shared_root=root, mode=mode,
                        calibration=calibration, trial_scene=trial_scene,
                        catalog=catalog,
                        from_pose_stem=args.from_pose_stem,
                        to_pose_stem=args.to_pose_stem,
                        height_cm=args.height_cm,
                        release_xy_robot_m=(args.release_x_m, args.release_y_m),
                        live_start_q=start,
                        observation_id="synthetic_reset_key_pose",
                        key_capture_timestamp_s=2.0,
                        start_q_acquisition_timestamp_s=2.0,
                        max_state_skew_s=.1, max_pose_error_deg=10.,
                        max_center_in_hand_drift_m=(
                            args.max_reset_drift_mm / 1000.),
                        max_symmetry_axis_tilt_deg=(
                            args.max_reset_axis_tilt_deg),
                        minimum_rest_socket_clearance_m=(
                            args.min_rest_socket_clearance_mm / 1000.),
                        minimum_board_edge_clearance_m=(
                            args.min_board_edge_clearance_mm / 1000.),
                        limits=limits, reset_candidate_root=reset_root,
                        max_seed_attempts=args.max_seed_attempts,
                        retreat_goal_arm_q=retreat_q,
                        minimum_release_key_clearance_m=(
                            None if args.min_release_key_clearance_mm is None
                            else args.min_release_key_clearance_mm / 1000.))
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=False)
            shutil.copy2(log, output / log.name)
            (output / "failure.json").write_text(json.dumps({
                "schema": "precision_insertion_synthetic_repose_failure_v1",
                "synthetic": True, "robot_ready": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "planner_log_sha256": _sha(output / log.name),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"error": str(exc), "output": str(output),
                              "robot_ready": False}, indent=2))
            return 1
        write_repose_preflight_artifacts(
            result=result, trial_scene=trial_scene, output_dir=output,
            source_files=source_files)
        shutil.copy2(log, output / log.name)
    context = {
        "schema": "precision_insertion_synthetic_repose_diagnostic_v1",
        "synthetic": True, "robot_ready": False,
        "planner_mode": args.planner_mode,
        "reset_candidate_dir": str(reset_root),
        "source_catalog_sha256": _sha(catalog_path),
        "source_tabletop_pose_sha256": _sha(tabletop_path),
        "retreat_goal_arm_q_sha256": (
            None if retreat_path is None else _sha(retreat_path)),
        "start_q_source": "stock_FR3_Inspire_diagnostic_home_not_measured",
        "board_source": "hypothetical_rectangle_not_measured_charuco",
        "limits_source": "exploratory_not_commissioned",
        "result_status": result.status,
        "scope": "offline_planner_diagnostic_not_session_or_motion_authorization",
        "planner_log_sha256": _sha(output / "planner_stdout_stderr.txt"),
        "attempted_reset_seed_files": {
            str(attempt["seed_id"]): {
                path.name: _sha(path)
                for path in sorted(Path(attempt["source"]).iterdir())
                if path.is_file()
            }
            for attempt in result.attempted_seeds
        },
    }
    (output / "synthetic_context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": result.status,
                      "attempted": list(result.attempted_seeds),
                      "output": str(output), "robot_ready": False},
                     indent=2))
    return (0 if result.status in {
        "held_reset_path_available_release_unplanned",
        "nominal_reset_preflight_pass_drop_unobserved",
    } else 2)


if __name__ == "__main__":
    raise SystemExit(main())
