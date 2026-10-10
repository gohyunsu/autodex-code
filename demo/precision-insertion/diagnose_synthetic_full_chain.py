#!/usr/bin/env python3
"""Run the real v8 pickup/insertion planner in an explicitly synthetic scene.

This diagnostic can expose code/IK/path failures before camera commissioning.
It NEVER creates a measured session, physical success label, or robot permit.
The socket/base, board, key pose and joint start here are hypothetical.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
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

from autodex.utils.tabletop_geometry import table_cuboid  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.candidates import select_pose_candidates  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.trial_preflight import (  # noqa: E402
    plan_fresh_key_trial, write_trial_preflight_artifacts,
)
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_calibration(*, root: Path, mode, table_z_m: float,
                          socket_xy_m: tuple[float, float]) -> SessionCalibration:
    """Build a labelled *hypothetical* fixed world using actual fixture CAD."""
    paths = AssetPaths(root, mode)
    board = {"table_surface_z_m": table_z_m}
    pose = np.eye(4)
    pose[:3, 3] = [*socket_xy_m, table_z_m]
    collision = paths.socket_collision_mesh.resolve()
    fixed = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {"table": table_cuboid(board)}},
        {"fixture_socket": {"collision_mesh": collision,
                            "pose_robot": pose}})
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "c2r": np.eye(4).tolist(),
        "socket_pose_robot": pose.tolist(),
        "socket_collision_mesh": str(collision),
        "socket_collision_mesh_sha256": _sha(collision),
        "socket_observations": [{"timestamp_s": 1.0,
                                 "source": "synthetic_not_camera_acquisition"}],
        "synthetic": True,
    }
    return SessionCalibration(board, pose, {"synthetic": True}, fixed, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mode", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--pose-stem", required=True)
    parser.add_argument("--table-z-m", type=float, required=True)
    parser.add_argument("--key-x-m", type=float, required=True)
    parser.add_argument("--key-y-m", type=float, required=True)
    parser.add_argument("--socket-x-m", type=float, required=True)
    parser.add_argument("--socket-y-m", type=float, required=True)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.shared_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.max_candidates < 1:
        raise ValueError("max-candidates must be positive")
    values = (args.table_z_m, args.key_x_m, args.key_y_m,
              args.socket_x_m, args.socket_y_m)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("synthetic placements must be finite")
    mode = select_mode(args.mode, args.gap_mm)
    paths = AssetPaths(root, mode)
    catalog_path = args.catalog.expanduser().resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    selection = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=args.pose_stem)
    if selection["status"] != "candidates_available":
        raise ValueError(f"current v8 pose has no usable pool: {selection['status']}")
    tabletop_path = paths.key_tabletop_dir / f"{args.pose_stem}.npy"
    tabletop = np.asarray(np.load(tabletop_path, allow_pickle=False), dtype=float)
    if tabletop.shape != (4, 4):
        raise ValueError("v8 tabletop pose is not a 4x4 transform")
    key_pose = tabletop.copy()
    key_pose[:3, 3] += [args.key_x_m, args.key_y_m, args.table_z_m]
    calibration = synthetic_calibration(
        root=root, mode=mode, table_z_m=args.table_z_m,
        socket_xy_m=(args.socket_x_m, args.socket_y_m))

    from autodex.planner.planner import GraspPlanner
    planner = GraspPlanner(hand="fr3_inspire", use_cuda_graph=False)
    start = planner._init_state.copy()  # stock FR3/Inspire diagnostic home
    # Exploratory numerical audit settings are intentionally not represented
    # as commissioned motion/force limits.  Every artifact remains synthetic.
    limits = PathAuditLimits(
        max_joint_step_rad=.12, max_wrist_step_m=.015,
        max_wrist_rotation_deg=8., goal_position_tolerance_m=.005,
        goal_rotation_tolerance_deg=5., axial_lateral_tolerance_m=.003,
        axial_rotation_tolerance_deg=5., minimum_hand_clearance_m=.0002)
    with tempfile.TemporaryDirectory(prefix="precision_planner_log.") as temp:
        log = Path(temp) / "planner_stdout_stderr.txt"
        try:
            with log.open("w", encoding="utf-8") as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = plan_fresh_key_trial(
                        planner=planner, mode=mode, shared_root=root,
                        calibration=calibration, catalog=catalog,
                        key_pose_world=key_pose,
                        key_observation_id="synthetic_pose_" + args.pose_stem,
                        key_capture_timestamp_s=2.0, live_start_q=start,
                        start_q_acquisition_timestamp_s=2.0,
                        max_key_state_skew_s=.1, limits=limits,
                        max_pose_error_deg=10., axial_waypoint_step_m=.005,
                        max_candidate_attempts=args.max_candidates)
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=False)
            shutil.copy2(log, output / log.name)
            failure = {
                "schema": "precision_insertion_synthetic_planner_failure_v1",
                "synthetic": True, "robot_ready": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "planner_log_sha256": _sha(output / log.name),
                "scope": "offline_code_failure_not_grasp_or_insertion_result",
            }
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(json.dumps({"failure": failure["error"],
                              "report_dir": str(output),
                              "robot_ready": False}, indent=2))
            return 1
        write_trial_preflight_artifacts(result, output)
        shutil.copy2(log, output / log.name)
    context = {
        "schema": "precision_insertion_synthetic_full_chain_diagnostic_v1",
        "synthetic": True, "robot_ready": False,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": _sha(catalog_path),
        "key_tabletop_pose": str(tabletop_path),
        "key_tabletop_pose_sha256": _sha(tabletop_path),
        "table_z_m": args.table_z_m,
        "key_xy_robot_m": [args.key_x_m, args.key_y_m],
        "socket_xy_robot_m": [args.socket_x_m, args.socket_y_m],
        "T_robot_key_hypothetical": key_pose.tolist(),
        "T_robot_socket_hypothetical": calibration.socket_pose_robot.tolist(),
        "start_q_source": "stock_FR3_Inspire_diagnostic_home_not_measured",
        "limits_source": "exploratory_not_commissioned",
        "result_status": result.status,
        "attempted_candidates": list(result.attempted_candidates),
        "planner_log_sha256": _sha(output / "planner_stdout_stderr.txt"),
        "scope": "offline_planner_diagnostic_not_session_or_motion_authorization",
    }
    with (output / "synthetic_context.json").open("x", encoding="utf-8") as stream:
        json.dump(context, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result.status,
                      "attempted": list(result.attempted_candidates),
                      "report_dir": str(output), "robot_ready": False},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
