#!/usr/bin/env python3
"""Run the real capture4 socket/key perception path without a robot.

For a fresh board-free session, leave only the fixed socket on the calibrated
table surface at startup and place the key after socket measurement. With
explicit saved-socket reuse,
the key may already be present. This diagnostic reuses the
same session, pose-only key admission, tabletop and catalog code as the live
pipeline; it neither builds a Franka executor nor plans or commands a
trajectory. Use probe_zerodex_inline_frames.py for raw image/mask diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from autodex.utils.conversion import cart2se3
from precision_insertion.assets import AssetPaths
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.candidates import select_pose_candidates
from precision_insertion.live_key_trial import _capture_admitted_key
from precision_insertion.live_orchestration import load_reused_socket_session
from precision_insertion.live_session_runner import (
    open_verified_session, preflight_endpoint_catalog_file,
)
from precision_insertion.pose_selection import classify_key_tabletop_pose
from precision_insertion.session_launch import (
    load_session_launch, open_session_cameras,
)
from precision_insertion.silhouette_compat import prepare_precision_silhouette
from precision_insertion.stage_policy import load_zerodex_stage_policy
from precision_insertion.table_surface import load_table_surface
from precision_insertion.timing_profile import profile_run, timed
from precision_insertion.world import build_trial_scene_from_session
from robot_pc_stages import _key_capture, _tabletop_pose_error_deg


def run_camera_only(
    *, launch_config: Path, endpoint_catalog: Path, output_dir: Path,
    wait_for_key: Callable[[], object],
    reuse_session_evidence: Path | None = None,
    gap_mm: int | None = None,
    socket_only: bool = False,
    key_present_during_socket_capture: bool = False,
) -> dict:
    """Capture a fixed socket, then optionally capture one key."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibration_dir = os.environ.get("PRECISION_CALIB_DIR")
    c2r_npy = os.environ.get("PRECISION_C2R_NPY")
    scene_mode = os.environ.get("PRECISION_SCENE_MODE")
    if gap_mm is not None or calibration_dir or c2r_npy or scene_mode is not None:
        # Mirror run_robot_pc.sh: keep the immutable stage template and save
        # the exact per-run mode and active rig calibration beside the
        # camera-only evidence.
        launch_record = json.loads(Path(launch_config).read_text(encoding="utf-8"))
        if gap_mm is not None:
            launch_record["mode"] = {"family": "cylinder", "gap_mm": gap_mm}
        if calibration_dir:
            launch_record["calib_dir"] = str(
                Path(calibration_dir).expanduser().resolve())
        if c2r_npy:
            launch_record["c2r_npy"] = str(
                Path(c2r_npy).expanduser().resolve())
        camera_ids = tuple(value.strip() for value in os.environ.get(
            "PRECISION_CAPTURE4_CAMERA_IDS", "").split(",") if value.strip())
        board_fit_camera_ids = tuple(value.strip() for value in os.environ.get(
            "PRECISION_BOARD_FIT_CAMERA_IDS", "").split(",") if value.strip())
        if camera_ids:
            launch_record["camera_ids"] = list(camera_ids)
        if board_fit_camera_ids:
            launch_record["board_fit_camera_ids"] = list(board_fit_camera_ids)
        if scene_mode is not None:
            launch_record["scene_mode"] = scene_mode
        if scene_mode == "board_free":
            table_z = os.environ.get("PRECISION_TABLE_SURFACE_Z_M")
            table_config = os.environ.get("PRECISION_TABLE_PLANE_CONFIG")
            if table_z is None:
                if not table_config:
                    table_config = str(Path(launch_config).resolve().parent /
                                       "table_plane.json")
                surface = load_table_surface(table_config)
                table_z = surface["table_surface_z_m"]
            else:
                surface = (load_table_surface(table_config) if table_config else {
                    "plane_point_robot_m": np.array([0.0, 0.0, float(table_z)]),
                    "normal_robot": np.array([0.0, 0.0, 1.0]),
                })
            launch_record["table_surface_z_m"] = float(table_z)
            launch_record["table_surface_point_robot_m"] = np.asarray(
                surface["plane_point_robot_m"], dtype=float).tolist()
            launch_record["table_surface_normal_robot"] = np.asarray(
                surface["normal_robot"], dtype=float).tolist()
            launch_record["socket_perception_mode"] = "table_constrained_body"
            for key in ("board_id", "minimum_board_corners",
                        "board_capture_phase", "board_fit_camera_ids"):
                launch_record.pop(key, None)
        elif scene_mode not in (None, "charuco_board"):
            raise ValueError("unknown precision scene mode")
        launch_config = output / "launch.json"
        with launch_config.open("x", encoding="utf-8") as stream:
            json.dump(launch_record, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
    with profile_run(output):
        with timed("startup.stage_policy", phase="startup", kind="io"):
            stage_policy = load_zerodex_stage_policy()
        with timed("startup.launch_config", phase="startup", kind="io"):
            launch = load_session_launch(
                launch_config, evidence_dir_override=output / "socket_evidence")
        valid_socket_mode = (
            launch.scene_mode == "charuco_board" and
            launch.socket_perception_mode == "board_constrained_body" or
            launch.scene_mode == "board_free" and
            launch.socket_perception_mode != "board_constrained_body")
        if (launch.mode.family != "cylinder" or not valid_socket_mode or
                tuple(launch.pc_list) != ("capture4",)):
            raise ValueError("camera-only probe needs the capture4 cylinder profile")
        with timed("startup.catalog_load", phase="startup", kind="io"):
            catalog = preflight_endpoint_catalog_file(
                mode=launch.mode, shared_root=launch.shared_root,
                catalog_path=endpoint_catalog)
        saved_session = None
        if reuse_session_evidence is not None:
            with timed("startup.saved_session_verify", phase="startup",
                       kind="io"):
                saved_session = load_reused_socket_session(
                    launch=launch, evidence_dir=reuse_session_evidence)
        paths = AssetPaths(launch.shared_root, launch.mode)
        with open_session_cameras(
                launch, init_ports=(15006, 15007, 16893)) as cameras:
            # The socket path is mask/body-only. Prewarm the key renderer as
            # the normal ZeroDex session does, before camera heartbeat starts.
            if not socket_only:
                with timed("startup.key_renderer_warmup", phase="startup"):
                    prepare_precision_silhouette(
                        init_orchestrator=cameras.init_orchestrator,
                        object_name=launch.mode.key_object,
                        raw_mesh=paths.raw_mesh(launch.mode.key_object))
            with open_autodex_camera_stream(
                    launch.pc_list, capture_ips=launch.capture_ips,
                    name="precision_zerodex_camera_only",
                    acquisition_mode=launch.camera_acquisition_mode):
                if saved_session is None:
                    with timed("startup.board_socket_session", phase="startup"):
                        started = cameras.start()
                else:
                    started = saved_session
                with timed("startup.session_ledger_open", phase="startup",
                           kind="io"):
                    opened = open_verified_session(
                        mode=launch.mode, shared_root=launch.shared_root,
                        evidence_dir=started.evidence_dir,
                        catalog_path=endpoint_catalog,
                        output_dir=output / "ledger", max_xy_retries=None,
                        catalog_snapshot=catalog,
                        producer_manifest=started.producer_manifest,
                        calibration_snapshot=started.calibration,
                        upright_cylinder_only=launch.upright_cylinder_only)
                if socket_only:
                    report = {
                        "schema": "precision_insertion_zerodex_socket_only_session_v1",
                        "scope": "real_camera_socket_no_key_or_robot",
                        "mode": {"family": launch.mode.family,
                                 "gap_mm": launch.mode.gap_mm},
                        "socket_evidence_dir": str(started.evidence_dir),
                        "socket_evidence_reused": saved_session is not None,
                        "key_present_during_socket_capture": key_present_during_socket_capture,
                        "robot_ready": False,
                    }
                    with (output / "report.json").open("x", encoding="utf-8") as stream:
                        json.dump(report, stream, indent=2, sort_keys=True,
                                  allow_nan=False)
                        stream.write("\n")
                    return report
                with timed("startup.key_placement_wait", phase="startup",
                           kind="operator"):
                    wait_for_key()
                capture_settings = _key_capture(stage_policy)
                # These settings are consumed by the live planning wrapper,
                # not by the shared camera/pose admission primitive.
                capture_settings.pop("max_key_state_skew_s")
                with timed("key.camera_only_capture", phase="key"):
                    observation = _capture_admitted_key(
                        runner=opened.runner,
                        init_orchestrator=cameras.init_orchestrator,
                        acquisition_metadata_for_request=(
                            cameras.acquisition_metadata_for_request),
                        capture_root=launch.capture_root,
                        output=output / "key_capture", capture_id="key_000",
                        calibrated_camera_ids=set(launch.camera_ids),
                        intrinsics_full=launch.intrinsics_full,
                        extrinsics_full=launch.extrinsics_full,
                        image_hw=launch.image_hw,
                        key_prompt=stage_policy.key_prompt,
                        view_limits=launch.view_limits,
                        request_id_factory=None,
                        registered_key_assistance=None,
                        pose_only=True,
                        silhouette_iterations=100,
                        silhouette_loss_threshold=0.003,
                        **capture_settings)
        with timed("key.scene_and_classification", phase="key"):
            tabletop_pose_error_deg = _tabletop_pose_error_deg(stage_policy)
            scene = build_trial_scene_from_session(
                mode=launch.mode, shared_root=launch.shared_root,
                calibration=started.calibration,
                key_pose_world=observation.pose_world,
                max_axis_error_deg=tabletop_pose_error_deg)
            pose_robot = cart2se3(np.asarray(
                scene["mesh"]["target"]["pose"], dtype=float))
            pose_class = classify_key_tabletop_pose(
                mode=launch.mode, shared_root=launch.shared_root,
                pose_robot_key=pose_robot,
                max_rotation_error_deg=tabletop_pose_error_deg)
            selected = select_pose_candidates(
                catalog, expected_mode=launch.mode,
                tabletop_pose_stem=pose_class["stem"])
        direct_preflight = (
            selected["status"] == "candidates_available" and
            (not launch.upright_cylinder_only or pose_class["stem"] == "000"))
        report = {
            "schema": "precision_insertion_zerodex_camera_only_session_v2",
            "scope": "real_camera_perception_and_catalog_not_robot_plan_or_motion",
            "socket_evidence_dir": str(started.evidence_dir),
            "socket_evidence_reused": saved_session is not None,
            "key_present_during_socket_capture": key_present_during_socket_capture,
            "key_evidence_dir": str(output / "key_capture"),
            "key_request_id": observation.request_id,
            "selected_camera_id": observation.selected_camera_id,
            "tabletop_pose": pose_class,
            "candidate_status": selected["status"],
            "endpoint_eligible_candidate_count": len(selected["candidates"]),
            "direct_candidate_preflight_available": direct_preflight,
            "live_pose_route": (
                "direct_candidate_preflight" if direct_preflight else
                "reorientation_preflight_if_alternative_pose_available"),
            "robot_ready": False,
        }
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--gap-mm", type=int, choices=(1, 3, 5, 10, 15, 20),
                        help="socket radial gap; saves the selected mode in the run launch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-session-evidence", type=Path,
                        help="verified fixed-socket session; skips board/socket recapture")
    parser.add_argument("--key-already-placed", action="store_true",
                        help="capture an existing key immediately; a fresh socket capture with the key present is diagnostic only")
    parser.add_argument("--socket-only", action="store_true",
                        help="measure and save the fixed socket, then exit before key capture")
    args = parser.parse_args(argv)
    if args.reuse_session_evidence is None:
        if args.key_already_placed:
            print("Key already in view: this fresh socket/key capture is a one-off diagnostic; its baseline includes the key.")
        else:
            print("Leave the fixed socket in place and REMOVE the key before starting.")
            input("Press Enter to measure the socket... ")
    else:
        print("Reusing the saved fixed-socket session; leave the socket unmoved.")
    def wait_for_key() -> None:
        if not args.key_already_placed:
            input("Place or confirm the key on the calibrated table surface, then press Enter to recognize it... ")

    report = run_camera_only(
        launch_config=args.launch_config,
        endpoint_catalog=args.endpoint_catalog,
        output_dir=args.output_dir,
        reuse_session_evidence=args.reuse_session_evidence,
        gap_mm=args.gap_mm,
        socket_only=args.socket_only,
        key_present_during_socket_capture=(
            args.key_already_placed and args.reuse_session_evidence is None),
        wait_for_key=wait_for_key)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
