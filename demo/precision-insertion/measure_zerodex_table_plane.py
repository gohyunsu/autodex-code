#!/usr/bin/env python3
"""Measure the current table support plane once from the temporary ChArUco board.

The resulting ``table_plane.json`` is a calibration artifact for later
board-free sessions.  This command opens only the existing capture4 camera
stack; it does not initialize an object model or connect to the robot.
"""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime
import json
import os
from pathlib import Path

import cv2
import numpy as np

from precision_insertion.board_view_selection import (
    measure_tabletop_with_view_selection,
)
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import collect_board_snapshot
from precision_insertion.session_launch import (
    _load_stock_camera_calibration, load_session_launch, open_session_cameras,
)
from precision_insertion.table_surface import TABLE_PLANE_SCHEMA
from precision_insertion.geometry import validate_se3


def _measure_from_reference_camera(
    *, serial: str, image: np.ndarray, intrinsics: dict,
    extrinsic_camera_world: np.ndarray, c2r: np.ndarray, board_id: str,
    minimum_board_corners: int,
) -> dict:
    """Measure the board in the calibrated world-anchor camera.

    The ZeroDex extrinsic solution fixes one physical camera as the world
    frame.  Its PnP board pose and the saved world-to-robot hand-eye transform
    are therefore sufficient to recover the support plane without depending
    on the other cameras' current relative extrinsics.
    """
    from paradex.image.aruco import detect_charuco
    from src.execution.charuco_tabletop import _board_corner_model

    info = detect_charuco(image).get(board_id)
    if info is None:
        raise RuntimeError(f"board {board_id} is not visible in {serial}")
    ids = np.asarray(info.get("checkerIDs", []), dtype=np.int32).reshape(-1)
    pixels = np.asarray(
        info.get("checkerCorner", []), dtype=np.float64).reshape(-1, 2)
    if len(ids) < minimum_board_corners or len(ids) != len(pixels):
        raise RuntimeError(
            f"board {board_id}: {serial} has {len(ids)} corners; "
            f"needs {minimum_board_corners}")
    model = _board_corner_model(board_id)
    K = np.asarray(intrinsics["K_orig"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(
        intrinsics["dist_params"], dtype=np.float64).reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(
        model[ids], pixels, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f"board {board_id}: PnP failed in {serial}")
    rotation, _ = cv2.Rodrigues(rvec)
    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = rotation
    T_camera_board[:3, 3] = tvec.reshape(3)
    extrinsic = np.asarray(extrinsic_camera_world, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, (0.0, 0.0, 0.0, 1.0)))
    T_robot_board = (
        np.linalg.inv(validate_se3(c2r, name="C2R")) @
        np.linalg.inv(validate_se3(extrinsic, name=f"{serial} extrinsic")) @
        T_camera_board)
    center_local = model.mean(axis=0)
    plane_point = (T_robot_board @ np.append(center_local, 1.0))[:3]
    normal = T_robot_board[:3, 2].copy()
    if normal[2] < 0:
        normal = -normal
    projected, _ = cv2.projectPoints(
        model[ids], rvec, tvec, K, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    return {
        "schema_version": 1,
        "source": "charuco_reference_camera_pnp",
        "board": board_id,
        "center_robot_m": plane_point.tolist(),
        "table_surface_z_m": float(plane_point[2]),
        "plane_point_robot_m": plane_point.tolist(),
        "normal_robot": normal.tolist(),
        "T_robot_board": T_robot_board.tolist(),
        "corner_ids": ids.tolist(),
        "metrics": {
            "reference_camera_id": serial,
            "corners_detected": int(len(ids)),
            "reprojection_median_px": float(np.median(errors)),
            "reprojection_max_px": float(np.max(errors)),
        },
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _register_board_config(path: Path, board_id: str) -> None:
    from paradex.image import aruco as paradex_aruco

    info = json.loads(path.read_text(encoding="utf-8"))
    paradex_aruco.check_boardinfo_valid({board_id: info})
    paradex_aruco.boardinfo_dict[board_id] = info
    paradex_aruco._charuco_board_cache.pop(board_id, None)
    paradex_aruco._charuco_detector_cache.pop(board_id, None)
    paradex_aruco.get_charuco_detector()


def _direct_snapshot(*, pc: str, capture_ip: str,
                     camera_ids: set[str], timeout_s: float):
    from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator
    from capture_zerodex_extrinsic_direct import _snapshot

    sender_module = importlib.import_module(
        "paradex.io.capture_pc.command_sender")
    original_get_pc_ip = sender_module.get_pc_ip
    sender_module.get_pc_ip = lambda name: (
        capture_ip if name == pc else original_get_pc_ip(name))
    orchestrator = None
    try:
        orchestrator = SnapshotOrchestrator([pc], [capture_ip])
        return _snapshot(orchestrator, camera_ids, timeout_s)
    finally:
        sender_module.get_pc_ip = original_get_pc_ip
        if orchestrator is not None:
            orchestrator.close()


def _measurement_launch(template: Path, output: Path, *, board_id: str,
                        minimum_board_corners: int) -> Path:
    record = json.loads(template.read_text(encoding="utf-8"))
    record["scene_mode"] = "charuco_board"
    record["board_id"] = board_id
    record["minimum_board_corners"] = minimum_board_corners
    record["board_capture_phase"] = (
        "empty_before_socket" if minimum_board_corners == 54
        else "socket_present_partial")
    record.pop("table_surface_z_m", None)
    if os.environ.get("PRECISION_CALIB_DIR"):
        record["calib_dir"] = str(Path(
            os.environ["PRECISION_CALIB_DIR"]).expanduser().resolve())
    if os.environ.get("PRECISION_C2R_NPY"):
        record["c2r_npy"] = str(Path(
            os.environ["PRECISION_C2R_NPY"]).expanduser().resolve())
    camera_ids = [value.strip() for value in os.environ.get(
        "PRECISION_CAPTURE4_CAMERA_IDS", "").split(",") if value.strip()]
    if camera_ids:
        record["camera_ids"] = camera_ids
    board_fit_ids = [value.strip() for value in os.environ.get(
        "PRECISION_BOARD_FIT_CAMERA_IDS", "").split(",") if value.strip()]
    if board_fit_ids:
        record["board_fit_camera_ids"] = board_fit_ids
    path = output / "measurement_launch.json"
    _write_json(path, record)
    return path


def _save_detection_overlays(output: Path, images: dict[str, np.ndarray],
                             board_id: str) -> None:
    from paradex.image.aruco import detect_charuco

    tiles = []
    for serial, image in sorted(images.items()):
        overlay = image.copy()
        info = detect_charuco(image).get(board_id)
        if info is not None:
            corners = np.asarray(info.get("checkerCorner", []), dtype=np.float32)
            ids = np.asarray(info.get("checkerIDs", []), dtype=np.int32)
            if len(corners) and len(corners) == len(ids):
                cv2.aruco.drawDetectedCornersCharuco(
                    overlay, corners.reshape(-1, 1, 2), ids.reshape(-1, 1),
                    cornerColor=(0, 255, 0))
        # The snapshot transport is already JPEG.  Avoid expanding every
        # evidence frame to multi-megabyte PNGs on the NAS.
        cv2.imwrite(str(output / f"{serial}_raw.jpg"), image,
                    (cv2.IMWRITE_JPEG_QUALITY, 95))
        cv2.imwrite(str(output / f"{serial}_board_overlay.jpg"), overlay,
                    (cv2.IMWRITE_JPEG_QUALITY, 95))
        tile = cv2.resize(overlay, (1024, 768), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, serial, (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if len(tiles) == 4:
        grid = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        cv2.imwrite(str(output / "board_overlays_4view.jpg"), grid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--board-id", choices=("9", "11"), default="9")
    parser.add_argument("--minimum-board-corners", type=int, default=36)
    parser.add_argument(
        "--reference-camera",
        help=("use the extrinsic solution's fixed world-anchor camera for "
              "single-view PnP instead of multi-view triangulation"))
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument("--c2r-npy", type=Path)
    parser.add_argument("--board-config", type=Path)
    parser.add_argument("--capture", default="capture4=147.46.219.136")
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    direct = args.launch_config is None
    if direct:
        if not all((args.calib_dir, args.c2r_npy, args.board_config,
                    args.camera_id)):
            parser.error("direct mode needs calib, C2R, board and camera IDs")
        try:
            pc, capture_ip = args.capture.split("=", 1)
        except ValueError:
            parser.error("--capture must be NAME=IP")
        camera_ids = set(args.camera_id)
        calibration_dir = args.calib_dir.expanduser().resolve()
        c2r_path = args.c2r_npy.expanduser().resolve()
        intrinsics, extrinsics, _image_hw = _load_stock_camera_calibration(
            calibration_dir, camera_ids)
        c2r = np.load(c2r_path, allow_pickle=False)
        _register_board_config(
            args.board_config.expanduser().resolve(), args.board_id)
        with open_autodex_camera_stream(
                (pc,), capture_ips=(capture_ip,),
                name="precision_table_plane_measurement",
                acquisition_mode="stream"):
            snapshot = _direct_snapshot(
                pc=pc, capture_ip=capture_ip, camera_ids=camera_ids,
                timeout_s=args.timeout_s)
        board_fit_camera_ids = tuple(sorted(camera_ids))
        calibration_inputs = {
            "extrinsic_dir": str(calibration_dir),
            "c2r_npy": str(c2r_path),
        }
    else:
        launch_path = _measurement_launch(
            args.launch_config.expanduser().resolve(), output,
            board_id=args.board_id,
            minimum_board_corners=args.minimum_board_corners)
        launch = load_session_launch(
            launch_path, evidence_dir_override=output / "unused_session_evidence")
        with open_autodex_camera_stream(
                launch.pc_list, capture_ips=launch.capture_ips,
                name="precision_table_plane_measurement",
                acquisition_mode=launch.camera_acquisition_mode):
            with open_session_cameras(
                    launch, init_ports=(15006, 15007, 16893)) as cameras:
                snapshot = collect_board_snapshot(
                    snapshot_orchestrator=cameras.snapshot_orchestrator,
                    calibrated_camera_ids=launch.camera_ids,
                    acquisition_metadata_for_request=(
                        cameras.acquisition_metadata_for_request),
                    timeout_s=launch.board_timeout_s)
        camera_ids = launch.camera_ids
        intrinsics = launch.intrinsics_full
        extrinsics = launch.extrinsics_full
        c2r = launch.c2r
        board_fit_camera_ids = launch.board_fit_camera_ids
        launch_record = json.loads(launch_path.read_text(encoding="utf-8"))
        calibration_inputs = {
            "extrinsic_dir": launch_record["calib_dir"],
            "c2r_npy": launch_record["c2r_npy"],
        }

    # Preserve the exact frames even when the geometric fit fails.  They are
    # the evidence needed to distinguish board visibility from calibration or
    # camera-serial correspondence problems.
    _save_detection_overlays(output, snapshot.images_bgr, args.board_id)
    if args.reference_camera:
        if args.reference_camera not in camera_ids:
            parser.error("reference camera is not active in this launch")
        measured = _measure_from_reference_camera(
            serial=args.reference_camera,
            image=snapshot.images_bgr[args.reference_camera],
            intrinsics=intrinsics[args.reference_camera],
            extrinsic_camera_world=extrinsics[
                args.reference_camera],
            c2r=c2r, board_id=args.board_id,
            minimum_board_corners=args.minimum_board_corners)
    else:
        fit_images = snapshot.images_bgr
        if board_fit_camera_ids is not None:
            fit_images = {
                serial: snapshot.images_bgr[serial]
                for serial in board_fit_camera_ids}
        measured = measure_tabletop_with_view_selection(
            fit_images, intrinsics, extrinsics,
            c2r, board_id=args.board_id,
            minimum_board_corners=args.minimum_board_corners)
    record = {
        "schema": TABLE_PLANE_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "one_time_charuco_measurement",
        "support_surface": "charuco_printed_face",
        "table_surface_z_m": measured["table_surface_z_m"],
        "plane_point_robot_m": measured["plane_point_robot_m"],
        "normal_robot": measured["normal_robot"],
        "T_robot_board": measured["T_robot_board"],
        "board_id": args.board_id,
        "request_id": snapshot.request_id,
        "frame_evidence": snapshot.frame_evidence,
        "measurement": measured,
        "runtime_use": "board_free_support_surface",
        "calibration_inputs": calibration_inputs,
    }
    target = output / "table_plane.json"
    _write_json(target, record)
    print(json.dumps({
        "table_plane": str(target),
        "table_surface_z_m": record["table_surface_z_m"],
        "normal_robot": record["normal_robot"],
        "metrics": measured["metrics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
