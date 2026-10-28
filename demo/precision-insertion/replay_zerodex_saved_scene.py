#!/usr/bin/env python3
"""Replay saved capture4 board, socket and key evidence without hardware.

The socket and key may come from different requests. Their separately measured
board poses and request IDs are retained so this diagnostic cannot be mistaken
for one live, synchronized session or a Franka trajectory validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.board_socket_fit import fit_board_socket_body
from precision_insertion.candidates import select_pose_candidates
from precision_insertion.session_launch import (
    load_session_launch, register_profile_charuco_board,
)
from precision_insertion.symmetry import match_axisymmetric_tabletop_pose
from src.execution.charuco_tabletop import measure_tabletop_from_images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_id(images_dir: Path) -> int:
    match = re.search(r"_request_(\d+)$", images_dir.parent.name)
    if match is None:
        raise ValueError(f"saved image directory lacks request ID: {images_dir}")
    return int(match.group(1))


def _board_from_undistorted_pngs(images_dir: Path, launch) -> tuple[dict, dict]:
    images = {}
    for serial in launch.camera_ids:
        image = cv2.imread(str(images_dir / f"{serial}.png"), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != launch.image_hw:
            raise ValueError(f"missing or wrong-sized saved image: {serial}")
        images[serial] = image
    # Capture4's saved PNGs are already undistorted. The stock board routine
    # expects original pixels, so supply the undistorted K and zero distortion
    # rather than undistorting those pixels a second time.
    intrinsics = {
        serial: {
            **launch.intrinsics_full[serial],
            "K_orig": launch.intrinsics_full[serial]["K_undist"],
            "dist_params": np.zeros_like(
                launch.intrinsics_full[serial]["dist_params"]),
        }
        for serial in launch.camera_ids
    }
    board = measure_tabletop_from_images(
        images, intrinsics, launch.extrinsics_full, launch.c2r,
        board_id="9", minimum_board_corners=launch.minimum_board_corners)
    return board, {serial: _sha256(images_dir / f"{serial}.png")
                   for serial in launch.camera_ids}


def replay(args: argparse.Namespace) -> dict:
    launch = load_session_launch(args.launch_config)
    if launch.board_id != "9" or launch.mode.family != "cylinder":
        raise ValueError("saved capture4 replay needs cylindrical board-9 mode")
    register_profile_charuco_board(launch)
    socket_board, socket_images = _board_from_undistorted_pngs(
        args.socket_images_dir, launch)
    key_board, key_images = _board_from_undistorted_pngs(
        args.key_images_dir, launch)

    mask_report = json.loads(args.socket_mask_dir.joinpath(
        "report.json").read_text(encoding="utf-8"))
    if mask_report.get("target_object") != launch.mode.socket_object:
        raise ValueError("saved socket masks belong to another object")
    if mask_report.get("socket_request_id") != _request_id(
            args.socket_images_dir):
        raise ValueError("socket masks and images come from different requests")
    masks = {}
    for serial in launch.camera_ids:
        mask = cv2.imread(str(args.socket_mask_dir / f"{serial}_mask.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != launch.image_hw:
            raise ValueError(f"missing or wrong-sized socket mask: {serial}")
        if np.count_nonzero(mask) != mask_report["views"][serial]["mask_pixels"]:
            raise ValueError(f"socket mask differs from saved report: {serial}")
        if np.any(mask):
            masks[serial] = mask > 0
    paths = AssetPaths(launch.shared_root, launch.mode)
    socket_world, socket_fit = fit_board_socket_body(
        masks=masks,
        intrinsics_undist={serial: launch.intrinsics_full[serial]["K_undist"]
                           for serial in masks},
        extrinsics_camera_world={serial: launch.extrinsics_full[serial]
                                 for serial in masks},
        c2r=launch.c2r,
        board_pose_robot=np.asarray(socket_board["T_robot_board"]),
        raw_mesh=paths.raw_mesh(launch.mode.socket_object))

    key_refine = json.loads(args.key_refinement_report.read_text(
        encoding="utf-8"))
    if key_refine.get("object_name") != launch.mode.key_object:
        raise ValueError("saved key refinement belongs to another object")
    foundpose_path = Path(key_refine["foundpose_report"])
    if _sha256(foundpose_path) != key_refine["foundpose_report_sha256"]:
        raise ValueError("key refinement's FoundPose source has changed")
    foundpose = json.loads(foundpose_path.read_text(encoding="utf-8"))
    if _request_id(args.key_images_dir) != _request_id(
            Path(foundpose["capture_dir"]) / "images"):
        raise ValueError("key refinement and images come from different requests")
    for row in foundpose["views"]:
        serial = row["serial"]
        if row["image_sha256"] != key_images[serial]:
            raise ValueError(f"saved key image differs from FoundPose: {serial}")
    key_world = np.asarray(key_refine["stock_refinement"]["post_refinement"][
        "pose_world_m"], dtype=np.float64)
    key_robot = np.linalg.inv(launch.c2r) @ key_world
    socket_robot = np.linalg.inv(launch.c2r) @ socket_world
    _, stem, axis_error_deg = match_axisymmetric_tabletop_pose(
        key_robot, object_root=launch.shared_root / "object_processing",
        object_name=launch.mode.key_object)
    catalog = json.loads(args.endpoint_catalog.read_text(encoding="utf-8"))
    selection = select_pose_candidates(
        catalog, expected_mode=launch.mode, tabletop_pose_stem=stem)
    board_delta_mm = 1000 * np.linalg.norm(
        np.asarray(socket_board["T_robot_board"])[:3, 3] -
        np.asarray(key_board["T_robot_board"])[:3, 3])
    return {
        "schema": "precision_insertion_zerodex_saved_scene_replay_v1",
        "scope": "saved_images_only_no_robot_state_or_motion_validation",
        "socket_request_id": mask_report.get("socket_request_id"),
        "key_request_id": _request_id(args.key_images_dir),
        "socket_image_sha256": socket_images,
        "key_image_sha256": key_images,
        "socket_board": {"metrics": socket_board["metrics"],
                         "T_robot_board": socket_board["T_robot_board"]},
        "key_board": {"metrics": key_board["metrics"],
                      "T_robot_board": key_board["T_robot_board"]},
        "between_capture_board_origin_delta_mm": float(board_delta_mm),
        "socket_fit": socket_fit,
        "T_robot_socket": socket_robot.tolist(),
        "key_refinement_report": str(args.key_refinement_report),
        "key_mask_mean_iou": key_refine["stock_refinement"][
            "post_refinement"]["mean_iou"],
        "T_robot_key": key_robot.tolist(),
        "key_tabletop_pose_stem": stem,
        "key_axis_residual_deg": axis_error_deg,
        "endpoint_catalog": str(args.endpoint_catalog),
        "candidate_status": selection["status"],
        "endpoint_eligible_candidate_keys": [row["key"] for row in
                                             selection["candidates"]],
        "current_scene_franka_plan_tested": False,
        "physical_insertion_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--socket-images-dir", type=Path, required=True)
    parser.add_argument("--socket-mask-dir", type=Path, required=True)
    parser.add_argument("--key-images-dir", type=Path, required=True)
    parser.add_argument("--key-refinement-report", type=Path, required=True)
    parser.add_argument("--endpoint-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output),
                      "board_delta_mm": result["between_capture_board_origin_delta_mm"],
                      "socket_iou": result["socket_fit"]["mean_body_iou"],
                      "tabletop_stem": result["key_tabletop_pose_stem"],
                      "eligible": len(result["endpoint_eligible_candidate_keys"])},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
