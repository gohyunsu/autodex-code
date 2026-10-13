#!/usr/bin/env python3
"""Camera-only board/socket/key diagnostic using current AutoDex inference.

The stock init daemon publishes SAM3 and FoundPose results without sensor
frame IDs. Optional separate-port precision init daemons add per-request frame
IDs and hashes. Either route retains every saved view and result for review;
neither admits a pose to the precision session or constructs a robot controller.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import secrets
import time

import cv2
import numpy as np

from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator
from precision_insertion.assets import AssetPaths
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.camera_transport import PrecisionInitOrchestrator
from precision_insertion.frame_provenance import image_sha256
from precision_insertion.prompt_contract import reject_incorrect_object_color
from precision_insertion.session_launch import load_session_launch
from precision_insertion.silhouette_compat import prepare_precision_silhouette
from src.execution.charuco_tabletop import measure_tabletop_from_images


SCHEMA = "precision_insertion_stock_camera_qa_v1"
KEY_PROMPT = "white plastic cylindrical key"


def _request_id() -> int:
    return secrets.randbelow(2**31 - 1) + 1


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_complete_png(path: Path, *, image_hw: tuple[int, int],
                       timeout_s: float) -> np.ndarray | None:
    """Wait for the daemon's asynchronous NFS PNG write to reach IEND."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with path.open("rb") as stream:
                stream.seek(-12, 2)
                complete = stream.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"
        except (FileNotFoundError, OSError):
            complete = False
        if complete:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None and image.shape[:2] == image_hw:
                return image
        time.sleep(0.1)
    return None


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip() == ""
    except (EOFError, KeyboardInterrupt):
        return False


def _bound_frame_identity(mask: dict, pose: dict,
                          image: np.ndarray) -> dict:
    """Verify precision-daemon metadata against the saved undistorted PNG."""
    digest = image_sha256(image)
    frame_id = mask.get("frame_id")
    if (type(frame_id) is not int or frame_id <= 0 or
            pose.get("frame_id") != frame_id or
            any(row.get("image_sha256") != digest or
                row.get("image_space") != "autodex_undistorted_full_frame"
                for row in (mask, pose))):
        raise ValueError("mask, pose and saved PNG do not share a sensor frame")
    return {"sensor_frame_id": frame_id, "undistorted_pixel_sha256": digest}


def _save_stock_request(*, init: InitOrchestrator, object_name: str,
                        prompt: str, label: str, launch, output: Path,
                        timeout_s: float,
                        require_bound_frames: bool = False) -> tuple[dict, dict, dict]:
    """Save same-request undistorted PNG, mask and per-view pose."""
    req_id = _request_id()
    source = launch.capture_root / f"stock_qa_{output.name}_{label}_request_{req_id}"
    if source.exists():
        raise FileExistsError(source)
    masks, poses, timing = init.collect_payloads(
        prompt=prompt, request_id=req_id,
        n_expected_serials=len(launch.camera_ids), timeout_s=timeout_s,
        save_capture_dir=str(source))
    folder = output / label
    folder.mkdir()
    mask_dir = folder / "masks"
    mask_dir.mkdir()
    ids = set(masks) | set(poses)
    views = {}
    for serial in sorted(ids):
        raw = source / "images" / f"{serial}.png"
        image = _read_complete_png(
            raw, image_hw=launch.image_hw,
            timeout_s=max(90.0, launch.image_write_timeout_s))
        mask = masks.get(serial, {}).get("mask")
        mask = np.asarray(mask, dtype=bool) if mask is not None else None
        mask_file = mask_dir / f"{serial}.png"
        if mask is not None and not cv2.imwrite(
                str(mask_file), mask.astype(np.uint8) * 255):
            raise OSError(f"could not save {serial} mask")
        pose = poses.get(serial, {})
        source_frame = (_bound_frame_identity(masks[serial], pose, image)
                        if require_bound_frames and image is not None else None)
        world = pose.get("pose_world")
        if world is not None:
            world = np.asarray(world, dtype=float)
            world = world.tolist() if world.shape == (4, 4) and np.isfinite(world).all() else None
        views[serial] = {
            "raw_image": ({"path": str(raw), "sha256": _sha(raw),
                           "pixel_sha256": image_sha256(image)}
                          if image is not None else None),
            "source_frame": source_frame,
            "sam": {"received": serial in masks,
                    "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else None,
                    "mask_png": ({"path": str(mask_file), "sha256": _sha(mask_file)}
                                 if mask is not None else None)},
            "foundpose": {
                "received": serial in poses, "ok": pose.get("ok") is True,
                "quality": float(pose["quality"]) if "quality" in pose else None,
                "inliers": int(pose["inliers"]) if "inliers" in pose else None,
                "pose_world": world,
            },
        }
    payload = {
        "object": object_name, "prompt": prompt, "request_id": req_id,
        "source_capture_dir": str(source), "timing": timing,
        "received_mask_count": len(masks), "received_pose_count": len(poses),
        "missing_camera_ids": sorted(launch.camera_ids - ids), "views": views,
        "source_frame_identity": (
            "received_mask_pose_png_bound_to_sensor_frame"
            if require_bound_frames else "unavailable_from_stock_init_daemon"),
        "robot_ready": False,
    }
    _write_report(folder / "report.json", payload)
    if any(view["raw_image"] is None for view in views.values()):
        raise RuntimeError(
            f"{label}: stock daemon did not finish every source PNG; "
            f"inspect {folder / 'report.json'}")
    return payload, masks, poses


def _stock_refinement(init: InitOrchestrator, masks: dict, poses: dict) -> dict:
    """Run the unchanged IoU/silhouette selector on diagnostic payloads."""
    try:
        pose, details = init.refine_from_payloads(
            masks, poses, selection_mode="iou")
        return {"pose_world": None if pose is None else np.asarray(pose).tolist(),
                "details": details}
    except Exception as exc:
        return {"pose_world": None,
                "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--precision-init-port-mask", type=int)
    parser.add_argument("--precision-init-port-pose", type=int)
    parser.add_argument("--precision-init-port-cmd", type=int)
    args = parser.parse_args(argv)
    ports = (args.precision_init_port_mask, args.precision_init_port_pose,
             args.precision_init_port_cmd)
    if any(port is not None for port in ports) and (
            any(port is None for port in ports) or
            any(not 0 < port < 65536 for port in ports) or
            len(set(ports)) != 3):
        parser.error("precision init needs three distinct TCP ports")
    precision_ports = None if all(port is None for port in ports) else ports
    output = args.evidence_dir.expanduser().resolve()
    if not args.evidence_dir.is_absolute() or output.exists():
        parser.error("evidence directory must be a new absolute path")
    launch = replace(load_session_launch(
        args.launch_config, evidence_dir_override=output),
        execution_scope="camera_only_diagnostic")
    if launch.mode.family != "cylinder" or launch.board_id != "9":
        parser.error("this prepared diagnostic requires the cylinder and board 9")
    reject_incorrect_object_color(launch.socket_prompt)
    reject_incorrect_object_color(KEY_PROMPT)
    if not _confirm("Keep the key absent; place the fixed white socket on board 9, then press Enter (q to cancel): "):
        return 2
    output.mkdir(parents=True, exist_ok=False)
    paths = AssetPaths(launch.shared_root, launch.mode)
    report = {"schema": SCHEMA,
              "scope": ("precision_frame_bound_camera_only_diagnostic"
                        if precision_ports else "stock_camera_only_diagnostic"),
              "board_id": "9", "mode": {"family": "cylinder", "gap_mm": 20},
              "expected_camera_ids": sorted(launch.camera_ids),
              "calibration_scope": "current_launch_files_not_independent_robot_accuracy",
              "source_frame_identity": (
                  "init_mask_pose_png_bound_board_snapshot_fid_unavailable"
                  if precision_ports else
                  "stock_snapshot_and_init_frame_ids_unavailable_in_diagnostic"),
              "precision_init_ports": (None if precision_ports is None else
                                       dict(zip(("mask", "pose", "cmd"),
                                                precision_ports))),
              "timing_scope": "stationary_scene_no_exposure_alignment_claim",
              "robot_ready": False, "status": "started"}
    _write_report(output / "report.json", report)
    with open_autodex_camera_stream(launch.pc_list, name="precision_stock_camera_qa"):
        snap = SnapshotOrchestrator(list(launch.pc_list), list(launch.capture_ips))
        init_kwargs = ({} if precision_ports is None else dict(zip(
            ("port_mask", "port_pose", "port_cmd"), precision_ports)))
        stock_init = InitOrchestrator(
            list(launch.pc_list), list(launch.capture_ips), **init_kwargs)
        init = (stock_init if precision_ports is None else
                PrecisionInitOrchestrator(stock_init))
        try:
            board_dir = output / "board"
            board_dir.mkdir()
            board_frames, board_timing = snap.snap(
                n_expected=len(launch.camera_ids), timeout_s=launch.board_timeout_s,
                request_id=_request_id(), decode=True,
                save_dir_local=str(board_dir / "jpeg"))
            images = {s: row["image"] for s, row in board_frames.items()}
            if set(images) != launch.camera_ids or any(x is None for x in images.values()):
                raise RuntimeError("board snapshot did not return all 20 cameras")
            board = measure_tabletop_from_images(
                images, launch.intrinsics_full, launch.extrinsics_full,
                launch.c2r, board_id="9",
                minimum_board_corners=launch.minimum_board_corners)
            _write_report(board_dir / "measurement.json", board)
            report["board"] = {
                "timing": board_timing, "measurement": str(board_dir / "measurement.json"),
                "jpeg_sha256": {s: _sha(board_dir / "jpeg" / f"{s}.jpg")
                                for s in sorted(images)}}
            report["status"] = "board_measured"
            _write_report(output / "report.json", report)

            socket = launch.mode.socket_object
            prepare_precision_silhouette(
                init_orchestrator=stock_init, object_name=socket,
                raw_mesh=paths.raw_mesh(socket))
            init.init_object(
                obj_name=socket, mesh_path=str(paths.raw_mesh(socket)),
                assets_root=str(paths.foundpose_assets_root(socket)),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, mode="live",
                load_silhouette=False)
            socket_requests = []
            for index in range(launch.socket_capture_count):
                row, masks, poses = _save_stock_request(
                    init=init, object_name=socket, prompt=launch.socket_prompt,
                    label=f"socket_{index:03d}", launch=launch, output=output,
                    timeout_s=max(120.0, launch.socket_timeout_s),
                    require_bound_frames=precision_ports is not None)
                if precision_ports is not None and row["missing_camera_ids"]:
                    raise RuntimeError("precision socket capture missed cameras")
                row["stock_refinement"] = _stock_refinement(stock_init, masks, poses)
                _write_report(output / f"socket_{index:03d}" / "report.json", row)
                socket_requests.append(str(output / f"socket_{index:03d}" / "report.json"))
            report["socket_requests"] = socket_requests
            report["status"] = "socket_saved_awaiting_key"
            _write_report(output / "report.json", report)
            if not _confirm("Socket captures saved. Place the white cylindrical key without moving the socket, then press Enter (q to cancel): "):
                return 2

            key = launch.mode.key_object
            prepare_precision_silhouette(
                init_orchestrator=stock_init, object_name=key,
                raw_mesh=paths.raw_mesh(key))
            init.init_object(
                obj_name=key, mesh_path=str(paths.raw_mesh(key)),
                assets_root=str(paths.foundpose_assets_root(key)),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, mode="live",
                load_silhouette=False)
            row, masks, poses = _save_stock_request(
                init=init, object_name=key, prompt=KEY_PROMPT,
                label="key", launch=launch, output=output,
                timeout_s=120.0,
                require_bound_frames=precision_ports is not None)
            if precision_ports is not None and row["missing_camera_ids"]:
                raise RuntimeError("precision key capture missed cameras")
            row["stock_refinement"] = _stock_refinement(stock_init, masks, poses)
            _write_report(output / "key" / "report.json", row)
            report["key_request"] = str(output / "key" / "report.json")
            report["status"] = "complete_camera_only"
            _write_report(output / "report.json", report)
        finally:
            stock_init.close()
            snap.close()
    print(json.dumps({"report": str(output / "report.json"),
                      "robot_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
