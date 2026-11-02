#!/usr/bin/env python3
"""Render one saved socket measurement in its four calibrated camera views.

The figure is diagnostic only. It overlays the saved SAM mask and the frozen
socket bore/axis on the exact captured pixels; it does not rerun perception or
change the session calibration. View labels report the per-session robust
socket-centre selection rather than a persistent camera blacklist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SERIAL_ORDER = ("26053248", "25322639", "25305462", "25322642")


def _project(record: dict, serial: str, points_robot: np.ndarray) -> np.ndarray:
    camera = record["camera_calibration"]
    intrinsic = camera["intrinsics_full"][serial]
    camera_from_world = np.asarray(
        camera["extrinsics_full"][serial], dtype=np.float64)
    if camera_from_world.shape == (4, 4):
        camera_from_world = camera_from_world[:3]
    world_from_robot = np.asarray(record["c2r"], dtype=np.float64)
    points_world = (
        world_from_robot[:3, :3] @ points_robot.T
    ).T + world_from_robot[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(camera_from_world[:, :3])
    pixels, _ = cv2.projectPoints(
        points_world, rotation_vector, camera_from_world[:, 3],
        np.asarray(intrinsic["K_orig"], dtype=np.float64),
        np.asarray(intrinsic["dist_params"], dtype=np.float64))
    return np.rint(pixels.reshape(-1, 2)).astype(np.int32)


def _task_geometry_path(record: dict) -> Path:
    socket_object = record["mode"]["socket_object"]
    collision_mesh = Path(record["socket_collision_mesh"]).resolve()
    shared_root = collision_mesh.parents[4]
    path = (shared_root / "AutoDex/precision_insertion/fixtures" /
            socket_object / "task_geometry.json")
    if not path.is_file():
        raise FileNotFoundError(f"task geometry not found: {path}")
    return path


def _socket_geometry(
        record: dict, geometry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.asarray(record["socket_pose_robot"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("socket_pose_robot must be 4x4")
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    bore_radius = float(geometry["socket_bore_radius_m"])
    rim_z = float(geometry["socket_rim_z_m"])
    local_rim = np.stack([
        bore_radius * np.cos(angles), bore_radius * np.sin(angles),
        np.full_like(angles, rim_z)], axis=1)
    rim = (transform[:3, :3] @ local_rim.T).T + transform[:3, 3]
    center = (transform[:3, :3] @ np.array([0.0, 0.0, rim_z]) +
              transform[:3, 3])
    axis_tip = center + transform[:3, 2] * .060
    return rim, center, axis_tip


def render(evidence_dir: Path, output: Path) -> dict:
    root = Path(evidence_dir).expanduser().resolve()
    calibration_path = root / "session_calibration.json"
    payload_path = root / "socket/socket_000/payloads.json"
    record = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    geometry_path = _task_geometry_path(record)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    refinement = record["socket_capture_refinements"][0]
    body_fit = refinement["stock_selection"]["body_fit"]
    inliers = set(body_fit["socket_pose_inlier_camera_ids"])
    excluded = set(body_fit["socket_pose_excluded_camera_ids"])
    rim, center, axis_tip = _socket_geometry(record, geometry)

    panels: dict[str, np.ndarray] = {}
    rows = {}
    for serial in SERIAL_ORDER:
        image_path = root / f"socket/socket_000/images/{serial}.png"
        mask_path = root / f"socket/socket_000/masks/{serial}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise FileNotFoundError(f"missing or mismatched saved view: {serial}")
        view = image.copy()
        selected = mask > 0
        view[selected] = np.rint(
            .62 * view[selected] + .38 * np.array([0, 210, 255])
        ).clip(0, 255).astype(np.uint8)
        rim_px = _project(record, serial, rim)
        center_px = _project(record, serial, center[None])[0]
        axis_px = _project(record, serial, np.stack([center, axis_tip]))
        cv2.polylines(view, [rim_px], True, (255, 235, 0), 5, cv2.LINE_AA)
        cv2.drawMarker(view, tuple(center_px), (255, 235, 0),
                       cv2.MARKER_CROSS, 30, 4, cv2.LINE_AA)
        cv2.arrowedLine(view, tuple(axis_px[0]), tuple(axis_px[1]),
                        (40, 230, 80), 5, cv2.LINE_AA, tipLength=.18)
        used = serial in inliers
        status = "used" if used else "excluded this session"
        color = (70, 230, 90) if used else (80, 120, 255)
        residual = float(body_fit["per_view_center_residual_mm"][serial])
        label = f"{serial}  {status}  {residual:.2f} mm"
        cv2.putText(view, label, (34, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, (20, 20, 20), 7, cv2.LINE_AA)
        cv2.putText(view, label, (34, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, color, 3, cv2.LINE_AA)
        panels[serial] = view
        rows[serial] = {
            "used_for_socket_center": used,
            "center_residual_mm": residual,
            "mask_iou": float(refinement["per_view_iou"][serial]),
            "frame_id": payload["frame_evidence"][serial]["frame_id"],
        }

    canvas = np.full((1080, 1920, 3), (26, 31, 36), dtype=np.uint8)
    for serial, (x0, y0) in zip(
            SERIAL_ORDER, ((0, 0), (960, 0), (0, 540), (960, 540))):
        image = panels[serial]
        scale = min(940 / image.shape[1], 520 / image.shape[0])
        resized = cv2.resize(
            image, (round(image.shape[1] * scale),
                    round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA)
        x = x0 + (960 - resized.shape[1]) // 2
        y = y0 + (540 - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 5]):
        raise OSError(f"could not write {output}")
    metadata = {
        "schema": "precision_insertion_socket_measurement_overlay_v1",
        "evidence_dir": str(root),
        "image": str(output),
        "socket_pose_robot": record["socket_pose_robot"],
        "socket_pose_source": record["socket_pose_source"],
        "task_geometry": str(geometry_path),
        "socket_bore_radius_m": geometry["socket_bore_radius_m"],
        "socket_rim_z_m": geometry["socket_rim_z_m"],
        "views": rows,
        "legend": {
            "yellow_fill": "saved SAM socket mask",
            "cyan_ring_and_cross": "measured 20 mm bore rim and centre",
            "green_arrow": "measured socket +Z axis",
        },
        "scope": "saved_measurement_diagnostic_no_perception_or_pose_change",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.evidence_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
