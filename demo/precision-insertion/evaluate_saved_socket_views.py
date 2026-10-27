#!/usr/bin/env python3
"""Read-only leave-one-camera-out check of the fixed socket body fit.

Uses saved undistorted socket-only frames, SAM masks, and frozen session
calibration. Each displayed green pose is fit without that camera's mask.
This measures cross-view consistency, not physical ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from precision_insertion.board_socket_fit import _barrel_dimensions, fit_board_socket_body
from precision_insertion.frame_provenance import image_sha256


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projection(pose: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
                dimensions: tuple[float, float, float], shape: tuple[int, int],
                *, region: str = "wall",
                flange_dimensions: tuple[float, float, float] | None = None,
                ) -> tuple[np.ndarray, list[np.ndarray]]:
    selected = []
    if region in {"wall", "whole_body"}:
        selected.append(dimensions)
    if region in {"flange", "whole_body"}:
        if flange_dimensions is None:
            raise ValueError("flange projection needs its mesh dimensions")
        selected.append(flange_dimensions)
    if not selected:
        raise ValueError(f"unknown socket mask region: {region}")
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rings = []
    for radius, bottom, top in selected:
        for z in (bottom, top):
            points = np.stack((radius * np.cos(angles), radius * np.sin(angles),
                               np.full_like(angles, z), np.ones_like(angles)))
            camera = (extrinsic @ pose @ points)[:3]
            if np.any(camera[2] <= 0):
                raise ValueError("projected socket is behind camera")
            pixels = intrinsic @ camera
            rings.append(np.rint((pixels[:2] / pixels[2]).T).astype(np.int32))
    silhouette = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(silhouette, cv2.convexHull(np.vstack(rings)), 1)
    return silhouette, rings


def _crop(image: np.ndarray, mask: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 3)
    for ring in rings:
        cv2.polylines(canvas, [ring], True, (0, 210, 0), 3, cv2.LINE_AA)
    yy, xx = np.nonzero(mask)
    side = max(int(max(np.ptp(xx), np.ptp(yy)) * 1.55), 330)
    h, w = mask.shape
    side = min(side, h, w)
    cx, cy = int((xx.min() + xx.max()) / 2), int((yy.min() + yy.max()) / 2)
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return cv2.cvtColor(canvas[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB)


def evaluate(run_dir: Path, output: Path) -> dict:
    calibration_path = run_dir / "frozen_session_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    evidence = Path(json.loads((run_dir / "source_evidence_binding.json").read_text())
                    ["session_evidence_dir"])
    capture = evidence / "socket" / "socket_000"
    payload = json.loads((capture / "payloads.json").read_text())
    admitted = sorted(calibration["socket_capture_refinements"][0]
                      ["accepted_camera_ids"])
    root = Path(calibration["socket_collision_mesh"]).parents[2]
    mesh = root / "raw_mesh" / f"{calibration['mode']['socket_object']}.obj"
    if _digest(mesh) != calibration["socket_capture_refinements"][0]["raw_mesh_sha256"]:
        raise ValueError("saved socket mesh differs from frozen calibration")
    board = np.asarray(calibration["board"]["T_robot_board"], dtype=float)
    c2r = np.asarray(calibration["c2r"], dtype=float)
    intrinsic = {s: np.asarray(calibration["camera_calibration"]
                                ["intrinsics_full"][s]["K_undist"], dtype=float)
                 for s in admitted}
    extrinsic = {s: np.asarray(calibration["camera_calibration"]
                                ["extrinsics_full"][s], dtype=float)
                 for s in admitted}
    masks, images = {}, {}
    for serial in admitted:
        masks[serial] = cv2.imread(str(capture / "masks" / f"{serial}.png"),
                                   cv2.IMREAD_GRAYSCALE) > 127
        images[serial] = cv2.imread(str(capture / "images" / f"{serial}.png"))
        if images[serial] is None or masks[serial].shape != images[serial].shape[:2]:
            raise ValueError(f"missing or mismatched saved image/mask: {serial}")
        if image_sha256(images[serial]) != payload["frame_evidence"][serial]["image_sha256"]:
            raise ValueError(f"saved pixels differ from acquired frame: {serial}")
    full_pose, full = fit_board_socket_body(
        masks=masks, intrinsics_undist=intrinsic,
        extrinsics_camera_world=extrinsic, c2r=c2r,
        board_pose_robot=board, raw_mesh=mesh)
    # A camera contributes only one XY point to the median. Recomputing the
    # median of all other points is exactly the N-1 fit, without N expensive
    # full-resolution silhouette passes per held-out camera.
    points = full["per_view_board_xy_m"]
    dimensions = _barrel_dimensions(mesh)
    flange_dimensions = (full["flange_radius_m"],
                         full["flange_bottom_z_m"],
                         full["flange_top_z_m"])
    rows = {}
    fig, axes = plt.subplots((len(admitted) + 4) // 5, 5,
                             figsize=(20, 4 * ((len(admitted) + 4) // 5)))
    for ax, serial in zip(axes.flat, admitted):
        other = np.asarray([points[s] for s in admitted if s != serial])
        xy = np.median(other, axis=0)
        robot_pose = np.linalg.inv(c2r) @ full_pose
        robot_pose[:3, 3] = (board[:3, 3] + xy[0] * board[:3, 0]
                             - xy[1] * board[:3, 1])
        heldout_pose = c2r @ robot_pose
        model, rings = _projection(heldout_pose, extrinsic[serial],
                                   intrinsic[serial], dimensions,
                                   masks[serial].shape,
                                   region=full["per_view_selected_region"][serial],
                                   flange_dimensions=flange_dimensions)
        intersection = np.count_nonzero(model & masks[serial])
        union = np.count_nonzero(model | masks[serial])
        score = float(intersection / union) if union else 0.0
        shift_mm = float(np.linalg.norm(xy - full["board_xy_m"]) * 1000)
        image_path = capture / "images" / f"{serial}.png"
        mask_path = capture / "masks" / f"{serial}.png"
        rows[serial] = {
            "heldout_body_iou": score,
            "in_sample_body_iou": full["per_view_iou"][serial],
            "selected_mask_region": full["per_view_selected_region"][serial],
            "heldout_fit_shift_from_full_mm": shift_mm,
            "heldout_board_xy_m": xy.tolist(),
            "frame_id": payload["frame_evidence"][serial]["frame_id"],
            "capture_timestamp_s": payload["frame_evidence"][serial]["timestamp_s"],
            "capture_timestamp_max_error_s": payload["frame_evidence"][serial]["max_error_s"],
            "acquisition_image_sha256": payload["frame_evidence"][serial]["image_sha256"],
            "image_file_sha256": _digest(image_path),
            "mask_file_sha256": _digest(mask_path),
        }
        ax.imshow(_crop(images[serial], masks[serial], rings))
        ax.set_title(f"{serial}  |  held out IoU {score:.3f}\n"
                     f"center shift {shift_mm:.2f} mm", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(admitted):]:
        ax.axis("off")
    fig.suptitle(f"{run_dir.name}: real undistorted captures | cyan SAM; "
                 "green prediction fitted WITHOUT this camera", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / f"heldout_all_views_{run_dir.name}.png", dpi=120)
    plt.close(fig)
    return {
        "run_id": run_dir.name,
        "calibration_sha256": _digest(calibration_path),
        "raw_mesh_sha256": _digest(mesh),
        "evidence_dir": str(evidence),
        "source_method": "2026-10-21 saved stock capture; board/body read-only replay",
        "full_board_xy_m": full["board_xy_m"],
        "full_mean_body_iou": full["mean_body_iou"],
        "views": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [evaluate(run_dir, args.output) for run_dir in args.run_dir]
    (args.output / "heldout_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n")
    for result in results:
        scores = [row["heldout_body_iou"] for row in result["views"].values()]
        shifts = [row["heldout_fit_shift_from_full_mm"]
                  for row in result["views"].values()]
        print(result["run_id"], "views", len(scores), "mean IoU",
              f"{np.mean(scores):.3f}", "min IoU", f"{min(scores):.3f}",
              "max shift mm", f"{max(shifts):.3f}")


if __name__ == "__main__":
    main()
