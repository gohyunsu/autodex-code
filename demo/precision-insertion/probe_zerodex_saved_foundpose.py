#!/usr/bin/env python3
"""Replay FoundPose on one saved ZeroDex four-view SAM capture, without cameras.

The PNGs came from one init request and are already undistorted. Masks are
saved from that request's live probe; its report binds the request ID. This
diagnostic does not measure a robot pose, certify object identity or move a
robot. Run it in capture4's existing gotrack environment, from our namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np


_REQUEST_SUFFIX = re.compile(r"_request_(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(capture_dir: Path, mask_dir: Path, calib_dir: Path,
                 probe_report: Path, object_name: str) -> list[dict]:
    match = _REQUEST_SUFFIX.search(capture_dir.name)
    if match is None:
        raise ValueError("capture directory needs its original request ID")
    report = json.loads(probe_report.read_text(encoding="utf-8"))
    if (report.get("socket_request_id") != int(match.group(1)) or
            report.get("target_object") != object_name):
        raise ValueError("saved probe and source frames describe different requests")
    intrinsics = json.loads((calib_dir / "intrinsics.json").read_text())
    extrinsics = json.loads((calib_dir / "extrinsics.json").read_text())
    camera_ids = sorted(report["views"])
    views = []
    for serial in camera_ids:
        image_path = capture_dir / "images" / f"{serial}.png"
        mask_path = mask_dir / f"{serial}_mask.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if (image is None or mask is None or
                image.shape[:2] != mask.shape or
                not np.isin(mask, (0, 255)).all() or
                int(np.count_nonzero(mask)) !=
                report["views"][serial]["mask_pixels"]):
            raise ValueError(f"saved image/mask mismatch for {serial}")
        K = np.asarray(intrinsics[serial]["intrinsics_undistort"], dtype=float)
        ext = np.asarray(extrinsics[serial], dtype=float)
        if ext.shape == (3, 4):
            ext = np.vstack((ext, [0., 0., 0., 1.]))
        if K.shape != (3, 3) or ext.shape != (4, 4):
            raise ValueError(f"calibration matrix shape mismatch for {serial}")
        views.append({
            "serial": serial,
            "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "mask": mask > 0,
            "K": K,
            "extrinsics": ext,
            "image_path": str(image_path),
            "image_sha256": _sha256(image_path),
            "mask_path": str(mask_path),
            "mask_sha256": _sha256(mask_path),
        })
    return views


def _refine_core(views: list[dict], rows: list[dict], mesh: Path) -> dict:
    """Diagnose core AutoDex IoU and capture4 silhouette on saved views.

    The capture4 checkout predates local InitOrchestrator.refine_from_payloads,
    so this is explicitly not the full current runtime method.
    """
    from autodex.perception.pose_select import (compute_cross_view_iou,
                                                select_best_pose_by_iou)
    from precision_insertion.silhouette_compat import SolidMtlSilhouetteOptimizer

    size = {view["mask"].shape for view in views}
    if len(size) != 1:
        raise ValueError("stock renderer needs same-sized views")
    height, width = size.pop()
    renderer = SolidMtlSilhouetteOptimizer(str(mesh), device="cuda")
    intrinsics = {view["serial"]: view["K"] for view in views}
    extrinsics = {view["serial"]: view["extrinsics"] for view in views}
    masks = {view["serial"]: view["mask"] for view in views}
    poses = {row["serial"]: np.asarray(row["pose_world_m"])
             for row in rows if row["status"] == "pose_estimated_not_validated"}
    started = time.perf_counter()
    selected, pose, best_iou, candidate_ious = select_best_pose_by_iou(
        poses, masks, intrinsics, extrinsics, height, width,
        renderer.glctx, renderer.mesh_tensors)
    if pose is None:
        return {"method": "AutoDex_core_IoU_capture4_silhouette",
                "selected": None, "candidate_mean_ious": candidate_ious,
                "scope": "diagnostic_not_current_runtime_refinement"}
    initial_pose = np.asarray(pose).tolist()
    fitted, loss = renderer.optimize(
        pose, [{"mask": mask.astype(np.uint8) * 255,
                "K": intrinsics[serial], "extrinsic": extrinsics[serial]}
               for serial, mask in masks.items()],
        iters=100, lr=0.002, antialias=True)
    fitted_iou, per_view_iou = compute_cross_view_iou(
        fitted, masks, intrinsics, extrinsics, height, width,
        renderer.glctx, renderer.mesh_tensors)
    return {
        "method": "AutoDex_core_IoU_capture4_silhouette",
        "elapsed_s": time.perf_counter() - started,
        "selected": selected, "candidate_mean_ious": candidate_ious,
        "selected_initial_mean_iou": best_iou,
        "selected_initial_pose_world_m": initial_pose,
        "fitted_mean_iou": fitted_iou, "fitted_per_view_iou": per_view_iou,
        "silhouette_loss": float(loss),
        "passes_current_silhouette_loss_0_003": float(loss) <= 0.003,
        "fitted_pose_world_m": np.asarray(fitted).tolist(),
        "scope": "capture4_core_diagnostic_not_current_runtime_refinement_or_metric_truth",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-refine", action="store_true",
                        help="diagnose AutoDex IoU and capture4 silhouette fit")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    views = _load_inputs(
        args.capture_dir.expanduser().resolve(),
        args.mask_dir.expanduser().resolve(),
        args.calib_dir.expanduser().resolve(),
        args.probe_report.expanduser().resolve(), args.object_name)
    # Match the namespaced live daemon: capture4's xFormers build cannot run
    # this DINOv2 float32 attention on its GPU, while stock DINOv2 supports
    # PyTorch SDPA without changing any shared package or source checkout.
    os.environ["XFORMERS_DISABLED"] = "1"
    from autodex.perception.foundpose_init import FoundPoseInit

    started = time.perf_counter()
    model = FoundPoseInit(
        str(args.mesh.expanduser().resolve()),
        str(args.assets_root.expanduser().resolve()), args.object_name)
    model_load_s = time.perf_counter() - started
    rows = []
    for view in views:
        started = time.perf_counter()
        result = model.estimate_one_view(
            view["rgb"], view["mask"], view["K"], view["extrinsics"])
        row = {key: value for key, value in view.items()
               if key not in {"rgb", "mask", "K", "extrinsics"}}
        row["mask_pixels"] = int(view["mask"].sum())
        row["elapsed_s"] = time.perf_counter() - started
        if result is None:
            row["status"] = "no_pose"
        else:
            row.update({
                "status": "pose_estimated_not_validated",
                "pose_world_m": np.asarray(result["pose_world"], dtype=float).tolist(),
                "quality": float(result["quality"]),
                "inliers": int(result["inliers"]),
                "template_id": int(result["template_id"]),
                "timings": result["timings"],
            })
        rows.append(row)
    record = {
        "schema": "precision_insertion_zerodex_saved_foundpose_probe_v1",
        "scope": "saved_same_request_camera_pose_diagnostic_not_robot_validation",
        "object_name": args.object_name,
        "capture_dir": str(args.capture_dir),
        "probe_report": str(args.probe_report),
        "model_load_s": model_load_s,
        "views": rows,
        "robot_ready": False,
    }
    if args.core_refine:
        record["core_refinement"] = _refine_core(
            views, rows, args.mesh.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "model_load_s": model_load_s,
                      "status": {row["serial"]: row["status"] for row in rows}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
