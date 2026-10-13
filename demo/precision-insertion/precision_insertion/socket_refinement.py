"""One-capture stock AutoDex socket refinement before fixture calibration.

The original per-view SAM/FoundPose evidence is admitted first. This module
selects/refines its poses and records the score against every admitted mask;
it does not capture images or command the robot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .calibration import (
    SocketCaptureRefinement, _file_sha256, _json_native,
)
from .geometry import validate_se3
from .perception_evidence import SocketCaptureEvidence
from .session_bootstrap import SocketCaptureInput


def refine_socket_capture(
    *, capture: SocketCaptureInput, admission: SocketCaptureEvidence,
    init_orchestrator, socket_object: str, raw_mesh: Path,
    minimum_refined_iou: float, silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
) -> SocketCaptureRefinement:
    """Run stock IoU selection and silhouette, retaining all view scores.

    The supplied IoU threshold labels strong supporting views. When a view
    misses it, use the stock silhouette result rather than imposing an extra
    all-view veto. Neither gate proves absolute metric accuracy or identity.
    """
    if (not isinstance(capture, SocketCaptureInput) or
            not isinstance(admission, SocketCaptureEvidence) or
            admission.capture_id != capture.capture_id):
        raise ValueError("refinement needs matching captured/admitted evidence")
    threshold = float(minimum_refined_iou)
    loss_limit = float(silhouette_loss_threshold)
    if (not math.isfinite(threshold) or not 0 < threshold <= 1 or
            not math.isfinite(loss_limit) or loss_limit <= 0 or
            type(silhouette_iterations) is not int or
            silhouette_iterations <= 0):
        raise ValueError("socket refinement needs commissioned positive limits")
    accepted = tuple(item.camera_id for item in admission.observations)
    if len(accepted) < 2 or len(set(accepted)) != len(accepted):
        raise ValueError("socket refinement needs at least two admitted views")
    stock = getattr(init_orchestrator, "stock", init_orchestrator)
    renderer = getattr(stock, "_sil", None)
    mesh = Path(raw_mesh).expanduser().resolve()
    mesh_sha = _file_sha256(mesh)
    if (getattr(stock, "obj_name", None) != socket_object or
            renderer is None or
            getattr(renderer, "_obj_name", None) != socket_object or
            getattr(renderer, "_precision_mesh_sha256", None) != mesh_sha):
        raise ValueError("socket stock renderer is not bound to the exact raw mesh")
    if (not all(camera in stock.intrinsics_undist and
                camera in stock.extrinsics and
                camera in capture.masks and camera in capture.poses
                for camera in accepted) or
            not isinstance(stock.H, int) or not isinstance(stock.W, int)):
        raise ValueError("socket refinement camera/calibration payload is incomplete")

    refined, selection = stock.refine_from_payloads(
        capture.masks, capture.poses, subset_serials=list(accepted),
        sil_iters=silhouette_iterations, sil_lr=0.002,
        sil_loss_threshold=loss_limit, selection_mode="iou")
    if (refined is None or not isinstance(selection, dict) or
            selection.get("best_serial") not in accepted or
            selection.get("n_candidates") != len(accepted) or
            selection.get("n_masks") != len(accepted) or
            selection.get("sil_reject") or
            selection.get("sil_skipped") or
            not math.isfinite(float(selection.get("sil_loss", float("nan")))) or
            float(selection["sil_loss"]) > loss_limit):
        raise ValueError("stock socket IoU/silhouette refinement did not pass")
    pose = validate_se3(refined, name="stock refined socket pose")

    from autodex.perception.pose_select import compute_cross_view_iou

    masks = {camera: np.asarray(capture.masks[camera]["mask"], dtype=bool)
             for camera in accepted}
    intrinsics = {camera: stock.intrinsics_undist[camera]
                  for camera in accepted}
    extrinsics = {camera: stock.extrinsics[camera] for camera in accepted}
    mean_iou, per_view = compute_cross_view_iou(
        pose, masks, intrinsics, extrinsics, stock.H, stock.W,
        renderer.glctx, renderer.mesh_tensors)
    if (set(per_view) != set(accepted) or
            not math.isfinite(float(mean_iou)) or
            any(not math.isfinite(float(score)) or
                    not 0 <= float(score) <= 1
                    for score in per_view.values())):
        raise ValueError("refined socket has invalid per-view IoU scores")
    gate = ("all_view_iou" if all(float(score) >= threshold
                                  for score in per_view.values())
            else "stock_silhouette")
    stock_record = _json_native(selection)
    stock_record["post_refinement_mean_iou"] = float(mean_iou)
    stock_record["post_refinement_iou_support_camera_ids"] = sorted(
        camera for camera, score in per_view.items() if float(score) >= threshold)
    stock_record["selection_mode"] = "iou"
    stock_record["silhouette_iterations"] = silhouette_iterations
    stock_record["silhouette_loss_threshold"] = loss_limit
    return SocketCaptureRefinement(
        capture_id=capture.capture_id, request_id=capture.request_id,
        pose_world=pose, accepted_camera_ids=accepted,
        selected_camera_id=selection["best_serial"],
        per_view_iou={camera: float(per_view[camera]) for camera in accepted},
        minimum_refined_iou=threshold, stock_selection=stock_record,
        raw_mesh_sha256=mesh_sha, refinement_gate=gate)
