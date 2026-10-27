"""One-capture socket localization before fixture calibration.

The stock path remains available for comparison.  The fixed-cylinder path
uses the acquired masks and board pose to avoid fitting the full flange mesh
to masks that contain only the body.  Neither path captures or moves a robot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .board_socket_fit import fit_board_socket_body
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
    perception_mode: str = "iou",
    board_pose_robot: np.ndarray | None = None,
    c2r: np.ndarray | None = None,
) -> SocketCaptureRefinement:
    """Localize one socket from admitted AutoDex camera evidence.

    Board-constrained body mode fits the fixed upright cylinder on CPU. An
    IoU limit of zero records the score without vetoing its estimated pose.
    The legacy stock
    modes retain their existing record format for saved-session replay.
    In those modes, the IoU threshold labels supporting views. A low-scoring
    view uses the stock silhouette result without an extra all-view veto.
    ``ignore_sil_loss`` matches stock run_auto's named mode:
    it still selects by IoU and optimizes the silhouette, but records that
    its absolute loss cutoff was not applied. Neither mode proves absolute
    metric accuracy or identity.
    """
    if (not isinstance(capture, SocketCaptureInput) or
            not isinstance(admission, SocketCaptureEvidence) or
            admission.capture_id != capture.capture_id):
        raise ValueError("refinement needs matching captured/admitted evidence")
    threshold = float(minimum_refined_iou)
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1 or
            (threshold == 0 and perception_mode != "board_constrained_body")):
        raise ValueError("socket refinement needs a valid body/mesh IoU limit")
    if (not isinstance(perception_mode, str) or
            perception_mode not in ("iou", "ignore_sil_loss",
                                    "board_constrained_body")):
        raise ValueError("unknown socket perception mode")
    if perception_mode != "board_constrained_body":
        loss_limit = float(silhouette_loss_threshold)
        if (not math.isfinite(loss_limit) or loss_limit <= 0 or
                type(silhouette_iterations) is not int or
                silhouette_iterations <= 0):
            raise ValueError("stock silhouette refinement needs positive limits")
    accepted = tuple(item.camera_id for item in admission.observations)
    if len(accepted) < 2 or len(set(accepted)) != len(accepted):
        raise ValueError("socket refinement needs at least two admitted views")
    stock = getattr(init_orchestrator, "stock", init_orchestrator)
    renderer = getattr(stock, "_sil", None)
    mesh = Path(raw_mesh).expanduser().resolve()
    mesh_sha = _file_sha256(mesh)
    if getattr(stock, "obj_name", None) != socket_object:
        raise ValueError("socket perception is bound to another object")
    if perception_mode != "board_constrained_body" and (
            renderer is None or
            getattr(renderer, "_obj_name", None) != socket_object or
            getattr(renderer, "_precision_mesh_sha256", None) != mesh_sha):
        raise ValueError("socket stock renderer is not bound to the exact raw mesh")
    if not all(camera in stock.intrinsics_undist and
               camera in stock.extrinsics and
               camera in capture.masks and
               (perception_mode == "board_constrained_body" or
                camera in capture.poses)
               for camera in accepted):
        raise ValueError("socket refinement camera/calibration payload is incomplete")

    if perception_mode == "board_constrained_body":
        if board_pose_robot is None or c2r is None:
            raise ValueError("board-constrained socket fit needs board pose and C2R")
        pose, body_fit = fit_board_socket_body(
            masks={camera: capture.masks[camera]["mask"] for camera in accepted},
            intrinsics_undist={camera: stock.intrinsics_undist[camera]
                               for camera in accepted},
            extrinsics_camera_world={camera: stock.extrinsics[camera]
                                     for camera in accepted},
            c2r=c2r, board_pose_robot=board_pose_robot, raw_mesh=mesh)
        if body_fit["mean_body_iou"] < threshold:
            raise ValueError(
                "socket body projection does not match acquired masks: "
                f"mean IoU {body_fit['mean_body_iou']:.3f} < {threshold:.3f}")
        per_view = body_fit["per_view_iou"]
        selected = max(per_view, key=per_view.__getitem__)
        return SocketCaptureRefinement(
            capture_id=capture.capture_id, request_id=capture.request_id,
            pose_world=pose, accepted_camera_ids=accepted,
            selected_camera_id=selected, per_view_iou=per_view,
            minimum_refined_iou=threshold,
            stock_selection={
                "selection_mode": "board_constrained_body",
                "best_serial": selected,
                "n_candidates": len(accepted), "n_masks": len(accepted),
                "sil_skipped": True, "silhouette_iterations": 0,
                "perception_mode": perception_mode,
                "body_fit": _json_native(body_fit),
            },
            raw_mesh_sha256=mesh_sha,
            refinement_gate="board_constrained_body")

    if not isinstance(stock.H, int) or not isinstance(stock.W, int):
        raise ValueError("stock socket refinement needs image dimensions")
    refined, selection = stock.refine_from_payloads(
        capture.masks, capture.poses, subset_serials=list(accepted),
        sil_iters=silhouette_iterations, sil_lr=0.002,
        sil_loss_threshold=(float("inf") if perception_mode == "ignore_sil_loss"
                            else loss_limit), selection_mode="iou")
    if (refined is None or not isinstance(selection, dict) or
            selection.get("best_serial") not in accepted or
            selection.get("n_candidates") != len(accepted) or
            selection.get("n_masks") != len(accepted) or
            selection.get("sil_reject") or
            selection.get("sil_skipped") or
            not math.isfinite(float(selection.get("sil_loss", float("nan")))) or
            (perception_mode == "iou" and
             float(selection["sil_loss"]) > loss_limit)):
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
            else ("stock_ignore_sil_loss" if perception_mode == "ignore_sil_loss"
                  else "stock_silhouette"))
    stock_record = _json_native(selection)
    stock_record["post_refinement_mean_iou"] = float(mean_iou)
    stock_record["post_refinement_iou_support_camera_ids"] = sorted(
        camera for camera, score in per_view.items() if float(score) >= threshold)
    stock_record["selection_mode"] = "iou"
    stock_record["perception_mode"] = perception_mode
    stock_record["silhouette_iterations"] = silhouette_iterations
    stock_record["silhouette_loss_gate_applied"] = perception_mode == "iou"
    stock_record["silhouette_loss_threshold"] = (
        loss_limit if perception_mode == "iou" else None)
    return SocketCaptureRefinement(
        capture_id=capture.capture_id, request_id=capture.request_id,
        pose_world=pose, accepted_camera_ids=accepted,
        selected_camera_id=selection["best_serial"],
        per_view_iou={camera: float(per_view[camera]) for camera in accepted},
        minimum_refined_iou=threshold, stock_selection=stock_record,
        raw_mesh_sha256=mesh_sha, refinement_gate=gate)
