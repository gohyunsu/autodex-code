"""Optional ChArUco-registered key ROI, separate from the stock live path.

The saved NAS evaluator localized a small tabletop key by registering a
socket-only board image to a later key image, excluding the socket, then
running SAM3 on the remaining crop. This module factors that *proposal* out
for a demo-local producer. Its masks are not FoundPose poses, and the baseline
must be from the same frozen board/camera session before live use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def detector_from_board_config(
    *, board_config_path: Path, board_id: str,
) -> Callable:
    """Build the exact ParaDex ChArUco board without mutating system/current.

    A workstation may have a different ``paradex/system/current`` symlink
    from the AutoDex rig. The caller must supply its rig's actual config;
    missing board IDs fail before processing any saved image.
    """
    if not isinstance(board_id, str) or not board_id:
        raise ValueError("ChArUco board ID is required")
    record = json.loads(Path(board_config_path).expanduser().resolve().read_text(
        encoding="utf-8"))
    item = record.get(board_id) if isinstance(record, dict) else None
    if not isinstance(item, dict):
        raise ValueError(f"board {board_id} is absent from explicit config")
    try:
        size = (int(item["numX"]), int(item["numY"]))
        checker = float(item["checkerLength"])
        marker = float(item["markerLength"])
        dictionary_name = "DICT_" + item["dict_type"]
        marker_ids = np.asarray(item["markerIDs"], dtype=np.int32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid explicit ChArUco board config") from exc
    if (min(size) < 2 or not 0 < marker < checker or
            marker_ids.ndim != 1 or len(marker_ids) == 0 or
            len(set(marker_ids.tolist())) != len(marker_ids) or
            not hasattr(cv2.aruco, dictionary_name)):
        raise ValueError("invalid explicit ChArUco geometry or dictionary")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name))
    board = cv2.aruco.CharucoBoard(
        size, checker, marker, dictionary, marker_ids)
    board.setLegacyPattern(bool(item.get("setLegacyPattern", False)))
    detector = cv2.aruco.CharucoDetector(board)

    def detect(image: np.ndarray, *, board_ids: list[str]) -> dict:
        if board_id not in board_ids:
            return {}
        corners, ids, _marker_corners, _marker_ids = detector.detectBoard(image)
        if ids is None or len(ids) == 0:
            return {}
        return {board_id: {"checkerCorner": corners.reshape(-1, 2),
                           "checkerIDs": ids.reshape(-1)}}

    return detect


@dataclass(frozen=True)
class RegisteredKeyROILimits:
    minimum_shared_corners: int
    minimum_homography_inliers: int
    homography_ransac_px: float
    maximum_board_shift_px: float
    difference_threshold: int
    socket_dilation_px: int
    closing_kernel_px: int
    minimum_component_pixels: int
    maximum_competitor_fraction: float
    crop_pad_px: int
    crop_pad_fraction: float
    minimum_sam_pixels: int
    maximum_sam_pixels: int
    minimum_difference_support_fraction: float
    maximum_socket_overlap_fraction: float

    def validate(self) -> None:
        ints = (self.minimum_shared_corners, self.minimum_homography_inliers,
                self.difference_threshold, self.socket_dilation_px,
                self.closing_kernel_px, self.minimum_component_pixels,
                self.crop_pad_px, self.minimum_sam_pixels,
                self.maximum_sam_pixels)
        if (any(type(value) is not int for value in ints) or
                self.minimum_shared_corners < 8 or
                not 4 <= self.minimum_homography_inliers <=
                self.minimum_shared_corners or
                not 0 < self.difference_threshold < 256 or
                self.socket_dilation_px < 0 or
                self.closing_kernel_px < 1 or
                self.closing_kernel_px % 2 != 1 or
                self.minimum_component_pixels < 1 or
                self.crop_pad_px < 0 or
                self.minimum_sam_pixels < 1 or
                self.maximum_sam_pixels < self.minimum_sam_pixels):
            raise ValueError("invalid registered-key ROI pixel limits")
        positives = (self.homography_ransac_px, self.maximum_board_shift_px)
        fractions = (self.maximum_competitor_fraction,
                     self.crop_pad_fraction,
                     self.minimum_difference_support_fraction,
                     self.maximum_socket_overlap_fraction)
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in positives) or
                any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in fractions) or
                not 0 <= self.maximum_competitor_fraction < 1 or
                not 0 <= self.crop_pad_fraction <= 1 or
                not 0 < self.minimum_difference_support_fraction <= 1 or
                not 0 <= self.maximum_socket_overlap_fraction < 1):
            raise ValueError("invalid registered-key ROI geometric limits")


@dataclass(frozen=True)
class KeyROIProposal:
    status: str
    reason: str
    crop_xyxy: tuple[int, int, int, int] | None
    difference_component: np.ndarray | None
    socket_exclusion: np.ndarray | None
    diagnostics: dict

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_registered_key_roi_v1",
            "status": self.status, "reason": self.reason,
            "crop_xyxy": (None if self.crop_xyxy is None else
                          list(self.crop_xyxy)),
            "diagnostics": self.diagnostics,
            "scope": "tabletop_roi_proposal_not_pose_or_robot_permission",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class KeyROIMask:
    status: str
    reason: str
    mask: np.ndarray | None
    diagnostics: dict

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_registered_key_mask_v1",
            "status": self.status, "reason": self.reason,
            "diagnostics": self.diagnostics,
            "scope": "assisted_sam_mask_not_foundpose_or_robot_permission",
            "robot_ready": False,
        }


def _board_points(detector: Callable, image: np.ndarray,
                  board_id: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    result = detector(image, board_ids=[board_id])
    item = result.get(board_id) if isinstance(result, dict) else None
    if not isinstance(item, dict):
        raise ValueError("ChArUco board was not detected")
    ids = np.asarray(item.get("checkerIDs")).reshape(-1)
    corners = np.asarray(item.get("checkerCorner"), dtype=float).reshape(-1, 2)
    if (len(ids) != len(corners) or len(ids) != len(set(ids.tolist())) or
            not np.all(np.isfinite(corners))):
        raise ValueError("ChArUco board corners or IDs are invalid")
    return {int(fid): point for fid, point in zip(ids, corners)}, corners


def propose_registered_key_roi(
    *, current_bgr: np.ndarray, socket_only_bgr: np.ndarray,
    socket_only_mask: np.ndarray, board_detector: Callable,
    board_id: str, limits: RegisteredKeyROILimits,
) -> KeyROIProposal:
    """Propose one tabletop-key crop from a registered same-camera baseline.

    The selected component is only a localization hint. It must be followed
    by SAM mask QA, FoundPose and multi-view/fixture checks; a large lighting
    change can also create a connected difference component.
    """
    limits.validate()
    if (not callable(board_detector) or not isinstance(board_id, str) or
            not board_id or not isinstance(current_bgr, np.ndarray) or
            current_bgr.dtype != np.uint8 or current_bgr.ndim != 3 or
            current_bgr.shape[2] != 3 or
            not isinstance(socket_only_bgr, np.ndarray) or
            socket_only_bgr.shape != current_bgr.shape or
            socket_only_bgr.dtype != np.uint8 or
            not isinstance(socket_only_mask, np.ndarray) or
            socket_only_mask.shape != current_bgr.shape[:2]):
        raise ValueError("ROI needs same-camera undistorted BGR and socket mask")
    if socket_only_mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ValueError("socket-only mask must be boolean or uint8")

    def abstain(reason: str, **diagnostics) -> KeyROIProposal:
        return KeyROIProposal("abstain", reason, None, None, None, diagnostics)

    try:
        old, _ = _board_points(board_detector, socket_only_bgr, board_id)
        new, current_corners = _board_points(
            board_detector, current_bgr, board_id)
    except (ValueError, TypeError):
        return abstain("board_corners_unavailable")
    common = sorted(old.keys() & new.keys())
    if len(common) < limits.minimum_shared_corners:
        return abstain("insufficient_shared_board_corners",
                       shared_corners=len(common))
    src = np.asarray([old[fid] for fid in common], dtype=np.float32)
    dst = np.asarray([new[fid] for fid in common], dtype=np.float32)
    board_shift = float(np.linalg.norm(dst - src, axis=1).max())
    if board_shift > limits.maximum_board_shift_px:
        return abstain("board_or_camera_moved_too_far",
                       maximum_board_shift_px=board_shift)
    homography, inlier = cv2.findHomography(
        src, dst, cv2.RANSAC, limits.homography_ransac_px)
    if (homography is None or inlier is None or
            not np.all(np.isfinite(homography)) or
            int(inlier.sum()) < limits.minimum_homography_inliers):
        return abstain("board_registration_failed",
                       shared_corners=len(common))
    h, w = current_bgr.shape[:2]
    aligned = cv2.warpPerspective(socket_only_bgr, homography, (w, h))
    board_hull = cv2.convexHull(
        current_corners.astype(np.float32)).astype(np.int32)
    board = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(board, board_hull, 1)
    registered_socket = cv2.warpPerspective(
        (socket_only_mask > 0).astype(np.uint8), homography, (w, h),
        flags=cv2.INTER_NEAREST)
    radius = limits.socket_dilation_px
    side = 2 * radius + 1
    exclusion = cv2.dilate(
        registered_socket, np.ones((side, side), dtype=np.uint8)) > 0
    difference = (np.max(cv2.absdiff(aligned, current_bgr), axis=2) >
                  limits.difference_threshold).astype(np.uint8)
    difference *= board
    difference[exclusion] = 0
    kernel = np.ones((limits.closing_kernel_px,
                      limits.closing_kernel_px), dtype=np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_CLOSE, kernel)
    difference[exclusion] = 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        difference, connectivity=8)
    components = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
         if stats[i, cv2.CC_STAT_AREA] >= limits.minimum_component_pixels),
        reverse=True)
    diagnostic = {
        "shared_corners": len(common),
        "homography_inliers": int(inlier.sum()),
        "maximum_board_shift_px": board_shift,
        "candidate_components": len(components),
        "homography_socket_only_to_current": homography.tolist(),
    }
    if not components:
        return abstain("no_key_difference_component", **diagnostic)
    first_area, first_id = components[0]
    diagnostic["largest_component_pixels"] = first_area
    if len(components) > 1:
        competitor = components[1][0] / first_area
        diagnostic["second_to_first_area_fraction"] = competitor
        if competitor > limits.maximum_competitor_fraction:
            return abstain("ambiguous_difference_components", **diagnostic)
    x, y, cw, ch = (int(stats[first_id, index]) for index in
                    (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                     cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    pad = max(limits.crop_pad_px,
              int(math.ceil(limits.crop_pad_fraction * max(cw, ch))))
    crop = (max(0, x - pad), max(0, y - pad),
            min(w, x + cw + pad), min(h, y + ch + pad))
    component = labels == first_id
    return KeyROIProposal(
        "roi_available", "largest_registered_difference_component", crop,
        component, exclusion, diagnostic)


def segment_registered_key_roi(
    *, current_rgb: np.ndarray, proposal: KeyROIProposal,
    sam_segmentor, prompt: str, limits: RegisteredKeyROILimits,
) -> KeyROIMask:
    """Run the existing SAM3 segmentor on the crop, then reject bad masks."""
    limits.validate()
    if (not isinstance(current_rgb, np.ndarray) or
            current_rgb.dtype != np.uint8 or current_rgb.ndim != 3 or
            current_rgb.shape[2] != 3 or
            not isinstance(proposal, KeyROIProposal) or
            not isinstance(prompt, str) or not prompt.strip() or
            not callable(getattr(sam_segmentor, "segment", None))):
        raise ValueError("ROI SAM needs an RGB frame, proposal and segmentor")
    if proposal.status != "roi_available":
        return KeyROIMask("abstain", "no_registered_key_roi", None,
                          {"roi_status": proposal.status,
                           "roi_reason": proposal.reason})
    if (proposal.crop_xyxy is None or
            proposal.difference_component is None or
            proposal.socket_exclusion is None or
            proposal.difference_component.shape != current_rgb.shape[:2] or
            proposal.socket_exclusion.shape != current_rgb.shape[:2]):
        raise ValueError("ROI proposal has inconsistent full-frame geometry")
    x0, y0, x1, y1 = proposal.crop_xyxy
    h, w = current_rgb.shape[:2]
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError("ROI crop lies outside original image")
    cropped = sam_segmentor.segment(current_rgb[y0:y1, x0:x1], prompt)
    if cropped is None:
        return KeyROIMask("abstain", "sam_returned_no_mask", None, {})
    cropped = np.asarray(cropped)
    if cropped.shape != (y1 - y0, x1 - x0):
        raise ValueError("cropped SAM mask has incorrect pixel geometry")
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = cropped > 0
    area = int(mask.sum())
    if area == 0:
        return KeyROIMask("abstain", "sam_mask_empty", None,
                          {"mask_pixels": 0})
    overlap = float(np.logical_and(mask, proposal.socket_exclusion).sum() / area)
    support = float(np.logical_and(
        mask, cv2.dilate(proposal.difference_component.astype(np.uint8),
                         np.ones((7, 7), dtype=np.uint8)) > 0).sum() / area)
    diagnostic = {
        "mask_pixels": area, "socket_overlap_fraction": overlap,
        "difference_support_fraction": support,
    }
    if not limits.minimum_sam_pixels <= area <= limits.maximum_sam_pixels:
        reason = "sam_mask_area_outside_limit"
    elif overlap > limits.maximum_socket_overlap_fraction:
        reason = "sam_mask_overlaps_socket"
    elif support < limits.minimum_difference_support_fraction:
        reason = "sam_mask_lacks_difference_support"
    else:
        return KeyROIMask("mask_available", "registered_crop_sam_qa_pass",
                          mask, diagnostic)
    return KeyROIMask("abstain", reason, None, diagnostic)
