"""Project fixed 1 mm XY hypotheses into calibrated AutoDex camera views.

This is a display aid for ZeroDex-style closed-set voting, not metric pose
estimation or motion authorization. A view whose proposals are too close in
the *original pixels* is rejected before resizing; magnification cannot
create camera information that was not captured.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import validate_se3
from .observer import XYView
from .xy_voting import XYChoice, project_choice_anchors, validate_choices


COLORS = (
    (255, 255, 255), (38, 207, 255), (255, 99, 71),
    (135, 255, 114), (255, 209, 87),
)


@dataclass(frozen=True)
class CalibratedXYFrame:
    camera_id: str
    timestamp_s: float
    raw: Image.Image
    T_camera_socket: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class XYOverlayBatch:
    status: str
    views: tuple[XYView, ...]
    per_camera: dict[str, dict]
    choice_ids: tuple[str, ...]
    minimum_anchor_separation_px: float

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_xy_overlay_batch_v1",
            "status": self.status,
            "per_camera": self.per_camera,
            "choice_ids": list(self.choice_ids),
            "minimum_anchor_separation_px": self.minimum_anchor_separation_px,
            "scope": "display_only_calibrated_projection_not_motion_authorization",
            "robot_ready": False,
        }


def camera_socket_transform(
    *, T_camera_world: np.ndarray, T_world_robot: np.ndarray,
    T_robot_socket: np.ndarray,
) -> np.ndarray:
    """Compose AutoDex world-to-camera extrinsics and frozen socket pose."""
    camera_world = validate_se3(T_camera_world, name="T_camera_world")
    world_robot = validate_se3(T_world_robot, name="session C2R")
    robot_socket = validate_se3(T_robot_socket, name="frozen T_robot_socket")
    return validate_se3(camera_world @ world_robot @ robot_socket,
                        name="T_camera_socket")


def _min_separation(points: dict[str, tuple[float, float]]) -> float:
    coords = list(points.values())
    if len(coords) < 2:
        return float("inf")
    return min(math.dist(first, second)
               for index, first in enumerate(coords)
               for second in coords[index + 1:])


def build_xy_candidate_overlays(
    *, choices: Sequence[XYChoice], frames: Sequence[CalibratedXYFrame],
    socket_plane_z_m: float, minimum_anchor_separation_px: float,
    crop_width_px: int, display_scale: int = 2,
    minimum_usable_views: int = 2,
) -> XYOverlayBatch:
    """Return paired raw/annotated crops for sufficiently resolving views.

    The caller supplies only geometry-screened candidate IDs. Pairwise
    separation is checked before crop/resize, so a 1 px displacement cannot
    be made "visible" by upscaling. Actual socket/key contours and occlusion
    still require calibrated mesh projection or separate perception evidence.
    """
    by_id = validate_choices(choices)
    if (not math.isfinite(minimum_anchor_separation_px) or
            minimum_anchor_separation_px <= 0):
        raise ValueError("minimum anchor separation needs a positive pixel budget")
    if crop_width_px < 32 or display_scale < 1 or minimum_usable_views < 2:
        raise ValueError("invalid crop, display scale, or view count")
    if not frames or len({frame.camera_id for frame in frames}) != len(frames):
        raise ValueError("calibrated frames need unique nonempty cameras")
    if any(not frame.camera_id for frame in frames):
        raise ValueError("camera ID must be nonempty")
    accepted: list[XYView] = []
    diagnostics: dict[str, dict] = {}
    for frame in frames:
        if (not isinstance(frame.raw, Image.Image) or
                not math.isfinite(float(frame.timestamp_s))):
            raise ValueError("every XY frame needs a PIL image and capture time")
        width, height = frame.raw.size
        projected = project_choice_anchors(
            by_id.values(), T_camera_socket=frame.T_camera_socket,
            intrinsics=frame.intrinsics, socket_plane_z_m=socket_plane_z_m,
            image_size_wh=(width, height))
        missing = sorted(set(by_id) - set(projected))
        separation = _min_separation(projected)
        diagnostic = {
            "anchors_original_px": {
                name: [float(x), float(y)] for name, (x, y)
                in projected.items()},
            "missing_choice_ids": missing,
            "minimum_pairwise_separation_original_px": (
                separation if math.isfinite(separation) else None),
            "crop_original_px": None,
            "display_scale": display_scale,
            "accepted": False,
            "reason": None,
        }
        if missing:
            diagnostic["reason"] = "candidate_outside_camera_view"
        elif separation < minimum_anchor_separation_px:
            diagnostic["reason"] = "one_mm_candidates_not_pixel_resolvable"
        else:
            center_x = float(np.mean([point[0] for point in projected.values()]))
            center_y = float(np.mean([point[1] for point in projected.values()]))
            crop_w = min(crop_width_px, width)
            crop_h = min(round(crop_width_px * 9 / 16), height)
            left = max(0, min(width - crop_w, round(center_x - crop_w / 2)))
            top = max(0, min(height - crop_h, round(center_y - crop_h / 2)))
            if any(not (left + 8 <= x < left + crop_w - 8 and
                        top + 8 <= y < top + crop_h - 8)
                   for x, y in projected.values()):
                diagnostic["reason"] = "candidate_too_close_to_crop_edge"
            else:
                box = (left, top, left + crop_w, top + crop_h)
                raw = frame.raw.convert("RGB").crop(box).resize(
                    (crop_w * display_scale, crop_h * display_scale),
                    getattr(Image, "Resampling", Image).NEAREST)
                overlay = raw.copy()
                draw = ImageDraw.Draw(overlay)
                font = ImageFont.load_default()
                for index, (name, _choice) in enumerate(by_id.items()):
                    x, y = projected[name]
                    px = (x - left) * display_scale
                    py = (y - top) * display_scale
                    color = COLORS[index % len(COLORS)]
                    radius = max(2, display_scale)
                    draw.ellipse((px - radius, py - radius,
                                  px + radius, py + radius),
                                 outline=color, width=2)
                    label = f"{index + 1}: {name}"
                    legend_y = 12 + index * 16
                    draw.rectangle((8, legend_y - 1, 8 + 7 * len(label),
                                    legend_y + 11), fill=(25, 29, 35))
                    draw.text((10, legend_y), label, fill=color, font=font)
                accepted.append(XYView(
                    frame.camera_id, float(frame.timestamp_s), raw, overlay))
                diagnostic["accepted"] = True
                diagnostic["crop_original_px"] = list(box)
        diagnostics[frame.camera_id] = diagnostic
    status = ("views_ready" if len(accepted) >= minimum_usable_views
              else "insufficient_pixel_resolvable_views")
    return XYOverlayBatch(
        status, tuple(accepted), diagnostics, tuple(by_id),
        float(minimum_anchor_separation_px))
