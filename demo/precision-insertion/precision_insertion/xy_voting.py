"""Read-only multi-view choice resolution for socket-frame XY retry proposals.

This module has no VLM or robot client. Its choices must be supplied by a
separate geometry-screened catalog; a positive vote is only a proposal to
replan, never permission to move or proof of insertion success. Unlike the
historical continuous-residual retry helper, it never invents a new metric
offset or averages tied candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .geometry import validate_se3


@dataclass(frozen=True)
class XYChoice:
    choice_id: str
    offset_socket_m: tuple[float, float]


@dataclass(frozen=True)
class ViewVote:
    camera_id: str
    timestamp_s: float
    visible: bool
    choice_id: str | None
    failure_class: str


@dataclass(frozen=True)
class ChoiceDecision:
    status: str
    reason: str
    choice_id: str | None
    offset_socket_m: tuple[float, float] | None
    supporting_cameras: tuple[str, ...]
    vote_counts: Mapping[str, int]

    def to_record(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "choice_id": self.choice_id,
            "offset_socket_m": (list(self.offset_socket_m)
                                if self.offset_socket_m is not None else None),
            "supporting_cameras": list(self.supporting_cameras),
            "vote_counts": dict(self.vote_counts),
            "scope": "read_only_xy_retry_proposal_not_motion_authorization",
        }


def _xy(value, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(x) for x in pair):
        raise ValueError(f"{name} must be finite")
    return pair


def validate_choices(choices: Iterable[XYChoice]) -> dict[str, XYChoice]:
    """Check identity and metric units without claiming geometry validation."""
    by_id: dict[str, XYChoice] = {}
    offsets: set[tuple[float, float]] = set()
    for choice in choices:
        if not isinstance(choice, XYChoice) or not choice.choice_id.strip():
            raise ValueError("each XY choice needs a nonempty ID")
        if choice.choice_id in by_id:
            raise ValueError(f"duplicate XY choice ID: {choice.choice_id}")
        offset = _xy(choice.offset_socket_m, choice.choice_id)
        if offset in offsets:
            raise ValueError("two IDs cannot describe the same XY offset")
        offsets.add(offset)
        by_id[choice.choice_id] = XYChoice(choice.choice_id, offset)
    if not by_id:
        raise ValueError("at least one XY choice is required")
    return by_id


def project_choice_anchors(
    choices: Iterable[XYChoice],
    *,
    T_camera_socket: np.ndarray,
    intrinsics: np.ndarray,
    socket_plane_z_m: float,
    image_size_wh: tuple[int, int],
) -> dict[str, tuple[float, float]]:
    """Project socket-frame candidate centers into one calibrated image.

    These are overlay anchors only. The VLM should see CAD key/socket outlines
    and real pixels too; a projected point is not a pose measurement.
    """
    by_id = validate_choices(choices)
    camera_socket = validate_se3(T_camera_socket, name="T_camera_socket")
    K = np.asarray(intrinsics, dtype=np.float64)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9)):
        raise ValueError("intrinsics must be a finite valid 3x3 matrix")
    if not math.isfinite(socket_plane_z_m):
        raise ValueError("socket_plane_z_m must be finite")
    width, height = image_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("image_size_wh must be positive")
    projected: dict[str, tuple[float, float]] = {}
    for choice_id, choice in by_id.items():
        x, y = choice.offset_socket_m
        camera_point = (camera_socket @ np.array(
            [x, y, socket_plane_z_m, 1.0], dtype=np.float64))[:3]
        if camera_point[2] <= 0:
            continue
        pixel = K @ camera_point
        u, v = (float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]))
        if 0 <= u < width and 0 <= v < height:
            projected[choice_id] = (u, v)
    return projected


def resolve_multiview_choice(
    choices: Iterable[XYChoice],
    votes: Iterable[ViewVote],
    *,
    current_offset_socket_m: tuple[float, float],
    grasp_held: bool,
    hard_abort: bool,
    max_step_m: float,
    max_total_m: float,
    max_timestamp_skew_s: float,
    decision_timestamp_s: float,
    max_frame_age_s: float,
    min_supporting_views: int = 2,
) -> ChoiceDecision:
    """Fuse camera choices; fail closed on ties, staleness, or unsafe evidence.

    Offsets are absolute target parameters in the frozen socket frame, not
    camera-pixel deltas and not incremental additions to the prior attempt.
    The caller must separately verify the choice catalog's geometry, live
    calibration, collision world, robot plan, and guarded-contact controller.
    """
    by_id = validate_choices(choices)
    old = _xy(current_offset_socket_m, "current_offset_socket_m")
    limits = (max_step_m, max_total_m, max_timestamp_skew_s,
              max_frame_age_s)
    if not all(math.isfinite(v) and v > 0 for v in limits):
        raise ValueError("step, total, and time limits must be positive")
    if not math.isfinite(decision_timestamp_s):
        raise ValueError("decision_timestamp_s must be finite")
    if min_supporting_views < 2:
        raise ValueError("at least two agreeing camera views are required")
    rows = list(votes)
    if len({vote.camera_id for vote in rows}) != len(rows):
        raise ValueError("one vote per unique camera is required")
    for vote in rows:
        if not vote.camera_id or not math.isfinite(vote.timestamp_s):
            raise ValueError("each view needs a camera ID and finite timestamp")
        if vote.choice_id not in (None, "abstain") and vote.choice_id not in by_id:
            raise ValueError(f"unknown XY choice ID from {vote.camera_id}")
        if not vote.visible and vote.choice_id not in (None, "abstain"):
            raise ValueError("an occluded view cannot cast a candidate vote")

    empty = {choice_id: 0 for choice_id in by_id}

    def result(status: str, reason: str, *, selected: str | None = None,
               supporters: tuple[str, ...] = (), counts=None) -> ChoiceDecision:
        return ChoiceDecision(
            status, reason, selected,
            by_id[selected].offset_socket_m if selected is not None else None,
            supporters, empty if counts is None else counts,
        )

    if hard_abort is not False or grasp_held is not True:
        return result("stop", "hard_abort_or_grasp_not_verified")
    if any(vote.failure_class == "slip" for vote in rows):
        return result("stop", "possible_slip")
    if len(rows) < min_supporting_views:
        return result("abstain", "insufficient_camera_views")
    timestamps = [vote.timestamp_s for vote in rows]
    if max(timestamps) - min(timestamps) > max_timestamp_skew_s:
        return result("abstain", "unsynchronized_images")
    if (any(timestamp > decision_timestamp_s + max_timestamp_skew_s
            for timestamp in timestamps) or
            decision_timestamp_s - min(timestamps) > max_frame_age_s):
        return result("abstain", "stale_or_future_images")

    informative = [vote for vote in rows if vote.visible and
                   vote.failure_class in {"misaligned", "rim_jam"}]
    if len(informative) < min_supporting_views:
        return result("abstain", "insufficient_informative_views")
    counts = dict(empty)
    for vote in informative:
        if vote.choice_id not in (None, "abstain"):
            counts[vote.choice_id] += 1
    highest = max(counts.values())
    winners = [choice_id for choice_id, n in counts.items() if n == highest]
    if (highest < min_supporting_views or len(winners) != 1 or
            highest * 2 <= len(informative)):
        return result("abstain", "no_strict_multiview_consensus", counts=counts)

    selected = winners[0]
    supporters = tuple(vote.camera_id for vote in informative
                       if vote.choice_id == selected)
    target = by_id[selected].offset_socket_m
    if math.hypot(*target) > max_total_m:
        return result("abstain", "total_offset_budget_exceeded", counts=counts)
    if math.dist(target, old) > max_step_m:
        return result("abstain", "step_budget_exceeded", counts=counts)
    if math.dist(target, old) <= 1e-12:
        return result("no_correction", "selected_current_offset", selected=selected,
                      supporters=supporters, counts=counts)
    return result("propose", "multiview_choice_requires_live_preflight",
                  selected=selected, supporters=supporters, counts=counts)
