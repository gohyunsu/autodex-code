"""Read-only 1 mm socket-frame XY hypotheses with exact endpoint screening.

This is a geometric *pre-filter*, not the final VLM choice list or motion
authorization. The live planner must still validate the full Franka/attached
key path and guarded contact before a retry is physically possible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .xy_voting import axis_1mm_proposals


def screen_axis_1mm_endpoint_choices(
    *, shared_root: Path, mode: TaskMode, candidate_dir: Path,
    current_offset_socket_m: tuple[float, float],
    max_total_offset_m: float, minimum_hand_clearance_m: float,
    screen: Callable = screen_grasp_endpoint,
) -> dict:
    """Test hold and four cardinal 1 mm absolute targets at 20 mm depth.

    The collision test treats key/socket intersection as a failed CAD fit and
    checks the posed Inspire hand against the exact socket mesh. Individual
    offsets are kept in the report even when rejected, so a VLM caller can
    offer *only* ``endpoint_pass`` rows after an additional live path check.
    A VLM vote cannot override a rejected row.
    """
    if not math.isfinite(max_total_offset_m) or max_total_offset_m <= 0:
        raise ValueError("max_total_offset_m must be positive and finite")
    if (not math.isfinite(minimum_hand_clearance_m) or
            minimum_hand_clearance_m <= 0):
        raise ValueError("minimum_hand_clearance_m must be positive and finite")
    choices = axis_1mm_proposals(current_offset_socket_m)
    current = choices[0].offset_socket_m
    rows = []
    for choice in choices:
        target = choice.offset_socket_m
        row = {
            "choice_id": choice.choice_id,
            "xy_offset_socket_m": list(target),
            "relative_step_m": math.dist(
                target, current),
            "within_total_offset_budget": (
                math.hypot(*target) <= max_total_offset_m + 1e-12),
            "endpoint_pass": False,
        }
        if row["within_total_offset_budget"]:
            report = screen(
                shared_root=shared_root, mode=mode,
                candidate_dir=candidate_dir,
                minimum_hand_clearance_m=minimum_hand_clearance_m,
                xy_offset_socket_m=target,
            )
            if (report.get("xy_offset_socket_m") != list(target) or
                    report.get("verification_depth_m") != mode.target_depth_m):
                raise ValueError("endpoint screen returned a different target")
            row["endpoint_pass"] = bool(report["endpoint_pass"])
            row["endpoint_report"] = report
        else:
            row["reason"] = "total_offset_budget_exceeded"
        rows.append(row)
    return {
        "schema": "precision_insertion_xy_endpoint_choices_v1",
        "scope": "read_only_20mm_endpoint_geometry_not_live_path_or_motion_authorization",
        "mode": {
            "family": mode.family, "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object,
        },
        "candidate_dir": str(Path(candidate_dir).expanduser().resolve()),
        "current_offset_socket_m": list(current),
        "max_total_offset_m": max_total_offset_m,
        "minimum_hand_clearance_m": minimum_hand_clearance_m,
        "rows": rows,
        "endpoint_clear_choice_ids": [
            row["choice_id"] for row in rows if row["endpoint_pass"]],
        "robot_ready": False,
        "not_validated": [
            "source grasp stability or tabletop suitability",
            "Franka IK, full path, or attached-key transfer",
            "continuous insertion contact or force response",
            "physical insertion success",
        ],
    }
