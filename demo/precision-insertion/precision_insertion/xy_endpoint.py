"""Read-only 1 mm socket-frame XY hypotheses with exact endpoint screening.

This is a geometric *pre-filter*, not the final VLM choice list or motion
authorization. The live planner must still validate the full Franka/attached
key path and guarded contact before a retry is physically possible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .geometry import validate_se3
from .xy_voting import axis_1mm_proposals


def screen_axis_1mm_endpoint_choices(
    *, shared_root: Path, mode: TaskMode, candidate_dir: Path,
    current_offset_socket_m: tuple[float, float],
    max_total_offset_m: float, minimum_hand_clearance_m: float,
    T_key_hand_override: np.ndarray | None = None,
    held_hand_q_measured: np.ndarray | None = None,
    relation_source: str = "multiview_key_pose_plus_live_wrist",
    screen: Callable = screen_grasp_endpoint,
) -> dict:
    """Test hold and four cardinal 1 mm absolute targets at 20 mm depth.

    The collision test reports both key/socket CAD fit and posed Inspire
    hand/socket clearance. With an observed key relation, a VLM caller offers
    only ``endpoint_pass`` rows. With an unobserved squeeze shift, a caller may
    expose hand-clear rows for *diagnostic* votes, but must not interpret them
    as 20 mm insertion-feasible targets or robot-motion authorization.
    """
    if not math.isfinite(max_total_offset_m) or max_total_offset_m <= 0:
        raise ValueError("max_total_offset_m must be positive and finite")
    if (not math.isfinite(minimum_hand_clearance_m) or
            minimum_hand_clearance_m <= 0):
        raise ValueError("minimum_hand_clearance_m must be positive and finite")
    if T_key_hand_override is not None:
        T_key_hand_override = validate_se3(
            T_key_hand_override, name="held T_key_hand hypothesis")
    if (not isinstance(relation_source, str) or
            relation_source not in {
                "multiview_key_pose_plus_live_wrist",
                "v8_nominal_unobserved_key",
            }):
        raise ValueError("unknown held-relation source")
    if held_hand_q_measured is not None:
        held_hand_q_measured = np.asarray(held_hand_q_measured, dtype=np.float64)
        if (T_key_hand_override is None or
                held_hand_q_measured.shape != (6,) or
                not np.all(np.isfinite(held_hand_q_measured))):
            raise ValueError("measured hand joints need a paired observed key/hand pose")
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
            kwargs = {}
            if T_key_hand_override is not None:
                kwargs["T_key_hand_override"] = T_key_hand_override
            if held_hand_q_measured is not None:
                kwargs["hand_poses_override"] = {
                    "measured_held": held_hand_q_measured}
                kwargs["override_source"] = relation_source
            report = screen(
                shared_root=shared_root, mode=mode,
                candidate_dir=candidate_dir,
                minimum_hand_clearance_m=minimum_hand_clearance_m,
                xy_offset_socket_m=target,
                **kwargs,
            )
            if (report.get("xy_offset_socket_m") != list(target) or
                    report.get("verification_depth_m") != mode.target_depth_m):
                raise ValueError("endpoint screen returned a different target")
            row["endpoint_pass"] = bool(report["endpoint_pass"])
            if "hand_socket_clear_at_20mm" in report:
                if type(report["hand_socket_clear_at_20mm"]) is not bool:
                    raise ValueError("endpoint hand/socket verdict must be boolean")
                row["hand_socket_clear_at_20mm"] = report[
                    "hand_socket_clear_at_20mm"]
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
        "T_key_hand_source": (
            "v8_candidate" if T_key_hand_override is None
            else relation_source),
        "hand_pose_source": (
            "v8_nominal" if held_hand_q_measured is None
            else "measured_inspire_feedback"),
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
