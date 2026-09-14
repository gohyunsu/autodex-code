"""Small, robot-neutral timing contracts for the execution adapters.

The runner owns episode-level timing, while an adapter owns the spans between
its own commands.  Keeping the schema here prevents xArm and FR3 from using
the same words for different work.
"""

from __future__ import annotations

import time
from typing import Iterable


PICKUP_SPANS = (
    "init_motion_s",
    "approach_monitor_warmup_s",
    "approach_motion_s",
    "pregrasp_motion_s",
    "grasp_motion_s",
    "squeeze_motion_s",
    "lift_start_check_s",
    "lift_runtime_replan_s",
    "lift_motion_s",
)

PLACE_PLAN_SPANS = (
    "preplace_plan_s",
    "descend_plan_s",
    "post_release_lift_plan_s",
    "retract_plan_s",
)
PLACE_MOTION_SPANS = (
    "preplace_motion_s",
    "descend_motion_s",
    "release_s",
    "post_release_lift_motion_s",
)
PLACE_OTHER_SPANS = ("control_setup_s", "contact_monitor_warmup_s")


def _sum(timing: dict, keys: Iterable[str]) -> float:
    return sum(max(0.0, float(timing.get(key, 0.0) or 0.0)) for key in keys)


def new_pickup_timing() -> dict:
    """Return a v2 pickup/lift timing object and its monotonic start clock."""
    return {
        "schema_version": 2,
        "clock": "time.perf_counter",
        "lift_plan_source": "not_requested",
        **{key: 0.0 for key in PICKUP_SPANS},
    }


def finish_pickup_timing(timing: dict, started: float) -> dict:
    """Close a pickup span without double-counting its child durations."""
    timing["accounted_s"] = round(_sum(timing, PICKUP_SPANS), 3)
    timing["total_s"] = round(max(0.0, time.perf_counter() - started), 3)
    timing["overhead_s"] = round(
        max(0.0, timing["total_s"] - timing["accounted_s"]), 3)
    return timing


def new_place_timing() -> dict:
    """Return one cross-arm placement timing object.

    FR3 fills the complete preflight/release chain. xArm fills descent planning
    or contact-control spans and marks release as an external runner action.
    A zero is an unavailable/not-used span, never an inferred residual.
    """
    return {
        "schema_version": 2,
        "clock": "time.perf_counter",
        "release_execution": "adapter",
        **{key: 0.0 for key in (*PLACE_PLAN_SPANS, *PLACE_MOTION_SPANS,
                                *PLACE_OTHER_SPANS)},
    }


def finish_place_timing(timing: dict, started: float) -> dict:
    """Close placement timing; preflight plans and later reset motion remain
    intentionally distinct in the caller's hierarchy."""
    timing["planning_s"] = round(_sum(timing, PLACE_PLAN_SPANS), 3)
    timing["motion_s"] = round(_sum(timing, PLACE_MOTION_SPANS), 3)
    timing["setup_s"] = round(_sum(timing, PLACE_OTHER_SPANS), 3)
    timing["accounted_s"] = round(
        timing["planning_s"] + timing["motion_s"] + timing["setup_s"], 3)
    timing["total_s"] = round(max(0.0, time.perf_counter() - started), 3)
    timing["overhead_s"] = round(
        max(0.0, timing["total_s"] - timing["accounted_s"]), 3)
    return timing
