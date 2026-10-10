"""Offline cylinder promotion must not turn a nominal-only grasp into v8."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promote_cylinder_tabletop_candidates import select_ids  # noqa: E402
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM  # noqa: E402


def _screens():
    labels = [f"{int(gap):02d}mm" for gap in CYLINDER_RADIAL_GAPS_MM]
    nominal = {
        "schema": "precision_insertion_cylinder_1000_per_scene_screen_v1",
        "total_raw_proposals": 2000,
        "physical_key_object": "precision_key_cylinder_r15_h80",
        "scene_counts": {"0": {"mujoco_stable": 2},
                         "1": {"mujoco_stable": 1}},
        "socket_gaps": {
            label: {"eligible_offline_grasp_ids": ["table/0/1", "table/0/2"]}
            for label in labels},
    }
    achieved = {
        "schema": "precision_insertion_cylinder_achieved_endpoint_comparison_v1",
        "socket_family_complete": True,
        "candidate_count": 3,
        "socket_gaps": {
            label: {
                "nominal_initial_pose_endpoint_pass_count": 2,
                "both_achieved_states_clear_ids": ["table/0/1", "table/0/2"],
            } for label in labels},
    }
    return nominal, achieved


def test_promotion_requires_nominal_and_achieved_clearance_for_every_gap():
    nominal, achieved = _screens()
    achieved["socket_gaps"]["15mm"]["both_achieved_states_clear_ids"] = [
        "table/0/1"]
    assert select_ids(nominal, achieved) == ["table/0/1"]


def test_promotion_rejects_incomplete_or_inconsistent_sources():
    nominal, achieved = _screens()
    del achieved["socket_gaps"]["20mm"]
    with pytest.raises(ValueError, match="every socket gap"):
        select_ids(nominal, achieved)
    nominal, achieved = _screens()
    achieved["socket_gaps"]["15mm"]["both_achieved_states_clear_ids"].append(
        "table/0/1")
    with pytest.raises(ValueError, match="duplicate/inconsistent"):
        select_ids(nominal, achieved)
