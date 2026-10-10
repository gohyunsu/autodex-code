"""One-millimetre XY endpoint pre-filter does not authorize robot motion."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.xy_endpoint import (  # noqa: E402
    screen_axis_1mm_endpoint_choices,
)


def test_one_mm_cardinal_targets_are_screened_before_vlm(tmp_path):
    mode = select_mode("cylinder", 20)
    tested = []

    def fake_screen(**kwargs):
        xy = tuple(kwargs["xy_offset_socket_m"])
        tested.append(xy)
        return {
            "xy_offset_socket_m": list(xy),
            "verification_depth_m": mode.target_depth_m,
            "endpoint_pass": xy in {(0.0, 0.0), (0.001, 0.0)},
        }

    report = screen_axis_1mm_endpoint_choices(
        shared_root=tmp_path, mode=mode,
        candidate_dir=tmp_path / "raw_seed",
        current_offset_socket_m=(0.0, 0.0),
        max_total_offset_m=0.001,
        minimum_hand_clearance_m=0.0002, screen=fake_screen,
    )
    assert [row["choice_id"] for row in report["rows"]] == [
        "hold", "x_plus_1mm", "x_minus_1mm", "y_plus_1mm", "y_minus_1mm",
    ]
    assert tested == [
        (0.0, 0.0), (0.001, 0.0), (-0.001, 0.0),
        (0.0, 0.001), (0.0, -0.001),
    ]
    assert report["endpoint_clear_choice_ids"] == ["hold", "x_plus_1mm"]
    assert all(row["relative_step_m"] == pytest.approx(0.001)
               for row in report["rows"][1:])
    assert report["robot_ready"] is False


def test_total_offset_budget_prunes_without_geom_call(tmp_path):
    mode = select_mode("square", 1.5)
    tested = []

    def fake_screen(**kwargs):
        xy = tuple(kwargs["xy_offset_socket_m"])
        tested.append(xy)
        return {
            "xy_offset_socket_m": list(xy),
            "verification_depth_m": mode.target_depth_m,
            "endpoint_pass": True,
        }

    report = screen_axis_1mm_endpoint_choices(
        shared_root=tmp_path, mode=mode,
        candidate_dir=tmp_path / "seed",
        current_offset_socket_m=(0.002, 0.0),
        max_total_offset_m=0.002,
        minimum_hand_clearance_m=0.0002, screen=fake_screen,
    )
    assert (0.003, 0.0) not in tested
    assert (0.002, 0.001) not in tested
    assert (0.002, -0.001) not in tested
    assert [row["choice_id"] for row in report["rows"]
            if not row["within_total_offset_budget"]] == [
                "x_plus_1mm", "y_plus_1mm", "y_minus_1mm",
            ]


def test_mismatched_geometry_result_is_rejected(tmp_path):
    mode = select_mode("cylinder", 5)

    def wrong_target(**_kwargs):
        return {
            "xy_offset_socket_m": [0.5, 0.0],
            "verification_depth_m": 0.020,
            "endpoint_pass": True,
        }

    with pytest.raises(ValueError, match="different target"):
        screen_axis_1mm_endpoint_choices(
            shared_root=tmp_path, mode=mode,
            candidate_dir=tmp_path / "seed",
            current_offset_socket_m=(0.0, 0.0),
            max_total_offset_m=0.002,
            minimum_hand_clearance_m=0.0002, screen=wrong_target,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        screen_axis_1mm_endpoint_choices(
            shared_root=tmp_path, mode=mode,
            candidate_dir=tmp_path / "seed",
            current_offset_socket_m=(0.0, 0.0),
            max_total_offset_m=0.0,
            minimum_hand_clearance_m=0.0002, screen=wrong_target,
        )
