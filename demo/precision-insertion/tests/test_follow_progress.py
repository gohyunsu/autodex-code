"""Regression for the measured-ahead transfer stall of 2026-10-20."""

import numpy as np

from src.execution.follow_progress import (
    advance_reference_index, measured_path_index,
)


def test_measured_arm_ahead_of_frozen_reference_catches_up_without_path_jump():
    path = np.zeros((132, 7), dtype=float)
    path[:, 6] = np.arange(132) * 0.035
    measured = path[44].copy()
    index = 32.0
    # The old distance-only gate held index 32 indefinitely: waypoint 33 is
    # 0.385 rad behind the measured arm, exceeding its 0.12 rad lead limit.
    assert np.linalg.norm(path[33] - measured) > 0.12
    for _ in range(12):
        index = advance_reference_index(path, measured, index, 1.0, 0.12)
    assert index == 44.0


def test_off_path_arm_does_not_advance_reference():
    path = np.zeros((132, 7), dtype=float)
    path[:, 6] = np.arange(132) * 0.035
    measured = path[44].copy()
    measured[0] = 0.2
    assert advance_reference_index(path, measured, 32.0, 1.0, 0.12) == 32.0


def test_lagging_arm_uses_available_lead_without_freezing_reference():
    path = np.zeros((30, 7), dtype=float)
    path[:, 0] = np.arange(30) * 0.01
    measured = path[0].copy()
    index = advance_reference_index(path, measured, 10.0, 3.0, 0.12)
    assert 11.99 < index <= 12.0
    assert np.linalg.norm(path[12] - measured) <= 0.12
    measured[0] += 0.005
    next_index = advance_reference_index(path, measured, index, 3.0, 0.12)
    assert index < next_index < index + 3.0


def test_hand_progress_uses_measured_arm_even_after_early_waypoint_is_passed():
    path = np.zeros((160, 7), dtype=float)
    path[:, 0] = np.arange(160) * 0.01
    measured = path[75].copy()
    assert np.linalg.norm(path[20] - measured) > 0.04
    assert measured_path_index(path, measured, 20, 82.0) == 75
    assert measured_path_index(path, path[130], 75, 82.0) == 82
