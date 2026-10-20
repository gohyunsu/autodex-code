"""Regression for the measured-ahead transfer stall of 2026-10-20."""

import numpy as np

from src.execution.follow_progress import advance_reference_index


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
