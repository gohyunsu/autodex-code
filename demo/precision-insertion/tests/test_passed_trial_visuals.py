"""The presentation exporter preserves saved path endpoints and rigid grasp."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_passed_cylinder_trial_visuals import (  # noqa: E402
    _joint_transition, _robot_key_poses, _rows,
)


def test_time_compression_keeps_saved_path_endpoints():
    source = np.arange(9 * 13, dtype=float).reshape(9, 13)
    compressed = _rows(source, 5)
    assert compressed.shape == (5, 13)
    assert np.array_equal(compressed[0], source[0])
    assert np.array_equal(compressed[-1], source[-1])


def test_commanded_hand_transition_keeps_both_endpoints():
    start = np.zeros(6)
    end = np.arange(6, dtype=float)
    result = _joint_transition(start, end, 9)
    assert np.array_equal(result[0], start)
    assert np.array_equal(result[-1], end)
    assert np.all(np.diff(result[:, 1:], axis=0) >= 0)


def test_held_key_is_derived_from_hand_fk_and_fixed_relative_transform():
    class FakeUrdf:
        base_link = "robot"

        def update_cfg(self, cfg):
            self.cfg = cfg

        def get_transform(self, link, base):
            assert link == "base_link" and base == "robot"
            result = np.eye(4)
            result[0, 3] = self.cfg["fr3_joint1"]
            return result

    qpos = np.zeros((3, 13))
    qpos[:, 0] = (0.1, 0.2, 0.3)
    T_key_hand = np.eye(4)
    T_key_hand[2, 3] = 0.08
    key = _robot_key_poses(FakeUrdf(), qpos, T_key_hand)
    for index, pose in enumerate(key):
        hand = np.eye(4)
        hand[0, 3] = qpos[index, 0]
        assert np.allclose(pose @ T_key_hand, hand)
