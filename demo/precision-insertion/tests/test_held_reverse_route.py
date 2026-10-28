"""A held recovery must undo XY before transfer and lift."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.held_reverse_route import (  # noqa: E402
    assemble_reverse_held_trajectory,
)
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.corrected_hold_recovery import (  # noqa: E402
    propagate_held_key_pose,
)


class LinearFK:
    def fk_wrist(self, q):
        pose = np.eye(4)
        pose[0, 3] = q[0] / 100.
        return pose


def _path(start, end, hand):
    path = np.zeros((2, 13))
    path[:, 0] = [start, end]
    path[:, 7:] = hand
    return path


def test_reverse_route_undoes_xy_then_transfer_then_lift_and_keeps_grip():
    lift = _path(0, 1, .1)
    transfer = _path(1, 2, .2)
    xy = [_path(2, 3, .3), _path(3, 4, .4)]
    current = np.zeros(13)
    current[0], current[7:] = 4, .7
    limits = PathAuditLimits(
        max_joint_step_rad=.12, max_wrist_step_m=.015,
        max_wrist_rotation_deg=8., goal_position_tolerance_m=.005,
        goal_rotation_tolerance_deg=5., axial_lateral_tolerance_m=.005,
        axial_rotation_tolerance_deg=5., minimum_hand_clearance_m=0.)
    path, segments = assemble_reverse_held_trajectory(
        planner=LinearFK(), current_q=current,
        executed_xy_trajectories=xy,
        executed_transfer_trajectory=transfer,
        executed_lift_trajectory=lift, limits=limits,
        arm_velocity_rad_s=1.2, traj_dt_s=.01)
    assert [name for name, _ in segments if name.startswith("reverse_")] == [
        "reverse_xy_1", "reverse_xy_0", "reverse_transfer", "reverse_lift"]
    assert path[0, 0] == 4 and path[-1, 0] == 0
    assert np.all(np.diff(path[:, 0]) <= 0)
    assert np.all(path[:, 7:] == .7)
    assert np.all(xy[0][:, 7:] == .3)


def test_fresh_joint_feedback_moves_held_key_hypothesis_with_wrist():
    failed = np.zeros(13)
    current = failed.copy()
    current[0] = 2.
    key = np.eye(4)
    key[0, 3] = .03
    moved = propagate_held_key_pose(
        planner=LinearFK(), failed_key_pose=key,
        failed_q=failed, current_q=current)
    assert np.isclose(moved[0, 3], .05)
