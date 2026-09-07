import unittest

import numpy as np

from src.demo.inference.run_demo import _arc_delta, _joint0_arc_trajectory


def _planar_wrist_fk(_planner, qpos):
    """Small FK stand-in: J0 rotates a wrist on a unit-radius base orbit."""
    angle = float(np.asarray(qpos)[0])
    T = np.eye(4)
    T[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]]
    T[:2, 3] = [np.cos(angle), np.sin(angle)]
    return T


class Joint0TransferTest(unittest.TestCase):
    def test_only_joint0_changes_and_object_bearing_is_used(self):
        start = np.array([0.20, -0.4, 0.3, -1.2, 0.1, 0.8, 1.0, 2.0])
        target_angle = 1.0
        reference_xy = np.array([np.cos(target_angle), np.sin(target_angle)])
        object_in_wrist = np.eye(4)
        # The object bearing differs from the wrist bearing by atan2(0.3, 1).
        object_in_wrist[:2, 3] = [0.0, 0.3]

        traj, info = _joint0_arc_trajectory(
            None, _planar_wrist_fk, start, 6, object_in_wrist, target_angle,
            reference_xy)

        self.assertGreaterEqual(len(traj), 2)
        np.testing.assert_allclose(
            traj[:, 1:], np.tile(start[1:], (len(traj), 1)))
        # target_angle sits counter-clockwise of the held object's bearing,
        # so the short way round is positive; the old clockwise-only rule
        # turned this into a >300 deg sweep.
        self.assertGreater(info["joint0_delta_deg"], 0.0)
        self.assertLess(abs(info["joint0_delta_deg"]), 180.0)
        self.assertEqual(info["direction"], "counter_clockwise")
        self.assertAlmostEqual(info["measured_joint0_deg"],
                               np.rad2deg(start[0]), places=6)
        self.assertGreater(info["target_joint0_deg"], info["measured_joint0_deg"])
        object_end = _planar_wrist_fk(None, traj[-1]) @ object_in_wrist
        self.assertAlmostEqual(np.arctan2(object_end[1, 3], object_end[0, 3]),
                               target_angle, places=6)
        self.assertGreater(info["box_xy_error_m"], 0.0)

    def test_arc_delta_takes_the_short_way_and_respects_joint_limits(self):
        cw = _arc_delta(np.deg2rad(19.0), np.deg2rad(-30.0))
        self.assertAlmostEqual(np.rad2deg(cw), -49.0, places=6)

        ccw = _arc_delta(np.deg2rad(19.0), np.deg2rad(45.0))
        self.assertAlmostEqual(np.rad2deg(ccw), 26.0, places=6)

        # Short way is +26 deg but J0 already sits 10 deg below its ceiling,
        # so the reachable sweep is the long one.
        limited = _arc_delta(np.deg2rad(19.0), np.deg2rad(45.0),
                             joint0=np.deg2rad(160.0),
                             joint0_limits=(np.deg2rad(-180.0),
                                            np.deg2rad(170.0)))
        self.assertAlmostEqual(np.rad2deg(limited), -334.0, places=6)

        # Neither sweep fits: report the short one rather than command a
        # 334 deg swing that is just as far out of range.
        boxed = _arc_delta(np.deg2rad(19.0), np.deg2rad(45.0),
                           joint0=np.deg2rad(160.0),
                           joint0_limits=(np.deg2rad(-170.0),
                                          np.deg2rad(170.0)))
        self.assertAlmostEqual(np.rad2deg(boxed), 26.0, places=6)


if __name__ == "__main__":
    unittest.main()


class Joint0ExplicitTurnTest(unittest.TestCase):
    """``turn_rad`` commands the sweep itself, not a direction to end up facing."""

    def test_turn_is_applied_verbatim_and_only_joint0_moves(self):
        start = np.array([0.20, -0.4, 0.3, -1.2, 0.1, 0.8, 1.0, 2.0])
        object_in_wrist = np.eye(4)
        object_in_wrist[:2, 3] = [0.0, 0.3]

        traj, info = _joint0_arc_trajectory(
            None, _planar_wrist_fk, start, 6, object_in_wrist,
            target_angle_rad=1.0, turn_rad=np.deg2rad(-100.0))

        np.testing.assert_allclose(traj[:, 1:], np.tile(start[1:], (len(traj), 1)))
        self.assertAlmostEqual(info["joint0_delta_deg"], -100.0, places=6)
        self.assertAlmostEqual(float(traj[-1, 0]),
                               float(start[0]) + np.deg2rad(-100.0), places=6)
        self.assertEqual(info["commanded"], "turn")
        self.assertEqual(info["direction"], "clockwise")

    def test_bearing_mode_is_unchanged_by_the_new_argument(self):
        start = np.array([0.20, -0.4, 0.3, -1.2, 0.1, 0.8, 1.0, 2.0])
        object_in_wrist = np.eye(4)
        object_in_wrist[:2, 3] = [0.0, 0.3]

        turned, turn_info = _joint0_arc_trajectory(
            None, _planar_wrist_fk, start, 6, object_in_wrist, 1.0, turn_rad=None)
        default, default_info = _joint0_arc_trajectory(
            None, _planar_wrist_fk, start, 6, object_in_wrist, 1.0)

        np.testing.assert_allclose(turned, default)
        self.assertEqual(default_info["commanded"], "bearing")
        # The two ways of stating the motion genuinely differ: asking for
        # bearing 1.0 rad here is not a 100 deg sweep.
        self.assertNotAlmostEqual(turn_info["joint0_delta_deg"], -100.0, places=1)


class GoalWristTest(unittest.TestCase):
    """Carrying a held object onto a measured goal pose."""

    @staticmethod
    def _rot_z(deg):
        a = np.deg2rad(deg)
        T = np.eye(4)
        T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        return T

    def test_yaw_mode_matches_the_goal_yaw_and_keeps_the_carried_tilt(self):
        from src.demo.inference.run_demo import _goal_wrist

        # Object picked tilted 20 deg about x, carried at z=0.45.
        tilt = np.eye(4)
        tilt[1:3, 1:3] = [[np.cos(0.35), -np.sin(0.35)], [np.sin(0.35), np.cos(0.35)]]
        T_obj_now = self._rot_z(10.0) @ tilt
        T_obj_now[:3, 3] = [0.55, 0.10, 0.45]
        T_goal = self._rot_z(70.0)
        T_goal[:3, 3] = [0.40, -0.30, 0.06]
        object_in_wrist = np.eye(4)
        object_in_wrist[:3, 3] = [0.0, 0.0, -0.05]

        wrist, info = _goal_wrist(T_obj_now, object_in_wrist, T_goal, mode="yaw")
        placed = wrist @ object_in_wrist
        # xy comes from the goal, z stays the carry height (the descent is the
        # separate lay-down), and the yaw difference is applied about world z.
        np.testing.assert_allclose(placed[:2, 3], [0.40, -0.30], atol=1e-9)
        self.assertAlmostEqual(placed[2, 3], 0.45, places=9)
        self.assertAlmostEqual(info["goal_yaw_deg"], 60.0, places=6)
        # The carried tilt is preserved: only rotation about z changed.
        relative = placed[:3, :3] @ T_obj_now[:3, :3].T
        np.testing.assert_allclose(relative[2, :2], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(relative[:2, 2], [0.0, 0.0], atol=1e-9)

    def test_full_mode_commands_the_recorded_rotation(self):
        from src.demo.inference.run_demo import _goal_wrist

        T_obj_now = np.eye(4)
        T_obj_now[:3, 3] = [0.55, 0.10, 0.45]
        T_goal = self._rot_z(70.0)
        T_goal[:3, 3] = [0.40, -0.30, 0.06]
        object_in_wrist = np.eye(4)

        wrist, info = _goal_wrist(T_obj_now, object_in_wrist, T_goal, mode="full")
        placed = wrist @ object_in_wrist
        np.testing.assert_allclose(placed[:3, :3], T_goal[:3, :3], atol=1e-9)
        self.assertEqual(info["mode"], "full")

    def test_unknown_mode_is_rejected(self):
        from src.demo.inference.run_demo import _goal_wrist

        with self.assertRaises(ValueError):
            _goal_wrist(np.eye(4), np.eye(4), np.eye(4), mode="whatever")
