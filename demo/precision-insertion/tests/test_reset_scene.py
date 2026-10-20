"""The missed-grasp reset must not invent a key attached to an empty hand."""

import numpy as np

from src.execution.reset_scene import scene_for_reset, vertical_clearance_poses


def test_missed_key_keeps_tabletop_pose_and_original_scene():
    original = {"mesh": {
        "target": {"pose": [0., 0., 0., 1., 0., 0., 0.]},
        "fixture_socket": {"pose": [0.2, 0., 0., 1., 0., 0., 0.]},
    }}
    saved_pose = original["mesh"]["target"]["pose"].copy()
    scene = scene_for_reset(original, key_was_carried=False)
    assert scene["mesh"]["target"]["pose"] == saved_pose
    assert set(scene["mesh"]) == {"target", "fixture_socket"}
    scene["mesh"]["target"]["pose"][0] = 1.
    assert original["mesh"]["target"]["pose"] == saved_pose


def test_carried_key_projects_from_grasp_to_current_wrist():
    original = {"mesh": {"target": {
        "pose": [0., 0., 0., 1., 0., 0., 0.]}}}
    grasp = np.eye(4)
    grasp[0, 3] = 1.
    current = np.eye(4)
    current[0, 3] = 2.
    scene = scene_for_reset(
        original, key_was_carried=True,
        grasp_wrist=grasp, current_wrist=current)
    np.testing.assert_allclose(scene["mesh"]["target"]["pose"],
                               [1., 0., 0., 1., 0., 0., 0.])
    assert original["mesh"]["target"]["pose"][0] == 0.


def test_reset_lift_preserves_measured_planner_wrist_orientation():
    wrist = np.eye(4)
    wrist[:3, :3] = np.array([
        [0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    wrist[:3, 3] = [0.5, 0.2, 0.3]

    class Planner:
        def fk_wrist(self, q):
            assert np.asarray(q).shape == (13,)
            return wrist

    start, end = vertical_clearance_poses(
        Planner(), np.zeros(13), 0.1)
    np.testing.assert_allclose(start, wrist)
    np.testing.assert_allclose(end[:3, :3], wrist[:3, :3])
    np.testing.assert_allclose(end[:3, 3], [0.5, 0.2, 0.4])
    assert wrist[2, 3] == 0.3
