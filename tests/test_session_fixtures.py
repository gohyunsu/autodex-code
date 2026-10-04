import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.execution.scene_cfg import add_fixed_mesh_fixtures
from src.execution.session_fixtures import freeze_pose_medoid, validate_se3


def _pose(x_mm=0.0, yaw_deg=0.0):
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[0, 3] = x_mm / 1000.0
    return pose


class SessionFixtureTest(unittest.TestCase):
    def test_medoid_is_observed_pose_and_reports_repeatability(self):
        poses = [_pose(0.0, 0.0), _pose(0.4, 0.2), _pose(0.8, 0.4)]
        selected, diagnostics = freeze_pose_medoid(
            poses, translation_limit_mm=1.0, rotation_limit_deg=1.0)
        np.testing.assert_allclose(selected, poses[1])
        self.assertEqual(diagnostics["selected_index"], 1)
        self.assertTrue(diagnostics["accepted"])

    def test_repeatability_gate_rejects_outlier(self):
        with self.assertRaisesRegex(ValueError, "not repeatable"):
            freeze_pose_medoid(
                [_pose(), _pose(0.2), _pose(4.0)],
                translation_limit_mm=1.0, rotation_limit_deg=1.0)

    def test_validate_se3_rejects_non_rotation(self):
        invalid = np.eye(4)
        invalid[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_se3(invalid)

    def test_fixed_fixture_is_added_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "socket.obj"
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            source = {"mesh": {"target": {"pose": [0] * 7,
                                           "file_path": "target.obj"}}}
            result = add_fixed_mesh_fixtures(source, {
                "fixture_unified_socket": {
                    "pose_robot": _pose(),
                    "collision_mesh": str(mesh),
                }
            })
            self.assertNotIn("fixture_unified_socket", source["mesh"])
            self.assertEqual(
                result["mesh"]["fixture_unified_socket"]["file_path"],
                str(mesh),
            )

    def test_fixed_fixture_cannot_replace_target(self):
        with self.assertRaisesRegex(ValueError, "invalid fixed fixture"):
            add_fixed_mesh_fixtures(
                {"mesh": {}},
                {"target": {"pose_robot": _pose(),
                            "collision_mesh": "/missing.obj"}},
            )


if __name__ == "__main__":
    unittest.main()
