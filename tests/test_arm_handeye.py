import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.execution.handeye import load_arm_C2R, save_arm_C2R


def _session(root: Path, name: str, arm=None, dof=None, matrix=None):
    index = root / name / "0"
    index.mkdir(parents=True)
    if arm is not None:
        with (index / "meta.json").open("w") as handle:
            json.dump({"arm": arm}, handle)
    if dof is not None:
        np.save(index / "qpos.npy", np.zeros(dof))
    if matrix is not None:
        np.save(index / "C2R.npy", matrix)
    return index


class ArmHandeyeTest(unittest.TestCase):
    def test_skips_newer_calibration_for_other_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xarm = np.eye(4); xarm[0, 3] = 0.1
            franka = np.eye(4); franka[0, 3] = 0.7
            _session(root, "20260101_120000", arm="xarm", dof=6, matrix=xarm)
            _session(root, "20260102_120000", arm="franka", dof=7, matrix=franka)

            actual, info = load_arm_C2R("xarm", calibration_root=root)
            np.testing.assert_allclose(actual, xarm)
            self.assertEqual(info["session"], "20260101_120000")
            self.assertEqual(info["arm"], "xarm")

    def test_legacy_session_uses_unambiguous_qpos_dof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = np.eye(4); expected[1, 3] = -0.2
            _session(root, "20250101_120000", dof=7, matrix=expected)

            actual, info = load_arm_C2R("franka", calibration_root=root)
            np.testing.assert_allclose(actual, expected)
            self.assertEqual(info["arm_evidence"], "qpos_dof:7")

    def test_save_records_calibration_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "calibrations"
            out = Path(tmp) / "episode"
            expected = np.eye(4); expected[2, 3] = 1.2
            _session(root, "20260103_120000", arm="xarm", dof=6, matrix=expected)

            save_arm_C2R(out, "xarm", calibration_root=root)
            np.testing.assert_allclose(np.load(out / "C2R.npy"), expected)
            with (out / "C2R_meta.json").open() as handle:
                meta = json.load(handle)
            self.assertEqual(meta["arm"], "xarm")
            self.assertEqual(meta["session"], "20260103_120000")


if __name__ == "__main__":
    unittest.main()
