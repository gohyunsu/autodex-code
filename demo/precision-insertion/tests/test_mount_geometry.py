"""Regression for the FR3/Inspire mount recovered from saved v8 execution."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.mount_geometry import (  # noqa: E402
    link7_to_hand, link8_to_hand,
)


URDF_PATH = (Path(__file__).resolve().parents[3] / "autodex" / "planner" /
             "src" / "curobo" / "content" / "assets" / "robot" /
             "fr3_inspire_description" / "fr3_inspire_oct12_mount.urdf")


def test_saved_october_grasp_fk_and_executor_mount_agree(monkeypatch):
    # Episode 20261012_104059_820384_attempt_0001, plan/lift_preflight.npz.
    q = np.array([0.5267223715782166, 0.549107551574707,
                  0.1237032413482666, -2.1028215885162354,
                  -0.7297736406326294, 2.298358917236328,
                  0.9479228854179382], dtype=np.float64)
    saved = np.array([
        [-0.019921795449367792, 0.2740596007494347,
         -0.9615063480305962, 0.43131157755851746],
        [0.023388033326315727, -0.9613062967658204,
         -0.2744871649012242, 0.2967628538608551],
        [-0.9995279495657942, -0.027956019663842813,
         0.012741232332633351, 0.09740141034126282],
        [0., 0., 0., 1.],
    ])
    robot = URDF.load(str(URDF_PATH), load_meshes=False,
                      build_scene_graph=True)
    robot.update_cfg(dict(zip(robot.actuated_joint_names[:7], q)))
    actual = robot.get_transform("base_link", "base")
    delta = np.linalg.inv(saved) @ actual
    assert np.linalg.norm(delta[:3, 3]) < 1e-6
    assert Rotation.from_matrix(delta[:3, :3]).magnitude() < 1e-6

    flange = robot.get_transform("fr3_link7", "base")
    np.testing.assert_allclose(flange @ link7_to_hand(URDF_PATH), actual,
                               atol=1e-9)
    monkeypatch.setenv("PRECISION_FR3_INSPIRE_URDF", str(URDF_PATH))
    paths = AssetPaths(Path("/unused/shared_root"), select_mode("cylinder", 20))
    assert paths.robot_urdf == URDF_PATH


def test_saved_fr3_otee_is_link8_frame_and_needs_only_flange_to_hand():
    # First raw arm state of v8_demo_lg/inspire/pepsi/
    # 20261012_104059_820384_attempt_0001. O_T_EE is column-major.
    q = np.array([-0.87761205, 0.08288148, -0.04027677,
                  -1.73657179, 1.01233649, 1.52333868, -0.35994095])
    observed_ee = np.array([
        0.76994199, -0.63709974, 0.03596206, 0.,
        -0.29769570, -0.40847391, -0.86285943, 0.,
        0.56441706, 0.65364593, -0.50416303, 0.,
        0.41980502, -0.39255905, 0.54899848, 1.,
    ]).reshape(4, 4, order="F")
    robot = URDF.load(str(URDF_PATH), load_meshes=False,
                      build_scene_graph=True)
    robot.update_cfg(dict(zip(robot.actuated_joint_names[:7], q)))
    link8 = robot.get_transform("fr3_link8", "base")
    hand = robot.get_transform("base_link", "base")
    assert np.linalg.norm(observed_ee[:3, 3] - link8[:3, 3]) < .0004
    predicted_hand = observed_ee @ link8_to_hand(URDF_PATH)
    assert np.linalg.norm(predicted_hand[:3, 3] - hand[:3, 3]) < .0004
    assert Rotation.from_matrix(
        hand[:3, :3].T @ predicted_hand[:3, :3]).magnitude() < 1e-4
    wrong_hand = observed_ee @ link7_to_hand(URDF_PATH)
    assert np.linalg.norm(wrong_hand[:3, 3] - hand[:3, 3]) > .106
