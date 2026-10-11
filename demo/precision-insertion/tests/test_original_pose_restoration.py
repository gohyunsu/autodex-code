"""Original-tabletop-target geometry and no unmeasured state substitution."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.original_pose_restoration import (  # noqa: E402
    original_center_release_xy, plan_original_pose_restoration,
)


def test_canonical_release_origin_preserves_original_cad_center(tmp_path):
    mode = select_mode("cylinder", 20)
    folder = AssetPaths(tmp_path, mode).key_tabletop_dir
    folder.mkdir(parents=True)
    canonical = np.eye(4)
    canonical[:3, :3] = np.array([[0., 0., 1.],
                                 [0., 1., 0.],
                                 [-1., 0., 0.]])
    np.save(folder / "001.npy", canonical)
    center = (0., 0., .04)
    xy = original_center_release_xy(
        shared_root=tmp_path, mode=mode, target_pose_stem="001",
        original_center_xy_robot_m=(.32, -.08),
        key_center_local_m=center)
    assert xy == pytest.approx((.28, -.08))
    assert np.asarray(xy) + (canonical[:3, :3] @ center)[:2] == pytest.approx(
        (.32, -.08))


def test_restoration_requires_exposure_aligned_measured_joints(tmp_path):
    with pytest.raises(ValueError, match="camera-synchronized measured joints"):
        plan_original_pose_restoration(
            planner=object(), shared_root=tmp_path, mode=select_mode("cylinder", 20),
            calibration=object(), catalog={},
            key_observation=SimpleNamespace(selected_acquisition_timestamp_s=10.),
            measured_start_q=np.zeros(13), start_q_timestamp_s=11.,
            max_state_skew_s=.02, target_pose_stem="000",
            target_xy_robot_m=(0., 0.), release_height_cm=12,
            max_pose_error_deg=10., minimum_rest_socket_clearance_m=.005,
            minimum_board_edge_clearance_m=.005,
            minimum_release_key_clearance_m=.005,
            retreat_goal_arm_q=np.zeros(7), limits=object())
