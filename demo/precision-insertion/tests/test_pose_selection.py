"""Live-key tabletop class binding to explicit v8 asset roots."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.pose_selection import classify_key_tabletop_pose  # noqa: E402


def _pose_x(degrees):
    theta = np.deg2rad(degrees)
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]]
    return pose


def _assets(tmp_path, mode):
    paths = AssetPaths(tmp_path, mode)
    paths.key_tabletop_dir.mkdir(parents=True)
    np.save(paths.key_tabletop_dir / "000.npy", np.eye(4))
    np.save(paths.key_tabletop_dir / "001.npy", _pose_x(90))
    if mode.family == "cylinder":
        info = paths.key_tabletop_dir.parent
        (info / "symmetry.json").write_text(json.dumps({
            "type": "Dinf", "axes": [
                {"axis": [0, 0, 1], "fold": "inf"},
                {"axis": [1, 0, 0], "fold": 2},
            ],
        }), encoding="utf-8")


@pytest.mark.parametrize("family,gap", [("square", 1.5), ("cylinder", 20)])
def test_fresh_key_pose_matches_only_explicit_v8_tabletop_class(
    tmp_path, family, gap,
):
    mode = select_mode(family, gap)
    _assets(tmp_path, mode)
    result = classify_key_tabletop_pose(
        mode=mode, shared_root=tmp_path, pose_robot_key=_pose_x(90),
        max_rotation_error_deg=10)
    assert result["stem"] == "001"
    assert result["rotation_error_deg"] == pytest.approx(0, abs=0.01)
    assert result["source"] == "fresh_key_pose_not_physical_pose_certification"
    with pytest.raises(ValueError, match="does not match"):
        classify_key_tabletop_pose(
            mode=mode, shared_root=tmp_path, pose_robot_key=_pose_x(45),
            max_rotation_error_deg=10)
