"""A complete stock v8 cell is audited before insertion screening."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.stock_tabletop import audit_tabletop_cell  # noqa: E402


def test_audit_requires_complete_raw_and_exact_stock_pass(tmp_path):
    raw = tmp_path / "raw/key/table/0"
    passed = tmp_path / "pass/key/table/0"
    for seed, clear, success, reason in (
            (0, False, False, "scene_collision"),
            (1, True, True, None)):
        directory = raw / str(seed)
        directory.mkdir(parents=True)
        np.save(directory / "coll_valid.npy", np.array(clear))
        (directory / "sim_eval.json").write_text(json.dumps({
            "hand": "inspire", "version": "v8", "success": success,
            "reason": reason,
        }))
    stable = passed / "1"
    stable.mkdir(parents=True)
    for name in ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
                 "bodex_info.npy"):
        np.save(raw / "1" / name, np.zeros(1))
        np.save(stable / name, np.zeros(1))

    counts, rows = audit_tabletop_cell(
        raw_root=tmp_path / "raw", pass_root=tmp_path / "pass",
        key_object="key", scene_id="0", expected_seed_count=2)
    assert counts["generated"] == 2
    assert counts["mujoco_stable"] == 1
    assert rows == [("1", stable)]
    with pytest.raises(ValueError, match="incomplete BODex cell"):
        audit_tabletop_cell(
            raw_root=tmp_path / "raw", pass_root=tmp_path / "pass",
            key_object="key", scene_id="0", expected_seed_count=3)
