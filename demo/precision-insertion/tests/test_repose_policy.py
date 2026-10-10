"""An insertable alternative pose is not yet an executable reorientation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.repose_policy import assess_repose_options  # noqa: E402


MODE = select_mode("cylinder", 20)


def _current_pose(root: Path):
    tabletop = (root / "object_processing" / MODE.key_object /
                "processed_data/info/tabletop")
    tabletop.mkdir(parents=True)
    np.save(tabletop / "000.npy", np.eye(4))


def _assess(root: Path, **overrides):
    args = dict(
        shared_root=root, mode=MODE, catalog={"shared_root": str(root)},
        current_pose_stem="000", target_stems=("001", "002"),
        T_robot_key=np.eye(4), max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0)
    args.update(overrides)
    return assess_repose_options(**args)


def test_repose_assessment_requires_both_insertable_grasp_and_reset_seed(
    tmp_path, monkeypatch,
):
    _current_pose(tmp_path)
    seen = []

    def insertion(_catalog, *, tabletop_pose_stem, **_):
        return {"status": "candidates_available",
                "candidates": [object()] * (2 if tabletop_pose_stem == "002" else 1)}

    def reset(**kwargs):
        seen.append((kwargs["height_cm"], kwargs["to_pose_stem"],
                     kwargs["attempted_ids"], kwargs["candidate_root"]))
        if kwargs["height_cm"] == 12 and kwargs["to_pose_stem"] == "001":
            return {"scene_info": [{"grasp_idx": "191",
                                    "source": "/staged/191"}]}
        return None

    monkeypatch.setattr(
        "precision_insertion.repose_policy.select_pose_candidates", insertion)
    monkeypatch.setattr(
        "precision_insertion.repose_policy.load_v8_reset_seeds", reset)
    handoff = tmp_path / "handoff"
    result = _assess(
        tmp_path, candidate_root=handoff,
        attempted_reset=((12, "001", "2"),))
    assert result["status"] == (
        "staged_reset_seed_requires_install_and_full_chain_preflight")
    assert result["candidate_root_is_canonical"] is False
    assert result["robot_ready"] is False
    assert result["targets"][0]["target_pose_stem"] == "001"
    assert result["targets"][0]["reset_seed_refs"] == [
        {"height_cm": 12, "seed_id": "191", "source": "/staged/191"}]
    assert result["targets"][1]["status"] == "no_reset_seed_for_insertable_pose"
    assert (12, "001", ("2",), handoff / "reset_12") in seen
    canonical = _assess(tmp_path)
    assert canonical["status"] == (
        "reset_seed_available_requires_full_chain_preflight")
    assert canonical["candidate_root_is_canonical"] is True


def test_repose_assessment_stops_for_missing_or_stale_evidence(tmp_path, monkeypatch):
    _current_pose(tmp_path)
    monkeypatch.setattr(
        "precision_insertion.repose_policy.select_pose_candidates",
        lambda *_, **__: {"status": "catalog_stale", "reason": "changed"})
    result = _assess(tmp_path)
    assert result["status"] == "catalog_unavailable"
    with pytest.raises(ValueError, match="drift and tilt limits"):
        _assess(tmp_path, max_center_in_hand_drift_m=0.0)
    with pytest.raises(ValueError, match="unique other v8"):
        _assess(tmp_path, target_stems=("000",))
    with pytest.raises(ValueError, match="unique other v8"):
        _assess(tmp_path, target_stems=("1", "001"))
    with pytest.raises(ValueError, match="different shared roots"):
        _assess(tmp_path, catalog={"shared_root": str(tmp_path / "other")})
