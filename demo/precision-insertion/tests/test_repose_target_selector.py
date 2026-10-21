"""Finite reset target search preserves key XY before nearby cells."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.repose_target_selector import (  # noqa: E402
    DirectedReposeTargetSelector, ReposeTargetsExhausted,
)
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from autodex.utils.conversion import se32cart  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402


def _inputs():
    trial = SimpleNamespace(
        pose_class={"stem": "000"},
        repose_target_stems=("001", "002"),
        trial_scene={"mesh": {"target": {
            "pose": [0.4, 0.2, 0.05, 1., 0., 0., 0.]}}})
    latest = SimpleNamespace(
        observation=SimpleNamespace(capture_id="key_1"), preflight=trial)
    catalog = {"candidates": [
        {"tabletop_pose_stem": "001", "eligible": True},
        {"tabletop_pose_stem": "002", "eligible": True},
        {"tabletop_pose_stem": "002", "eligible": True}]}
    return SimpleNamespace(catalog=catalog), latest


def _decision(action):
    return SessionDecision(action, "test", None)


def test_reset_selector_ranks_coverage_and_preserves_first_release_xy():
    runner, latest = _inputs()
    selector = DirectedReposeTargetSelector(
        heights_cm=(12,), xy_offsets_m=((0., 0.), (.03, 0.)))
    first = selector(runner, _decision("preflight_repose"), latest)
    assert first == {"to_pose_stem": "002", "height_cm": 12,
                     "release_xy_robot_m": (.4, .2)}
    assert selector(runner, _decision("continue_repose_seed_preflight"),
                    latest) == first
    assert selector(runner, _decision("preflight_repose_release"),
                    latest) == first
    second = selector(runner, _decision("preflight_alternative_repose_cell"),
                      latest)
    assert second["to_pose_stem"] == "002"
    assert second["release_xy_robot_m"][0] == pytest.approx(.43)
    assert selector(runner, _decision("preflight_alternative_repose_cell"),
                    latest)["to_pose_stem"] == "001"
    selector(runner, _decision("preflight_alternative_repose_cell"), latest)
    with pytest.raises(ReposeTargetsExhausted, match="all 4"):
        selector(runner, _decision("preflight_alternative_repose_cell"),
                 latest)
    assert len(selector.attempted_choices()) == 4


def test_cylinder_repose_preserves_cad_center_when_standing_key(tmp_path,
                                                                 monkeypatch):
    mode = select_mode("cylinder", 20)
    canonical = AssetPaths(tmp_path, mode).key_tabletop_dir
    canonical.mkdir(parents=True)
    np.save(canonical / "000.npy", np.eye(4))
    monkeypatch.setattr(
        "precision_insertion.repose_target_selector._load_mesh",
        lambda _path: SimpleNamespace(bounds=np.array([
            [-.015, -.015, 0.], [.015, .015, .08]])))
    lying = np.eye(4)
    lying[:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
    lying[:3, 3] = [.4, .2, .015]
    trial = SimpleNamespace(
        pose_class={"stem": "001"}, repose_target_stems=("000",),
        trial_scene={"mesh": {"target": {
            "pose": se32cart(lying).tolist()}}})
    latest = SimpleNamespace(
        observation=SimpleNamespace(capture_id="lying_1"), preflight=trial)
    runner = SimpleNamespace(
        mode=mode, shared_root=tmp_path,
        catalog={"candidates": [{"tabletop_pose_stem": "000",
                                 "eligible": True}]})
    choice = DirectedReposeTargetSelector(
        xy_offsets_m=((0., 0.),))(
            runner, _decision("preflight_repose"), latest)
    assert choice["release_xy_robot_m"] == pytest.approx((.44, .2))


def test_reset_selector_rejects_invalid_heights_and_resets_on_new_capture():
    with pytest.raises(ValueError, match="v8 heights"):
        DirectedReposeTargetSelector(heights_cm=(10,))
    runner, latest = _inputs()
    selector = DirectedReposeTargetSelector(
        heights_cm=(12,), xy_offsets_m=((0., 0.),))
    selector(runner, _decision("preflight_repose"), latest)
    another = SimpleNamespace(
        observation=SimpleNamespace(capture_id="key_2"),
        preflight=latest.preflight)
    assert selector(runner, _decision("preflight_repose"), another)[
        "to_pose_stem"] == "002"


def test_missing_directed_seed_skips_other_xy_cells_at_same_height():
    runner, latest = _inputs()
    selector = DirectedReposeTargetSelector(
        heights_cm=(12,), xy_offsets_m=((0., 0.), (.03, 0.)))
    assert selector(runner, _decision("preflight_repose"), latest)[
        "to_pose_stem"] == "002"
    next_choice = selector(
        runner, SessionDecision("preflight_alternative_repose_cell",
                                "no_verified_reset_seed", None), latest)
    assert next_choice["to_pose_stem"] == "001"
    assert len(selector.attempted_choices()) == 2


def test_selector_skips_pose_without_eligible_reset_seed():
    runner, latest = _inputs()
    selector = DirectedReposeTargetSelector(
        heights_cm=(12,), xy_offsets_m=((0., 0.), (.03, 0.)))
    seen = []

    def available(stem, height):
        seen.append((stem, height))
        return stem == "001"

    first = selector(runner, _decision("preflight_repose"), latest,
                     seed_available=available)
    assert first["to_pose_stem"] == "001"
    assert seen == [("002", 12), ("001", 12)]
    selector(runner, _decision("preflight_alternative_repose_cell"), latest)
    with pytest.raises(ReposeTargetsExhausted, match="all 2"):
        selector(runner, _decision("preflight_alternative_repose_cell"), latest)


def test_selector_stops_without_repeating_unavailable_reset_cells():
    runner, latest = _inputs()
    selector = DirectedReposeTargetSelector()
    with pytest.raises(ReposeTargetsExhausted, match="all 0"):
        selector(runner, _decision("preflight_repose"), latest,
                 seed_available=lambda _stem, _height: False)
    assert selector.attempted_choices() == ()


def test_exhausted_repose_search_ends_session_with_saved_reason(tmp_path):
    runner = SessionRunner.__new__(SessionRunner)
    runner._attempt = None
    runner._preflight = SimpleNamespace(
        repose_target_stems=("001",), pose_class={"stem": "000"})
    runner._capture_id = "key_1"
    runner._repose_search_exhausted = False
    runner.output_dir = tmp_path
    runner.current_decision = lambda: _decision(
        "preflight_alternative_repose_cell")
    runner.exhaust_repose_targets((
        {"to_pose_stem": "001", "height_cm": 12,
         "release_xy_robot_m": (.4, .2)},))
    del runner.current_decision
    assert runner.current_decision().action == "stop_for_review"
    assert (tmp_path / "repose_targets_exhausted.json").is_file()
