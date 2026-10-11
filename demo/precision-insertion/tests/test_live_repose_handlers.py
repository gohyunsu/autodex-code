"""Pose exhaustion reuses the saved AutoDex key capture for v8 reset plans."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.live_key_trial import PreparedLiveKeyTrial  # noqa: E402
from precision_insertion.live_repose_handlers import (  # noqa: E402
    LiveReposePlanningHandlers,
)
from precision_insertion.session_policy import SessionDecision  # noqa: E402


class Runner:
    def __init__(self):
        self.decision = SessionDecision(
            "preflight_repose", "current_pose_has_no_remaining_plannable_grasp",
            None)
        self.calls = []

    def current_decision(self):
        return self.decision

    def preflight_repose(self, **kwargs):
        self.calls.append(kwargs)
        self.decision = SessionDecision(
            "repose_execution_gate_required", "nominal_reset_plan", None)


def _handler():
    runner = Runner()
    observed = object()
    state = SimpleNamespace(full_q=np.zeros(13), sample_timestamp_s=10.)
    latest = PreparedLiveKeyTrial(
        observation=observed, state=state, preflight=object(),
        key_evidence_dir=Path("/saved/key"))
    planner = object()
    limits = object()
    bound = LiveReposePlanningHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        key_planning=SimpleNamespace(latest=latest), planner=planner,
        path_limits=limits,
        select_target=lambda *_args: {
            "to_pose_stem": "table_pose_001", "height_cm": 12,
            "release_xy_robot_m": (.25, .05)},
        parameters={"max_state_skew_s": .02, "max_pose_error_deg": 10.})
    return bound, runner, latest, planner, limits


def test_repose_plans_from_same_exhausted_capture():
    bound, runner, latest, planner, limits = _handler()
    bound.handlers()["preflight_repose"](runner, runner.current_decision())
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["key_observation"] is latest.observation
    assert call["live_start_q"] is latest.state.full_q
    assert call["start_q_acquisition_timestamp_s"] == 10.
    assert call["planner"] is planner and call["limits"] is limits
    assert call["to_pose_stem"] == "table_pose_001"
    assert runner.current_decision().action == "repose_execution_gate_required"


def test_wrong_session_does_not_plan_repose():
    bound, runner, *_ = _handler()
    with pytest.raises(ValueError, match="another session or decision"):
        bound.plan(object(), runner.current_decision())
    assert runner.calls == []
