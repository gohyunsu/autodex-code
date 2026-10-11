"""Frozen-session key planning reuses one capture for pilot continuation."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.action_loop import run_session_actions  # noqa: E402
from precision_insertion.live_key_trial import PreparedLiveKeyTrial  # noqa: E402
from precision_insertion import live_planning_handlers  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


PARAMETERS = {
    "maximum_multiview_center_error_mm": 2.,
    "maximum_multiview_angle_error_deg": 5.,
    "maximum_socket_mask_overlap_fraction": .1,
    "socket_projection_dilation_px": 2,
    "minimum_refinement_iou": .7,
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 30.,
    "max_arm_velocity_rad_s": .05,
    "max_key_state_skew_s": .02,
    "max_pose_error_deg": 10.,
    "axial_waypoint_step_m": .005,
    "capture_timeout_s": 5.,
    "image_write_timeout_s": 5.,
    "planning_options": {"reset_candidate_root": Path("/staged/reset")},
    "max_candidate_attempts": 1,
}


class Runner:
    def __init__(self, path):
        self.output_dir = path
        self.decision = SessionDecision("capture_fresh_key", "ready", None)
        self.continuation_calls = []

    def current_decision(self):
        return self.decision

    def preflight_next_key(self, **kwargs):
        self.continuation_calls.append(kwargs)
        self.decision = SessionDecision(
            "execution_gate_required", "selected", "table/0/194")
        return SimpleNamespace(status="sampled_planning_pass")


def _live(runner):
    launch = SimpleNamespace(
        capture_root=Path("/shared/capture"), camera_ids={"front", "side"},
        intrinsics_full={"front": object(), "side": object()},
        extrinsics_full={"front": object(), "side": object()},
        image_hw=(480, 640), view_limits=object())
    cameras = SimpleNamespace(
        launch=launch, init_orchestrator=object(),
        acquisition_metadata_for_request=lambda request_id: {})
    clear_view = SimpleNamespace(
        before_key_capture=lambda: None,
        verify_state=lambda state, **_kwargs: state,
        prepare_failed_grasp_for_capture=lambda **_kwargs: None)
    return SimpleNamespace(
        opened=SimpleNamespace(runner=runner), cameras=cameras,
        clear_view=clear_view)


def test_capture_then_continue_uses_same_key_and_measured_state(
        tmp_path, monkeypatch):
    runner = Runner(tmp_path / "session")
    live = _live(runner)
    state = SimpleNamespace(full_q=np.zeros(13), sample_timestamp_s=10.)
    observation = SimpleNamespace(capture_id="key_001")
    capture_calls = []

    def prepare_next_live_key(**kwargs):
        capture_calls.append(kwargs)
        runner.decision = SessionDecision(
            "continue_candidate_preflight", "pilot_prefix", None)
        return PreparedLiveKeyTrial(
            observation, state, SimpleNamespace(status="candidate_budget_exhausted"),
            kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_planning_handlers, "prepare_next_live_key",
                        prepare_next_live_key)
    ids = iter(["key_001", "key_002"])
    bound = live_planning_handlers.LiveKeyPlanningHandlers(
        live=live, planner=object(),
        state_for_observation=lambda _: state,
        limits=object(), key_prompt="blue key excluding red socket",
        key_parameters=PARAMETERS,
        capture_id_factory=lambda: next(ids))
    result = run_session_actions(runner, handlers=bound.handlers())
    assert result.status == "needs_handler"
    assert result.decision.action == "execution_gate_required"
    assert len(capture_calls) == 1
    assert capture_calls[0]["capture_id"] == "key_001"
    assert capture_calls[0]["runner"] is runner
    assert capture_calls[0]["capture_root"] == live.cameras.launch.capture_root
    assert capture_calls[0]["max_candidate_attempts"] == 1
    assert len(runner.continuation_calls) == 1
    continued = runner.continuation_calls[0]
    assert continued["key_observation"] is observation
    assert continued["measured_start_state"] is state
    assert continued["key_evidence_dir"] == capture_calls[0]["key_evidence_dir"]
    assert continued["max_candidate_attempts"] is None
    assert continued["reset_candidate_root"] == Path("/staged/reset")
    assert bound.latest.preflight.status == "sampled_planning_pass"


def test_failed_grasp_reobserves_with_new_capture(tmp_path, monkeypatch):
    runner = Runner(tmp_path / "session")
    live = _live(runner)
    captured = []
    recovery = []
    live.clear_view.prepare_failed_grasp_for_capture = (
        lambda **kwargs: recovery.append(kwargs))

    def prepare_next_live_key(**kwargs):
        captured.append(kwargs["capture_id"])
        runner.decision = SessionDecision("execution_gate_required", "selected",
                                          f"table/0/{len(captured)}")
        return PreparedLiveKeyTrial(
            SimpleNamespace(capture_id=kwargs["capture_id"]),
            SimpleNamespace(full_q=np.zeros(13), sample_timestamp_s=10.),
            SimpleNamespace(status="sampled_planning_pass",
                            pickup_plan=SimpleNamespace(success=True),
                            trial_scene={"mesh": {"target": {},
                                                  "fixture_socket": {}}}),
            kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_planning_handlers, "prepare_next_live_key",
                        prepare_next_live_key)
    ids = iter(["key_001", "key_002"])
    bound = live_planning_handlers.LiveKeyPlanningHandlers(
        live=live, planner=object(),
        state_for_observation=lambda _: None,
        limits=object(), key_prompt="blue key excluding red socket",
        key_parameters=PARAMETERS,
        capture_id_factory=lambda: next(ids))
    bound.capture_and_preflight(runner, runner.current_decision())
    runner.decision = SessionDecision(
        "reobserve_key_and_preflight", "grasp_failed_try_next_candidate",
        "table/0/1")
    bound.capture_and_preflight(runner, runner.current_decision())
    assert captured == ["key_001", "key_002"]
    assert len(recovery) == 1
    assert recovery[0]["planner"] is bound.planner
    assert recovery[0]["pickup_plan"].success is True
    assert "fixture_socket" in recovery[0]["trial_scene"]["mesh"]
    assert bound.latest.observation.capture_id == "key_002"


def test_fresh_and_reobserved_key_check_clear_view_before_capture_and_exposure(
        tmp_path, monkeypatch):
    runner = Runner(tmp_path / "session")
    live = _live(runner)
    events = []
    state = object()
    live.clear_view.before_key_capture = lambda: events.append("before")

    def verify_state(value, **kwargs):
        events.append(("exposure", kwargs["phase"], kwargs["require_fresh"]))
        assert value is state
        return value

    live.clear_view.verify_state = verify_state

    def capture(**kwargs):
        events.append("capture")
        assert kwargs["state_for_observation"](object()) is state
        return PreparedLiveKeyTrial(
            SimpleNamespace(capture_id=kwargs["capture_id"]), state,
            SimpleNamespace(status="sampled_planning_pass",
                            pickup_plan=SimpleNamespace(success=True),
                            trial_scene={"mesh": {"target": {},
                                                  "fixture_socket": {}}}),
            kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_planning_handlers, "prepare_next_live_key",
                        capture)
    bound = live_planning_handlers.LiveKeyPlanningHandlers(
        live=live, planner=object(), state_for_observation=lambda _: state,
        limits=object(), key_prompt="blue key excluding socket",
        key_parameters=PARAMETERS)
    bound.capture_and_preflight(runner, runner.current_decision())
    runner.decision = SessionDecision(
        "reobserve_key_and_preflight", "grasp_failed_try_next_candidate",
        "table/0/1")
    bound.capture_and_preflight(runner, runner.current_decision())
    assert events == ["before", "capture", ("exposure", "key_exposure", False),
                      "before", "capture", ("exposure", "key_exposure", False)]


def test_continuation_without_saved_capture_and_parameter_override_rejected(
        tmp_path):
    runner = Runner(tmp_path / "session")
    live = _live(runner)
    with pytest.raises(ValueError, match="override"):
        live_planning_handlers.LiveKeyPlanningHandlers(
            live=live, planner=object(), state_for_observation=lambda _: None,
            limits=object(), key_prompt="key", key_parameters={
                **PARAMETERS, "runner": object()})
    bound = live_planning_handlers.LiveKeyPlanningHandlers(
        live=live, planner=object(), state_for_observation=lambda _: None,
        limits=object(), key_prompt="key", key_parameters=PARAMETERS)
    runner.decision = SessionDecision(
        "continue_candidate_preflight", "pilot_prefix", None)
    with pytest.raises(ValueError, match="no saved key"):
        bound.continue_same_capture(runner, runner.current_decision())
