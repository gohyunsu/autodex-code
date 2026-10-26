"""A failed held insertion must return, clear the camera, then observe reset."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_recovery_handlers  # noqa: E402
from precision_insertion.live_key_trial import ReturnedKeyCapture  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402


CAPTURE = {
    "maximum_multiview_center_error_mm": 2.,
    "maximum_multiview_angle_error_deg": 5.,
    "maximum_socket_mask_overlap_fraction": .1,
    "socket_projection_dilation_px": 2,
    "minimum_refinement_iou": .7,
    "max_key_state_skew_s": .02,
    "capture_timeout_s": 5.,
    "image_write_timeout_s": 5.,
}
RETURN = {
    "release_height_m": .1,
    "minimum_rest_socket_clearance_m": .01,
    "minimum_board_edge_clearance_m": .01,
    "minimum_release_key_clearance_m": .01,
    "max_withdrawal_state_gap_s": .2,
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 30.,
    "max_arm_velocity_rad_s": .05,
}
LANDING = {
    "max_pose_error_deg": 10.,
    "max_return_center_shift_m": .02,
    "support_tolerance_m": .005,
    "minimum_rest_socket_clearance_m": .01,
    "minimum_board_edge_clearance_m": .01,
}


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.decision = SessionDecision(
            "recover_held_key_then_reobserve", "insertion_failed", "table/0/3")
        self.active_attempt = SimpleNamespace(
            attempt_id="attempt_001", candidate_id="table/0/3",
            started_at_s=10.)
        self.session_sha256 = "a" * 64
        self.catalog_sha256 = "b" * 64
        self.events = []

    def current_decision(self):
        return self.decision

    def reset_landing_geometry(self, **_kwargs):
        return {"classification": {"stem": "000"},
                "target_pose_stem": "000", "center_xy_displacement_m": 0.0}

    def prepare_failed_insertion_return(self, **kwargs):
        self.events.append(("plan", kwargs))
        return object(), object(), self.output_dir / "return_plan.json"

    def observe_reset_landing(self, **kwargs):
        self.events.append(("landing", kwargs))
        self.decision = SessionDecision(
            "reobserve_key_and_preflight", "reset_observed", None)


def _state():
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=20., arm_timestamp_s=20.,
        arm_robot_uptime_s=10., hand_timestamp_s=20.,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


def _setup(tmp_path):
    runner = Runner(tmp_path)
    events = []
    clear_view = SimpleNamespace(
        complete_failed_insertion_return_with_autodex=lambda **kwargs:
        events.append(("motion", kwargs)),
        before_key_capture=lambda: events.append("clear_view"))
    launch = SimpleNamespace(
        capture_root=tmp_path, camera_ids={"front", "side"},
        intrinsics_full={"front": object(), "side": object()},
        extrinsics_full={"front": object(), "side": object()},
        image_hw=(480, 640), view_limits=object())
    cameras = SimpleNamespace(
        launch=launch, init_orchestrator=object(),
        acquisition_metadata_for_request=lambda _id: {})
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        clear_view=clear_view, cameras=cameras)

    def withdraw(_runner, _decision):
        events.append("withdraw")
        return _state(), tmp_path / "withdrawal.json"

    def record(run, _report, _runner, _decision):
        events.append("record_start")
        run()
        events.append("record_end")
        return tmp_path / "execution.json"

    bound = live_recovery_handlers.LiveHeldReturnHandlers(
        live=live, planner=object(), limits=object(),
        state_for_observation=lambda _obs: _state(),
        withdraw_guarded=withdraw, record_execution=record,
        key_prompt="white precision key excluding white socket",
        capture_parameters=CAPTURE, return_parameters=RETURN,
        landing_parameters=LANDING, capture_id_factory=lambda: "reset_001")
    return bound, runner, events


def test_held_return_clears_view_then_recaptures_same_session(
        tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    observation = SimpleNamespace(acquisition_interval_s=(30., 31.))

    def verify(**kwargs):
        events.append("verified_execution")
        assert kwargs["attempt_id"] == "attempt_001"
        return {"plan_report": {"path": str((tmp_path / "return_plan.json").resolve())}}

    def capture(**kwargs):
        assert kwargs["runner"] is runner
        assert kwargs["capture_id"] == "reset_001"
        assert kwargs["key_prompt"] == bound.key_prompt
        kwargs["clear_view"].before_key_capture()
        events.append("capture")
        return ReturnedKeyCapture(observation, _state(),
                                  kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_recovery_handlers,
                        "verify_failed_return_execution", verify)
    monkeypatch.setattr(live_recovery_handlers,
                        "capture_returned_key_for_reset", capture)
    bound.recover(runner, runner.current_decision())
    assert [item if isinstance(item, str) else item[0] for item in events] == [
        "withdraw", "record_start", "motion", "record_end",
        "verified_execution", "clear_view", "capture"]
    assert runner.events[0][0] == "plan"
    assert runner.events[0][1]["withdrawn_state"].source == "robot_joint_feedback"
    assert runner.events[1][0] == "landing"
    assert runner.events[1][1]["key_observation"] is observation
    assert runner.events[1][1]["recovery_log_path"] == tmp_path / "execution.json"
    assert runner.current_decision().action == "reobserve_key_and_preflight"


def test_held_return_reuses_withdrawal_after_inspire_fingers_settle(
        tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    def measured(raw_value, timestamp):
        raw = np.full(6, raw_value)
        commanded = np.full(6, 500.)
        state = LiveRobotState(
            np.r_[np.zeros(7), convert_inspire_raw(raw[None])[0]],
            np.zeros(7), timestamp, timestamp, 10., timestamp,
            raw, commanded, abs(raw_value - 500.), np.zeros(6))
        state.validate(max_arm_hand_skew_s=.1,
                       max_hand_command_error_raw=30.,
                       max_arm_velocity_rad_s=.05)
        return state

    earlier = measured(500., 20.)
    current = measured(520., 21.)
    assert np.max(np.abs(current.full_q[7:] - earlier.full_q[7:])) > .02
    bound.read_state = lambda: current
    bound.reuse_withdrawal = lambda: (
        "attempt_001", earlier, tmp_path / "withdrawal.json")
    observation = SimpleNamespace(acquisition_interval_s=(30., 31.))
    monkeypatch.setattr(
        live_recovery_handlers, "verify_failed_return_execution",
        lambda **_kwargs: {"plan_report": {
            "path": str((tmp_path / "return_plan.json").resolve())}})
    monkeypatch.setattr(
        live_recovery_handlers, "capture_returned_key_for_reset",
        lambda **kwargs: ReturnedKeyCapture(
            observation, current, kwargs["key_evidence_dir"]))
    bound.recover(runner, runner.current_decision())
    assert "withdraw" not in events
    assert runner.events[0][1]["withdrawn_state"] is current
    assert runner.events[0][1]["withdrawal_evidence_path"] == (
        tmp_path / "withdrawal.json")


def test_reused_withdrawal_rejects_a_moved_arm_before_return(tmp_path):
    bound, runner, events = _setup(tmp_path)
    current = replace(_state(), sample_timestamp_s=21.,
                      arm_timestamp_s=21., hand_timestamp_s=21.)
    current.full_q[0] = .1
    bound.read_state = lambda: current
    bound.reuse_withdrawal = lambda: (
        "attempt_001", _state(), tmp_path / "withdrawal.json")
    with pytest.raises(ValueError, match="robot moved"):
        bound.recover(runner, runner.current_decision())
    assert "withdraw" not in events
    assert not runner.events


def test_controller_must_invoke_return_before_any_camera_capture(
        tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    bound.record_execution = lambda *_args: tmp_path / "execution.json"
    monkeypatch.setattr(live_recovery_handlers,
                        "capture_returned_key_for_reset",
                        lambda **_kwargs: events.append("capture"))
    with pytest.raises(ValueError, match="skipped the physical return"):
        bound.recover(runner, runner.current_decision())
    assert "capture" not in events


def test_rejected_execution_cannot_recapture_landed_key(tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    monkeypatch.setattr(live_recovery_handlers,
                        "verify_failed_return_execution",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            ValueError("invalid controller record")))
    monkeypatch.setattr(live_recovery_handlers,
                        "capture_returned_key_for_reset",
                        lambda **_kwargs: events.append("capture"))
    with pytest.raises(ValueError, match="invalid controller record"):
        bound.recover(runner, runner.current_decision())
    assert "capture" not in events
    assert "clear_view" not in events


def test_wrong_session_or_action_cannot_move_robot(tmp_path):
    bound, runner, events = _setup(tmp_path)
    with pytest.raises(ValueError, match="another session or decision"):
        bound.recover(object(), runner.current_decision())
    wrong = SessionDecision("capture_fresh_key", "ready", None)
    with pytest.raises(ValueError, match="another session or decision"):
        bound.recover(runner, wrong)
    assert not events


def test_misplaced_return_is_restored_before_trial_reset_label(tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    monkeypatch.setattr(live_recovery_handlers,
                        "verify_failed_return_execution",
                        lambda **_kwargs: {"plan_report": {
                            "path": str((tmp_path / "return_plan.json").resolve())}})
    captures = iter(("reset_initial", "reset_corrected"))
    bound.capture_id_factory = lambda: next(captures)

    def capture(**kwargs):
        events.append(("capture", kwargs["capture_id"]))
        return ReturnedKeyCapture(
            SimpleNamespace(acquisition_interval_s=(30., 31.)),
            _state(), kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_recovery_handlers,
                        "capture_returned_key_for_reset", capture)
    states = iter((.035, .0))
    runner.reset_landing_geometry = lambda **_kwargs: {
        "classification": {"stem": "000"}, "target_pose_stem": "000",
        "center_xy_displacement_m": next(states)}
    execution = tmp_path / "restoration_execution.json"
    bound._restore_original = lambda *_args: (
        events.append("restore") or (execution, 25.0))
    bound.recover(runner, runner.current_decision())
    assert [item[0] if isinstance(item, tuple) else item for item in events].count(
        "restore") == 1
    assert runner.events[-1][1]["restoration_evidence_paths"] == (execution,)


def test_released_failure_is_perceived_and_restored_without_withdrawal(
        tmp_path, monkeypatch):
    bound, runner, events = _setup(tmp_path)
    release = tmp_path / "poststroke_release.json"
    release.write_text('{"released":true}\n', encoding="utf-8")
    runner.decision = SessionDecision(
        "recover_released_key_then_reobserve", "xy_advice_saved",
        "table/0/3")
    runner._attempt_dir = tmp_path / "attempts" / "attempt_001"
    runner._attempt_dir.mkdir(parents=True)
    runner.active_attempt.events = [{
        "stage": "insertion_success", "value": False,
        "timestamp_s": 10.5,
        "evidence_refs": {"release_execution": str(release.resolve())},
    }, {"stage": "next_trial_xy_advice", "timestamp_s": 20.0}]
    state = replace(
        _state(), sample_timestamp_s=19.95,
        arm_timestamp_s=19.95, hand_timestamp_s=19.95)
    monkeypatch.setattr(live_recovery_handlers.time, "time", lambda: 20.1)
    bound.read_state = lambda: state
    bound.live.clear_view.before_key_capture = lambda: (
        events.append("clear_view") or state)
    bound.live.clear_view.verify_state = lambda measured, **_kwargs: measured
    observation = SimpleNamespace(acquisition_interval_s=(30., 31.))
    landed = ReturnedKeyCapture(observation, state, tmp_path / "released_key")
    bound._capture_landing = lambda *_args: (
        events.append("capture_released") or landed)
    bound.restore_until_original = lambda *_args: (
        events.append("restore_original") or (landed, ()))

    bound.recover_released(runner, runner.current_decision())

    assert events == ["capture_released", "restore_original"]
    assert "withdraw" not in events
    record = json.loads((runner._attempt_dir /
                         "released_failure_clear_view.json").read_text())
    assert record["hand_open"] is True
    assert record["clear_view_reached"] is True
    assert record["measured_state"]["sample_timestamp_s"] == 19.95
    assert record["completed_at_s"] == 20.1
    assert record["poststroke_release"]["path"] == str(release.resolve())
    assert runner.events[-1][0] == "landing"
    assert runner.events[-1][1]["key_observation"] is observation
    assert runner.current_decision().action == "reobserve_key_and_preflight"
