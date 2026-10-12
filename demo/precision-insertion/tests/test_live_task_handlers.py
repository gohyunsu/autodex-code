"""One selected grasp proceeds through observed hold and 20 mm verdict."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_task_handlers  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def _state():
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=10., arm_timestamp_s=10.,
        arm_robot_uptime_s=1., hand_timestamp_s=10.,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


def _capture(label):
    return RawCameraCapture(
        capture_id=label, request_id=10,
        images_bgr={camera: np.zeros((8, 8, 3), dtype=np.uint8)
                    for camera in ("front", "side")},
        frame_ids={"front": 10, "side": 10}, acquisition_metadata={})


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.active_attempt = SimpleNamespace(
            attempt_id="attempt_001", candidate_id="table/0/3")
        self.decision = SessionDecision(
            "transfer_execution_gate_required", "bounded_plan", "table/0/3")
        self._postlift_report_path = root / "postlift.json"
        self._preinsert_report_path = root / "preinsert.json"
        self.initial_key_evidence_dir = root / "initial_key"
        self.session_sha256 = "a" * 64
        self.events = []

    def current_decision(self):
        return self.decision

    def prepare_observed_preinsert_label(self, **kwargs):
        self.events.append(("assess_hold", kwargs))
        return SimpleNamespace(preinsert_reached=True)

    def observe_stage(self, stage, status, **kwargs):
        self.events.append(("label_hold", stage, status, kwargs))
        self.decision = SessionDecision(
            "await_guarded_insertion_and_observation", "hold_observed",
            "table/0/3")

    def prepare_guarded_axial_handoff(self, **kwargs):
        self.events.append(("handoff", kwargs))
        return self.output_dir / "handoff.json"

    def prepare_released_insertion_label(self, **kwargs):
        self.events.append(("assess_insertion", kwargs))
        self.decision = SessionDecision("session_success", "observed_20mm", "table/0/3")


def _handler(tmp_path, enabled=True):
    runner = Runner(tmp_path)
    first = _capture("preinsert_001")
    final = _capture("final_001")
    bound = live_task_handlers.LiveTaskHandlers(
        live=SimpleNamespace(
            opened=SimpleNamespace(runner=runner),
            cameras=SimpleNamespace(launch=SimpleNamespace(
                camera_ids={"front", "side"})),
            clear_view=SimpleNamespace(before_key_capture=lambda: None)),
        transfer_adapter=object(), insertion_adapter=object(),
        read_state=_state, state_for_capture=lambda _capture: _state(),
        capture_preinsert=lambda *_args: first,
        capture_final=lambda *_args: final,
        backend=object(), motion_interlock=lambda: True,
        transfer_limits=object(), insertion_limits=object(),
        contact_limits=object(),
        transfer_commissioning_path=tmp_path / "transfer_commissioning.json",
        insertion_watchdog_path=tmp_path / "watchdog.json",
        insertion_contact_path=tmp_path / "contact.json",
        preinsert_parameters={}, handoff_parameters={},
        insertion_parameters={"max_handoff_age_s": 1.},
        final_parameters={},
        release_and_clear=lambda *_args: tmp_path / "release.json",
        enable_robot_motion=enabled,
        decision_time=lambda: 20.)
    return bound, runner, first, final


def test_transfer_hold_then_guarded_20mm_and_vlm_label(tmp_path, monkeypatch):
    bound, runner, first, final = _handler(tmp_path)
    actions = []
    bound.release_and_clear = lambda *_args: (
        actions.append(("release_and_clear",)) or tmp_path / "release.json")
    bound.live.clear_view.before_key_capture = lambda: actions.append(
        ("clear_view_gate",))
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_held_transfer",
        lambda **kwargs: actions.append(("transfer", kwargs)))
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_guarded_insertion",
        lambda **kwargs: (actions.append(("insertion", kwargs)) or {
            "metric_record": {"path": str(tmp_path / "metric.json")}}))
    monkeypatch.setattr(
        live_task_handlers, "write_raw_camera_capture",
        lambda capture, output, **kwargs:
        actions.append(("write", capture, output, kwargs["phase"])))
    monkeypatch.setattr(
        live_task_handlers, "verify_raw_camera_capture",
        lambda output, **kwargs:
        actions.append(("verify", output, kwargs["phase"])))
    monkeypatch.setattr(
        live_task_handlers, "verify_post_insertion_release",
        lambda *args, **kwargs: {})
    bound.handlers()["transfer_execution_gate_required"](
        runner, runner.current_decision())
    assert runner.current_decision().action == "await_guarded_insertion_and_observation"
    assert [row[0] for row in runner.events] == ["assess_hold", "label_hold"]
    assert runner.events[1][1:3] == ("preinsert_reached", True)
    assert actions[1][1] is first and actions[1][3] == "preinsert"
    bound.handlers()["await_guarded_insertion_and_observation"](
        runner, runner.current_decision())
    assert [row[0] for row in runner.events] == [
        "assess_hold", "label_hold", "handoff", "assess_insertion"]
    assert [row[0] for row in actions] == [
        "transfer", "write", "verify", "insertion", "release_and_clear",
        "clear_view_gate", "write", "verify"]
    assert actions[6][1] is final and actions[6][3] == "final_clear_view"
    assert runner.events[-1][1]["metric_record_path"] == tmp_path / "metric.json"
    assert runner.events[-1][1]["initial_key_bundle"] == tmp_path / "initial_key"
    capture_root = tmp_path / "attempts" / "attempt_001" / "raw_captures"
    assert runner.events[-1][1]["preinsert_bundle"] == (
        capture_root / "preinsert" / "preinsert_001")
    assert runner.events[-1][1]["final_bundle"] == (
        capture_root / "final_clear_view" / "final_001")
    assert runner.current_decision().action == "session_success"


def test_motion_disabled_before_transfer_or_insertion(tmp_path):
    bound, runner, *_ = _handler(tmp_path, enabled=False)
    with pytest.raises(PermissionError, match="not enabled"):
        bound.transfer_and_observe(runner, runner.current_decision())
    assert runner.events == []
