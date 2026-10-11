"""The square live retry uses one pending plan and one new arrival image."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_observed_xy_retry_handlers as module  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402
from test_measured_lift_preflight import _state  # noqa: E402


def test_square_live_retry_connects_withdrawal_transfer_arrival_and_label(
    tmp_path, monkeypatch,
):
    state = _state(102.0, np.full(7, .03))
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    raw = RawCameraCapture(
        "held_1", 42, {"cam_a": image, "cam_b": image.copy()},
        {"cam_a": 1, "cam_b": 2},
        {"request_id": 42, "source": "camera_acquisition", "frames": {
            "cam_a": {"timestamp_s": 102.1},
            "cam_b": {"timestamp_s": 102.1}}})
    attempt = SimpleNamespace(attempt_id="a1", candidate_id="table/0/3")
    first = SessionDecision(
        "guarded_withdrawal_then_xy_assessment", "failed", "table/0/3")
    retry = SessionDecision(
        "await_retry_execution_and_observation", "pending", "table/0/3")
    calls = []
    runner = SimpleNamespace(
        mode=SimpleNamespace(family="square"), active_attempt=attempt,
        _attempt_dir=tmp_path / "attempt", _postlift_report_path=(
            tmp_path / "postlift.json"), _retry_assessment_index=0,
        current_decision=lambda: first)
    runner._attempt_dir.mkdir()
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text(json.dumps({"completed_at_s": 101.9}),
                          encoding="utf-8")
    preflight = object()

    def prepare_retry(**kwargs):
        calls.append(("prepare", kwargs["withdrawal_evidence_path"]))
        runner._retry_assessment_index = 1
        runner.current_decision = lambda: retry
        return SimpleNamespace(status="ready_to_record_pending_retry",
                               preflight=preflight)

    runner.prepare_observed_xy_retry = prepare_retry
    runner.assess_observed_xy_retry_arrival = lambda **kwargs: (
        calls.append(("arrival", kwargs["raw_bundle"])) or
        SimpleNamespace(preinsert_reached=True))
    runner.prepare_observed_xy_retry_axial_handoff = lambda **kwargs: (
        calls.append(("handoff", kwargs["arrival_report_path"])) or
        (tmp_path / "handoff.json"))
    runner.prepare_observed_insertion_label = lambda **kwargs: (
        calls.append(("label", kwargs["retry_arrival_report_path"])))
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(launch=SimpleNamespace(
            camera_ids=("cam_a", "cam_b"), intrinsics_full={},
            extrinsics_full={})))
    monkeypatch.setattr(module, "KeyPoseObservation", object)
    monkeypatch.setattr(
        module, "execute_bound_observed_retry_transfer",
        lambda **kwargs: calls.append(("transfer", kwargs["preflight"])))
    metric = tmp_path / "metric.json"
    metric.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "execute_bound_guarded_insertion",
                        lambda **kwargs: (calls.append((
                            "guarded", kwargs["observed_xy_retry"])) or
                            {"metric_record": {"path": str(metric)}}))
    handler = module.LiveObservedXYRetryHandlers(
        live=live, planner=object(), backend=object(),
        read_state=lambda: state, state_for_capture=lambda _: state,
        withdraw_guarded=lambda *_: (state, withdrawal),
        capture_withdrawn_key=lambda *_: module.HeldKeyRetryCapture(
            object(), tmp_path / "held_evidence", raw, state),
        capture_arrival=lambda *_: raw, capture_final=lambda *_: raw,
        transfer_adapter=object(), insertion_adapter=object(),
        motion_interlock=lambda: True, retry_limits=object(),
        transfer_limits=object(), insertion_limits=object(),
        contact_limits=object(), transfer_commissioning_path=tmp_path,
        insertion_watchdog_path=tmp_path,
        insertion_contact_path=tmp_path, arrival_parameters={},
        handoff_parameters={}, insertion_parameters={}, final_parameters={},
        enable_robot_motion=True, decision_time=lambda: 102.2)
    handler._capture = lambda _runner, _decision, _callback, phase: (
        tmp_path / phase, state)
    handler.assess_and_shift(runner, first)
    handler.insert_and_observe(runner, retry)
    assert [name for name, _ in calls] == [
        "prepare", "transfer", "arrival", "handoff", "guarded", "label"]
    assert handler._pending is None
