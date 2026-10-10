"""A retry needs new camera bytes around its own guarded contact stroke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from precision_insertion.retry_guarded_execution import (  # noqa: E402
    execute_bound_retry_guarded_insertion,
)
from precision_insertion.records import AttemptRecord, STAGES  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.session_policy import decide_after_attempt  # noqa: E402
from precision_insertion.retry_insertion_checkpoint import (  # noqa: E402
    assess_retry_insertion_checkpoint, verify_retry_insertion_checkpoint,
    write_retry_insertion_checkpoint,
)
from test_retry_guarded_execution import _setup  # noqa: E402


class _Backend:
    model = "test-retry-vlm"

    def __init__(self, visual_class: str):
        self.visual_class = visual_class
        self.calls = 0

    def infer(self, _images, _prompt):
        self.calls += 1
        return json.dumps({
            "visual_class": self.visual_class,
            "evidence_views": ["a", "b"],
            "evidence": "synthetic test observation",
        })


def _capture(tmp_path, *, source: Path, phase: str,
             request_id: int, first_id: int, timestamp_s: float):
    images = {camera: cv2.imread(str(source / "images" / f"{camera}.png"))
              for camera in ("a", "b")}
    ids = {camera: first_id + i for i, camera in enumerate(images)}
    frames = {camera: {
        "frame_id": ids[camera],
        "image_sha256": image_sha256(image),
        "timestamp_s": timestamp_s + .002 * i,
        "max_error_s": .001,
        "timestamp_method": "hardware_exposure",
        "clock_domain": "unix_utc",
    } for i, (camera, image) in enumerate(images.items())}
    raw = RawCameraCapture(
        f"retry_{phase}_{request_id}", request_id, images, ids,
        {"request_id": request_id,
         "source": "camera_acquisition", "frames": frames})
    return write_raw_camera_capture(
        raw, tmp_path / f"{phase}_{request_id}", phase=phase)


def _case(tmp_path, monkeypatch, *, visual_class="normal_appearance"):
    execution = _setup(tmp_path, monkeypatch)
    execute_bound_retry_guarded_insertion(
        **execution, enable_robot_motion=True)
    arrival = execution["arrival"]
    before = _capture(
        tmp_path, source=arrival.capture_dir, phase="preinsert",
        request_id=20, first_id=61, timestamp_s=101.27)
    after = _capture(
        tmp_path, source=arrival.capture_dir, phase="final_or_abort",
        request_id=21, first_id=71, timestamp_s=101.54)
    backend = _Backend(visual_class)
    call = dict(
        execution_log_path=(
            tmp_path / "retry_guarded_executions/000/execution.json"),
        handoff_report_path=execution["handoff_report_path"],
        expected=execution["expected"],
        previous=execution["previous"], arrival=arrival,
        checkpoint=execution["checkpoint"],
        shift_plan=execution["shift_plan"],
        mode=execution["runner"].mode,
        shared_root=execution["runner"].shared_root,
        calibration=execution["runner"].calibration,
        preinsert_bundle=before, final_bundle=after,
        decision_timestamp_s=101.56, backend=backend,
        max_phase_skew_s=.01, max_preinsert_age_s=.1,
        max_final_observation_gap_s=.1)
    return call, backend, execution["runner"]


def _verify_kwargs(call):
    return {name: call[name] for name in (
        "expected", "previous", "arrival", "checkpoint",
        "shift_plan", "mode", "shared_root", "calibration")}


def _attach_pending_attempt(tmp_path, call, runner):
    pending = json.loads((tmp_path / "state_004.json").read_text())
    runner._attempt = AttemptRecord(
        attempt_id=pending["attempt_id"], mode=call["mode"],
        session_calibration_sha256=pending[
            "session_calibration_sha256"],
        candidate_id=pending["candidate_id"],
        tabletop_pose_stem=pending["tabletop_pose_stem"],
        xy_offset_socket_m=tuple(pending["xy_offset_socket_m"]),
        started_at_s=99.,
        labels={name: pending[name] for name in STAGES},
        events=pending["events"], failure_code=pending["failure_code"],
        _pending_retry=True)
    runner._attempted = set()
    runner._insertion_assessment_index = 0


def _record_retry_label(call, runner):
    return runner.prepare_observed_retry_insertion_label(
        replan=call["expected"], previous=call["previous"],
        arrival=call["arrival"], checkpoint=call["checkpoint"],
        shift_plan=call["shift_plan"],
        handoff_report_path=call["handoff_report_path"],
        execution_log_path=call["execution_log_path"],
        preinsert_bundle=call["preinsert_bundle"],
        final_bundle=call["final_bundle"], backend=call["backend"],
        decision_timestamp_s=call["decision_timestamp_s"],
        max_phase_skew_s=call["max_phase_skew_s"],
        max_preinsert_age_s=call["max_preinsert_age_s"],
        max_final_observation_gap_s=call["max_final_observation_gap_s"])


def test_retry_observation_replays_new_frames_but_cannot_claim_success(
        tmp_path, monkeypatch):
    call, backend, _ = _case(tmp_path, monkeypatch)
    observed = assess_retry_insertion_checkpoint(**call)
    assert backend.calls == 1
    assert observed["effective_vlm_class"] == "normal_appearance"
    assert observed["outcome"]["insertion_success"] is None
    assert observed["key_depth_admissibility"]["status"] == "not_admitted"
    path = write_retry_insertion_checkpoint(
        observed, tmp_path / "retry_assessment")
    assert verify_retry_insertion_checkpoint(
        path, **_verify_kwargs(call)) == observed


def test_retry_observed_rim_jam_is_failure_without_invented_depth(
        tmp_path, monkeypatch):
    call, _, _ = _case(tmp_path, monkeypatch, visual_class="rim_jam")
    observed = assess_retry_insertion_checkpoint(**call)
    assert observed["outcome"]["insertion_success"] is False
    assert observed["evidence"]["key_depth_interval_m"] is None


def test_retry_rejects_reused_arrival_capture_before_vlm(
        tmp_path, monkeypatch):
    call, backend, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="new pre-contact camera capture"):
        assess_retry_insertion_checkpoint(**{
            **call, "preinsert_bundle": call["arrival"].capture_dir})
    assert backend.calls == 0


def test_retry_rejects_reused_frame_id_before_vlm(tmp_path, monkeypatch):
    call, backend, _ = _case(tmp_path, monkeypatch)
    before = call["preinsert_bundle"]
    manifest = before / "manifest.json"
    report = json.loads(manifest.read_text())
    report["frame_evidence"]["a"]["frame_id"] = 51
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        assess_retry_insertion_checkpoint(**call)
    assert backend.calls == 0


def test_retry_replay_rejects_changed_vlm_answer_or_image(
        tmp_path, monkeypatch):
    call, _, _ = _case(tmp_path, monkeypatch)
    report = assess_retry_insertion_checkpoint(**call)
    path = write_retry_insertion_checkpoint(
        report, tmp_path / "retry_assessment")
    changed = json.loads(path.read_text())
    changed["visual"]["raw_answer"] = "not json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from replay"):
        verify_retry_insertion_checkpoint(path, **_verify_kwargs(call))
    path.write_text(json.dumps(report))
    image = call["final_bundle"] / "images" / "a.png"
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="PNG changed"):
        verify_retry_insertion_checkpoint(path, **_verify_kwargs(call))


def test_session_runner_records_second_verdict_without_false_success(
        tmp_path, monkeypatch):
    call, _, runner = _case(tmp_path, monkeypatch)
    _attach_pending_attempt(tmp_path, call, runner)
    assert isinstance(runner, SessionRunner)
    with pytest.raises(ValueError, match="XY retry must use"):
        runner.prepare_observed_insertion_label(
            preinsert_bundle=call["preinsert_bundle"],
            final_bundle=call["final_bundle"],
            metric_record_path=tmp_path / "retry_metric.json",
            backend=call["backend"],
            decision_timestamp_s=call["decision_timestamp_s"],
            max_phase_skew_s=call["max_phase_skew_s"],
            max_preinsert_age_s=call["max_preinsert_age_s"],
            max_final_observation_gap_s=call[
                "max_final_observation_gap_s"])
    observed = _record_retry_label(call, runner)
    assert observed["outcome"]["insertion_success"] is None
    assert runner._attempt.labels["insertion_success"] is None
    assert runner._attempt.events[-1]["stage"] == "insertion_success"
    assert (tmp_path / "state_005.json").is_file()
    assert (tmp_path / "insertion_assessments/000/report.json").is_file()


def test_session_runner_rim_jam_reenters_retry_policy_with_budget(
        tmp_path, monkeypatch):
    call, _, runner = _case(tmp_path, monkeypatch, visual_class="rim_jam")
    _attach_pending_attempt(tmp_path, call, runner)
    observed = _record_retry_label(call, runner)
    assert observed["outcome"]["insertion_success"] is False
    assert runner._attempt.labels["insertion_success"] is False
    decision = decide_after_attempt(runner._attempt, max_xy_retries=2)
    assert decision.action == "guarded_withdrawal_then_xy_assessment"
