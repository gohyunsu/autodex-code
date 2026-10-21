"""Failed insertion reuses saved preinsert frames without robot motion."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_retry_handlers  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


@pytest.mark.parametrize("status", [
    "abstain", "diagnostic_metric_xy_correction",
    "diagnostic_planar_xy_hypothesis", "diagnostic_exploratory_xy_step",
])
def test_live_grounded_advice_never_moves_xy_in_same_attempt(
        tmp_path, status):
    decision = SessionDecision(
        "assess_saved_preinsert_xy", "failed", "table/0/1")
    events = []

    class Runner:
        mode = SimpleNamespace(family="cylinder")
        xy_update_weight = 1.0
        active_attempt = SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/1",
            wrist_correction_socket_m=(0.0, 0.0))
        _postlift_report_path = tmp_path / "postlift.json"
        _attempt_dir = tmp_path
        _retry_assessment_index = 0

        def current_decision(self):
            return decision

        def prepare_saved_preinsert_xy_diagnostic(self, **kwargs):
            events.append(("diagnostic", kwargs["backend"]))
            report = tmp_path / "xy_retry_assessments/000/report.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"status":"saved"}', encoding="utf-8")
            self._retry_assessment_index += 1
            return SimpleNamespace(
                status=status,
                alignment={"bounded_xy_increment_socket_m": [0.001, 0.0]})

        def record_xy_retry_unavailable(self, **kwargs):
            events.append(("unavailable", kwargs["reason"]))

        def record_next_trial_xy_advice(self, **kwargs):
            events.append(("next_trial", kwargs["diagnostic"].status))

    runner = Runner()
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))
    backend = object()
    params = dict(
        live=live, backend=backend,
        retry_limits=SimpleNamespace(max_total_offset_m=0.003),
        alignment_limits=object(),
        diagnostic_parameters={},
        enable_robot_motion=True, decision_time=lambda: 10.0)
    handler = live_retry_handlers.LiveGroundedRetryHandlers(**params)
    handler.assess_for_next_trial(runner, decision)
    assert events == [("diagnostic", backend), (
        ("unavailable", "vlm_abstain") if status == "abstain" else
        ("next_trial", status))]
    assert set(handler.handlers()) == {"assess_saved_preinsert_xy"}


def test_optional_xy_diagnostic_records_capture_model_outage(tmp_path):
    decision = SessionDecision(
        "assess_saved_preinsert_xy", "failed", "table/0/1")
    events = []

    class Runner:
        mode = SimpleNamespace(family="cylinder")
        active_attempt = SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/1")
        _postlift_report_path = tmp_path / "postlift.json"
        _attempt_dir = tmp_path
        _retry_assessment_index = 0

        def current_decision(self):
            return decision

        def prepare_saved_preinsert_xy_diagnostic(self, **_kwargs):
            raise RuntimeError(
                "capture-PC Qwen 24122734: OutOfMemoryError('CUDA out of memory')")

        def record_xy_retry_unavailable(self, **kwargs):
            events.append(kwargs)

    runner = Runner()
    handler = live_retry_handlers.LiveGroundedRetryHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        backend=object(), retry_limits=object(), alignment_limits=object(),
        diagnostic_parameters={}, decision_time=lambda: 10.0)
    handler.assess_for_next_trial(runner, decision)
    assert len(events) == 1
    assert events[0]["reason"] == "vlm_backend_unavailable"
    report = events[0]["retry_assessment_path"]
    assert '"reason": "vlm_backend_unavailable"' in report.read_text()
