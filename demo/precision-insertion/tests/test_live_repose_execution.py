"""Directed reset execution is followed by same-session landing capture."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_repose_execution  # noqa: E402
from precision_insertion.live_key_trial import ReturnedKeyCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.session_sha256 = "a" * 64
        self._repose_report_path = root / "repose_plan.json"
        self._repose_report_path.write_text("{}", encoding="utf-8")
        self._repose_preflight = SimpleNamespace(selected_seed={"seed_id": "r1"})
        self.decision = SessionDecision(
            "repose_execution_gate_required", "nominal_reset", None)
        self.active_attempt = None
        self.events = []

    def current_decision(self):
        return self.decision

    def begin_repose_attempt(self, **kwargs):
        self.events.append(("begin", kwargs))
        self.active_attempt = SimpleNamespace(
            attempt_id=kwargs["attempt_id"], started_at_s=kwargs["started_at_s"])
        self.decision = SessionDecision("await_repose_observation", "reset_executing",
                                        None)
        return self.active_attempt

    def observe_repose_landing(self, **kwargs):
        self.events.append(("landed", kwargs))
        self.decision = SessionDecision("reobserve_key_and_preflight",
                                        "repose_landing_observed", None)


def _handler(tmp_path):
    runner = Runner(tmp_path)
    log = tmp_path / "repose_execution.json"
    log.write_text("{}", encoding="utf-8")
    events = []
    clear_view = SimpleNamespace(
        before_key_capture=lambda: events.append("clear_view"))
    launch = SimpleNamespace(
        capture_root=tmp_path, camera_ids={"front", "side"},
        intrinsics_full={"front": object(), "side": object()},
        extrinsics_full={"front": object(), "side": object()},
        image_hw=(480, 640), view_limits=object())
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner), clear_view=clear_view,
        cameras=SimpleNamespace(
            launch=launch, init_orchestrator=object(),
            acquisition_metadata_for_request=lambda _request: {}))

    def execute(_runner, _decision, plan):
        assert plan is runner._repose_preflight
        events.append("execute_repose")
        return log

    bound = live_repose_execution.LiveReposeExecutionHandlers(
        live=live, execute_repose=execute,
        state_for_observation=lambda _observation: object(),
        key_prompt="white key excluding white socket",
        capture_parameters={}, landing_parameters={},
        attempt_id_factory=lambda: "repose_001",
        capture_id_factory=lambda: "repose_key_001",
        started_at_factory=lambda: 10., decision_time=lambda: 31.)
    return bound, runner, events, log


def test_repose_execution_then_clear_view_and_new_key_capture(
        tmp_path, monkeypatch):
    bound, runner, events, log = _handler(tmp_path)
    observation = SimpleNamespace(acquisition_interval_s=(29., 30.))

    def verify(**kwargs):
        events.append("verify_execution")
        assert kwargs["selected_seed"] == {"seed_id": "r1"}
        assert kwargs["path"] == log
        return {"release_completed_at_s": 20.}

    def capture(**kwargs):
        events.append("capture")
        assert kwargs["expected_action"] == "await_repose_observation"
        assert kwargs["runner"] is runner
        return ReturnedKeyCapture(observation, object(),
                                  kwargs["key_evidence_dir"])

    monkeypatch.setattr(live_repose_execution,
                        "verify_repose_execution_evidence", verify)
    monkeypatch.setattr(live_repose_execution,
                        "capture_returned_key_for_reset", capture)
    bound.begin(runner, runner.current_decision())
    bound.observe(runner, runner.current_decision())
    assert events == ["execute_repose", "verify_execution", "clear_view", "capture"]
    assert runner.events[0][0] == "begin"
    assert runner.events[1][0] == "landed"
    assert runner.events[1][1]["key_observation"] is observation
    assert runner.events[1][1]["release_completed_at_s"] == 20.
    assert runner.current_decision().action == "reobserve_key_and_preflight"


def test_no_landing_capture_without_this_attempt_execution(
        tmp_path, monkeypatch):
    bound, runner, events, _ = _handler(tmp_path)
    monkeypatch.setattr(live_repose_execution,
                        "capture_returned_key_for_reset",
                        lambda **_kwargs: pytest.fail("captured without execution"))
    with pytest.raises(ValueError, match="completed reset"):
        bound.observe(runner, SessionDecision("await_repose_observation",
                                              "reset_executing", None))
    assert events == []
