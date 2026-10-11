"""The live dispatcher follows SessionRunner decisions, never guesses labels."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.action_loop import run_session_actions  # noqa: E402
from precision_insertion.live_orchestration import (  # noqa: E402
    run_live_precision_session,
)
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def _decision(action, candidate=None):
    return SessionDecision(action, "test", candidate)


class Runner:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def current_decision(self):
        return self.actions[self.index]

    def advance(self):
        self.index += 1


def test_repeated_grasp_failure_uses_new_capture_and_next_candidate():
    runner = Runner([
        _decision("capture_fresh_key"),
        _decision("execution_gate_required", "table/0/1"),
        _decision("await_lift_observation", "table/0/1"),
        _decision("reobserve_key_and_preflight", "table/0/1"),
        _decision("execution_gate_required", "table/0/2"),
        _decision("hold_for_supervised_completion", "table/0/2"),
    ])
    called = []

    def advance(name):
        def handler(current, decision):
            assert current is runner
            called.append((name, decision.candidate_id))
            current.advance()
        return handler

    result = run_session_actions(runner, handlers={
        name: advance(name) for name in (
            "capture_fresh_key", "execution_gate_required",
            "await_lift_observation", "reobserve_key_and_preflight")},
        max_steps=10)
    assert result.status == "terminal"
    assert result.decision.action == "hold_for_supervised_completion"
    assert called == [
        ("capture_fresh_key", None),
        ("execution_gate_required", "table/0/1"),
        ("await_lift_observation", "table/0/1"),
        ("reobserve_key_and_preflight", "table/0/1"),
        ("execution_gate_required", "table/0/2"),
    ]
    assert result.to_record()["robot_ready"] is False


def test_insertion_failure_routes_to_xy_retry_without_implicit_execution():
    runner = Runner([
        _decision("await_guarded_insertion_and_observation", "table/0/1"),
        _decision("guarded_withdrawal_then_xy_assessment", "table/0/1"),
        _decision("await_retry_execution_and_observation", "table/0/1"),
        _decision("hold_for_supervised_completion", "table/0/1"),
    ])
    result = run_session_actions(runner, handlers={
        "await_guarded_insertion_and_observation": lambda row, _: row.advance(),
    })
    assert result.status == "needs_handler"
    assert result.decision.action == "guarded_withdrawal_then_xy_assessment"
    assert runner.index == 1  # no XY command or insertion label was fabricated

    result = run_session_actions(runner, handlers={
        "guarded_withdrawal_then_xy_assessment": lambda row, _: row.advance(),
        "await_retry_execution_and_observation": lambda row, _: row.advance(),
    })
    assert result.status == "terminal"
    assert [step.action for step in result.visited] == [
        "guarded_withdrawal_then_xy_assessment",
        "await_retry_execution_and_observation",
        "hold_for_supervised_completion",
    ]


def test_unchanged_decision_and_handler_errors_do_not_repeat_actions():
    runner = Runner([_decision("continue_candidate_preflight")])
    result = run_session_actions(
        runner, handlers={"continue_candidate_preflight": lambda *_: None},
        max_steps=2)
    assert result.status == "decision_unchanged"
    assert result.decision.action == "continue_candidate_preflight"
    with pytest.raises(RuntimeError, match="camera lost"):
        run_session_actions(runner, handlers={
            "continue_candidate_preflight": lambda *_: (
                _ for _ in ()).throw(RuntimeError("camera lost"))})


def test_step_budget_reports_nonterminal_decision():
    runner = Runner([_decision("capture_fresh_key"),
                     _decision("reobserve_key_and_preflight")])
    result = run_session_actions(runner, handlers={
        "capture_fresh_key": lambda row, _: row.advance(),
    }, max_steps=1)
    assert result.status == "step_budget_exhausted"
    assert result.decision.action == "reobserve_key_and_preflight"


def test_live_wrapper_keeps_cameras_open_for_repeated_actions(monkeypatch):
    from precision_insertion import live_orchestration

    calls = []
    runner = Runner([
        _decision("capture_fresh_key"),
        _decision("reobserve_key_and_preflight"),
        _decision("stop_for_review"),
    ])
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))

    @contextmanager
    def opened(**kwargs):
        calls.append("camera_open")
        try:
            yield live
        finally:
            calls.append("camera_closed")

    monkeypatch.setattr(live_orchestration, "open_live_precision_session", opened)

    with pytest.raises(ValueError, match="max_steps"):
        run_live_precision_session(
            launch_config_path=Path("/unused/launch.json"),
            endpoint_catalog_path=Path("/unused/catalog.json"),
            run_output_dir=Path("/unused/run"), max_xy_retries=2,
            build_handlers=lambda _: {}, max_steps=0)
    assert calls == []  # invalid loop configuration does not start cameras

    def build_handlers(received):
        assert received is live

        def advance(current, decision):
            assert current is runner
            calls.append(decision.action)
            current.advance()

        return {"capture_fresh_key": advance,
                "reobserve_key_and_preflight": advance}

    result = run_live_precision_session(
        launch_config_path=Path("/unused/launch.json"),
        endpoint_catalog_path=Path("/unused/catalog.json"),
        run_output_dir=Path("/unused/run"), max_xy_retries=2,
        build_handlers=build_handlers)
    assert result.status == "terminal"
    assert calls == ["camera_open", "capture_fresh_key",
                     "reobserve_key_and_preflight", "camera_closed"]
