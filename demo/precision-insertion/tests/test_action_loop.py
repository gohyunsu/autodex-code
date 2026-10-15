"""The live dispatcher follows SessionRunner decisions, never guesses labels."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.action_loop import run_session_actions  # noqa: E402
from precision_insertion.live_orchestration import (  # noqa: E402
    compose_live_handlers, run_live_precision_session,
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


def test_repeated_grasp_failure_restores_before_new_capture_and_candidate():
    runner = Runner([
        _decision("capture_fresh_key"),
        _decision("execution_gate_required", "table/0/1"),
        _decision("await_lift_observation", "table/0/1"),
        _decision("recover_key_then_reobserve", "table/0/1"),
        _decision("reobserve_key_and_preflight", "table/0/1"),
        _decision("execution_gate_required", "table/0/2"),
        _decision("session_success", "table/0/2"),
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
            "await_lift_observation", "recover_key_then_reobserve",
            "reobserve_key_and_preflight")},
        max_steps=10)
    assert result.status == "success"
    assert result.decision.action == "session_success"
    assert called == [
        ("capture_fresh_key", None),
        ("execution_gate_required", "table/0/1"),
        ("await_lift_observation", "table/0/1"),
        ("recover_key_then_reobserve", "table/0/1"),
        ("reobserve_key_and_preflight", "table/0/1"),
        ("execution_gate_required", "table/0/2"),
    ]
    assert result.to_record()["robot_ready"] is False


def test_insertion_failure_saves_next_trial_xy_then_returns_key():
    runner = Runner([
        _decision("await_guarded_insertion_and_observation", "table/0/1"),
        _decision("guarded_withdrawal_then_xy_assessment", "table/0/1"),
        _decision("recover_held_key_then_reobserve", "table/0/1"),
        _decision("reobserve_key_and_preflight", "table/0/1"),
        _decision("execution_gate_required", "table/0/1"),
        _decision("session_success", "table/0/1"),
    ])
    result = run_session_actions(runner, handlers={
        "await_guarded_insertion_and_observation": lambda row, _: row.advance(),
    })
    assert result.status == "needs_handler"
    assert result.decision.action == "guarded_withdrawal_then_xy_assessment"
    assert runner.index == 1  # no XY command or insertion label was fabricated

    result = run_session_actions(runner, handlers={
        "guarded_withdrawal_then_xy_assessment": lambda row, _: row.advance(),
        "recover_held_key_then_reobserve": lambda row, _: row.advance(),
        "reobserve_key_and_preflight": lambda row, _: row.advance(),
        "execution_gate_required": lambda row, _: row.advance(),
    })
    assert result.status == "success"
    assert [step.action for step in result.visited] == [
        "guarded_withdrawal_then_xy_assessment",
        "recover_held_key_then_reobserve",
        "reobserve_key_and_preflight",
        "execution_gate_required",
        "session_success",
    ]


def test_success_stops_before_any_reset_or_next_trial_handler():
    runner = Runner([
        _decision("await_guarded_insertion_and_observation", "table/0/1"),
        _decision("session_success", "table/0/1"),
    ])
    called = []

    def observe(current, _decision):
        called.append("observe_insertion")
        current.advance()

    def must_not_run(_current, _decision):
        called.append("unexpected_motion")
        raise AssertionError("success must not run reset or another trial")

    result = run_session_actions(runner, handlers={
        "await_guarded_insertion_and_observation": observe,
        "session_success": must_not_run,
        "reobserve_key_and_preflight": must_not_run,
        "preflight_repose": must_not_run,
    })
    assert result.status == "success"
    assert result.decision.action == "session_success"
    assert called == ["observe_insertion"]


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


def test_plan_only_continues_candidate_search_and_stops_before_pickup():
    runner = Runner([_decision("capture_fresh_key"),
                     _decision("continue_candidate_preflight"),
                     _decision("execution_gate_required", "table/0/1")])
    called = []

    def plan(current, decision):
        called.append(decision.action)
        current.advance()

    result = run_session_actions(runner, handlers={
        "capture_fresh_key": plan,
        "continue_candidate_preflight": plan,
        "execution_gate_required": lambda *_: pytest.fail("pickup started"),
    }, stop_before_actions={"execution_gate_required": "planning_ready"})
    assert result.status == "planning_ready"
    assert result.decision.candidate_id == "table/0/1"
    assert called == ["capture_fresh_key", "continue_candidate_preflight"]


def test_plan_only_stops_before_repose_when_pickup_planning_fails():
    runner = Runner([_decision("capture_fresh_key"),
                     _decision("continue_candidate_preflight"),
                     _decision("preflight_repose")])
    result = run_session_actions(runner, handlers={
        "capture_fresh_key": lambda row, _: row.advance(),
        "continue_candidate_preflight": lambda row, _: row.advance(),
        "preflight_repose": lambda *_: pytest.fail("repose planning started"),
    }, stop_before_actions={
        "execution_gate_required": "planning_ready",
        "preflight_repose": "planning_no_candidate",
    })
    assert result.status == "planning_no_candidate"
    assert result.decision.action == "preflight_repose"


def test_default_loop_does_not_truncate_repeated_physical_attempts():
    runner = Runner([_decision("capture_fresh_key", str(i))
                     for i in range(125)] +
                    [_decision("session_success")])
    result = run_session_actions(runner, handlers={
        "capture_fresh_key": lambda row, _: row.advance(),
    })
    assert result.status == "success"
    assert len(result.visited) == 126


def test_complete_live_handler_composition_rejects_missing_or_duplicate_stages():
    from precision_insertion.live_orchestration import _LIVE_ACTIONS

    live = object()

    class Component:
        def __init__(self, names):
            self.live = live
            self.names = names

        def handlers(self):
            return {name: lambda *_: None for name in self.names}

    with pytest.raises(ValueError, match="unconfigured actions"):
        compose_live_handlers(live, Component({"capture_fresh_key"}))
    with pytest.raises(ValueError, match="duplicate live actions"):
        compose_live_handlers(
            live, Component(_LIVE_ACTIONS), Component({"capture_fresh_key"}))
    assert set(compose_live_handlers(live, Component(_LIVE_ACTIONS))) == _LIVE_ACTIONS


def test_real_cylinder_stage_handlers_cover_exactly_the_active_live_actions():
    """Catch stale mandatory actions that no assembled stage can execute."""
    from precision_insertion import runtime_assembly
    from precision_insertion.live_nominal_relation_handlers import (
        LiveNominalRelationHandlers,
    )
    from precision_insertion.live_orchestration import _LIVE_ACTIONS

    live = object()
    classes = dict(runtime_assembly.STAGE_CLASSES)
    classes["held_relation"] = LiveNominalRelationHandlers
    components = []
    for cls in classes.values():
        component = cls.__new__(cls)
        component.live = live
        components.append(component)
    handlers = compose_live_handlers(live, *components)
    assert set(handlers) == _LIVE_ACTIONS
    assert "await_retry_execution_and_observation" not in handlers


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
            clear_view=object(), build_handlers=lambda _: {}, max_steps=0)
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
        clear_view=object(),
        build_handlers=build_handlers)
    assert result.status == "terminal"
    assert calls == ["camera_open", "capture_fresh_key",
                     "reobserve_key_and_preflight", "camera_closed"]


def test_initial_key_is_placed_after_socket_capture_before_first_key_image(
        monkeypatch):
    from precision_insertion import live_orchestration

    calls = []
    runner = Runner([
        _decision("capture_fresh_key"), _decision("stop_for_review")])
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))

    @contextmanager
    def opened(**_kwargs):
        calls.append("socket_captured")
        try:
            yield live
        finally:
            calls.append("camera_closed")

    monkeypatch.setattr(live_orchestration, "open_live_precision_session", opened)

    def build(_live):
        def capture(current, _decision):
            calls.append("key_captured")
            current.advance()
        return {"capture_fresh_key": capture}

    result = run_live_precision_session(
        launch_config_path=Path("/unused/launch.json"),
        endpoint_catalog_path=Path("/unused/catalog.json"),
        run_output_dir=Path("/unused/run"), max_xy_retries=None,
        clear_view=object(), build_handlers=build,
        confirm_initial_key_placement=lambda: (
            calls.append("operator_enter") or True))
    assert result.status == "terminal"
    assert calls == ["socket_captured", "operator_enter", "key_captured",
                     "camera_closed"]

    calls.clear()
    live.opened.runner = Runner([
        _decision("capture_fresh_key"), _decision("stop_for_review")])
    with pytest.raises(RuntimeError, match="initial key placement"):
        run_live_precision_session(
            launch_config_path=Path("/unused/launch.json"),
            endpoint_catalog_path=Path("/unused/catalog.json"),
            run_output_dir=Path("/unused/run"), max_xy_retries=None,
            clear_view=object(), build_handlers=build,
            confirm_initial_key_placement=lambda: (
                calls.append("operator_quit") or False))
    assert calls == ["socket_captured", "operator_quit", "camera_closed"]


def test_live_wrapper_closes_cameras_immediately_on_success(monkeypatch):
    from precision_insertion import live_orchestration

    calls = []
    runner = Runner([
        _decision("await_guarded_insertion_and_observation", "table/0/1"),
        _decision("session_success", "table/0/1"),
    ])
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))

    @contextmanager
    def opened(**_kwargs):
        calls.append("camera_open")
        try:
            yield live
        finally:
            calls.append("camera_closed")

    monkeypatch.setattr(live_orchestration, "open_live_precision_session", opened)

    def observe(current, _decision):
        calls.append("observe_insertion")
        current.advance()

    result = run_live_precision_session(
        launch_config_path=Path("/unused/launch.json"),
        endpoint_catalog_path=Path("/unused/catalog.json"),
        run_output_dir=Path("/unused/run"), max_xy_retries=2,
        clear_view=object(),
        build_handlers=lambda _live: {
            "await_guarded_insertion_and_observation": observe,
        },
        max_steps=1)
    assert result.status == "success"
    assert result.decision.action == "session_success"
    assert calls == ["camera_open", "observe_insertion", "camera_closed"]
