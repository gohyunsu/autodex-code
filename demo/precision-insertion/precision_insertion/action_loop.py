"""Drive the existing SessionRunner decisions without reimplementing policy.

Each handler owns exactly one *complete* decision boundary: it calls the
existing capture, planning, execution and observation functions and updates
the runner before returning. This loop never infers a label from a plan and
contains no robot or camera implementation. A missing handler leaves the
current decision visible instead of silently skipping a physical phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Callable, Mapping, Protocol

from .session_policy import SessionDecision
from .timing_profile import timed


class DecisionRunner(Protocol):
    def current_decision(self) -> SessionDecision: ...


ActionHandler = Callable[[DecisionRunner, SessionDecision], None]
_TERMINAL_STATUSES = {
    "stop_for_review": "terminal",
    "session_success": "success",
}


@dataclass(frozen=True)
class ActionLoopResult:
    status: str
    decision: SessionDecision
    visited: tuple[SessionDecision, ...]

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_action_loop_result_v1",
            "status": self.status,
            "decision": self.decision.to_record(),
            "visited": [item.to_record() for item in self.visited],
            "scope": "decision_dispatch_from_observed_labels_not_robot_motion_authorization",
            "robot_ready": False,
        }


def run_session_actions(
    runner: DecisionRunner, *, handlers: Mapping[str, ActionHandler],
    max_steps: int | None = None,
) -> ActionLoopResult:
    """Dispatch until the session is terminal, unhandled, or step-limited.

    Handlers are intentionally supplied by the AutoDex-PC integration. For
    example, ``capture_fresh_key`` calls ``prepare_next_live_key`` using the
    already-open camera session, whereas ``await_lift_observation`` must
    complete the existing pickup/lift/checkpoint boundary. The runner alone
    decides the next action after each handler; this module neither chooses
    grasp candidates nor manufactures an outcome to advance the loop.
    """
    if not callable(getattr(runner, "current_decision", None)):
        raise TypeError("action loop needs a SessionRunner decision provider")
    if not isinstance(handlers, Mapping) or any(
            not isinstance(name, str) or not name or not callable(handler)
            for name, handler in handlers.items()):
        raise TypeError("action handlers need nonempty action names and callables")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    visited: list[SessionDecision] = []
    for _ in (count() if max_steps is None else range(max_steps)):
        decision = runner.current_decision()
        if not isinstance(decision, SessionDecision):
            raise TypeError("runner returned no SessionDecision")
        visited.append(decision)
        if decision.action in _TERMINAL_STATUSES:
            return ActionLoopResult(
                _TERMINAL_STATUSES[decision.action], decision, tuple(visited))
        handler = handlers.get(decision.action)
        if handler is None:
            return ActionLoopResult("needs_handler", decision, tuple(visited))
        with timed(f"action.{decision.action}", phase="trial",
                   kind="action", candidate_id=decision.candidate_id):
            handler(runner, decision)
        after = runner.current_decision()
        if not isinstance(after, SessionDecision):
            raise TypeError("runner returned no SessionDecision")
        if after == decision:
            visited.append(after)
            return ActionLoopResult("decision_unchanged", after,
                                    tuple(visited))
    decision = runner.current_decision()
    if not isinstance(decision, SessionDecision):
        raise TypeError("runner returned no SessionDecision")
    visited.append(decision)
    status = _TERMINAL_STATUSES.get(
        decision.action, "step_budget_exhausted")
    return ActionLoopResult(status, decision, tuple(visited))
