"""Separate grasp validation from downstream task evaluation.

Candidate ``result.json`` files retain their historical meaning: ``success``
there records whether the grasp/lift succeeded.  Episode result files use this
module to additionally record whether the downstream manipulation task
succeeded.  The default :class:`LiftTask` preserves today's behavior by making
the lift itself the task.

This interface is intentionally about outcome semantics.  The motion hook for
precision insertion will be added with the insertion controller, without
changing the grasp-candidate contract introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskContext:
    """Stable trial metadata available to a task evaluator."""

    object_name: str
    arm: str
    hand: str
    trial_dir: str
    scene_info: Optional[tuple[str, ...]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    """Serializable result of a manipulation task."""

    task_name: str
    success: Optional[bool]
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.success is True:
            return "success"
        if self.success is False:
            return "failure"
        return "unjudgeable"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.task_name,
            "success": self.success,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@runtime_checkable
class TaskInterface(Protocol):
    """Evaluate downstream task success independently of grasp success."""

    name: str

    def evaluate(
        self,
        *,
        context: TaskContext,
        grasp_success: Optional[bool],
        grasp_evidence: Optional[Mapping[str, Any]] = None,
    ) -> TaskOutcome:
        """Return the task outcome for one physically attempted grasp."""


class LiftTask:
    """Backward-compatible task whose goal is a successful grasp and lift."""

    name = "grasp_lift"

    def evaluate(
        self,
        *,
        context: TaskContext,
        grasp_success: Optional[bool],
        grasp_evidence: Optional[Mapping[str, Any]] = None,
    ) -> TaskOutcome:
        del context
        reason = (
            "lift_succeeded" if grasp_success is True else
            "lift_failed" if grasp_success is False else
            "lift_unjudgeable"
        )
        return TaskOutcome(
            task_name=self.name,
            success=grasp_success,
            reason=reason,
            evidence=dict(grasp_evidence or {}),
        )


def attach_task_outcome(
    record: Mapping[str, Any],
    *,
    grasp_success: Optional[bool],
    task_outcome: TaskOutcome,
) -> dict[str, Any]:
    """Attach explicit grasp/task fields to an episode result.

    The top-level ``success`` key remains as a compatibility alias, but now
    unambiguously means episode/task success.  Code updating grasp candidates
    or grasp coverage must use ``grasp_success`` instead.
    """

    result = dict(record)
    result["grasp_success"] = grasp_success
    result["task_success"] = task_outcome.success
    result["task"] = task_outcome.to_record()
    result["success"] = task_outcome.success
    return result
