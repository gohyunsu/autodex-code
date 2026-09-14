"""Robot-neutral, canonical timing trace for one AutoDex trial.

The trace is intentionally the source of truth.  Convenience summaries are
derived only when a trial is saved, which keeps planner preflight, executor
replanning, and physical motion from being double counted.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterator, Optional


_PHASES = {
    "preparation", "perception", "planning", "execution", "post_execution",
    "recovery", "artifacts",
}
_KINDS = {"setup", "inference", "plan", "motion", "check", "decision", "io"}
_OUTCOMES = {"success", "failure", "skipped", "aborted"}


def _json_value(value: Any) -> Any:
    """Convert common numeric containers without importing numpy at module load."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class TimingSpan:
    """One elapsed operation inside a trial timing trace."""

    span_id: str
    parent_id: Optional[str]
    phase: str
    kind: str
    name: str
    start_s: float
    end_s: Optional[float] = None
    outcome: str = "success"
    attributes: dict[str, Any] = field(default_factory=dict)

    def close(self, end_s: float, *, outcome: str = "success", **attributes: Any) -> None:
        if self.end_s is not None:
            return
        self.end_s = max(float(end_s), self.start_s)
        self.outcome = outcome if outcome in _OUTCOMES else "failure"
        self.attributes.update(_json_value(attributes))

    def as_dict(self) -> dict[str, Any]:
        end_s = self.start_s if self.end_s is None else self.end_s
        return {
            "id": self.span_id,
            "parent_id": self.parent_id,
            "phase": self.phase,
            "kind": self.kind,
            "name": self.name,
            "start_s": round(self.start_s, 6),
            "end_s": round(end_s, 6),
            "duration_s": round(max(0.0, end_s - self.start_s), 6),
            "outcome": self.outcome,
            "attributes": _json_value(self.attributes),
        }


class TimingRecorder:
    """Append-only span recorder with deterministic parent/child ownership.

    ``clock`` is injectable so unit tests can assert exact snapshots without
    sleeping.  The recorder does not use wall-clock timestamps: all values are
    relative to one trial's monotonic origin.
    """

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._origin = float(clock())
        self._spans: dict[str, TimingSpan] = {}
        self._order: list[str] = []
        self._next_id = 0

    def now_s(self) -> float:
        return max(0.0, float(self._clock()) - self._origin)

    def begin(self, *, phase: str, kind: str, name: str,
              parent_id: Optional[str] = None, **attributes: Any) -> str:
        if phase not in _PHASES:
            raise ValueError(f"unknown timing phase: {phase}")
        if kind not in _KINDS:
            raise ValueError(f"unknown timing kind: {kind}")
        if parent_id is not None and parent_id not in self._spans:
            raise ValueError(f"unknown parent span: {parent_id}")
        span_id = f"s{self._next_id:04d}"
        self._next_id += 1
        self._spans[span_id] = TimingSpan(
            span_id=span_id, parent_id=parent_id, phase=phase, kind=kind,
            name=name, start_s=self.now_s(), attributes=_json_value(attributes))
        self._order.append(span_id)
        return span_id

    def end(self, span_id: str, *, outcome: str = "success", **attributes: Any) -> None:
        try:
            span = self._spans[span_id]
        except KeyError as exc:
            raise ValueError(f"unknown timing span: {span_id}") from exc
        span.close(self.now_s(), outcome=outcome, **attributes)

    @contextmanager
    def span(self, *, phase: str, kind: str, name: str,
             parent_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
        span_id = self.begin(phase=phase, kind=kind, name=name,
                             parent_id=parent_id, **attributes)
        try:
            yield span_id
        except BaseException as exc:
            self.end(span_id, outcome="failure", exception=type(exc).__name__)
            raise
        else:
            self.end(span_id)

    def record(self, *, phase: str, kind: str, name: str, duration_s: float,
               parent_id: Optional[str] = None, outcome: str = "success",
               **attributes: Any) -> str:
        """Add an already measured sequential interval.

        This is for third-party SDK calls that report their elapsed time after
        completion.  New core code should prefer :meth:`span`.
        """
        span_id = self.begin(phase=phase, kind=kind, name=name,
                             parent_id=parent_id, **attributes)
        span = self._spans[span_id]
        span.end_s = span.start_s + max(0.0, float(duration_s))
        span.outcome = outcome if outcome in _OUTCOMES else "failure"
        return span_id

    def record_interval(self, *, phase: str, kind: str, name: str,
                        start_s: float, end_s: Optional[float] = None,
                        parent_id: Optional[str] = None,
                        outcome: str = "success", **attributes: Any) -> str:
        """Add a span measured by a foreign clock at its real trace offset."""
        if phase not in _PHASES or kind not in _KINDS:
            raise ValueError("invalid timing phase or kind")
        if parent_id is not None and parent_id not in self._spans:
            raise ValueError(f"unknown parent span: {parent_id}")
        span_id = f"s{self._next_id:04d}"
        self._next_id += 1
        end = self.now_s() if end_s is None else float(end_s)
        self._spans[span_id] = TimingSpan(
            span_id=span_id, parent_id=parent_id, phase=phase, kind=kind,
            name=name, start_s=max(0.0, float(start_s)),
            end_s=max(max(0.0, float(start_s)), end),
            outcome=outcome if outcome in _OUTCOMES else "failure",
            attributes=_json_value(attributes),
        )
        self._order.append(span_id)
        return span_id

    def close_open_spans(self, *, outcome: str = "aborted") -> None:
        now = self.now_s()
        for span in self._spans.values():
            if span.end_s is None:
                span.close(now, outcome=outcome)

    @staticmethod
    def _union_duration(intervals: list[tuple[float, float]]) -> float:
        if not intervals:
            return 0.0
        total = 0.0
        start, end = sorted(intervals)[0]
        for cur_start, cur_end in sorted(intervals)[1:]:
            if cur_start <= end:
                end = max(end, cur_end)
            else:
                total += end - start
                start, end = cur_start, cur_end
        return total + end - start

    def as_dict(self, *, trial_total_s: Optional[float] = None) -> dict[str, Any]:
        self.close_open_spans()
        spans = [self._spans[span_id].as_dict() for span_id in self._order]
        total = self.now_s() if trial_total_s is None else max(0.0, float(trial_total_s))
        phases: dict[str, dict[str, float]] = {}
        for phase in sorted(_PHASES):
            intervals = [(item["start_s"], item["end_s"])
                         for item in spans if item["phase"] == phase]
            inclusive = self._union_duration(intervals)
            phases[phase] = {"inclusive_s": round(inclusive, 6),
                             "span_count": len(intervals)}
        classified = self._union_duration([
            (item["start_s"], item["end_s"]) for item in spans
        ])
        return {
            "schema_version": 1,
            "clock": {"name": "time.perf_counter", "origin": "trial_start"},
            "trial_total_s": round(total, 6),
            "spans": spans,
            "summary": {
                "phases": phases,
                "classified_s": round(classified, 6),
                "unattributed_s": round(max(0.0, total - classified), 6),
            },
        }
