"""Run-scoped timing for the live precision-insertion pipeline.

The trace is a sibling of the eventual run directory so startup failures and
operator waits remain visible even when no session ledger is opened. Timings
reuse PipelineTrace's monotonic event format; UTC is only for log correlation.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterator

from autodex.pipeline_trace import PipelineTrace


_TRACE: ContextVar[PipelineTrace | None] = ContextVar(
    "precision_insertion_timing_trace", default=None)
_PARENT: ContextVar[str | None] = ContextVar(
    "precision_insertion_timing_parent", default=None)


@dataclass
class RunProfile:
    trace: PipelineTrace
    outcome: str = "success"


def timing_directory(run_output_dir: Path) -> Path:
    """Return a crash-persistent sidecar without creating the run ledger."""
    target = Path(run_output_dir).expanduser().resolve()
    return target.with_name(target.name + ".timing")


def _active_timing_directory(run_output_dir: Path) -> Path:
    local_root = os.environ.get("PRECISION_TIMING_LOCAL_ROOT", "").strip()
    if not local_root:
        return timing_directory(run_output_dir)
    root = Path(local_root).expanduser().resolve()
    return root / timing_directory(run_output_dir).name


@contextmanager
def profile_run(
    run_output_dir: Path, *, process_started_ns: int | None = None,
) -> Iterator[RunProfile]:
    """Keep one trace open across startup, actions and cleanup.

    ``process_started_ns`` optionally records the untraced Python bootstrap
    before the profile was opened. It must come from ``time.perf_counter_ns``
    in the same process. This is timing metadata only; it neither adds a gate
    nor changes session sequencing.
    """
    profile_opened_ns = time.perf_counter_ns()
    if (process_started_ns is not None and
            (type(process_started_ns) is not int or
             process_started_ns <= 0 or
             process_started_ns > profile_opened_ns)):
        raise ValueError("process timing marker must precede profile startup")
    destination = _active_timing_directory(run_output_dir)
    trace = PipelineTrace(destination)
    profile = RunProfile(trace)
    token = _TRACE.set(trace)
    parent_token = _PARENT.set(None)
    try:
        yield profile
    except BaseException:
        profile.outcome = "failure"
        raise
    finally:
        profile_body_ended_ns = time.perf_counter_ns()
        _PARENT.reset(parent_token)
        _TRACE.reset(token)
        trace.close(outcome=profile.outcome)
        summary = summarize_events(destination / "events.jsonl")
        if process_started_ns is not None:
            pre_profile_s = (profile_opened_ns - process_started_ns) / 1e9
            profile_body_s = (profile_body_ended_ns - profile_opened_ns) / 1e9
            summary["python_pre_profile_s"] = round(pre_profile_s, 9)
            summary["profile_body_s"] = round(profile_body_s, 9)
            summary["python_process_to_profile_body_end_s"] = round(
                pre_profile_s + profile_body_s, 9)
        (destination / "timing_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        remote = timing_directory(run_output_dir)
        if destination != remote:
            # One bulk copy after the run keeps per-span flushes off NAS.
            # On an interrupted run, the local append-only trace remains.
            remote.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(destination, remote)


@contextmanager
def timed(name: str, *, phase: str, kind: str = "computation",
          **attributes: object) -> Iterator[None]:
    """Record one nested monotonic span; be a no-op outside a profiled run."""
    trace = _TRACE.get()
    if trace is None:
        yield
        return
    span_id = trace.begin(
        phase=phase, kind=kind, name=name, parent_id=_PARENT.get(),
        **attributes)
    token = _PARENT.set(span_id)
    try:
        yield
    except BaseException as exc:
        trace.end(span_id, outcome="failure", exception=repr(exc))
        raise
    else:
        trace.end(span_id)
    finally:
        _PARENT.reset(token)


def summarize_events(events_path: Path) -> dict:
    """Summarize completed spans without confusing work time with latency.

    ``inclusive_s``/``cumulative_s`` count every span and are useful for
    resource accounting. ``occupied_s`` is the union of monotonic intervals
    and therefore measures elapsed wall time even when camera, GPU and file
    workers overlap.  The additive phase/kind/attempt aggregates make a live
    run directly usable for experiment analysis without changing execution.
    """
    def occupied(intervals: list[tuple[float, float]]) -> float:
        total, end = 0.0, float("-inf")
        for start, stop in sorted(intervals):
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
        return total

    starts: dict[str, dict] = {}
    completed: list[dict] = []
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        span_id = row.get("span_id")
        if not span_id:
            continue
        if row.get("edge") == "start":
            starts[span_id] = row
        elif row.get("edge") == "end" and span_id in starts:
            begun = starts.pop(span_id)
            completed.append({
                "span_id": span_id,
                "parent_id": begun.get("parent_id"),
                "attempt_id": begun.get("attempt_id"),
                "name": row["name"], "phase": row["phase"],
                "kind": row["kind"], "outcome": row.get("outcome"),
                "attributes": begun.get("attributes", {}),
                "duration_s": row["duration_s"],
                "start_pipeline_s": begun["pipeline_time_s"],
                "end_pipeline_s": row["pipeline_time_s"],
            })
    children: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in completed:
        if row["parent_id"] is not None:
            children[row["parent_id"]].append((
                row["start_pipeline_s"], row["end_pipeline_s"]))
    exclusive_intervals: dict[str, list[tuple[float, float]]] = {}
    for row in completed:
        clipped = [
            (max(row["start_pipeline_s"], start),
             min(row["end_pipeline_s"], stop))
            for start, stop in children[row["span_id"]]
            if min(row["end_pipeline_s"], stop) >
            max(row["start_pipeline_s"], start)]
        covered = occupied(clipped)
        cumulative = sum(stop - start for start, stop in clipped)
        gaps = []
        cursor = row["start_pipeline_s"]
        for start, stop in sorted(clipped):
            start = max(start, cursor)
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, stop)
        if cursor < row["end_pipeline_s"]:
            gaps.append((cursor, row["end_pipeline_s"]))
        exclusive_intervals[row["span_id"]] = gaps
        row["child_occupied_s"] = round(covered, 9)
        row["child_cumulative_s"] = round(cumulative, 9)
        row["child_overlap_s"] = round(max(0.0, cumulative - covered), 9)
        row["self_s"] = round(max(0.0, row["duration_s"] - covered), 9)
    indexed = {row["span_id"]: row for row in completed}
    for row in completed:
        parent = row
        effective_attempt = row["attempt_id"]
        effective_candidate = row["attributes"].get("candidate_id")
        while parent is not None and (
                effective_attempt is None or effective_candidate is None):
            if effective_attempt is None:
                effective_attempt = parent["attempt_id"]
            if effective_candidate is None:
                effective_candidate = parent["attributes"].get("candidate_id")
            parent = indexed.get(parent["parent_id"])
        row["effective_attempt_id"] = effective_attempt
        row["effective_candidate_id"] = effective_candidate

    def aggregate(rows: list[dict], *, exclusive: bool = False) -> dict:
        intervals = ([interval for row in rows
                      for interval in exclusive_intervals[row["span_id"]]]
                     if exclusive else
                     [(row["start_pipeline_s"], row["end_pipeline_s"])
                      for row in rows])
        cumulative = sum(row["self_s"] if exclusive else row["duration_s"]
                         for row in rows)
        wall = occupied(intervals)
        return {
            "count": len(rows),
            "cumulative_s": round(cumulative, 9),
            "occupied_s": round(wall, 9),
            "overlap_s": round(max(0.0, cumulative - wall), 9),
            "self_s": round(sum(row["self_s"] for row in rows), 9),
            "failures": sum(row["outcome"] != "success" for row in rows),
        }

    def aggregate_by(field: str, *, exclusive: bool = False) -> dict[str, dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in completed:
            buckets[str(row[field])].append(row)
        return {key: aggregate(rows, exclusive=exclusive)
                for key, rows in sorted(buckets.items())}

    groups = aggregate_by("name")
    for group in groups.values():
        # Keep the established public name while adding the wall-time union
        # and overlap needed to interpret repeated/parallel operations.
        group["inclusive_s"] = group.pop("cumulative_s")

    attempt_rows: dict[str, list[dict]] = defaultdict(list)
    for row in completed:
        if row["effective_attempt_id"] is not None:
            attempt_rows[str(row["effective_attempt_id"])].append(row)
    attempts = {}
    for attempt_id, rows in sorted(attempt_rows.items()):
        candidates = sorted({str(row["effective_candidate_id"])
                             for row in rows
                             if row["effective_candidate_id"] is not None})
        phase_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            phase_rows[str(row["phase"])].append(row)
        attempts[attempt_id] = {
            **aggregate(rows),
            "candidate_ids": candidates,
            "start_pipeline_s": round(min(
                row["start_pipeline_s"] for row in rows), 9),
            "end_pipeline_s": round(max(
                row["end_pipeline_s"] for row in rows), 9),
            "phases": {phase: aggregate(items)
                       for phase, items in sorted(phase_rows.items())},
        }

    human_rows = [row for row in completed if row["kind"] == "human"]

    def descendants(ancestor_id: str, rows: list[dict]) -> list[dict]:
        """Return rows nested below one span without relying on list order."""
        selected = []
        for row in rows:
            parent_id = row["parent_id"]
            while parent_id is not None:
                if parent_id == ancestor_id:
                    selected.append(row)
                    break
                parent = indexed.get(parent_id)
                parent_id = None if parent is None else parent["parent_id"]
        return selected

    ordered_actions = []
    for row in sorted(completed, key=lambda item: item["start_pipeline_s"]):
        if not row["name"].startswith("action."):
            continue
        action_human = descendants(row["span_id"], human_rows)
        human_occupied = occupied([
            (max(row["start_pipeline_s"], item["start_pipeline_s"]),
             min(row["end_pipeline_s"], item["end_pipeline_s"]))
            for item in action_human
            if min(row["end_pipeline_s"], item["end_pipeline_s"]) >
            max(row["start_pipeline_s"], item["start_pipeline_s"])
        ])
        ordered_actions.append({
            "name": row["name"],
            "attempt_id": row["effective_attempt_id"],
            "candidate_id": row["effective_candidate_id"],
            "outcome": row["outcome"],
            "start_pipeline_s": row["start_pipeline_s"],
            "end_pipeline_s": row["end_pipeline_s"],
            "inclusive_s": row["duration_s"],
            "self_s": row["self_s"],
            "human_wait_s": round(human_occupied, 9),
            "automatic_elapsed_s": round(max(
                0.0, row["duration_s"] - human_occupied), 9),
        })

    if completed:
        envelope_start = min(row["start_pipeline_s"] for row in completed)
        envelope_end = max(row["end_pipeline_s"] for row in completed)
    else:
        envelope_start = envelope_end = 0.0
    inference = [row for row in completed if row["name"] == "vlm.infer"]
    inference_sum = sum(row["duration_s"] for row in inference)
    inference_wall = occupied([
        (row["start_pipeline_s"], row["end_pipeline_s"])
        for row in inference])
    human_sum = sum(row["duration_s"] for row in human_rows)
    human_wall = occupied([
        (row["start_pipeline_s"], row["end_pipeline_s"])
        for row in human_rows])
    envelope = max(0.0, envelope_end-envelope_start)
    return {
        "schema": "precision_insertion_timing_summary_v1",
        "clock": "time.perf_counter_ns",
        "vlm_inference_cumulative_s": round(inference_sum, 9),
        "vlm_inference_occupied_s": round(inference_wall, 9),
        "vlm_inference_overlap_s": round(max(0.0, inference_sum -
                                             inference_wall), 9),
        "span_envelope_start_s": round(envelope_start, 9),
        "span_envelope_end_s": round(envelope_end, 9),
        "span_envelope_s": round(envelope, 9),
        "human_wait_cumulative_s": round(human_sum, 9),
        "human_wait_occupied_s": round(human_wall, 9),
        "span_envelope_non_human_s": round(max(
            0.0, envelope - human_wall), 9),
        "unfinished_span_count": len(starts),
        "unfinished_spans": [row["name"] for row in starts.values()],
        "groups": groups,
        "phases": aggregate_by("phase"),
        "kinds": aggregate_by("kind", exclusive=True),
        "attempts": attempts,
        "ordered_actions": ordered_actions,
        "root_span_ids": [row["span_id"] for row in completed
                          if row["parent_id"] is None],
        "spans": sorted(completed, key=lambda row: row["start_pipeline_s"]),
    }
