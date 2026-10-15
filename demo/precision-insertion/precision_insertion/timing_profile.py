"""Run-scoped timing for the live demo, using stock AutoDex PipelineTrace.

The trace is a sibling of the eventual run directory so startup failures and
operator waits remain visible even when no session ledger is opened. Timings
use PipelineTrace's monotonic clock; UTC is only for log correlation.
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
def profile_run(run_output_dir: Path) -> Iterator[RunProfile]:
    """Keep a stock AutoDex trace open across startup, actions and cleanup."""
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
        _PARENT.reset(parent_token)
        _TRACE.reset(token)
        trace.close(outcome=profile.outcome)
        summary = summarize_events(destination / "events.jsonl")
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
    """Summarize completed spans, unioning overlapping direct child intervals."""
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
    groups: dict[str, dict] = {}
    for row in completed:
        covered = occupied([
            (max(row["start_pipeline_s"], start),
             min(row["end_pipeline_s"], stop))
            for start, stop in children[row["span_id"]]
            if min(row["end_pipeline_s"], stop) >
            max(row["start_pipeline_s"], start)])
        row["self_s"] = round(max(0.0, row["duration_s"] - covered), 9)
        group = groups.setdefault(row["name"], {
            "count": 0, "inclusive_s": 0.0, "self_s": 0.0,
            "failures": 0})
        group["count"] += 1
        group["inclusive_s"] += row["duration_s"]
        group["self_s"] += row["self_s"]
        group["failures"] += row["outcome"] != "success"
    for group in groups.values():
        group["inclusive_s"] = round(group["inclusive_s"], 9)
        group["self_s"] = round(group["self_s"], 9)
    inference = [row for row in completed if row["name"] == "vlm.infer"]
    inference_sum = sum(row["duration_s"] for row in inference)
    inference_wall = occupied([
        (row["start_pipeline_s"], row["end_pipeline_s"])
        for row in inference])
    return {
        "schema": "precision_insertion_timing_summary_v1",
        "clock": "time.perf_counter_ns",
        "vlm_inference_cumulative_s": round(inference_sum, 9),
        "vlm_inference_occupied_s": round(inference_wall, 9),
        "vlm_inference_overlap_s": round(max(0.0, inference_sum -
                                             inference_wall), 9),
        "unfinished_span_count": len(starts),
        "unfinished_spans": [row["name"] for row in starts.values()],
        "groups": dict(sorted(groups.items())),
        "spans": sorted(completed, key=lambda row: row["start_pipeline_s"]),
    }
