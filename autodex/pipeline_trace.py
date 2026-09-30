"""One append-only timeline for a complete AutoDex pipeline process.

The timeline is deliberately run-scoped rather than episode-scoped.  Every
record carries both a monotonic timestamp (the ordering/duration authority)
and a UTC timestamp (the bridge to independently recorded video).  Episode
results only reference this file; they never contain a second timing tree.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterator, Optional


def json_value(value: Any) -> Any:
    """Convert common scientific-Python values to bounded JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _utc_iso(timestamp_ns: int) -> str:
    return dt.datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000, tz=dt.timezone.utc
    ).isoformat(timespec="microseconds")


class PipelineTrace:
    """Crash-tolerant JSONL recorder for one complete pipeline run.

    ``bind`` may be delayed until CLI arguments determine the experiment
    directory.  Events emitted before then are retained in memory and flushed
    in sequence when the run directory becomes available.
    """

    schema_version = 1

    def __init__(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
        run_id: Optional[str] = None,
        origin_monotonic_ns: Optional[int] = None,
        origin_wall_ns: Optional[int] = None,
    ) -> None:
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self._origin_monotonic_ns = int(
            monotonic_ns() if origin_monotonic_ns is None
            else origin_monotonic_ns)
        self._origin_wall_ns = int(
            wall_time_ns() if origin_wall_ns is None else origin_wall_ns)
        stamp = dt.datetime.fromtimestamp(
            self._origin_wall_ns / 1_000_000_000, tz=dt.timezone.utc
        ).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_id = run_id or f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.output_dir: Optional[Path] = None
        self._stream = None
        self._pending: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._episode_events: dict[str, list[dict[str, Any]]] = {}
        self._episode_streams: dict[str, Any] = {}
        self._open_spans: dict[str, dict[str, Any]] = {}
        self._closed_spans: dict[str, dict[str, Any]] = {}
        self._next_seq = 0
        self._next_span = 0
        self._lock = threading.RLock()
        self._closed = False
        self._manifest: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": "running",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "argv": list(sys.argv),
            "started_utc_ns": self._origin_wall_ns,
            "started_utc": _utc_iso(self._origin_wall_ns),
            "clock": {
                "duration_clock": "time.perf_counter_ns",
                "sync_clock": "time.time_ns",
                "origin_monotonic_ns": self._origin_monotonic_ns,
                "origin_utc_ns": self._origin_wall_ns,
            },
        }
        self.event("pipeline.process_start", phase="startup", kind="lifecycle")
        self.clock_anchor("process_start")
        if output_dir is not None:
            self.bind(output_dir)

    @property
    def is_bound(self) -> bool:
        return self.output_dir is not None

    def now_s(self) -> float:
        return max(
            0.0,
            (int(self._monotonic_ns()) - self._origin_monotonic_ns)
            / 1_000_000_000,
        )

    def _sample(self) -> tuple[int, int]:
        # Read monotonic on both sides so the UTC bridge has a known bound.
        before = int(self._monotonic_ns())
        wall = int(self._wall_time_ns())
        after = int(self._monotonic_ns())
        return (before + after) // 2, wall

    def _base_record(self) -> dict[str, Any]:
        monotonic, wall = self._sample()
        record = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self._next_seq,
            "monotonic_ns": monotonic,
            "pipeline_time_s": round(
                max(0, monotonic - self._origin_monotonic_ns) / 1_000_000_000,
                9,
            ),
            "utc_ns": wall,
            "utc": _utc_iso(wall),
        }
        self._next_seq += 1
        return record

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._events.append(record)
            episode_id = record.get("episode_id")
            if episode_id is not None:
                self._episode_events.setdefault(str(episode_id), []).append(record)
            if self._stream is None:
                self._pending.append(record)
            else:
                encoded = json.dumps(record, separators=(",", ":")) + "\n"
                self._stream.write(encoded)
                self._stream.flush()
                if episode_id is not None:
                    episode_stream = self._episode_stream(str(episode_id))
                    episode_stream.write(encoded)
                    episode_stream.flush()
            return record

    @staticmethod
    def _safe_episode_id(episode_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._")
        if not safe:
            safe = hashlib.sha256(episode_id.encode()).hexdigest()[:16]
        return safe

    def _episode_dir(self, episode_id: str) -> Path:
        if self.output_dir is None:
            raise RuntimeError("pipeline trace must be bound")
        return self.output_dir / "episodes" / self._safe_episode_id(episode_id)

    def episode_timeline_path(self, episode_id: str) -> Path:
        """Return the run-relative path of an episode's identical event view."""
        return Path("episodes") / self._safe_episode_id(episode_id) / "events.jsonl"

    def _episode_stream(self, episode_id: str):
        stream = self._episode_streams.get(episode_id)
        if stream is None:
            directory = self._episode_dir(episode_id)
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                self._atomic_json(
                    manifest_path,
                    {
                        "schema_version": self.schema_version,
                        "run_id": self.run_id,
                        "episode_id": episode_id,
                        "status": "running",
                        "canonical_run_timeline": "../../events.jsonl",
                        "episode_timeline": "events.jsonl",
                    },
                )
            stream = (directory / "events.jsonl").open(
                "a", encoding="utf-8", buffering=1)
            self._episode_streams[episode_id] = stream
        return stream

    def bind(self, output_dir: Path | str, **manifest: Any) -> Path:
        """Bind the boot trace to its permanent run directory exactly once."""
        with self._lock:
            target = Path(output_dir).expanduser().resolve()
            if self.output_dir is not None:
                if target != self.output_dir:
                    raise RuntimeError(
                        f"pipeline trace already bound to {self.output_dir}, not {target}"
                    )
                return target
            target.mkdir(parents=True, exist_ok=False)
            (target / "artifacts" / "coverage").mkdir(parents=True)
            (target / "artifacts" / "planner").mkdir(parents=True)
            (target / "artifacts" / "recovery").mkdir(parents=True)
            (target / "artifacts" / "scene").mkdir(parents=True)
            (target / "sync").mkdir(parents=True)
            (target / "edit").mkdir(parents=True)
            (target / "episodes").mkdir(parents=True)
            self.output_dir = target
            self._stream = (target / "events.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
            for record in self._pending:
                encoded = json.dumps(record, separators=(",", ":")) + "\n"
                self._stream.write(encoded)
                episode_id = record.get("episode_id")
                if episode_id is not None:
                    self._episode_stream(str(episode_id)).write(encoded)
            self._stream.flush()
            self._pending.clear()
            self._manifest.update(json_value(manifest))
            self._write_manifest()
            self._write_episodes()
        self.event(
            "pipeline.trace_bound",
            phase="startup",
            kind="io",
            attributes={"output_dir": str(target)},
        )
        self.clock_anchor("trace_bound")
        return target

    def _atomic_json(self, path: Path, value: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(json_value(value), indent=2) + "\n")
        os.replace(tmp, path)

    def _write_manifest(self) -> None:
        if self.output_dir is not None:
            self._atomic_json(self.output_dir / "manifest.json", self._manifest)

    def _write_episodes(self) -> None:
        if self.output_dir is not None:
            self._atomic_json(
                self.output_dir / "episodes.json",
                {"schema_version": 1, "run_id": self.run_id,
                 "episodes": self._episodes},
            )

    def _refresh_episode_bounds(self) -> None:
        for record in self._episodes:
            episode_id = str(record.get("episode_id"))
            events = self._episode_events.get(episode_id, [])
            if not events:
                continue
            if "span_duration_s" not in record and "duration_s" in record:
                record["span_duration_s"] = record["duration_s"]
            start_s = min(float(event["pipeline_time_s"]) for event in events)
            end_s = max(float(event["pipeline_time_s"]) for event in events)
            record["start_pipeline_s"] = start_s
            record["end_pipeline_s"] = end_s
            record["duration_s"] = round(max(0.0, end_s - start_s), 9)

    def _write_episode_view(self, episode_id: str) -> None:
        if self.output_dir is None:
            return
        events = self._episode_events.get(episode_id, [])
        directory = self._episode_dir(episode_id)
        directory.mkdir(parents=True, exist_ok=True)
        episode_record = next(
            (item for item in reversed(self._episodes)
             if str(item.get("episode_id")) == episode_id),
            {"episode_id": episode_id},
        )
        self._atomic_json(
            directory / "manifest.json",
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                **episode_record,
                "status": episode_record.get("outcome", "running"),
                "canonical_run_timeline": "../../events.jsonl",
                "episode_timeline": "events.jsonl",
                "event_count": len(events),
            },
        )
        self._atomic_json(
            directory / "timeline.json",
            {"schema_version": self.schema_version, "run_id": self.run_id,
             "episode_id": episode_id, "events": events},
        )

    def event(
        self,
        name: str,
        *,
        phase: str,
        kind: str,
        outcome: Optional[str] = None,
        parent_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        **extra_attributes: Any,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._base_record()
            attrs = dict(attributes or {})
            attrs.update(extra_attributes)
            record.update({
                "edge": "instant",
                "name": str(name),
                "phase": str(phase),
                "kind": str(kind),
                "outcome": outcome,
                "parent_id": parent_id,
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "attributes": json_value(attrs),
            })
            return self._append(record)

    def begin(
        self,
        *,
        phase: str,
        kind: str,
        name: str,
        parent_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        **attributes: Any,
    ) -> str:
        with self._lock:
            span_id = f"p{self._next_span:06d}"
            self._next_span += 1
            record = self._base_record()
            record.update({
                "edge": "start",
                "span_id": span_id,
                "parent_id": parent_id,
                "name": str(name),
                "phase": str(phase),
                "kind": str(kind),
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "attributes": json_value(attributes),
            })
            self._open_spans[span_id] = dict(record)
            self._append(record)
            return span_id

    def end(self, span_id: str, *, outcome: str = "success", **attributes: Any) -> dict[str, Any]:
        with self._lock:
            if span_id in self._closed_spans:
                return self._closed_spans[span_id]
            start = self._open_spans.pop(span_id, None)
            if start is None:
                raise ValueError(f"unknown pipeline span: {span_id}")
            record = self._base_record()
            record.update({
                "edge": "end",
                "span_id": span_id,
                "parent_id": start.get("parent_id"),
                "name": start["name"],
                "phase": start["phase"],
                "kind": start["kind"],
                "episode_id": start.get("episode_id"),
                "attempt_id": start.get("attempt_id"),
                "outcome": str(outcome),
                "duration_s": round(
                    max(0, record["monotonic_ns"] - start["monotonic_ns"])
                    / 1_000_000_000,
                    9,
                ),
                "attributes": json_value(attributes),
            })
            self._closed_spans[span_id] = record
            return self._append(record)

    @contextmanager
    def span(self, *, phase: str, kind: str, name: str,
             parent_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
        span_id = self.begin(
            phase=phase, kind=kind, name=name, parent_id=parent_id, **attributes
        )
        try:
            yield span_id
        except BaseException as exc:
            self.end(span_id, outcome="failure", exception=repr(exc))
            raise
        else:
            self.end(span_id)

    def scoped(
        self, *, episode_id: Optional[str] = None,
        attempt_id: Optional[str] = None, parent_id: Optional[str] = None,
    ) -> "ScopedPipelineTrace":
        return ScopedPipelineTrace(
            self, episode_id=episode_id, attempt_id=attempt_id,
            parent_id=parent_id,
        )

    def clock_anchor(self, label: str) -> dict[str, Any]:
        before = int(self._monotonic_ns())
        wall = int(self._wall_time_ns())
        after = int(self._monotonic_ns())
        midpoint = (before + after) // 2
        return self.event(
            "pipeline.clock_anchor",
            phase="sync",
            kind="clock",
            label=label,
            monotonic_midpoint_ns=midpoint,
            utc_ns_sample=wall,
            uncertainty_ns=max(0, after - before) // 2,
        )

    def add_episode(self, record: dict[str, Any]) -> None:
        with self._lock:
            normalized = json_value(record)
            self._episodes.append(normalized)
            self._refresh_episode_bounds()
            self._write_episodes()
            episode_id = normalized.get("episode_id")
            if episode_id is not None:
                self._write_episode_view(str(episode_id))

    def close_episode_spans(
        self, episode_id: str, *, exclude: Optional[set[str]] = None,
        reason: str = "episode_finalized",
    ) -> None:
        """Close leaked child spans at the episode boundary.

        Hardware exceptions can bypass an adapter's normal ``end`` call. They
        must not remain open until process shutdown because that would assign
        the whole remaining run to the failed operation.
        """
        excluded = exclude or set()
        with self._lock:
            targets = [
                span_id for span_id, start in self._open_spans.items()
                if start.get("episode_id") == episode_id
                and span_id not in excluded
            ]
            # End children before parents when nested operations leaked.
            for span_id in reversed(targets):
                self.end(span_id, outcome="aborted", reason=reason)

    def write_artifact_json(self, relative_path: str | Path, value: Any) -> dict[str, Any]:
        if self.output_dir is None:
            raise RuntimeError("pipeline trace must be bound before writing artifacts")
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("artifact path must be relative and cannot contain '..'")
        path = self.output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(json_value(value), indent=2) + "\n").encode("utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        return {
            "path": rel.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    def close(self, *, outcome: str = "success", **attributes: Any) -> None:
        with self._lock:
            if self._closed:
                return
            for span_id in list(self._open_spans):
                self.end(span_id, outcome="aborted", reason="pipeline_closed")
            self.clock_anchor("process_end")
            self.event(
                "pipeline.process_end", phase="cleanup", kind="lifecycle",
                outcome=outcome, attributes=attributes,
            )
            self._manifest.update({
                "status": outcome,
                "ended_utc_ns": int(self._wall_time_ns()),
                "ended_utc": _utc_iso(int(self._wall_time_ns())),
                "duration_s": round(self.now_s(), 9),
                "event_count": len(self._events),
                "episode_count": len(self._episodes),
            })
            self._refresh_episode_bounds()
            self._write_manifest()
            self._write_episodes()
            if self.output_dir is not None:
                self._atomic_json(
                    self.output_dir / "timeline.json",
                    {"schema_version": self.schema_version,
                     "run_id": self.run_id, "events": self._events},
                )
                for episode_id in self._episode_events:
                    self._write_episode_view(episode_id)
                try:
                    from autodex.pipeline_edit import build_edit_package
                    build_edit_package(self.output_dir)
                    self._manifest["edit_package"] = "edit/storyboard.json"
                    self._write_manifest()
                except Exception as exc:
                    self._manifest["edit_package_error"] = repr(exc)
                    self._write_manifest()
            for episode_stream in self._episode_streams.values():
                episode_stream.flush()
                try:
                    os.fsync(episode_stream.fileno())
                except OSError:
                    pass
                episode_stream.close()
            if self._stream is not None:
                self._stream.flush()
                try:
                    os.fsync(self._stream.fileno())
                except OSError:
                    pass
                self._stream.close()
            self._closed = True


class ScopedPipelineTrace:
    """A lightweight view which supplies episode/attempt ownership."""

    def __init__(self, root: PipelineTrace, *, episode_id: Optional[str],
                 attempt_id: Optional[str], parent_id: Optional[str]) -> None:
        self.root = root
        self.episode_id = episode_id
        self.attempt_id = attempt_id
        self.parent_id = parent_id

    @property
    def run_id(self) -> str:
        return self.root.run_id

    def now_s(self) -> float:
        return self.root.now_s()

    def event(self, name: str, *, phase: str, kind: str,
              outcome: Optional[str] = None,
              parent_id: Optional[str] = None,
              attributes: Optional[dict[str, Any]] = None,
              **extra_attributes: Any) -> dict[str, Any]:
        return self.root.event(
            name, phase=phase, kind=kind, outcome=outcome,
            parent_id=parent_id or self.parent_id,
            episode_id=self.episode_id, attempt_id=self.attempt_id,
            attributes=attributes, **extra_attributes,
        )

    def begin(self, *, phase: str, kind: str, name: str,
              parent_id: Optional[str] = None, **attributes: Any) -> str:
        return self.root.begin(
            phase=phase, kind=kind, name=name,
            parent_id=parent_id or self.parent_id,
            episode_id=self.episode_id, attempt_id=self.attempt_id,
            **attributes,
        )

    def end(self, span_id: str, *, outcome: str = "success", **attributes: Any) -> dict[str, Any]:
        return self.root.end(span_id, outcome=outcome, **attributes)

    @contextmanager
    def span(self, *, phase: str, kind: str, name: str,
             parent_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
        span_id = self.begin(
            phase=phase, kind=kind, name=name, parent_id=parent_id, **attributes
        )
        try:
            yield span_id
        except BaseException as exc:
            self.end(span_id, outcome="failure", exception=repr(exc))
            raise
        else:
            self.end(span_id)

    def scoped(self, *, episode_id: Optional[str] = None,
               attempt_id: Optional[str] = None,
               parent_id: Optional[str] = None) -> "ScopedPipelineTrace":
        return self.root.scoped(
            episode_id=self.episode_id if episode_id is None else episode_id,
            attempt_id=self.attempt_id if attempt_id is None else attempt_id,
            parent_id=self.parent_id if parent_id is None else parent_id,
        )

    def write_artifact_json(self, relative_path: str | Path, value: Any) -> dict[str, Any]:
        return self.root.write_artifact_json(relative_path, value)
