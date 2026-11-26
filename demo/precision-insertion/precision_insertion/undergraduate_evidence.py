"""Read-only audit of physical evidence for the cylinder thesis protocol.

This module does not decide whether a robot motion is safe and is never used
by the live action loop.  It joins the small launch/recording receipts written
by the ZeroDex runner with episode-owned tracker artifacts so a demonstration
video cannot be mistaken for evidence of continuous 6D tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRACKER_INITIALIZATION_SCHEMA = "precision_insertion_mvgotrack_initialization_v1"
RECORDING_RECEIPT_SCHEMA = "precision_insertion_zerodex_recording_receipt_v1"
CONTINUITY_AUDIT_SCHEMA = "precision_insertion_mvgotrack_continuity_audit_v1"


@dataclass(frozen=True)
class RunPaths:
    """Canonical locations associated with one ZeroDex run identifier."""

    run_id: str
    control: Path
    receipt: Path
    recording: Path
    extras: tuple[Path, ...] = ()

    @property
    def roots(self) -> tuple[Path, ...]:
        values = (self.control, self.recording, *self.extras)
        return tuple(path for path in values if path.exists())


def _read_json(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_recording_path(run_root: Path, run_id: str) -> Path:
    """Return ``YYYY/MM/DD/RUN_ID`` without searching the entire NAS."""

    try:
        date = datetime.strptime(run_id[:8], "%Y%m%d")
    except (ValueError, TypeError):
        return run_root / "_unresolved" / run_id
    return run_root / date.strftime("%Y/%m/%d") / run_id


def discover_run_ids(run_root: Path) -> list[str]:
    """Discover run IDs from the two bounded metadata directories."""

    result: set[str] = set()
    for name in ("_control", "_recording_receipts"):
        directory = Path(run_root) / name
        if directory.is_dir():
            result.update(child.name for child in directory.iterdir()
                          if child.is_dir() and child.name[:8].isdigit())
    return sorted(result)


def resolve_run_paths(
    run_root: Path,
    run_id: str,
    *,
    extra_episode_roots: Iterable[Path] = (),
) -> RunPaths:
    extras = tuple(Path(root) / run_id for root in extra_episode_roots)
    return RunPaths(
        run_id=run_id,
        control=Path(run_root) / "_control" / run_id,
        receipt=Path(run_root) / "_recording_receipts" / run_id /
        "recording_ref.json",
        recording=_run_recording_path(Path(run_root), run_id),
        extras=extras,
    )


def _iter_files(roots: Sequence[Path]) -> Iterable[Path]:
    """Yield episode metadata without walking large image/video trees."""

    seen: set[Path] = set()
    media_directories = {
        "raw_captures", "video", "videos", "images", "frames",
        "visualizations",
    }
    for root in roots:
        if root.is_file():
            if root not in seen:
                seen.add(root)
                yield root
            continue
        if not root.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames
                           if name.lower() not in media_directories]
            parent = Path(directory)
            for name in filenames:
                path = parent / name
                if path not in seen:
                    seen.add(path)
                    yield path


def _relative_or_absolute(path: Path, run_root: Path) -> str:
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)


def _is_tracker_trace(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".jsonl", ".csv", ".npz"}:
        return False
    return ("mvgotrack" in name or "tracker" in name) and any(
        token in name for token in ("pose", "trace", "stream", "track"))


def _valid_continuity_audit(payload: Mapping[str, object], run_id: str) -> bool:
    counts = payload.get("phase_sample_counts")
    interval = payload.get("interval_unix_ns")
    references = payload.get("evidence_refs")
    calibrations = payload.get("calibration_sessions")
    objects = payload.get("objects")
    required_phases = ("pickup", "lift", "transfer", "precontact_hold")
    return bool(
        payload.get("schema") == CONTINUITY_AUDIT_SCHEMA and
        payload.get("run_id") == run_id and
        payload.get("interval") == "pickup_start_to_precontact_hold" and
        payload.get("verdict") == "pass" and
        isinstance(payload.get("first_seq"), int) and
        isinstance(payload.get("last_seq"), int) and
        payload["last_seq"] > payload["first_seq"] and
        isinstance(payload.get("sample_count"), int) and
        payload["sample_count"] >= len(required_phases) and
        payload.get("sequence_strictly_increasing") is True and
        isinstance(calibrations, list) and len(calibrations) == 1 and
        isinstance(calibrations[0], str) and bool(calibrations[0]) and
        isinstance(objects, list) and len(objects) == 1 and
        isinstance(objects[0], str) and bool(objects[0]) and
        isinstance(counts, dict) and
        all(isinstance(counts.get(phase), int) and counts[phase] >= 1
            for phase in required_phases) and
        isinstance(interval, dict) and
        isinstance(interval.get("pickup_start"), int) and
        isinstance(interval.get("precontact_hold"), int) and
        interval["precontact_hold"] > interval["pickup_start"] and
        isinstance(references, list) and bool(references) and
        all(isinstance(reference, str) and reference for reference in references)
    )


_CONTINUITY_PHASE_SPANS = {
    "pickup": "trial.pickup_squeeze",
    "lift": "trial.lift_motion",
    "transfer": "trial.transfer_motion",
    "precontact_hold": "trial.same_trial_alignment",
}


def _completed_timing_spans(path: Path) -> list[dict]:
    starts: dict[str, dict] = {}
    completed: list[dict] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        span_id = row.get("span_id")
        if not isinstance(span_id, str):
            continue
        if row.get("edge") == "start":
            starts[span_id] = row
        elif row.get("edge") == "end" and span_id in starts:
            start = starts.pop(span_id)
            completed.append({
                "span_id": span_id,
                "name": start.get("name"),
                "attempt_id": start.get("attempt_id"),
                "start_utc_ns": start.get("utc_ns"),
                "end_utc_ns": row.get("utc_ns"),
                "outcome": row.get("outcome"),
            })
    return completed


def _tracker_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if (isinstance(row, dict) and
                row.get("schema") == "forgehand.tracked_pose.v1" and
                isinstance(row.get("seq"), int) and
                isinstance(row.get("received_unix_ns"), int)):
            rows.append(row)
    return rows


def compile_continuity_audit(
    *,
    run_id: str,
    pose_stream: Path,
    timing_events: Path,
    attempt_id: str | None = None,
) -> dict:
    """Join tracker and motion clocks into the offline U1 continuity record.

    The computation uses the main-PC receive timestamp recorded beside each
    tracker message, so no capture-PC wall-clock assumption is needed.  It
    applies no score, jump, workspace or inter-frame-gap threshold.
    """

    spans = _completed_timing_spans(timing_events)
    if attempt_id is None:
        candidates = sorted({
            str(span["attempt_id"]) for span in spans
            if span.get("attempt_id") is not None and
            span.get("name") in _CONTINUITY_PHASE_SPANS.values()
        })
        complete = []
        for candidate in candidates:
            names = {span["name"] for span in spans
                     if span.get("attempt_id") == candidate and
                     span.get("outcome") == "success"}
            if set(_CONTINUITY_PHASE_SPANS.values()) <= names:
                complete.append(candidate)
        if len(complete) != 1:
            return {
                "schema": CONTINUITY_AUDIT_SCHEMA,
                "run_id": run_id,
                "interval": "pickup_start_to_precontact_hold",
                "verdict": "fail",
                "reason": "expected_exactly_one_complete_attempt",
                "candidate_attempt_ids": complete,
                "evidence_refs": [str(pose_stream), str(timing_events)],
            }
        attempt_id = complete[0]

    selected: dict[str, dict] = {}
    for phase, name in _CONTINUITY_PHASE_SPANS.items():
        matches = [span for span in spans
                   if span.get("attempt_id") == attempt_id and
                   span.get("name") == name and
                   span.get("outcome") == "success"]
        if len(matches) != 1:
            return {
                "schema": CONTINUITY_AUDIT_SCHEMA,
                "run_id": run_id,
                "interval": "pickup_start_to_precontact_hold",
                "verdict": "fail",
                "attempt_id": attempt_id,
                "reason": f"missing_or_ambiguous_phase:{phase}",
                "evidence_refs": [str(pose_stream), str(timing_events)],
            }
        span = matches[0]
        if (not isinstance(span["start_utc_ns"], int) or
                not isinstance(span["end_utc_ns"], int) or
                span["end_utc_ns"] <= span["start_utc_ns"]):
            raise ValueError(f"invalid UTC interval for {phase}")
        selected[phase] = span

    rows = _tracker_rows(pose_stream)
    interval_start = selected["pickup"]["start_utc_ns"]
    interval_end = selected["precontact_hold"]["end_utc_ns"]
    interval_rows = [row for row in rows
                     if interval_start <= row["received_unix_ns"] <= interval_end]
    valid_rows = [row for row in interval_rows if row.get("valid") is True]
    phase_counts = {
        phase: sum(
            span["start_utc_ns"] <= row["received_unix_ns"] <=
            span["end_utc_ns"] and row.get("valid") is True
            for row in interval_rows)
        for phase, span in selected.items()
    }
    sequences = [row["seq"] for row in interval_rows]
    monotonic_sequences = all(
        later > earlier for earlier, later in zip(sequences, sequences[1:]))
    calibrations = sorted({str(row["calib_session"])
                           for row in valid_rows
                           if isinstance(row.get("calib_session"), str) and
                           row["calib_session"]})
    objects = sorted({str(row["object"]) for row in valid_rows
                      if isinstance(row.get("object"), str) and row["object"]})
    passed = bool(
        valid_rows and all(count >= 1 for count in phase_counts.values()) and
        monotonic_sequences and len(calibrations) == 1 and len(objects) == 1)
    result = {
        "schema": CONTINUITY_AUDIT_SCHEMA,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "interval": "pickup_start_to_precontact_hold",
        "verdict": "pass" if passed else "fail",
        "first_seq": valid_rows[0]["seq"] if valid_rows else 0,
        "last_seq": valid_rows[-1]["seq"] if valid_rows else 0,
        "sample_count": len(valid_rows),
        "native_invalid_sample_count": len(interval_rows) - len(valid_rows),
        "phase_sample_counts": phase_counts,
        "interval_unix_ns": {
            "pickup_start": interval_start,
            "precontact_hold": interval_end,
        },
        "sequence_strictly_increasing": monotonic_sequences,
        "calibration_sessions": calibrations,
        "objects": objects,
        "thresholds": {
            "score": None,
            "pose_jump": None,
            "workspace": None,
            "maximum_interframe_gap": None,
        },
        "evidence_refs": [str(pose_stream), str(timing_events)],
    }
    if not passed:
        result["reason"] = "phase_coverage_or_identity_incomplete"
    return result


def audit_run(paths: RunPaths, *, run_root: Path) -> dict:
    """Audit one run without promoting file presence to task success."""

    receipt = _read_json(paths.receipt)
    launch_files = sorted(paths.control.glob("launch_*.json")) \
        if paths.control.is_dir() else []
    launch = _read_json(launch_files[-1]) if launch_files else None

    candidate_roots = list(paths.roots)
    if launch:
        evidence_dir = launch.get("evidence_dir")
        if isinstance(evidence_dir, str):
            candidate_roots.append(Path(evidence_dir))
    files = list(_iter_files(candidate_roots))

    initialization: list[Path] = []
    first_pose: list[Path] = []
    tracker_trace: list[Path] = []
    continuity_audits: list[Path] = []
    for path in files:
        if path.name == "initialization.json":
            payload = _read_json(path)
            if payload and payload.get("schema") == TRACKER_INITIALIZATION_SCHEMA:
                initialization.append(path)
        elif path.name == "first_pose.json":
            first_pose.append(path)
        elif path.name == "mvgotrack_continuity_audit.json":
            payload = _read_json(path)
            if payload and _valid_continuity_audit(payload, paths.run_id):
                continuity_audits.append(path)
        elif _is_tracker_trace(path):
            tracker_trace.append(path)

    recording_file: Path | None = None
    recording_completed = False
    release_and_clear = False
    pipeline_exit_code: int | None = None
    if receipt and receipt.get("schema") == RECORDING_RECEIPT_SCHEMA:
        pipeline_exit_code = receipt.get("pipeline_exit_code") \
            if isinstance(receipt.get("pipeline_exit_code"), int) else None
        recording = receipt.get("recording")
        if isinstance(recording, dict):
            video = recording.get("video")
            if isinstance(video, str):
                candidate = paths.recording / "video" / video
                if candidate.is_file():
                    recording_file = candidate
            recording_completed = recording.get("status") == "completed"
        release_and_clear = (
            receipt.get("recording_stop_reason") ==
            "release_and_clear_completed")

    u1_checks = {
        "accepted_foundpose_initialization": bool(initialization),
        "first_native_tracker_pose": bool(first_pose),
        "continuous_tracker_pose_stream": bool(tracker_trace),
        "episode_video": recording_file is not None and recording_completed,
        "pickup_to_precontact_continuity_audit": bool(continuity_audits),
    }
    artifact_bundle_complete = all(
        u1_checks[name] for name in (
            "accepted_foundpose_initialization",
            "first_native_tracker_pose",
            "continuous_tracker_pose_stream",
            "episode_video",
        ))
    u1_complete = artifact_bundle_complete and bool(continuity_audits)
    return {
        "run_id": paths.run_id,
        "pipeline_exit_code": pipeline_exit_code,
        "release_and_clear_completed": release_and_clear,
        "u1_visual_commissioning": {
            "complete": u1_complete,
            "artifact_bundle_complete": artifact_bundle_complete,
            "checks": u1_checks,
            "interpretation": (
                "continuous_tracking_evidence"
                if u1_complete else
                "partial_or_demonstration_evidence_only"),
        },
        "artifacts": {
            "launch": [_relative_or_absolute(path, run_root)
                       for path in launch_files],
            "recording_receipt": (
                _relative_or_absolute(paths.receipt, run_root)
                if paths.receipt.is_file() else None),
            "video": (
                _relative_or_absolute(recording_file, run_root)
                if recording_file else None),
            "tracker_initialization": [
                _relative_or_absolute(path, run_root)
                for path in sorted(initialization)],
            "tracker_first_pose": [
                _relative_or_absolute(path, run_root)
                for path in sorted(first_pose)],
            "tracker_pose_stream": [
                _relative_or_absolute(path, run_root)
                for path in sorted(tracker_trace)],
            "continuity_audit": [
                _relative_or_absolute(path, run_root)
                for path in sorted(continuity_audits)],
        },
    }


def build_report(
    run_root: Path,
    *,
    run_ids: Sequence[str] | None = None,
    extra_episode_roots: Iterable[Path] = (),
) -> dict:
    """Build the machine-readable U1 audit for the undergraduate protocol."""

    root = Path(run_root).expanduser().resolve()
    ids = list(run_ids) if run_ids is not None else discover_run_ids(root)
    runs = [
        audit_run(resolve_run_paths(
            root, run_id, extra_episode_roots=extra_episode_roots),
            run_root=root)
        for run_id in ids
    ]
    complete = [run["run_id"] for run in runs
                if run["u1_visual_commissioning"]["complete"]]
    videos = [run["run_id"] for run in runs
              if run["u1_visual_commissioning"]["checks"]["episode_video"]]
    return {
        "schema": "undergraduate_precision_evidence_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "scope": "offline_research_evidence_only",
        "runtime_effect": "none",
        "u1_contract": {
            "unit": "single_episode",
            "required": [
                "accepted_foundpose_initialization",
                "first_native_tracker_pose",
                "continuous_tracker_pose_stream",
                "episode_video",
                "pickup_to_precontact_continuity_audit",
            ],
            "note": (
                "File presence is only the artifact bundle. Completion also "
                "requires an episode-owned continuity audit tying tracker "
                "samples and video/phase timestamps to pickup start through "
                "the stopped precontact hold. A video or release event alone "
                "is demonstration evidence."),
        },
        "summary": {
            "run_count": len(runs),
            "runs_with_video": videos,
            "u1_complete_runs": complete,
            "u1_complete_count": len(complete),
        },
        "runs": runs,
    }
