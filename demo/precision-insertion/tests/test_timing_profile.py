"""Run-level monotonic timing uses stock AutoDex trace without robot access."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.timing_profile import (  # noqa: E402
    profile_run, summarize_events, timed, timing_directory,
)


def test_profile_groups_nested_spans_without_double_counting(tmp_path):
    output = tmp_path / "session"
    with profile_run(output):
        with timed("live.session", phase="session"):
            with timed("socket.capture", phase="socket", capture_index=0):
                with timed("vlm.infer", phase="vlm", camera_id="cam-1",
                           stage="lift", image_count=2):
                    pass
    directory = timing_directory(output)
    summary = json.loads((directory / "timing_summary.json").read_text())
    assert (directory / "events.jsonl").is_file()
    assert (directory / "manifest.json").is_file()
    assert summary["unfinished_span_count"] == 0
    assert summary["groups"]["vlm.infer"]["count"] == 1
    assert summary["groups"]["live.session"]["self_s"] <= (
        summary["groups"]["live.session"]["inclusive_s"])
    assert next(row for row in summary["spans"] if row["name"] ==
                "socket.capture")["attributes"]["capture_index"] == 0
    assert summarize_events(directory / "events.jsonl") == summary
    assert not output.exists()


def test_profile_keeps_failure_trace(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="capture failed"):
        with profile_run(output):
            with timed("socket.capture", phase="socket"):
                raise RuntimeError("capture failed")
    directory = timing_directory(output)
    summary = json.loads((directory / "timing_summary.json").read_text())
    assert summary["groups"]["socket.capture"]["failures"] == 1
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failure"


def test_profile_can_record_nonexception_terminal_status(tmp_path):
    output = tmp_path / "not_yet_done"
    with profile_run(output) as profile:
        profile.outcome = "needs_handler"
    manifest = json.loads((timing_directory(output) / "manifest.json").read_text())
    assert manifest["status"] == "needs_handler"


def test_timed_without_live_profile_is_noop():
    with timed("standalone", phase="test"):
        pass
