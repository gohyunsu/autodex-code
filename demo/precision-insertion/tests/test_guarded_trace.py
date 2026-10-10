"""A saved sensor stream must reproduce its abort/hold decision."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.guarded_contact import (  # noqa: E402
    GuardedContactLimits, GuardedContactSample,
)
from precision_insertion.guarded_trace import (  # noqa: E402
    replay_guarded_contact_trace, verify_guarded_contact_trace,
    verify_guarded_contact_trace_record, write_guarded_contact_trace,
)


def _limits():
    return GuardedContactLimits(
        .020, 10., 5., .5, .002, 4., 5., .05, .1, 1.,
        .005, .0001, .0005)


def _sample(t, depth):
    return GuardedContactSample(
        t, depth, (0., 0., 1.), (0., 0., .01),
        .0005, 1., 1., True)


def _stroke():
    return [(_sample(1.01 + .02 * i, .004 * i),
             1.02 + .02 * i) for i in range(6)]


def test_trace_replays_complete_hold_and_detects_changed_decision(tmp_path):
    trace = replay_guarded_contact_trace(
        attempt_id="trial", candidate_id="table/0/3", family="square",
        session_calibration_sha256="0" * 64,
        limits=_limits(), started_at_s=1., events=_stroke())
    assert trace["terminal_action"] == "hold_for_key_depth_and_vlm"
    assert trace["safety_abort"] is False
    path = write_guarded_contact_trace(trace, tmp_path / "trace.json")
    assert verify_guarded_contact_trace(path) == json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed["events"][-1]["decision"]["reason"] = "fabricated_success"
    with pytest.raises(ValueError, match="differs from replay"):
        verify_guarded_contact_trace_record(changed)


def test_trace_abort_and_missing_sample_deadline():
    events = _stroke()[:1] + [
        (replace(_sample(1.03, .004), force_socket_n=(0., 0., 20.)), 1.04)]
    trace = replay_guarded_contact_trace(
        attempt_id="trial", candidate_id="table/0/3", family="square",
        session_calibration_sha256="0" * 64,
        limits=_limits(), started_at_s=1., events=events)
    assert trace["terminal_action"] == "abort_hold_for_supervised_recovery"
    assert trace["safety_abort"] is True
    timeout = replay_guarded_contact_trace(
        attempt_id="trial", candidate_id="table/0/3", family="square",
        session_calibration_sha256="0" * 64,
        limits=_limits(), started_at_s=1.,
        events=[(_sample(1.01, 0.), 1.02), (None, 1.13)])
    assert timeout["events"][-1]["decision"]["reason"] == (
        "sensor_deadline_missed")


def test_trace_rejects_incomplete_or_late_extra_samples():
    with pytest.raises(ValueError, match="without a hold or abort"):
        replay_guarded_contact_trace(
            attempt_id="trial", candidate_id="table/0/3", family="square",
            session_calibration_sha256="0" * 64,
            limits=_limits(), started_at_s=1., events=_stroke()[:1])
    with pytest.raises(ValueError, match="after a latched stop"):
        replay_guarded_contact_trace(
            attempt_id="trial", candidate_id="table/0/3", family="square",
            session_calibration_sha256="0" * 64,
            limits=_limits(), started_at_s=1.,
            events=_stroke() + [(_sample(1.13, .021), 1.14)])
    with pytest.raises(ValueError, match="event order"):
        replay_guarded_contact_trace(
            attempt_id="trial", candidate_id="table/0/3", family="square",
            session_calibration_sha256="0" * 64,
            limits=_limits(), started_at_s=1.,
            events=[_stroke()[0], (_sample(1.03, .004), 1.02)])
