"""Retry stroke metrics must refer to the same fresh axial path and trace."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.guarded_contact import (  # noqa: E402
    GuardedContactLimits, GuardedContactSample,
)
from precision_insertion.guarded_trace import (  # noqa: E402
    replay_guarded_contact_trace, write_guarded_contact_trace,
)
from precision_insertion.retry_axial_handoff import (  # noqa: E402
    prepare_retry_axial_handoff,
)
from precision_insertion.retry_guarded_metric import (  # noqa: E402
    verify_retry_guarded_metric,
)
from test_retry_axial_handoff import _case  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_metric(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    packet = prepare_retry_axial_handoff(
        output_dir=tmp_path / "retry_guarded_axial_handoffs/000", **args)
    handoff = json.loads(packet.read_text())
    limits = GuardedContactLimits(
        .020, 10., 5., .5, .002, 4., None, .05, .1, .5, .005,
        .0001, .0005)
    samples = [(
        GuardedContactSample(
            101.31 + .03 * i, .004 * i, (0., 0., 1.),
            (0., 0., .01), .0005, 1., None, True),
        101.32 + .03 * i) for i in range(6)]
    trace = replay_guarded_contact_trace(
        attempt_id=handoff["attempt_id"],
        candidate_id=handoff["candidate_id"], family="cylinder",
        session_calibration_sha256=handoff["session_calibration_sha256"],
        limits=limits, started_at_s=101.3, events=samples,
        axial_handoff_sha256=_sha(packet),
        trajectory_archive_sha256=handoff["trajectory_archive_sha256"])
    trace_path = write_guarded_contact_trace(
        trace, tmp_path / "retry_force_trace.json")
    measurement = {
        "key_depth_interval_m": None, "key_depth_source": None,
        "alignment_within_limits": True,
        "safety_abort": False, "grasp_held": True,
    }
    metric = {
        "schema": "precision_insertion_guarded_retry_execution_v1",
        "attempt_id": handoff["attempt_id"],
        "candidate_id": handoff["candidate_id"],
        "session_calibration_sha256":
            handoff["session_calibration_sha256"],
        "retry_axial_handoff": {"path": str(packet), "sha256": _sha(packet)},
        "trajectory_archive_sha256": handoff["trajectory_archive_sha256"],
        "started_at_s": 101.3, "completed_at_s": 101.5,
        "measurement": measurement, "contact_limits": asdict(limits),
        "source_records": {"force_trace": {
            "path": str(trace_path), "sha256": _sha(trace_path)}},
        "scope": "external_retry_samples_not_physical_key_depth_certification",
        "robot_ready": False,
    }
    fields = {
        "key_depth": ("key_depth_interval_m", "key_depth_source"),
        "alignment": ("alignment_within_limits",),
        "grasp_state": ("grasp_held",),
    }
    for name, names in fields.items():
        raw = tmp_path / f"retry_{name}_raw.json"
        raw.write_text(json.dumps({"synthetic_test_evidence": name}))
        claim = tmp_path / f"retry_{name}.json"
        claim.write_text(json.dumps({
            "schema": "precision_insertion_external_metric_claim_v1",
            "source_name": name,
            "attempt_id": metric["attempt_id"],
            "candidate_id": metric["candidate_id"],
            "session_calibration_sha256":
                metric["session_calibration_sha256"],
            "recorded_at_s": 101.4,
            "producer_id": "synthetic_test_only",
            "source_method": "synthetic_not_physical_sensor",
            "measurement": {key: measurement[key] for key in names},
            "raw_evidence": [{"path": str(raw), "sha256": _sha(raw)}],
        }))
        metric["source_records"][name] = {
            "path": str(claim), "sha256": _sha(claim)}
    metric_path = tmp_path / "retry_metric.json"
    metric_path.write_text(json.dumps(metric))
    verify_args = dict(
        handoff_report_path=packet, expected=args["expected"],
        previous=args["previous"], arrival=args["arrival"],
        checkpoint=args["checkpoint"], shift_plan=args["shift_plan"],
        mode=args["mode"], shared_root=tmp_path,
        calibration=args["calibration"])
    return metric_path, verify_args


def test_retry_metric_replays_exact_new_path_and_external_trace(
        tmp_path, monkeypatch):
    path, context = _case_metric(tmp_path, monkeypatch)
    metric, started, completed = verify_retry_guarded_metric(path, **context)
    assert (started, completed) == (101.3, 101.5)
    assert metric["measurement"]["key_depth_interval_m"] is None
    assert metric["robot_ready"] is False


def test_retry_metric_rejects_foreign_handoff_or_trace(
        tmp_path, monkeypatch):
    path, context = _case_metric(tmp_path, monkeypatch)
    metric = json.loads(path.read_text())
    metric["retry_axial_handoff"]["sha256"] = "a" * 64
    path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="fresh axial handoff"):
        verify_retry_guarded_metric(path, **context)
    metric["retry_axial_handoff"]["sha256"] = _sha(
        context["handoff_report_path"])
    trace_ref = metric["source_records"]["force_trace"]
    trace_path = Path(trace_ref["path"])
    trace = json.loads(trace_path.read_text())
    trace["path_binding"]["trajectory_archive_sha256"] = "b" * 64
    trace_path.write_text(json.dumps(trace))
    trace_ref["sha256"] = _sha(trace_path)
    path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="another path or stroke"):
        verify_retry_guarded_metric(path, **context)


def test_retry_metric_rejects_changed_raw_claim(tmp_path, monkeypatch):
    path, context = _case_metric(tmp_path, monkeypatch)
    metric = json.loads(path.read_text())
    claim_path = Path(metric["source_records"]["alignment"]["path"])
    raw_path = Path(json.loads(claim_path.read_text())["raw_evidence"][0]["path"])
    raw_path.write_text("changed")
    with pytest.raises(ValueError, match="alignment raw evidence changed"):
        verify_retry_guarded_metric(path, **context)


def test_retry_metric_rejects_unpaired_depth_claim(tmp_path, monkeypatch):
    path, context = _case_metric(tmp_path, monkeypatch)
    metric = json.loads(path.read_text())
    metric["measurement"]["key_depth_source"] = "exposed_length_cad"
    path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="key-depth source is invalid"):
        verify_retry_guarded_metric(path, **context)
