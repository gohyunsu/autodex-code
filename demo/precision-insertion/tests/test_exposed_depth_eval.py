"""Offline depth evaluation stays source-bound and never admits success."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.exposed_depth_eval import (  # noqa: E402
    evaluate_exposed_depth_manifest,
)
from precision_insertion.saved_exposed_depth import (  # noqa: E402
    assess_saved_exposed_depth, write_saved_exposed_depth,
)
from test_saved_exposed_depth import _setup, _Backend  # noqa: E402


def _ref(path):
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _sample(tmp_path, monkeypatch, *, truth_interval=(.0209, .0211),
            observable=True):
    inputs = _setup(tmp_path, monkeypatch)
    if not observable:
        inputs["backend"] = _Backend([json.dumps({
            "rear_px": None, "axis_line_px": None,
            "evidence": "rear face is occluded"})] * 3)
    report = assess_saved_exposed_depth(**inputs)
    saved = write_saved_exposed_depth(report, tmp_path / "depth_estimate")
    raw = tmp_path / "instrument_measurements.csv"
    raw.write_text("depth_m\n0.0210\n", encoding="utf-8")
    calibration = tmp_path / "instrument_calibration.json"
    calibration.write_text('{"instrument":"synthetic-test"}\n',
                           encoding="utf-8")
    source = report["source"]
    truth = {
        "schema": "precision_insertion_independent_depth_v1",
        "attempt_id": report["attempt_id"],
        "candidate_id": report["candidate_id"], "mode": report["mode"],
        "session_calibration_sha256": source["session_calibration_sha256"],
        "camera_calibration_sha256": source["camera_calibration_sha256"],
        "task_geometry_sha256": report["task_geometry_sha256"],
        "final_manifest_sha256": source["final_manifest_sha256"],
        "final_capture_id": source["capture_id"],
        "final_request_id": source["request_id"],
        "measurement_method": "calibrated_depth_gauge",
        "measurement_time_s": 103.005,
        "clock_domain": "unix_utc",
        "depth_interval_m": list(truth_interval),
        "raw_evidence": [_ref(raw)],
        "instrument_calibration": _ref(calibration),
    }
    truth_path = tmp_path / "independent_depth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    manifest = {
        "schema": "precision_insertion_exposed_depth_eval_manifest_v1",
        "max_reference_capture_skew_s": .02,
        "samples": [{"depth_report": "depth_estimate/report.json",
                     "independent_depth": "independent_depth.json"}],
    }
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return inputs, manifest_path, truth_path, raw


def _evaluate(inputs, manifest):
    return evaluate_exposed_depth_manifest(
        manifest, mode=inputs["mode"], shared_root=inputs["shared_root"],
        calibration=inputs["calibration"])


def test_depth_eval_scores_bounded_result_without_promoting_label(
        tmp_path, monkeypatch):
    inputs, manifest, _, _ = _sample(tmp_path, monkeypatch)
    result = _evaluate(inputs, manifest)
    assert result["total"] == result["bounded"] == 1
    assert result["rows"][0]["truth_definitely_reached_20mm"] is True
    assert result["rows"][0]["truth_interval_contained"] is True
    assert result["definite_false_successes"] == 0
    assert result["possibly_false_successes"] == 0
    assert result["depth_source_admissible_for_task_label"] is False
    assert result["robot_ready"] is False


def test_depth_eval_catches_definite_false_success(tmp_path, monkeypatch):
    inputs, manifest, _, _ = _sample(
        tmp_path, monkeypatch, truth_interval=(.017, .019))
    result = _evaluate(inputs, manifest)
    assert result["definite_false_successes"] == 1
    assert result["possibly_false_successes"] == 1
    assert result["truth_interval_not_contained"] == 1
    assert result["rows"][0]["required_overestimate_bound_m"] > .001


def test_depth_eval_preserves_abstention(tmp_path, monkeypatch):
    inputs, manifest, _, _ = _sample(
        tmp_path, monkeypatch, observable=False)
    result = _evaluate(inputs, manifest)
    assert result["bounded"] == 0
    assert result["abstained"] == 1
    assert result["rows"][0]["estimated_interval_m"] is None


def test_depth_eval_rejects_tampered_or_stale_reference(tmp_path, monkeypatch):
    inputs, manifest, truth_path, raw = _sample(tmp_path, monkeypatch)
    raw.write_text("depth_m\n0.013\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw metrology file changed"):
        _evaluate(inputs, manifest)
    raw.write_text("depth_m\n0.0210\n", encoding="utf-8")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["measurement_time_s"] = 104.
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="not synchronized"):
        _evaluate(inputs, manifest)


def test_depth_eval_rejects_duplicate_sample(tmp_path, monkeypatch):
    inputs, manifest_path, _, _ = _sample(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"].append(dict(manifest["samples"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate depth trial"):
        _evaluate(inputs, manifest_path)


def test_depth_eval_rejects_same_camera_pixels_as_independent_source(
        tmp_path, monkeypatch):
    inputs, manifest, truth_path, _ = _sample(tmp_path, monkeypatch)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    image = inputs["final_bundle"] / "images" / "front.png"
    truth["raw_evidence"] = [_ref(image)]
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="reused a VLM camera image"):
        _evaluate(inputs, manifest)


def test_depth_eval_rejects_different_clock_domain(tmp_path, monkeypatch):
    inputs, manifest, truth_path, _ = _sample(tmp_path, monkeypatch)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["clock_domain"] = "another_clock"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="another timestamp clock"):
        _evaluate(inputs, manifest)
