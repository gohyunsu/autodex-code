"""ZeroDex recording references remain separate from immutable assets."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


def _module():
    source = (Path(__file__).resolve().parents[1] /
              "run_zerodex_recorded.py")
    spec = importlib.util.spec_from_file_location("run_zerodex_recorded", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recording_receipt_uses_run_tree(tmp_path):
    path = _module().recording_receipt_directory(
        tmp_path / "runs/zerodex_capture4/_recording_receipts",
        "20261030_123456_123")
    assert path == (tmp_path / "runs/zerodex_capture4/_recording_receipts" /
                    "20261030_123456_123")


def test_recording_receipt_rejects_path_components(tmp_path):
    with pytest.raises(ValueError, match="path-safe"):
        _module().recording_receipt_directory(tmp_path, "../other")


def test_all_view_recording_spans_first_motion_to_final_capture(
        tmp_path, monkeypatch):
    module = _module()
    receipt_root = tmp_path / "receipts"
    trigger_root = tmp_path / "triggers"
    monkeypatch.setenv("PRECISION_RUN_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("PRECISION_RECORDING_RECEIPT_ROOT", str(receipt_root))
    monkeypatch.setenv("PRECISION_RECORDING_TRIGGER_ROOT", str(trigger_root))
    monkeypatch.setenv("PRECISION_RUN_TAG", "20261030_123456_test")
    monkeypatch.setenv("PRECISION_CODE_COMMIT", "a" * 40)
    monkeypatch.setenv(
        "PRECISION_CAPTURE4_CAMERA_IDS", "100,101,102,103")
    events = []

    class Pipeline:
        def __init__(self, _argv, *, env, **_kwargs):
            self.env = env
            self.polls = 0
            self.stdout = io.StringIO("pipeline started\npipeline finished\n")
            events.append("pipeline_started")

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                marker = Path(self.env["PRECISION_EXECUTION_READY_MARKER"])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({
                    "schema": "precision_insertion_execution_ready_v1",
                    "attempt_id": "attempt_1",
                    "candidate_id": "table/0/20195",
                    "confirmed_at_unix_ns": 10,
                }), encoding="utf-8")
                events.append("execution_ready")
                return None
            if self.polls == 2:
                marker = Path(self.env[
                    "PRECISION_FINAL_CAPTURE_COMPLETED_MARKER"])
                marker.write_text(json.dumps({
                    "schema": "precision_insertion_final_capture_completed_v1",
                    "attempt_id": "attempt_1",
                    "candidate_id": "table/0/20195",
                    "confirmed_at_unix_ns": 20,
                }), encoding="utf-8")
                events.append("final_capture_completed")
                return None
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Pipeline)
    monkeypatch.setattr(module, "_start_recorder",
                        lambda **_kwargs: events.append("recorder_started"))
    monkeypatch.setattr(
        module, "_stop_recorder",
        lambda **_kwargs: (events.append("recorder_stopped") or {
            "schema": "precision_insertion_multiview_recording_manifest_v1",
            "camera_ids": ["100", "101", "102", "103"],
            "views": {}, "missing_camera_ids": []}))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module.run("--run") == 0
    assert events == [
        "pipeline_started", "execution_ready", "recorder_started",
        "final_capture_completed", "recorder_stopped"]
    receipt = json.loads((receipt_root / "20261030_123456_test" /
                          "recording_ref.json").read_text())
    assert receipt["recording_policy"] == (
        "after_planning_before_first_motion_until_final_capture_before_vlm")
    assert receipt["camera_ids"] == ["100", "101", "102", "103"]
    assert receipt["recording_trigger"]["candidate_id"] == "table/0/20195"
    assert receipt["recording_started_unix_ns"] is not None
    assert receipt["recording_stop_trigger"]["candidate_id"] == (
        "table/0/20195")
    assert receipt["recording_stopped_unix_ns"] is not None
    assert receipt["recording_stop_reason"] == (
        "final_capture_completed_before_vlm_label")
    assert receipt["pipeline_log"]["path"] == "pipeline.log"
    assert receipt["pipeline_log"]["size_bytes"] > 0
    assert len(receipt["pipeline_log"]["sha256"]) == 64
    assert (receipt_root / "20261030_123456_test" /
            "pipeline.log").read_text() == (
                "pipeline started\npipeline finished\n")


def test_failed_plan_preserves_pipeline_diagnostic(tmp_path, monkeypatch):
    module = _module()
    receipt_root = tmp_path / "receipts"
    trigger_root = tmp_path / "triggers"
    monkeypatch.setenv("PRECISION_RUN_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("PRECISION_RECORDING_RECEIPT_ROOT", str(receipt_root))
    monkeypatch.setenv("PRECISION_RECORDING_TRIGGER_ROOT", str(trigger_root))
    monkeypatch.setenv("PRECISION_RUN_TAG", "20261127_failed_plan")
    monkeypatch.setenv("PRECISION_CODE_COMMIT", "b" * 40)

    class FailedPipeline:
        def __init__(self, _argv, *, env, **_kwargs):
            self.stdout = io.StringIO(
                "planning initialized\nplanner failed: no solution\n")

        def poll(self):
            return 1

    monkeypatch.setattr(module.subprocess, "Popen", FailedPipeline)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert module.run("--plan-only") == 1
    run_dir = receipt_root / "20261127_failed_plan"
    receipt = json.loads((run_dir / "recording_ref.json").read_text())
    assert receipt["pipeline_exit_code"] == 1
    assert receipt["recording"] is None
    assert receipt["pipeline_log"]["path"] == "pipeline.log"
    assert (run_dir / "pipeline.log").read_text() == (
        "planning initialized\nplanner failed: no solution\n")
