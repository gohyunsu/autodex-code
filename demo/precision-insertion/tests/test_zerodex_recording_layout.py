"""ZeroDex recording references remain separate from immutable assets."""

import importlib.util
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


def test_recording_starts_only_after_preinsert_marker(
        tmp_path, monkeypatch):
    module = _module()
    receipt_root = tmp_path / "receipts"
    trigger_root = tmp_path / "triggers"
    monkeypatch.setenv("PRECISION_RUN_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("PRECISION_RECORDING_RECEIPT_ROOT", str(receipt_root))
    monkeypatch.setenv("PRECISION_RECORDING_TRIGGER_ROOT", str(trigger_root))
    monkeypatch.setenv("PRECISION_RUN_TAG", "20261030_123456_test")
    monkeypatch.setenv("PRECISION_CODE_COMMIT", "a" * 40)
    events = []

    class Pipeline:
        def __init__(self, _argv, *, env):
            self.env = env
            self.polls = 0
            events.append("pipeline_started")

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                marker = Path(self.env["PRECISION_PREINSERT_CONFIRMED_MARKER"])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({
                    "schema": "precision_insertion_preinsert_confirmed_v1",
                    "attempt_id": "attempt_1",
                    "candidate_id": "table/0/20195",
                    "confirmed_at_unix_ns": 10,
                }), encoding="utf-8")
                events.append("preinsert_confirmed")
                return None
            if self.polls == 2:
                marker = Path(self.env[
                    "PRECISION_RELEASE_CLEAR_COMPLETED_MARKER"])
                marker.write_text(json.dumps({
                    "schema": "precision_insertion_release_clear_completed_v1",
                    "attempt_id": "attempt_1",
                    "candidate_id": "table/0/20195",
                    "confirmed_at_unix_ns": 20,
                }), encoding="utf-8")
                events.append("release_clear_completed")
                return None
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Pipeline)
    monkeypatch.setattr(module, "_start_recorder",
                        lambda **_kwargs: events.append("recorder_started"))
    monkeypatch.setattr(
        module, "_stop_recorder",
        lambda **_kwargs: (events.append("recorder_stopped") or {
            "video": "26053248.avi", "frame_count": 1}))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module.run("--run") == 0
    assert events == [
        "pipeline_started", "preinsert_confirmed", "recorder_started",
        "release_clear_completed", "recorder_stopped"]
    receipt = json.loads((receipt_root / "20261030_123456_test" /
                          "recording_ref.json").read_text())
    assert receipt["recording_policy"] == (
        "after_preinsert_until_release_and_clear_completed")
    assert receipt["recording_trigger"]["candidate_id"] == "table/0/20195"
    assert receipt["recording_started_unix_ns"] is not None
    assert receipt["recording_stop_trigger"]["candidate_id"] == (
        "table/0/20195")
    assert receipt["recording_stopped_unix_ns"] is not None
    assert receipt["recording_stop_reason"] == "release_and_clear_completed"
