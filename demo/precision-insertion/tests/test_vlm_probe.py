"""Saved-image local/Gemini mode smoke test without model download/API call."""

import json
from pathlib import Path
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_vlm  # noqa: E402


def test_local_probe_uses_selected_backend_and_saves_read_only_record(
        tmp_path, monkeypatch):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    Image.new("RGB", (24, 16), "blue").save(before)
    Image.new("RGB", (24, 16), "red").save(after)
    calls = []

    class Backend:
        def infer(self, images, prompt):
            calls.append((images, prompt))
            return json.dumps({"class": "held", "evidence_views": ["front"],
                               "evidence": "visible object moves with hand"})

    def load(**kwargs):
        calls.append(kwargs)
        return Backend()

    monkeypatch.setattr(probe_vlm, "load_vlm_backend", load)
    report = tmp_path / "probe.json"
    assert probe_vlm.main([
        "--backend", "local", "--model-id", "Qwen/test",
        "--task", "lift", "--view", "front", str(before), str(after),
        "--output", str(report)]) == 0
    saved = json.loads(report.read_text())
    assert saved["backend"] == "local"
    assert saved["robot_ready"] is False
    assert saved["timestamp_source"] == (
        "synthetic_order_only_not_camera_acquisition")
    assert saved["observation"]["parsed"]["class"] == "held"
    assert calls[0]["require_native_pixels"] is False
    assert len(calls[1][0]) == 2
    with pytest.raises(FileExistsError):
        probe_vlm.main([
            "--backend", "local", "--model-id", "Qwen/test",
            "--task", "lift", "--view", "front", str(before), str(after),
            "--output", str(report)])


def test_gemini_probe_requires_explicit_external_image_opt_in(tmp_path):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    Image.new("RGB", (10, 10)).save(before)
    Image.new("RGB", (10, 10)).save(after)
    with pytest.raises(SystemExit, match="2"):
        probe_vlm.main([
            "--backend", "gemini", "--model-id", "test-model",
            "--task", "lift", "--view", "front", str(before), str(after),
            "--output", str(tmp_path / "probe.json")])
