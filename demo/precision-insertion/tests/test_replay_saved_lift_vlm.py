from pathlib import Path
import json
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import replay_saved_lift_vlm as replay  # noqa: E402
from replay_saved_lift_vlm import (  # noqa: E402
    _latest_lift_reports, _runtime_vlm, _summary,
)


def test_replay_summary_keeps_unobservable_separate_from_errors():
    result = _summary([
        {"truth": "held", "prediction": "held"},
        {"truth": "held", "prediction": "miss"},
        {"truth": "miss", "prediction": "held"},
        {"truth": "miss", "prediction": "unobservable"},
    ])
    assert result["count"] == 4
    assert result["correct"] == 1
    assert result["false_held"] == 1
    assert result["false_miss"] == 1
    assert result["unobservable"] == 1


@pytest.mark.parametrize("mode", ["local", "distributed"])
def test_runtime_vlm_accepts_current_backend_modes(tmp_path, mode):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({
        "vlm": {"mode": mode, "model_id": "/model",
                "max_new_tokens": 128}}))
    vlm, digest = _runtime_vlm(path)
    assert vlm["mode"] == mode
    assert len(digest) == 64


def test_latest_lift_report_selects_one_per_attempt(tmp_path):
    first = tmp_path / "attempt_a" / "lift_assessments"
    second = tmp_path / "attempt_b" / "lift_assessments"
    for root, indexes in ((first, ("000", "001")), (second, ("000",))):
        for index in indexes:
            report = root / index / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}")
    assert _latest_lift_reports(tmp_path) == [
        first / "001" / "report.json", second / "000" / "report.json"]


def test_open_backend_allows_cpu_only_when_explicit(tmp_path, monkeypatch):
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(replay, "load_vlm_backend", fake_loader)
    vlm = {"mode": "local", "model_id": "/model", "max_new_tokens": 8,
           "require_native_pixels": True}
    replay._open_backend(
        vlm, ceiling=(2048, 1536), output=tmp_path / "out.json",
        model_override=None, allow_cpu=True)
    assert calls[-1]["require_cuda"] is False

    replay._open_backend(
        vlm, ceiling=(2048, 1536), output=tmp_path / "out.json",
        model_override=None)
    assert calls[-1]["require_cuda"] is True
