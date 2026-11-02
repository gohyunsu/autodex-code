from pathlib import Path
import json
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
