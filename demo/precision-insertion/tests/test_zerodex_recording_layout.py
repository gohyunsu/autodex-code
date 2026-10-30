"""ZeroDex recording references remain separate from immutable assets."""

import importlib.util
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
