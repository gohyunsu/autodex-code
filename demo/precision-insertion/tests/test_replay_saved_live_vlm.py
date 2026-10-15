"""Read-only live-checkpoint replay rejects missing or changed evidence first."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "replay_saved_live_vlm.py"
SPEC = importlib.util.spec_from_file_location("replay_saved_live_vlm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_source_requires_unchanged_absolute_bytes(tmp_path):
    source = tmp_path / "record.json"
    source.write_bytes(b"{}")
    ref = {"path": str(source),
           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert replay._source(ref) == source
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        replay._source(ref)


def test_missing_checkpoint_fails_before_any_model_load(tmp_path, monkeypatch):
    def forbidden(**_kwargs):
        raise AssertionError("model must not load")
    monkeypatch.setattr(replay, "load_vlm_backend", forbidden)
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        replay.run(argparse.Namespace(
            output_root=output, released_report=tmp_path / "missing.json",
            xy_input=None, model_id=tmp_path / "model", repeats=4,
            max_new_tokens=512))
    assert not output.exists()
