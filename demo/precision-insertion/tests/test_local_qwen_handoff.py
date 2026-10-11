"""Standalone local-Qwen handoff preserves bytes and rejects stale copies."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_local_qwen_handoff import export, verify  # noqa: E402


def test_export_dereferences_cache_links_and_detects_tamper(tmp_path: Path):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    data = {
        "config.json": json.dumps({"model_type": "qwen3_vl"}).encode(),
        "model.safetensors": b"small-test-model",
        "preprocessor_config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "generation_config.json": b"{}",
    }
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name, content in data.items():
        blob = blobs / name
        blob.write_bytes(content)
        (snapshot / name).symlink_to(blob)

    output = tmp_path / "handoff"
    exported = export(snapshot, output)
    assert exported["file_count"] == len(data)
    assert verify(output) == exported
    assert not (output / "payload/model/model.safetensors").is_symlink()
    with pytest.raises(FileExistsError):
        export(snapshot, output)
    (output / "payload/model/tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        verify(output)


def test_export_rejects_wrong_model(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in (
        "config.json", "model.safetensors", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not the configured"):
        export(snapshot, tmp_path / "handoff")
