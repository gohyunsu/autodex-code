"""Manifest read-back rejects changed bytes and unsafe source paths."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_handoff_manifest import verify  # noqa: E402


def test_declared_handoff_bytes_and_tamper(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    item = payload / "candidate.bin"
    item.write_bytes(b"v8 seed")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "schema": "test", "file_sha256": {
            "payload/candidate.bin": hashlib.sha256(item.read_bytes()).hexdigest()}}))
    assert verify(tmp_path)["files_verified"] == 1
    item.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        verify(tmp_path)


def test_generated_archive_and_path_traversal(tmp_path):
    archive = tmp_path / "foundpose.tar"
    archive.write_bytes(b"representation")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "test", "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256_read_back": hashlib.sha256(
            archive.read_bytes()).hexdigest()}))
    assert verify(tmp_path)["bytes_verified"] == len(b"representation")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "file_sha256": {"../outside": "0" * 64}}))
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify(tmp_path)
