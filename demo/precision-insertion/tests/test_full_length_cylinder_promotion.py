"""Append-only promotion preserves source evidence and prior candidate IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promote_full_length_cylinder_candidates import FILES, install  # noqa: E402


def test_install_is_append_only_and_binds_source_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in FILES:
        (source / name).write_bytes(name.encode())
    destination = tmp_path / "candidates/table/0/20017"
    old = tmp_path / "candidates/table/0/17"
    old.mkdir(parents=True)
    (old / "keep.txt").write_text("historical", encoding="utf-8")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in FILES}
    row = {
        "source": source, "destination": destination,
        "file_sha256": hashes,
        "eligible_gaps": ["20mm"],
        "endpoint_report_sha256": {"20mm": "endpoint-hash"},
    }
    report = {
        "full_key_planning_mesh_sha256": "full-key-mesh-hash",
        "screen_summary_sha256": "summary-hash",
    }
    manifest = tmp_path / "promotion.json"
    install(report, [row], manifest)
    assert manifest.is_file() and (old / "keep.txt").read_text() == "historical"
    assert {name: (destination / name).read_bytes() for name in FILES} == {
        name: (source / name).read_bytes() for name in FILES}
    validation = json.loads((destination / "simulation_validation.json").read_text())
    assert validation["source_file_sha256"] == hashes
    assert validation["eligible_socket_gaps"] == ["20mm"]
    assert validation["physical_validation"] is False
    with pytest.raises(FileExistsError):
        install(report, [row], manifest)
