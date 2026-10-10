"""The square bundle manifest binds exactly the packaged file set."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_square_tabletop_handoff import SCHEMA, _sha, verify  # noqa: E402


def test_square_bundle_verifier_rejects_tamper_and_extra_manifest(tmp_path):
    root = tmp_path / "handoff"
    file = root / "payload/shared_data/asset.obj"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"CAD")
    manifest = {
        "schema": SCHEMA, "candidate_ids": ["table/4/104"],
        "robot_ready": False,
        "file_sha256": {"payload/shared_data/asset.obj": _sha(file)},
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify(root)["candidate_ids"] == ["table/4/104"]
    nested = root / "audit_only/MANIFEST.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify(root)
    nested.unlink()
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file set"):
        verify(root)
