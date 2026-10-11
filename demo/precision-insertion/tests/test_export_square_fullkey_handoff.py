"""The four-gap square handoff rejects changed or extra payload bytes."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_square_fullkey_handoff import SCHEMA, verify  # noqa: E402
from export_square_tabletop_handoff import _sha  # noqa: E402


def test_fullkey_bundle_verifier_rejects_tamper_and_extra_files(tmp_path):
    bundle = tmp_path / "square_handoff"
    payload = bundle / "payload/shared_data/object_processing/key/raw_mesh/key.obj"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"mesh")
    manifest = {
        "schema": SCHEMA, "modes": [], "robot_ready": False,
        "file_sha256": {str(payload.relative_to(bundle)): _sha(payload)},
    }
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify(bundle)["schema"] == SCHEMA
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file set"):
        verify(bundle)
    payload.write_bytes(b"mesh")
    (bundle / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify(bundle)
