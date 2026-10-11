"""The FoundPose handoff is exclusive and verifies the archived payload."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import publish_foundpose_handoff as handoff  # noqa: E402


def test_publish_is_non_overwriting_and_read_back_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "MISSING", ("fake",))
    representation = (tmp_path / "shared/AutoDex/foundpose_assets/fake/"
                      "object_repre/v1/fake/1/repre.pth")
    representation.parent.mkdir(parents=True)
    representation.write_bytes(b"test representation")
    digest = hashlib.sha256(representation.read_bytes()).hexdigest()
    (representation.parents[4] / "ONBOARDING_PROVENANCE.json").write_text(
        json.dumps({"object": "fake", "physical_pose_qualified": False,
                    "representation_sha256": digest}))

    output = tmp_path / "handoff"
    result = handoff.publish(shared_root=tmp_path / "shared",
                             output_root=output)
    assert result["objects"][0]["representation_sha256"] == digest
    assert result["archive_bytes"] > len(b"test representation")
    with pytest.raises(FileExistsError):
        handoff.publish(shared_root=tmp_path / "shared", output_root=output)
