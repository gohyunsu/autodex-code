"""Selected full-key v8 reset cells are staged without touching sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion import reorient_handoff as handoff  # noqa: E402


def _source(tmp_path, mode, *, success=True):
    shared = tmp_path / "shared"
    shared.mkdir()
    source = tmp_path / "source"
    seed = (source / "reset_12" / mode.key_object / "reorient_12" /
            "0_4" / "7")
    seed.mkdir(parents=True)
    (seed / "sim_eval.json").write_text(
        json.dumps({"success": success}), encoding="utf-8")
    (seed / "source_evidence.json").write_text(
        json.dumps({"source": "test"}), encoding="utf-8")
    (seed / "wrist_se3.npy").write_bytes(b"verified fixture bytes")
    return shared, source, seed


def test_verified_reset_seed_is_copied_and_rechecked(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    shared, source, seed = _source(tmp_path, mode)
    called = []

    def verify(**kwargs):
        called.append(kwargs["candidate_root"])
        path = (kwargs["candidate_root"] / mode.key_object /
                "reorient_12/0_4/7/source_evidence.json")
        return {"source_evidence_sha256": hashlib.sha256(
            path.read_bytes()).hexdigest()}

    monkeypatch.setattr(handoff, "verify_v8_reset_seed", verify)
    monkeypatch.setattr(handoff, "audit_v8_reorient_assets", lambda **_kwargs: {
        "directed_pairs": [{
            "from_v8_pose": 0, "to_v8_pose": 4,
            "stable_reset_seed_counts_by_height_cm": {"12": 1},
        }],
    })
    output = tmp_path / "new_handoff"
    report = handoff.assemble_v8_reorient_handoff(
        shared_root=shared, mode=mode,
        sources_by_cell={(12, 0, 4): source}, output_root=output)
    copied = (output / "reset_12" / mode.key_object /
              "reorient_12/0_4/7")
    assert report["candidate_count"] == 1
    assert (copied / "wrist_se3.npy").read_bytes() == (
        seed / "wrist_se3.npy").read_bytes()
    assert len(called) == 2  # source, then destination
    assert report["robot_ready"] is False
    with pytest.raises(FileExistsError):
        handoff.assemble_v8_reorient_handoff(
            shared_root=shared, mode=mode,
            sources_by_cell={(12, 0, 4): source}, output_root=output)


def test_no_pass_and_source_nested_output_leave_no_handoff(tmp_path):
    mode = select_mode("square", 1.5)
    shared, source, _seed = _source(tmp_path, mode, success=False)
    output = tmp_path / "new_handoff"
    with pytest.raises(ValueError, match="no verified pass"):
        handoff.assemble_v8_reorient_handoff(
            shared_root=shared, mode=mode,
            sources_by_cell={(12, 0, 4): source}, output_root=output)
    assert not output.exists()
    with pytest.raises(ValueError, match="inside a source tree"):
        handoff.assemble_v8_reorient_handoff(
            shared_root=shared, mode=mode,
            sources_by_cell={(12, 0, 4): source},
            output_root=source / "nested")


def test_resolved_canonical_nas_symlink_is_never_a_handoff_target(tmp_path):
    mode = select_mode("square", 1.5)
    shared, source, _seed = _source(tmp_path, mode)
    canonical_target = tmp_path / "nas" / "reset_12"
    canonical_target.mkdir(parents=True)
    link = shared / "AutoDex/candidates/inspire/reset_12"
    link.parent.mkdir(parents=True)
    link.symlink_to(canonical_target, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot write into canonical"):
        handoff.assemble_v8_reorient_handoff(
            shared_root=shared, mode=mode,
            sources_by_cell={(12, 0, 4): source},
            output_root=canonical_target / "precision_key_new")
