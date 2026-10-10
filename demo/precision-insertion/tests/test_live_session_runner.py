"""Opening a saved live session never bypasses source evidence or moves a robot."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion import live_session_runner as opening  # noqa: E402
from precision_insertion import session_runner  # noqa: E402


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    mode = select_mode("square", 1.5)
    for name in (mode.key_object, mode.socket_object):
        _write(root / "object_processing" / name / "raw_mesh" /
               f"{name}.obj", b"CAD")
        _write(root / "AutoDex/foundpose_assets" / name /
               "object_repre/v1" / name / "1/repre.pth", b"PTH")
    _write(root / "object_processing" / mode.socket_object /
           "processed_data/mesh/static_collision.obj", b"collision")
    _write(root / "AutoDex/content/assets/robot/fr3_inspire_description" /
           "fr3_inspire.urdf", b"robot")
    evidence = tmp_path / "session_evidence"
    _write(evidence / "evidence_manifest.json", b"{\"schema\":\"test\"}")
    _write(evidence / "session_calibration.json", b"{\"schema\":\"test\"}")
    catalog_file = _write(tmp_path / "catalog.json", json.dumps({
        "shared_root": str(root), "complete_scan": True,
        "candidates": [{"tabletop_pose_stem": "004"}],
    }).encode())
    calibration = SessionCalibration(
        board={}, socket_pose_robot=np.eye(4), socket_diagnostics={},
        collision_scene={}, record={"schema": "test_frozen_session"})
    events = []

    def verify(_path):
        events.append("verify_acquisition_bundle")
        return {"all_frames_bound_to_acquisition_evidence": True,
                "socket_capture_ids": ["socket_000", "socket_001"]}

    def load(_path, *, mode, shared_root):
        events.append("load_frozen_world")
        assert shared_root == root
        return calibration

    def validate(_catalog, *, mode, session_record):
        events.append("validate_v8_catalog")
        assert session_record is calibration.record

    monkeypatch.setattr(opening, "verify_session_evidence_bundle", verify)
    monkeypatch.setattr(opening, "load_session_calibration", load)
    monkeypatch.setattr(opening, "validate_catalog_session", validate)
    monkeypatch.setattr(session_runner, "validate_catalog_session", validate)
    monkeypatch.setattr(opening, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available"})
    args = {
        "mode": mode, "shared_root": root, "evidence_dir": evidence,
        "catalog_path": catalog_file, "output_dir": tmp_path / "run_001",
        "max_xy_retries": 2,
    }
    return args, events


def test_opening_binds_saved_evidence_and_frozen_catalog(tmp_path, monkeypatch):
    args, events = _setup(tmp_path, monkeypatch)
    opened = opening.open_verified_session(**args)
    assert events == ["verify_acquisition_bundle", "load_frozen_world",
                      "validate_v8_catalog", "validate_v8_catalog"]
    assert opened.next_decision.action == "capture_fresh_key"
    record = json.loads(opened.source_binding_path.read_text())
    assert record["robot_ready"] is False
    assert record["session_evidence_dir"] == str(args["evidence_dir"])
    assert record["catalog_source_path"] == str(args["catalog_path"])
    assert record["key_foundpose_sha256"] == record["socket_foundpose_sha256"]
    assert (opened.runner.output_dir / "frozen_session_calibration.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        opening.open_verified_session(**args)


def test_missing_pth_or_unbound_camera_evidence_fails_before_run(
        tmp_path, monkeypatch):
    args, events = _setup(tmp_path, monkeypatch)
    key = args["mode"].key_object
    pth = (args["shared_root"] / "AutoDex/foundpose_assets" / key /
           "object_repre/v1" / key / "1/repre.pth")
    pth.unlink()
    with pytest.raises(FileNotFoundError, match="key FoundPose"):
        opening.open_verified_session(**args)
    assert events == []
    assert not args["output_dir"].exists()
    _write(pth, b"PTH")
    monkeypatch.setattr(opening, "verify_session_evidence_bundle",
                        lambda _path: {
                            "all_frames_bound_to_acquisition_evidence": False,
                            "socket_capture_ids": ["socket_000", "socket_001"]})
    with pytest.raises(ValueError, match="acquisition-bound"):
        opening.open_verified_session(**args)
    assert not args["output_dir"].exists()


def test_wrong_root_catalog_fails_before_session_ledger(tmp_path, monkeypatch):
    args, _events = _setup(tmp_path, monkeypatch)
    args["catalog_path"].write_text(json.dumps({
        "shared_root": str(tmp_path / "other"), "complete_scan": True,
        "candidates": [{"tabletop_pose_stem": "004"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="another shared root"):
        opening.open_verified_session(**args)
    assert not args["output_dir"].exists()


def test_stale_candidate_catalog_fails_before_session_ledger(tmp_path, monkeypatch):
    args, _events = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(opening, "select_pose_candidates",
                        lambda *_args, **_kwargs: {"status": "catalog_stale"})
    with pytest.raises(ValueError, match="stale for pose 004"):
        opening.open_verified_session(**args)
    assert not args["output_dir"].exists()
