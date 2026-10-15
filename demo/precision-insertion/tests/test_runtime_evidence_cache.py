"""Live producer evidence is reused only within the current session."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.key_perception import verify_key_capture_artifacts  # noqa: E402
from precision_insertion.raw_camera_capture import verify_raw_camera_capture  # noqa: E402
from precision_insertion.runtime_evidence_cache import (  # noqa: E402
    live_evidence_cache, remember,
)
from precision_insertion import path_audit  # noqa: E402


def test_live_manifest_reuse_is_scoped_and_offline_verifiers_remain_strict(tmp_path):
    raw = tmp_path / "raw"
    key = tmp_path / "key"
    raw_manifest = {"schema": "precision_insertion_raw_capture_v1",
                    "phase": "preinsert"}
    key_manifest = {"schema": "precision_insertion_key_capture_evidence_v1"}
    with live_evidence_cache():
        remember("raw_camera", raw, "preinsert", raw_manifest)
        remember("key_capture", key, "key", key_manifest)
        assert verify_raw_camera_capture(raw, phase="preinsert") is raw_manifest
        assert verify_key_capture_artifacts(key) is key_manifest
    with pytest.raises(FileNotFoundError):
        verify_raw_camera_capture(raw, phase="preinsert")
    with pytest.raises(FileNotFoundError):
        verify_key_capture_artifacts(key)


def test_fixed_world_models_are_built_once_per_frozen_session(tmp_path, monkeypatch):
    class Frozen:
        pass

    class Mode:
        family = "cylinder"
        gap_mm = 20

    builds = []

    def build(calibration, *, mode, shared_root):
        builds.append(calibration)
        return ({"fixed": object()}, {"source": "once"})

    monkeypatch.setattr(path_audit, "_build_fixed_world_models", build)
    first, second = Frozen(), Frozen()
    a = path_audit._fixed_world_models(first, mode=Mode(), shared_root=tmp_path)
    assert path_audit._fixed_world_models(
        first, mode=Mode(), shared_root=tmp_path) is a
    path_audit._fixed_world_models(second, mode=Mode(), shared_root=tmp_path)
    assert builds == [first, second]
