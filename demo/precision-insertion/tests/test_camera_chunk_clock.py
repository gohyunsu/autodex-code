"""Raw chunk ticks need independent UTC fit and held-out validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_chunk_clock import (  # noqa: E402
    ChunkUTCClock, evaluate_chunk_utc_calibration,
)
from precision_insertion.camera_chunk_tap import (  # noqa: E402
    ChunkTimestampJournal,
)
from precision_insertion.camera_time import AcquisitionTimeProvider  # noqa: E402
from precision_insertion.camera_transport import FrameIdentityRegistry  # noqa: E402
from test_camera_chunk_tap import _Image  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(tmp_path):
    journal_path = tmp_path / "camera.jsonl"
    journal = ChunkTimestampJournal(journal_path, camera_serial="cam1")
    origin_ticks = 10**15
    for frame_id in range(1, 13):
        journal.record(_Image(frame_id, origin_ticks + frame_id * 10_000_000))
    journal.close()
    source = tmp_path / "independent_trigger_log.csv"
    source.write_text("synthetic external UTC reference\n", encoding="utf-8")
    base = 1_800_000_000.
    def reference(fid):
        return {"frame_id": fid, "exposure_utc_s": base + fid * .01,
                "max_error_s": .0001}
    record = {
        "schema": "precision_insertion_chunk_utc_calibration_v1",
        "calibration_id": "synthetic_cam1_a", "camera_serial": "cam1",
        "chunk_journal": {"path": str(journal_path),
                          "sha256": _sha(journal_path)},
        "source_method": "independent_per_camera_trigger_metrology",
        "source_files": [{"path": str(source), "sha256": _sha(source)}],
        "tick_frequency_hz": 1_000_000_000,
        "max_clock_rate_error_ppm": 100,
        "fit_samples": [reference(fid) for fid in (1, 3, 5)],
        "validation_samples": [reference(fid) for fid in (2, 4, 6)],
        "clock_offset_error_s": .0002,
        "drift_error_s_per_s": .00001,
        "max_total_error_s": .001,
        "max_extrapolation_s": .05,
        "valid_until_utc_s": base + 1.,
    }
    path = tmp_path / "utc_fit.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, journal_path, source, record


def test_chunk_clock_fits_independent_utc_and_replays_journal(tmp_path):
    path, _journal, _source, _record = _setup(tmp_path)
    clock = ChunkUTCClock.load(path)
    stamp, error = clock.exposure(8)
    assert stamp == pytest.approx(1_800_000_000.08, abs=1e-6)
    assert .00029 <= error <= .00031
    with pytest.raises(ValueError, match="extrapolation"):
        clock.exposure(12)
    audit = evaluate_chunk_utc_calibration(path)
    assert audit["journaled_frames"] == 12
    assert audit["robot_ready"] is False
    with pytest.raises(ValueError, match="CameraTimeCalibration records"):
        AcquisitionTimeProvider(FrameIdentityRegistry(), {"cam1": clock})


def test_chunk_clock_rejects_changed_or_fake_utc_sources(tmp_path):
    path, journal, source, record = _setup(tmp_path)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="independent UTC source changed"):
        ChunkUTCClock.load(path)
    source.write_text("synthetic external UTC reference\n", encoding="utf-8")
    record["source_method"] = "host_pc_time_after_GetNextImage"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid independent"):
        ChunkUTCClock.load(path)
    record["source_method"] = "independent_per_camera_trigger_metrology"
    record["source_files"] = [{"path": str(journal),
                               "sha256": _sha(journal)}]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="independent UTC source changed"):
        ChunkUTCClock.load(path)


def test_chunk_clock_rejects_overlapping_or_poor_validation(tmp_path):
    path, _journal, _source, record = _setup(tmp_path)
    record["validation_samples"][0]["frame_id"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        ChunkUTCClock.load(path)
    record["validation_samples"][0]["frame_id"] = 2
    record["validation_samples"][1]["exposure_utc_s"] += .01
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="error budget"):
        ChunkUTCClock.load(path)


def test_chunk_clock_rejects_expiry_and_postload_tampering(tmp_path):
    path, journal, source, record = _setup(tmp_path)
    record["valid_until_utc_s"] = 1_800_000_000.04
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="expires before validation"):
        ChunkUTCClock.load(path)
    record["valid_until_utc_s"] = 1_800_000_001.
    path.write_text(json.dumps(record), encoding="utf-8")
    clock = ChunkUTCClock.load(path)
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="source files changed"):
        clock.exposure(4)
    source.write_text("synthetic external UTC reference\n", encoding="utf-8")
    with journal.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source files changed"):
        clock.exposure(4)
