"""Chunk records come from the exact unreleased image, never receipt time."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_chunk_tap import (  # noqa: E402
    ChunkTimestampJournal, TimestampedCameraPointer, verify_chunk_journal,
)


class _Chunk:
    def __init__(self, ticks):
        self.ticks = ticks

    def GetTimestamp(self):
        return self.ticks


class _Image:
    def __init__(self, frame_id, ticks):
        self.frame_id, self.ticks = frame_id, ticks
        self.released = False

    def GetFrameID(self):
        return self.frame_id

    def GetChunkData(self):
        return _Chunk(self.ticks)

    def Release(self):
        self.released = True


class _Camera:
    def __init__(self, images):
        self.images = iter(images)

    def GetNextImage(self, *args):
        return next(self.images)

    def BeginAcquisition(self):
        return "unchanged"

    def EndAcquisition(self):
        return "stopped"


def test_proxy_preserves_image_and_journals_same_frame(tmp_path):
    path = tmp_path / "cam_1.jsonl"
    journal = ChunkTimestampJournal(path, camera_serial="cam1")
    first, second = _Image(7, 3000), _Image(8, 4000)
    proxy = TimestampedCameraPointer(_Camera([first, second]), journal)
    assert proxy.BeginAcquisition() == "unchanged"
    assert proxy.GetNextImage(500) is first
    assert proxy.GetNextImage(500) is second
    assert first.released is second.released is False
    journal.close()
    audited = verify_chunk_journal(path, camera_serial="cam1")
    assert audited["frame_count"] == 2
    assert audited["first_frame_id"] == 7
    assert audited["last_chunk_ticks"] == 4000
    assert audited["exposure_utc_admissible"] is False
    with pytest.raises(FileExistsError):
        ChunkTimestampJournal(path, camera_serial="cam1")


@pytest.mark.parametrize("frame_id,ticks", [(0, 100), (2, 0), (2, 100)])
def test_proxy_releases_image_if_chunk_metadata_is_invalid(
        tmp_path, frame_id, ticks):
    path = tmp_path / "cam.jsonl"
    journal = ChunkTimestampJournal(path, camera_serial="cam1")
    first = _Image(2, 100)
    bad = _Image(frame_id, ticks)
    proxy = TimestampedCameraPointer(_Camera([first, bad]), journal)
    proxy.BeginAcquisition()
    assert proxy.GetNextImage() is first
    with pytest.raises(ValueError):
        proxy.GetNextImage()
    assert bad.released is True
    journal.close()
    assert verify_chunk_journal(path, camera_serial="cam1")["frame_count"] == 1


def test_journal_rejects_nonmonotonic_or_corrupt_saved_rows(tmp_path):
    path = tmp_path / "cam.jsonl"
    journal = ChunkTimestampJournal(path, camera_serial="cam1")
    journal.record(_Image(1, 100))
    with pytest.raises(ValueError, match="did not advance"):
        journal.record(_Image(2, 99))
    journal.close()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "frame_id": 1, "chunk_timestamp_ticks": 200,
            "host_received_utc_s": 5.0}) + "\n")
    with pytest.raises(ValueError, match="nonmonotonic"):
        verify_chunk_journal(path, camera_serial="cam1")


def test_numpy_integral_vendor_values_are_recorded_exactly(tmp_path):
    path = tmp_path / "cam.jsonl"
    journal = ChunkTimestampJournal(path, camera_serial="cam1")
    journal.record(_Image(np.int64(3), np.uint64(10**15)))
    journal.close()
    audit = verify_chunk_journal(path, camera_serial="cam1")
    assert audit["last_frame_id"] == 3
    assert audit["last_chunk_ticks"] == 10**15


def test_buffer_drain_after_stop_is_not_journaled(tmp_path):
    path = tmp_path / "cam.jsonl"
    journal = ChunkTimestampJournal(path, camera_serial="cam1")
    used, discarded = _Image(10, 1000), _Image(11, 1100)
    proxy = TimestampedCameraPointer(_Camera([used, discarded]), journal)
    proxy.BeginAcquisition()
    assert proxy.GetNextImage() is used
    assert proxy.EndAcquisition() == "stopped"
    assert proxy.GetNextImage(1) is discarded
    journal.close()
    assert verify_chunk_journal(path, camera_serial="cam1")["frame_count"] == 1


def test_restarted_acquisition_gets_new_journal_and_fresh_id_epoch(tmp_path):
    paths = []

    def new_journal():
        path = tmp_path / f"epoch_{len(paths)}.jsonl"
        paths.append(path)
        return ChunkTimestampJournal(path, camera_serial="cam1")

    proxy = TimestampedCameraPointer(
        _Camera([_Image(1, 100), _Image(1, 100)]),
        journal_factory=new_journal)
    for _ in range(2):
        proxy.BeginAcquisition()
        proxy.GetNextImage()
        proxy.EndAcquisition()
    assert len(paths) == 2
    assert all(verify_chunk_journal(path, camera_serial="cam1")[
        "frame_count"] == 1 for path in paths)
