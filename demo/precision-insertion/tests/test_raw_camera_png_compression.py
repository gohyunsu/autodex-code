"""Checkpoint evidence remains byte-bound and pixel-lossless after encoding."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.raw_camera_capture import (  # noqa: E402
    _write_png_hash,
    write_png_jobs,
)


def test_fast_png_evidence_keeps_exact_pixels_and_encoded_hash(
        tmp_path, monkeypatch):
    source = np.random.default_rng(7).integers(
        0, 256, (24, 32, 3), dtype=np.uint8)
    path = tmp_path / "view.png"
    actual_encode = cv2.imencode
    parameters = []

    def encode(extension, image, params=None):
        parameters.append(params)
        return actual_encode(extension, image, params)

    monkeypatch.setattr(cv2, "imencode", encode)
    digest = _write_png_hash(path, source)
    assert parameters == [[cv2.IMWRITE_PNG_COMPRESSION, 1]]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_COLOR), source)


def test_saved_source_png_is_hard_linked_without_reencoding(tmp_path, monkeypatch):
    source = np.random.default_rng(9).integers(
        0, 256, (24, 32, 3), dtype=np.uint8)
    original = tmp_path / "raw.png"
    digest = _write_png_hash(original, source)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    target = evidence_dir / "view.png"
    monkeypatch.setattr(
        cv2, "imencode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hard-linked PNG must not be reencoded")))

    results, workers = write_png_jobs(
        [("cam", target, source, "undistorted", original)], phase="key")

    assert workers == 1
    assert results == [("cam", "undistorted", digest)]
    assert target.stat().st_ino == original.stat().st_ino
