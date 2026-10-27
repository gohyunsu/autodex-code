"""Checkpoint evidence remains byte-bound and pixel-lossless after encoding."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.raw_camera_capture import _write_png_hash  # noqa: E402


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
