"""The project board profile must be self-contained and detector-consistent."""

from pathlib import Path

import cv2

from calibration.charuco import detect, load_board


def test_board11_50mm_profile_detects_all_generated_corners():
    config = (Path(__file__).resolve().parents[1] / "config" /
              "zerodex_charuco_board11_50mm.json")
    info, board, detector = load_board(config)
    assert info["checkerLengthM"] == 0.05
    assert info["markerLengthM"] == 0.03
    assert info["markerIDs"] == list(range(70, 105))
    gray = board.generateImage((1200, 840), marginSize=30)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    corners, ids = detect(detector, image)
    assert corners.shape == (54, 2)
    assert ids.tolist() == list(range(54))


def test_existing_handeye_board2_profile_detects_all_generated_corners():
    config = (Path(__file__).resolve().parents[1] / "config" /
              "zerodex_charuco_handeye_board2_60mm.json")
    info, board, detector = load_board(config)
    assert info["checkerLengthM"] == 0.06
    assert info["markerLengthM"] == 0.05
    assert info["markerIDs"] == list(range(15, 30))
    assert info["setLegacyPattern"] is True
    gray = board.generateImage((750, 900), marginSize=30)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    corners, ids = detect(detector, image)
    assert corners.shape == (20, 2)
    assert ids.tolist() == list(range(20))
