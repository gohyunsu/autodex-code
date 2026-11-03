"""Metric ChArUco board construction and detection without ParaDex globals."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


_DICTIONARIES = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def load_board(path: str | Path):
    """Return the board JSON, metric OpenCV board and detector."""
    path = Path(path).expanduser().resolve()
    info = json.loads(path.read_text(encoding="utf-8"))
    dictionary_name = info["dict_type"]
    if dictionary_name not in _DICTIONARIES:
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[dictionary_name])
    marker_ids = np.asarray(info["markerIDs"], dtype=np.int32)
    expected_markers = (int(info["numX"]) * int(info["numY"])) // 2
    if len(marker_ids) != expected_markers:
        raise ValueError(
            f"board needs {expected_markers} marker IDs, got {len(marker_ids)}")
    board = cv2.aruco.CharucoBoard(
        (int(info["numX"]), int(info["numY"])),
        float(info["checkerLengthM"]), float(info["markerLengthM"]),
        dictionary, marker_ids)
    board.setLegacyPattern(bool(info.get("setLegacyPattern", False)))
    return info, board, cv2.aruco.CharucoDetector(board)


def detect(detector, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return full-resolution ChArUco pixels and board-local corner IDs."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    return (np.asarray(corners, dtype=np.float32).reshape(-1, 2),
            np.asarray(ids, dtype=np.int32).reshape(-1))


def draw_detection(image_bgr: np.ndarray, corners: np.ndarray,
                   ids: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if len(ids):
        cv2.aruco.drawDetectedCornersCharuco(
            overlay, corners.reshape(-1, 1, 2), ids.reshape(-1, 1),
            cornerColor=(0, 255, 0))
    return overlay
