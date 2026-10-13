"""A partly hidden board 9 still needs multi-view geometric agreement."""

import numpy as np
import pytest

from src.execution import charuco_tabletop as tabletop


def test_board_9_can_use_distributed_visible_corners_without_weakening_11(
        monkeypatch):
    model = np.array([[x * .05, y * .05, 0.]
                      for y in range(6) for x in range(9)])
    hidden = {15, 16, 23, 24, 25, 26, 32, 33, 34, 35, 42, 43}
    ids = np.array(sorted(set(range(54)) - hidden), dtype=np.int32)
    K = np.array([[1000., 0., 500.], [0., 1000., 400.], [0., 0., 1.]])
    extrinsics = {}
    pixels = {}
    images = {}
    for index, serial in enumerate(("cam_a", "cam_b")):
        ext = np.eye(4)
        ext[:3, 3] = [index * .2, 0., 1.]
        extrinsics[serial] = ext
        camera_points = model[ids] @ ext[:3, :3].T + ext[:3, 3]
        projected = camera_points @ K.T
        pixels[index] = projected[:, :2] / projected[:, 2:]
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0, 0] = index
        images[serial] = image

    def detect(image):
        return {"9": {"checkerIDs": ids,
                      "checkerCorner": pixels[int(image[0, 0, 0])]}}

    monkeypatch.setattr(tabletop, "_board_corner_model", lambda _id: model)
    monkeypatch.setattr("paradex.image.aruco.detect_charuco", detect)
    intrinsics = {serial: {"K_orig": K, "K_undist": K,
                           "dist_params": np.zeros(5)} for serial in images}
    result = tabletop.measure_tabletop_from_images(
        images, intrinsics, extrinsics, np.eye(4),
        board_id="9", minimum_board_corners=36)
    assert result["board"] == "9"
    assert result["source"] == "charuco_partial_board_preflight"
    assert result["metrics"]["corners_triangulated"] == 42
    assert result["metrics"]["missing_corner_ids"] == sorted(hidden)
    assert result["metrics"]["rigid_fit_rms_mm"] < 1e-5
    with pytest.raises(RuntimeError, match="needs at least 54"):
        tabletop.measure_tabletop_from_images(
            images, intrinsics, extrinsics, np.eye(4), board_id="9")
    with pytest.raises(ValueError, match="all 54"):
        tabletop.measure_tabletop_from_images(
            images, intrinsics, extrinsics, np.eye(4),
            board_id="11", minimum_board_corners=36)

    # A count alone cannot certify board registration if all visible corners
    # are clustered along one side of the socket.
    clustered = np.array([i for i in range(54) if i % 9 < 6], dtype=np.int32)
    def detect_clustered(image):
        index = int(image[0, 0, 0])
        ext = extrinsics[("cam_a", "cam_b")[index]]
        projected = (model[clustered] + ext[:3, 3]) @ K.T
        return {"9": {"checkerIDs": clustered,
                      "checkerCorner": projected[:, :2] / projected[:, 2:]}}
    monkeypatch.setattr("paradex.image.aruco.detect_charuco", detect_clustered)
    with pytest.raises(RuntimeError, match="do not span the board"):
        tabletop.measure_tabletop_from_images(
            images, intrinsics, extrinsics, np.eye(4),
            board_id="9", minimum_board_corners=36)
