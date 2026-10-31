import numpy as np
import pytest

from precision_insertion.board_view_selection import (
    measure_tabletop_with_view_selection,
)


def _measurement(cameras):
    return {
        "board": "9",
        "metrics": {
            "reprojection_median_px": len(cameras) / 10,
            "rigid_fit_rms_mm": 0.2,
            "plane_rms_mm": 0.1,
        },
    }


def test_all_views_remain_the_fast_path(monkeypatch):
    calls = []

    def measure(images, *_args, **_kwargs):
        calls.append(tuple(sorted(images)))
        return _measurement(images)

    monkeypatch.setattr(
        "src.execution.charuco_tabletop.measure_tabletop_from_images", measure)
    images = {camera: np.zeros((2, 2, 3), dtype=np.uint8)
              for camera in ("a", "b", "c", "d")}
    result = measure_tabletop_with_view_selection(
        images, {}, {}, np.eye(4), board_id="9", minimum_board_corners=36)
    assert calls == [("a", "b", "c", "d")]
    assert result["camera_view_selection"]["selected_camera_ids"] == [
        "a", "b", "c", "d"]
    assert result["camera_view_selection"]["excluded_camera_ids"] == []


def test_failed_all_view_fit_selects_largest_consistent_subset(monkeypatch):
    def measure(images, *_args, **_kwargs):
        cameras = set(images)
        if "bad" in cameras:
            raise RuntimeError("reprojection mismatch")
        return _measurement(images)

    monkeypatch.setattr(
        "src.execution.charuco_tabletop.measure_tabletop_from_images", measure)
    images = {camera: np.zeros((2, 2, 3), dtype=np.uint8)
              for camera in ("a", "b", "c", "bad")}
    result = measure_tabletop_with_view_selection(
        images, {}, {}, np.eye(4), board_id="9", minimum_board_corners=36)
    selection = result["camera_view_selection"]
    assert selection["method"] == "largest_stock_consistent_camera_subset"
    assert selection["selected_camera_ids"] == ["a", "b", "c"]
    assert selection["excluded_camera_ids"] == ["bad"]
    assert selection["scope"] == (
        "session_board_fit_not_persistent_camera_blacklist")


def test_original_all_view_error_is_preserved_when_no_subset_fits(monkeypatch):
    def measure(images, *_args, **_kwargs):
        raise RuntimeError("original board fit error: " + "".join(sorted(images)))

    monkeypatch.setattr(
        "src.execution.charuco_tabletop.measure_tabletop_from_images", measure)
    images = {camera: np.zeros((2, 2, 3), dtype=np.uint8)
              for camera in ("a", "b", "c")}
    with pytest.raises(RuntimeError, match="original board fit error: abc"):
        measure_tabletop_with_view_selection(
            images, {}, {}, np.eye(4), board_id="9", minimum_board_corners=36)
