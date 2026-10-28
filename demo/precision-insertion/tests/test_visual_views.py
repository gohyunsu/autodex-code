"""The live VLM uses the operator's exact camera selection."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.visual_views import (  # noqa: E402
    assessment_cameras, final_assessment_cameras,
    final_cameras_with_visible_socket,
)


def test_selected_triplet_is_used_instead_of_all_captured_views(monkeypatch):
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", "25322651,24122734,25305463")
    assert assessment_cameras({
        "25322651", "24122734", "25305463", "24080331", "25322638"
    }) == ["24122734", "25305463", "25322651"]


def test_selected_four_capture4_views_are_all_assessed(monkeypatch):
    selected = "26053248,25322639,25305462,25322642"
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", selected)
    assert assessment_cameras(set(selected.split(","))) == sorted(
        selected.split(","))


def test_final_label_uses_its_reviewed_subset_without_changing_other_views(
    monkeypatch,
):
    four = "26053248,25322639,25305462,25322642"
    three = "26053248,25322639,25305462"
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", four)
    monkeypatch.setenv("PRECISION_FINAL_VLM_CAMERA_IDS", three)
    assert assessment_cameras(set(four.split(","))) == sorted(four.split(","))
    assert final_assessment_cameras(set(four.split(","))) == sorted(
        three.split(","))


def test_final_label_defaults_to_existing_views(monkeypatch):
    monkeypatch.delenv("PRECISION_FINAL_VLM_CAMERA_IDS", raising=False)
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", "100,200,300")
    assert final_assessment_cameras({"100", "200", "300"}) == [
        "100", "200", "300"]


def test_saved_replay_uses_recorded_final_views_even_with_other_profile(monkeypatch):
    monkeypatch.setenv(
        "PRECISION_FINAL_VLM_CAMERA_IDS", "25305462,25322639,25322642")
    available = {"25305462", "25322639", "25322642", "26053248"}
    assert final_assessment_cameras(
        available, selected=["25322639", "26053248", "25305462"]
    ) == ["25305462", "25322639", "26053248"]


def test_one_view_per_capture_pc_is_selected(monkeypatch):
    selected = "24122734,25305463,25322638,25322649,25305461"
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", selected)
    assert assessment_cameras(set(selected.split(",")) | {"25322651"}) == (
        sorted(selected.split(",")))


def test_moved_socket_replaces_only_an_offscreen_final_view():
    preferred = ["100", "200", "300"]
    available = preferred + ["400"]
    projections = {
        camera: {"image_size_wh": [2048, 1536],
                 "projected_hull_bbox_px": bbox}
        for camera, bbox in {
            "100": [100, 100, 200, 200],
            "200": [2100, 100, 2200, 200],
            "300": [-20, 100, 20, 200],
            "400": [500, 500, 650, 650],
        }.items()
    }
    selected, visible = final_cameras_with_visible_socket(
        preferred, available, projections)
    assert selected == ["100", "300", "400"]
    assert visible == {"100": True, "200": False,
                       "300": True, "400": True}
    projections["200"]["projected_hull_bbox_px"] = [110, 110, 210, 210]
    assert final_cameras_with_visible_socket(
        preferred, available, projections)[0] == preferred


@pytest.mark.parametrize("selection", [
    "25322651,24122734", "25322651,25322651,25305463",
    "25322651,24122734,unknown",
])
def test_invalid_or_missing_selected_camera_cannot_silently_substitute(
    monkeypatch, selection,
):
    monkeypatch.setenv("PRECISION_VLM_CAMERA_IDS", selection)
    with pytest.raises(ValueError):
        assessment_cameras({"25322651", "24122734", "25305463"})
