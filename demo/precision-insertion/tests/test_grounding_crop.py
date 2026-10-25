"""Lossless VLM ROI pixels must map exactly back to full camera pixels."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.grounding_crop import (  # noqa: E402
    make_native_grounding_crop, nominal_preinsert_rois,
    verify_native_crop_observation,
)
from precision_insertion.grounded_alignment import (  # noqa: E402
    observe_grounded_cylinder_axis_cropped,
)
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402


def test_right_bottom_padding_does_not_rescale_source_pixels():
    raw = Image.new("RGB", (1280, 720), (10, 20, 30))
    raw.putpixel((1270, 710), (231, 54, 6))
    crop = make_native_grounding_crop(raw, (1200, 650, 1280, 720))
    assert crop.content_size_px == (80, 70)
    assert crop.model_size_px == (256, 256)
    assert crop.image.getpixel((70, 60)) == (231, 54, 6)
    assert crop.image.getpixel((80, 70)) == (127, 127, 127)
    assert crop.to_full_pixel((70., 60.)) == (1270., 710.)
    with pytest.raises(ValueError, match="padding"):
        crop.to_full_pixel((80., 60.))
    assert crop.mapping_record()["resampled"] is False


def test_nominal_preinsert_roi_uses_calibrated_projection_only():
    frame = CalibratedXYFrame(
        "camera1", 1., Image.new("RGB", (2048, 1536)), np.eye(4),
        np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]]))
    rois = nominal_preinsert_rois(
        [frame], tip_socket_m=np.array([.01, -.02, 1.]),
        rear_socket_m=np.array([.01, -.02, .7]),
        socket_rim_z_m=.5)
    assert rois == {"camera1": (903, 626, 1159, 882)}
    crop = make_native_grounding_crop(frame.raw, rois["camera1"])
    assert crop.model_size_px == (256, 256)
    assert crop.to_full_pixel((126., 132.)) == (1029., 758.)


def test_nominal_preinsert_roi_keeps_long_shaft_without_square_padding():
    frame = CalibratedXYFrame(
        "camera1", 1., Image.new("RGB", (2048, 1536)), np.eye(4),
        np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]]))
    tip = np.array([.01, -.02, 1.])
    rear = np.array([.01, -.02, .2])
    roi = nominal_preinsert_rois(
        [frame], tip_socket_m=tip, rear_socket_m=rear,
        socket_rim_z_m=.5)["camera1"]
    x0, y0, x1, y1 = roi
    assert x1 - x0 < y1 - y0
    assert (x1 - x0) * (y1 - y0) < 640 * 640
    for point in (tip, rear, np.array([0., 0., .5])):
        camera = point
        projected = frame.intrinsics @ camera
        u, v = projected[:2] / projected[2]
        assert x0 <= u < x1 and y0 <= v < y1


@pytest.mark.parametrize("roi", [
    (0, 0, 129, 10), (5, 2, 5, 20), (-1, 0, 10, 10),
    (1.5, 0, 10, 10),
])
def test_bad_or_out_of_frame_roi_fails_closed(roi):
    with pytest.raises(ValueError, match="metric ROI"):
        make_native_grounding_crop(Image.new("RGB", (128, 96)), roi)


def test_vlm_mark_in_padding_cannot_become_metric_landmark():
    class Backend:
        native_pixel_coordinates = True

        def infer(self, _images, _prompt):
            return ('{"tip_px":[100,20],"axis_line_px":'
                    '[[10,10],[20,20]],"evidence":"test"}')

    frame = CalibratedXYFrame(
        "camera1", 1., Image.new("RGB", (1280, 720)), np.eye(4),
        np.array([[1000., 0., 640.], [0., 1000., 360.], [0., 0., 1.]]))
    views, observations = observe_grounded_cylinder_axis_cropped(
        Backend(), [frame], roi_xyxy_by_camera={"camera1": (1200, 650, 1280, 720)})
    assert views[0].tip_uv_px is None
    assert views[0].axis_line_uv_px is None
    assert "padding" in observations[0].parse_error
    assert observations[0].pixel_mapping["mapped_full_pixels"] is None
    assert verify_native_crop_observation(
        observations[0].to_record(), full_image_size_px=frame.raw.size) is None


def test_invalid_crop_canvas_mark_stays_rejected_without_aborting_other_views():
    class Backend:
        native_pixel_coordinates = True

        def infer(self, _images, prompt):
            x = 9999 if "Camera bad." in prompt else 30
            return (f'{{"tip_px":[{x},20],"axis_line_px":'
                    '[[40,30],[50,40]],"evidence":"visible shaft"}')

    image = Image.new("RGB", (1280, 720))
    intrinsics = np.array([[1000., 0., 640.], [0., 1000., 360.],
                           [0., 0., 1.]])
    frames = [CalibratedXYFrame(camera, 1., image, np.eye(4), intrinsics)
              for camera in ("good1", "bad", "good2")]
    views, observations = observe_grounded_cylinder_axis_cropped(
        Backend(), frames,
        roi_xyxy_by_camera={camera: (100, 100, 356, 356)
                            for camera in ("good1", "bad", "good2")})
    assert [view.frame.camera_id for view in views if view.tip_uv_px is not None] == [
        "good1", "good2"]
    rejected = observations[1].to_record()
    assert rejected["parse_error"] == "line landmark outside original pixels"
    assert rejected["pixel_mapping"]["mapped_full_pixels"] is None
    assert verify_native_crop_observation(
        rejected, full_image_size_px=image.size) is None
    for index in (0, 2):
        assert verify_native_crop_observation(
            observations[index].to_record(),
            full_image_size_px=image.size)["tip_px"] == [130., 120.]


def test_saved_crop_mapping_replays_and_detects_tampering():
    class Backend:
        native_pixel_coordinates = True

        def infer(self, _images, _prompt):
            return ('{"tip_px":[30,20],"axis_line_px":'
                    '[[40,30],[50,40]],"evidence":"visible shaft"}')

    frame = CalibratedXYFrame(
        "camera1", 1., Image.new("RGB", (1280, 720)), np.eye(4),
        np.array([[1000., 0., 640.], [0., 1000., 360.], [0., 0., 1.]]))
    views, rows = observe_grounded_cylinder_axis_cropped(
        Backend(), [frame],
        roi_xyxy_by_camera={"camera1": (1200, 650, 1280, 720)})
    record = rows[0].to_record()
    assert views[0].tip_uv_px == (1230., 670.)
    assert verify_native_crop_observation(
        record, full_image_size_px=frame.raw.size)["tip_px"] == [1230., 670.]
    record["pixel_mapping"]["mapped_full_pixels"]["tip_px"][0] += 1
    with pytest.raises(ValueError, match="mapping differs"):
        verify_native_crop_observation(
            record, full_image_size_px=frame.raw.size)
