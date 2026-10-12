"""The assisted ROI is a proposal; wrong socket masks must fail closed."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.registered_key_roi import (  # noqa: E402
    RegisteredKeyROILimits, detector_from_board_config,
    propose_registered_key_roi,
    segment_registered_key_roi,
)


def _limits() -> RegisteredKeyROILimits:
    return RegisteredKeyROILimits(
        minimum_shared_corners=8, minimum_homography_inliers=7,
        homography_ransac_px=2., maximum_board_shift_px=5.,
        difference_threshold=40, socket_dilation_px=5,
        closing_kernel_px=3, minimum_component_pixels=100,
        maximum_competitor_fraction=.8, crop_pad_px=8,
        crop_pad_fraction=.15, minimum_sam_pixels=100,
        maximum_sam_pixels=1000,
        minimum_difference_support_fraction=.8,
        maximum_socket_overlap_fraction=.05)


def _board(_image, *, board_ids):
    assert board_ids == ["9"]
    xy = np.asarray([(x, y) for y in (20., 100., 220.)
                     for x in (20., 120., 220.)], dtype=np.float32)
    return {"9": {"checkerIDs": np.arange(len(xy)),
                  "checkerCorner": xy}}


def _scene():
    baseline = np.full((240, 240, 3), 100, dtype=np.uint8)
    socket = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(socket, (100, 100), 12, 255, -1)
    current = baseline.copy()
    current[145:165, 170:190] = (255, 255, 255)  # white BGR key
    return baseline, current, socket


def test_explicit_board_config_avoids_global_paradex_symlink(tmp_path):
    config = tmp_path / "board.json"
    config.write_text(json.dumps({"9": {
        "numX": 10, "numY": 7, "checkerLength": 1.0,
        "markerLength": .6, "dict_type": "6X6_250",
        "markerIDs": list(range(35)),
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        detector_from_board_config(
            board_config_path=config, board_id="10")
    detect = detector_from_board_config(
        board_config_path=config, board_id="9")
    assert detect(np.zeros((64, 64, 3), dtype=np.uint8),
                  board_ids=["9"]) == {}
    assert detect(np.zeros((64, 64, 3), dtype=np.uint8),
                  board_ids=["10"]) == {}


def test_registered_difference_localizes_key_and_maps_sam_to_full_frame():
    baseline, current, socket = _scene()
    proposal = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket, board_detector=_board,
        board_id="9", limits=_limits())
    assert proposal.status == "roi_available"
    assert proposal.crop_xyxy is not None
    x0, y0, x1, y1 = proposal.crop_xyxy
    assert x0 <= 170 < 190 <= x1 and y0 <= 145 < 165 <= y1
    assert proposal.to_record()["robot_ready"] is False

    class WhiteKey:
        def segment(self, crop, prompt):
            assert prompt == "white tabletop key"
            return np.all(crop > 200, axis=2)

    mask = segment_registered_key_roi(
        current_rgb=cv2.cvtColor(current, cv2.COLOR_BGR2RGB),
        proposal=proposal, sam_segmentor=WhiteKey(),
        prompt="white tabletop key", limits=_limits())
    assert mask.status == "mask_available"
    assert mask.mask is not None and mask.mask.shape == socket.shape
    assert int(mask.mask.sum()) == 400
    assert not np.logical_and(mask.mask, socket).any()


def test_socket_only_change_is_not_a_key_roi():
    baseline, current, socket = _scene()
    current[:] = baseline
    current[95:105, 95:105] = (0, 0, 255)
    proposal = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket, board_detector=_board,
        board_id="9", limits=_limits())
    assert proposal.status == "abstain"
    assert proposal.reason == "no_key_difference_component"


def test_ambiguous_differences_and_changed_board_abstain():
    baseline, current, socket = _scene()
    current[55:75, 170:190] = (0, 0, 255)
    ambiguous = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket, board_detector=_board,
        board_id="9", limits=_limits())
    assert ambiguous.reason == "ambiguous_difference_components"

    calls = 0
    def shifted(image, *, board_ids):
        nonlocal calls
        calls += 1
        result = _board(image, board_ids=board_ids)
        if calls == 2:
            result["9"]["checkerCorner"] += 50
        return result

    moved = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket, board_detector=shifted,
        board_id="9", limits=_limits())
    assert moved.reason == "board_or_camera_moved_too_far"


def test_sam_spill_and_unsupported_mask_are_rejected():
    baseline, current, socket = _scene()
    proposal = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket, board_detector=_board,
        board_id="9", limits=_limits())
    rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)

    class WholeCrop:
        def segment(self, crop, _prompt):
            return np.ones(crop.shape[:2], dtype=bool)

    bad = segment_registered_key_roi(
        current_rgb=rgb, proposal=proposal,
        sam_segmentor=WholeCrop(), prompt="key", limits=replace(
            _limits(), maximum_sam_pixels=10000))
    assert bad.status == "abstain"
    assert bad.reason == "sam_mask_lacks_difference_support"

    class WrongShape:
        def segment(self, crop, _prompt):
            return np.ones((1, 1), dtype=bool)

    with pytest.raises(ValueError, match="incorrect pixel geometry"):
        segment_registered_key_roi(
            current_rgb=rgb, proposal=proposal,
            sam_segmentor=WrongShape(), prompt="key", limits=_limits())
