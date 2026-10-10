"""Saved final frames, frozen calibration and VLM pixels remain source-bound."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import (  # noqa: E402
    SessionCalibration, _canonical_sha256,
)
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.exposed_depth import ExposedDepthLimits  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from precision_insertion.saved_exposed_depth import (  # noqa: E402
    assess_saved_exposed_depth, verify_saved_exposed_depth,
    write_saved_exposed_depth,
)
from precision_insertion import saved_exposed_depth  # noqa: E402
from test_grounded_alignment import _line_rig  # noqa: E402


class _Backend:
    model = "saved-test-vlm"
    native_pixel_coordinates = True

    def __init__(self, answers):
        self.answers = list(answers)

    def infer(self, _images, prompt):
        assert "rear_px" in prompt
        return self.answers.pop(0)


def _limits():
    return ExposedDepthLimits(
        3, .02, 5., 1., 1., 5., 4.,
        .0001, .1, .0001, .0001)


def _setup(tmp_path, monkeypatch):
    mode = select_mode("cylinder", 15.)
    rear = np.array([0., 0., .114])
    rows = _line_rig(rear, np.array([0., 0., -1.]))[:3]
    snapshot = {
        "intrinsics_full": {
            row.frame.camera_id: {
                "K_undist": row.frame.intrinsics.tolist()}
            for row in rows},
        "extrinsics_full": {
            row.frame.camera_id: row.frame.T_camera_socket.tolist()
            for row in rows},
    }
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "camera_calibration": snapshot,
        "camera_calibration_sha256": _canonical_sha256(snapshot),
        "c2r": np.eye(4).tolist(),
    }
    calibration = SessionCalibration(
        {}, np.eye(4), {}, {"mesh": {}, "cuboid": {}}, record)
    session_path = tmp_path / "session_calibration.json"
    session_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(saved_exposed_depth, "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    path = AssetPaths(tmp_path, mode).task_geometry
    path.parent.mkdir(parents=True)
    entry = np.eye(4)
    entry[:3, :3] = np.diag([1., -1., -1.])
    entry[2, 3] = .135
    verification = entry.copy()
    verification[2, 3] -= .020
    geometry = {
        "units": "m", "socket_pose_object": mode.socket_object,
        "key_object": mode.key_object,
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_verification": verification.tolist(),
        "verification_insertion_depth_m": .020,
        "insertion_direction_socket": [0., 0., -1.],
        "key_frame": {"insertion_axis": [0., 0., 1.],
                      "grasp_rear_z_m": 0., "tip_z_m": .080},
        "socket_entry_plane_z_m": .055,
        "socket_rim_z_m": .055,
    }
    path.write_text(json.dumps(geometry), encoding="utf-8")
    images = {row.frame.camera_id: np.zeros((720, 1280, 3), dtype=np.uint8)
              for row in rows}
    frame_ids = {row.frame.camera_id: i + 1
                 for i, row in enumerate(rows)}
    evidence = {camera: {
        "frame_id": frame_ids[camera],
        "image_sha256": image_sha256(image),
        "timestamp_s": 103. + i * .005,
        "max_error_s": .001,
        "timestamp_method": "hardware_exposure",
        "clock_domain": "unix_utc",
    } for i, (camera, image) in enumerate(images.items())}
    capture = RawCameraCapture(
        "final_1", 71, images, frame_ids,
        {"request_id": 71, "source": "camera_acquisition",
         "frames": evidence})
    final = write_raw_camera_capture(
        capture, tmp_path / "final", phase="final_or_abort")
    answers = [json.dumps({
        "rear_px": list(row.tip_uv_px),
        "axis_line_px": [list(point) for point in row.axis_line_uv_px],
        "evidence": "visible rear centre and shaft"}) for row in rows]
    inputs = dict(
        attempt_id="trial_1", candidate_id="table/0/3", mode=mode,
        shared_root=tmp_path, calibration=calibration,
        calibration_path=session_path, final_bundle=final,
        completed_at_s=102.8, decision_timestamp_s=103.1,
        max_capture_skew_s=.02, max_final_gap_s=.5,
        max_frame_age_s=1., backend=_Backend(answers), limits=_limits())
    return inputs


def test_saved_exposed_depth_replays_pixels_vlm_and_cad(tmp_path, monkeypatch):
    inputs = _setup(tmp_path, monkeypatch)
    report = assess_saved_exposed_depth(**inputs)
    assert report["diagnostic"]["status"] == "bounded_visual_depth"
    assert report["diagnostic"]["key_depth_interval_m"][0] > .020
    assert report["depth_source_admissible_for_task_label"] is False
    path = write_saved_exposed_depth(report, tmp_path / "depth_report")
    assert verify_saved_exposed_depth(
        path, mode=inputs["mode"], shared_root=tmp_path,
        calibration=inputs["calibration"]) == report


def test_saved_exposed_depth_rejects_changed_vlm_text_or_image(
        tmp_path, monkeypatch):
    inputs = _setup(tmp_path, monkeypatch)
    report = assess_saved_exposed_depth(**inputs)
    path = write_saved_exposed_depth(report, tmp_path / "depth_report")
    changed = json.loads(path.read_text())
    changed["vlm_observations"][0]["raw_answer"] = "not json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="VLM text/landmarks differ"):
        verify_saved_exposed_depth(
            path, mode=inputs["mode"], shared_root=tmp_path,
            calibration=inputs["calibration"])
    image = inputs["final_bundle"] / "images" / "front.png"
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="final raw camera PNG changed"):
        verify_saved_exposed_depth(
            path, mode=inputs["mode"], shared_root=tmp_path,
            calibration=inputs["calibration"])


def test_saved_exposed_depth_rejects_wrong_time_before_vlm(
        tmp_path, monkeypatch):
    inputs = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="do not follow"):
        assess_saved_exposed_depth(**{
            **inputs, "completed_at_s": 103.1})
    assert len(inputs["backend"].answers) == 3


def test_saved_exposed_depth_preserves_occluded_vlm_abstention(
        tmp_path, monkeypatch):
    inputs = _setup(tmp_path, monkeypatch)
    inputs["backend"] = _Backend([json.dumps({
        "rear_px": None, "axis_line_px": None,
        "evidence": "hand occludes rear face"})] * 3)
    report = assess_saved_exposed_depth(**inputs)
    assert report["diagnostic"]["status"] == "abstain"
    assert report["diagnostic"]["key_depth_interval_m"] is None
    path = write_saved_exposed_depth(report, tmp_path / "occluded_report")
    assert verify_saved_exposed_depth(
        path, mode=inputs["mode"], shared_root=tmp_path,
        calibration=inputs["calibration"])["diagnostic"]["status"] == "abstain"
