"""A VLM vote can propose, but cannot authorize, a one-millimetre retry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.conversion import se32cart  # noqa: E402
from precision_insertion.candidates import build_endpoint_catalog  # noqa: E402
from precision_insertion.observer import LabeledFrame  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.xy_retry import assess_xy_retry  # noqa: E402
from test_candidates import _candidate_fixture, _session_record  # noqa: E402


class FixedBackend:
    model = "test-vlm"

    def __init__(self):
        self.calls = 0

    def infer(self, _images, _prompt):
        self.calls += 1
        return json.dumps({
            "visible": True, "choice_id": "x_plus_1mm",
            "failure_class": "misaligned", "evidence": "visible lateral residual",
        })


def _setup(tmp_path, *, focal=4000):
    mode, paths, candidates, catalog_screen = _candidate_fixture(tmp_path)
    paths.task_geometry.write_text(json.dumps({
        "T_socket_key_preinsert": np.eye(4).tolist(),
    }), encoding="utf-8")
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=catalog_screen)
    record = _session_record(mode, paths)
    record["c2r"] = np.eye(4).tolist()
    world = {
        "mesh": {"fixture_socket": {
            "pose": se32cart(np.eye(4)).tolist(),
            "file_path": str(paths.socket_collision_mesh.resolve()),
        }},
        "cuboid": {},
    }
    calibration = SimpleNamespace(
        record=record, socket_pose_robot=np.eye(4), collision_scene=world)
    K = [[focal, 0, 500], [0, focal, 400], [0, 0, 1]]
    intrinsic = {serial: {"K_undist": K} for serial in ("a", "b")}
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 1
    extrinsic = {serial: camera_pose for serial in ("a", "b")}
    camera_snapshot = {
        "intrinsics_full": json.loads(json.dumps(intrinsic)),
        "extrinsics_full": {serial: matrix.tolist()
                            for serial, matrix in extrinsic.items()},
    }
    record["camera_calibration"] = camera_snapshot
    record["camera_calibration_sha256"] = hashlib.sha256(json.dumps(
        camera_snapshot, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    frames = [LabeledFrame(
        serial, "preinsert_hold", 100.0,
        Image.new("RGB", (1000, 800), (100, 110, 120)))
        for serial in ("a", "b")]
    bgr = np.asarray(frames[0].image, dtype=np.uint8)[:, :, ::-1].copy()
    frame_ids = {"a": 31, "b": 32}
    acquisition_metadata = {
        "request_id": 15, "source": "camera_acquisition",
        "frames": {serial: {
            "frame_id": frame_ids[serial],
            "image_sha256": image_sha256(bgr),
            "timestamp_s": 100.0,
            "max_error_s": 0.001,
            "timestamp_method": "hardware_exposure",
            "clock_domain": "unix_utc",
        } for serial in frame_ids},
    }

    def xy_screen(**kwargs):
        return {
            "endpoint_pass": True,
            "xy_offset_socket_m": list(kwargs["xy_offset_socket_m"]),
            "verification_depth_m": mode.target_depth_m,
            "T_key_hand": kwargs["T_key_hand_override"].tolist(),
        }

    return {
        "shared_root": tmp_path, "mode": mode, "calibration": calibration,
        "catalog": catalog, "candidate_key": ("table", "0", "1"),
        "tabletop_pose_stem": "000",
        "current_offset_socket_m": (0.0, 0.0),
        "observed_T_key_hand": np.load(
            candidates[0] / "wrist_se3.npy", allow_pickle=False),
        "held_hand_q_measured": np.full(6, 0.2),
        "observed_key_hand_source": "multiview_key_pose_plus_live_wrist",
        "max_grasp_translation_drift_m": 0.002,
        "max_grasp_rotation_drift_deg": 5.0,
        "failed_insertion_observed": True,
        "guarded_withdrawal_complete": True,
        "grasp_held": True, "hard_abort": False,
        "frames": frames, "intrinsics_full": intrinsic,
        "extrinsics_full": extrinsic,
        "frame_timestamp_source": "camera_acquisition",
        "frame_request_id": 15, "frame_ids": frame_ids,
        "acquisition_metadata": acquisition_metadata,
        "backend": FixedBackend(), "max_total_offset_m": 0.002,
        "minimum_anchor_separation_px": 3.0, "crop_width_px": 320,
        "decision_timestamp_s": 100.05, "max_frame_age_s": 0.2,
        "max_capture_skew_s": 0.02, "screen": xy_screen,
    }


def test_vlm_multiview_choice_is_only_a_replan_proposal(tmp_path):
    args = _setup(tmp_path)
    original_screen = args["screen"]
    checked = []
    def measured_screen(**kwargs):
        assert kwargs["override_source"] == (
            "multiview_key_pose_plus_live_wrist")
        assert kwargs["hand_poses_override"]["measured_held"] == (
            pytest.approx(np.full(6, 0.2)))
        checked.append(kwargs["xy_offset_socket_m"])
        return original_screen(**kwargs)
    args["screen"] = measured_screen
    result = assess_xy_retry(**args)
    assert result.status == "proposal_requires_live_preflight"
    assert result.decision.status == "propose"
    assert result.decision.offset_socket_m == pytest.approx((0.001, 0.0))
    assert result.decision.supporting_cameras == ("a", "b")
    assert args["backend"].calls == 2
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["frame_binding"]["request_id"] == 15
    assert len(result.endpoint_screen["endpoint_clear_choice_ids"]) == 5
    assert len(checked) == 5
    assert result.endpoint_screen["hand_pose_source"] == (
        "measured_inspire_feedback")


def test_unobserved_squeeze_offset_yields_direction_only_not_motion(tmp_path):
    args = _setup(tmp_path)
    args["observed_T_key_hand"] = None
    args["observed_key_hand_source"] = "v8_nominal_unobserved_key"
    checked = []

    def nominal_hand_screen(**kwargs):
        assert kwargs["override_source"] == "v8_nominal_unobserved_key"
        assert kwargs["hand_poses_override"]["measured_held"] == (
            pytest.approx(np.full(6, 0.2)))
        checked.append(kwargs["xy_offset_socket_m"])
        return {
            "endpoint_pass": False,  # The *nominal* key collides at every XY.
            "hand_socket_clear_at_20mm": True,
            "xy_offset_socket_m": list(kwargs["xy_offset_socket_m"]),
            "verification_depth_m": args["mode"].target_depth_m,
        }

    args["screen"] = nominal_hand_screen
    result = assess_xy_retry(**args)
    assert len(checked) == 5
    assert result.status == "diagnostic_xy_hypothesis_only"
    assert result.decision.offset_socket_m == pytest.approx((0.001, 0.0))
    assert result.endpoint_screen["endpoint_clear_choice_ids"] == []
    assert "x_plus_1mm" in result.endpoint_screen["vlm_advisory_choice_ids"]
    assert result.endpoint_screen["choice_basis"] == (
        "nominal_hand_socket_clearance_only")
    assert result.endpoint_screen["key_socket_fit_unverified"] is True
    assert result.endpoint_screen["observed_relation_translation_drift_m"] is None
    assert result.to_record()["robot_ready"] is False


def test_unobserved_route_requires_explicit_hand_clearance(tmp_path):
    args = _setup(tmp_path)
    args["observed_T_key_hand"] = None
    args["observed_key_hand_source"] = "v8_nominal_unobserved_key"
    with pytest.raises(ValueError, match="explicit hand/socket verdicts"):
        assess_xy_retry(**args)
    assert args["backend"].calls == 0

    args = _setup(tmp_path / "unsafe")
    args["observed_T_key_hand"] = None
    args["observed_key_hand_source"] = "v8_nominal_unobserved_key"

    def blocked_hand(**kwargs):
        return {
            "endpoint_pass": False,
            "hand_socket_clear_at_20mm": False,
            "xy_offset_socket_m": list(kwargs["xy_offset_socket_m"]),
            "verification_depth_m": args["mode"].target_depth_m,
        }

    args["screen"] = blocked_hand
    result = assess_xy_retry(**args)
    assert result.status == "no_safe_direction"
    assert args["backend"].calls == 0


def test_nominal_source_cannot_claim_observed_key_pose(tmp_path):
    args = _setup(tmp_path)
    args["observed_key_hand_source"] = "v8_nominal_unobserved_key"
    with pytest.raises(ValueError, match="observed pose only"):
        assess_xy_retry(**args)


def test_retry_rejects_changed_or_missing_frozen_camera_calibration(tmp_path):
    args = _setup(tmp_path)
    args["intrinsics_full"]["a"]["K_undist"][0][0] += 1
    with pytest.raises(ValueError, match="camera calibration changed"):
        assess_xy_retry(**args)
    args = _setup(tmp_path / "other")
    del args["calibration"].record["camera_calibration"]
    with pytest.raises(ValueError, match="frozen camera calibration"):
        assess_xy_retry(**args)


def test_pixel_unresolvable_views_abstain_before_vlm(tmp_path):
    args = _setup(tmp_path, focal=1000)
    result = assess_xy_retry(**args)
    assert result.status == "visual_abstain"
    assert result.decision is None
    assert args["backend"].calls == 0


def test_hard_abort_stops_without_polling_vlm(tmp_path):
    args = _setup(tmp_path)
    args["hard_abort"] = True
    result = assess_xy_retry(**args)
    assert result.status == "stop"
    assert args["backend"].calls == 0


def test_endpoint_geometry_can_veto_all_one_mm_retry_directions(tmp_path):
    args = _setup(tmp_path)

    def tight_socket(**kwargs):
        offset = tuple(kwargs["xy_offset_socket_m"])
        return {
            "endpoint_pass": offset == (0.0, 0.0),
            "xy_offset_socket_m": list(offset),
            "verification_depth_m": args["mode"].target_depth_m,
        }

    args["screen"] = tight_socket
    result = assess_xy_retry(**args)
    assert result.status == "no_safe_direction"
    assert args["backend"].calls == 0
    assert result.endpoint_screen["endpoint_clear_choice_ids"] == ["hold"]


def test_retry_rejects_publish_timestamps_or_wrong_grasp(tmp_path):
    args = _setup(tmp_path)
    args["frame_timestamp_source"] = "foundpose_publish"
    with pytest.raises(ValueError, match="acquisition-time"):
        assess_xy_retry(**args)
    args = _setup(tmp_path / "other")
    args["candidate_key"] = ("table", "0", "2")
    with pytest.raises(ValueError, match="not endpoint eligible"):
        assess_xy_retry(**args)


def test_observed_key_hand_drift_stops_retry_before_vlm(tmp_path):
    args = _setup(tmp_path)
    shifted = args["observed_T_key_hand"].copy()
    shifted[0, 3] += 0.003
    args["observed_T_key_hand"] = shifted
    result = assess_xy_retry(**args)
    assert result.status == "stop"
    assert "relation_drift" in result.reason
    assert args["backend"].calls == 0


def test_retry_rejects_mismatched_frame_and_uncertain_timing(tmp_path):
    args = _setup(tmp_path)
    args["acquisition_metadata"]["frames"]["a"]["frame_id"] = 99
    with pytest.raises(ValueError, match="frame ID mismatch"):
        assess_xy_retry(**args)
    assert args["backend"].calls == 0

    args = _setup(tmp_path / "other")
    args["acquisition_metadata"]["frames"]["a"]["max_error_s"] = 0.02
    result = assess_xy_retry(**args)
    assert result.status == "visual_abstain"
    assert args["backend"].calls == 0
