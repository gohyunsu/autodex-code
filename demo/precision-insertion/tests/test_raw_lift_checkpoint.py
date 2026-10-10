"""Raw post-lift images can be judged without a new held-key FoundPose."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.lift_checkpoint import (  # noqa: E402
    assess_raw_lift_checkpoint, verify_lift_checkpoint,
    write_lift_checkpoint,
)
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion import session_runner  # noqa: E402
from test_lift_checkpoint import _setup  # noqa: E402
from test_key_perception import CAMERAS, _session  # noqa: E402


def _raw_case(tmp_path, *, answer=None):
    old, backend = _setup(tmp_path)
    if answer is not None:
        backend.answer = answer
    images = {
        camera: cv2.imread(str(old["after_bundle"] / "images" /
                                    f"{camera}.png"))
        for camera in CAMERAS}
    ids = {"cam_a": 31, "cam_b": 32}
    rows = {
        camera: {"frame_id": ids[camera],
                 "image_sha256": image_sha256(images[camera]),
                 "timestamp_s": 101.0 + 0.005 * i,
                 "max_error_s": 0.001,
                 "timestamp_method": "hardware_exposure",
                 "clock_domain": "unix_utc"}
        for i, camera in enumerate(CAMERAS)}
    after = write_raw_camera_capture(
        RawCameraCapture(
            "raw_after_lift_2", 72, images, ids,
            {"request_id": 72, "source": "camera_acquisition",
             "frames": rows}),
        tmp_path / "raw_after_lift", phase="after_lift")
    args = dict(
        mode=old["mode"], attempt_id=old["attempt_id"],
        candidate_id=old["candidate_id"],
        attempt_started_at_s=old["attempt_started_at_s"],
        lift_completed_at_s=old["lift_completed_at_s"],
        decision_timestamp_s=old["decision_timestamp_s"],
        before_capture_id=old["before_capture_id"],
        before_bundle=old["before_bundle"], after_bundle=after,
        joint_sample=old["joint_sample"], backend=backend,
        max_phase_skew_s=0.02, max_lift_observation_gap_s=2.0,
        max_arm_hand_skew_s=0.02, max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05)
    return args, backend


def test_raw_visible_held_report_is_not_a_key_pose(tmp_path):
    args, backend = _raw_case(tmp_path)
    result = assess_raw_lift_checkpoint(**args)
    assert result.grasp_success is True
    assert result.evidence_kind == "raw_visual"
    assert result.center_rise_m is None
    assert len(backend.calls) == 1
    report_path = write_lift_checkpoint(result, tmp_path / "raw_report") / "report.json"
    saved = verify_lift_checkpoint(report_path)
    assert saved["grasp_success"] is True
    assert saved["center_rise_m"] is None
    assert saved["scope"] == "saved_raw_multiview_visible_key_not_6d_pose_or_contact_proof"
    assert saved["visual"]["raw_answer"]


def test_one_view_raw_held_abstains(tmp_path):
    args, _backend = _raw_case(tmp_path, answer={
        "class": "held", "evidence_views": ["cam_a"],
        "evidence": "only one view shows the key"})
    result = assess_raw_lift_checkpoint(**args)
    assert result.grasp_success is None
    path = write_lift_checkpoint(result, tmp_path / "raw_report") / "report.json"
    assert verify_lift_checkpoint(path)["grasp_success"] is None


def test_raw_lift_changed_pixels_or_verdict_are_rejected(tmp_path):
    args, _backend = _raw_case(tmp_path)
    result = assess_raw_lift_checkpoint(**args)
    path = write_lift_checkpoint(result, tmp_path / "raw_report") / "report.json"
    original = path.read_bytes()
    changed = json.loads(original)
    changed["grasp_success"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="label conflicts"):
        verify_lift_checkpoint(path)
    path.write_bytes(original)
    (args["after_bundle"] / "images" / "cam_a.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="raw camera PNG changed"):
        verify_lift_checkpoint(path)


def test_session_can_label_visible_lift_without_held_foundpose(
        monkeypatch, tmp_path):
    args, backend = _raw_case(tmp_path)
    calibration = _session(tmp_path, args["mode"])
    calibration.record["schema"] = "precision_insertion_session_calibration_v1"
    monkeypatch.setattr(session_runner, "validate_catalog_session",
                        lambda *_args, **_kwargs: None)
    runner = session_runner.SessionRunner(
        mode=args["mode"], calibration=calibration,
        catalog={"shared_root": str(tmp_path), "complete_scan": True},
        shared_root=tmp_path, output_dir=tmp_path / "session",
        max_xy_retries=1)
    runner._attempt = begin_attempt(
        attempt_id=args["attempt_id"], mode=args["mode"],
        session_record=calibration.record,
        candidate_id=args["candidate_id"], tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0),
        started_at_s=args["attempt_started_at_s"])
    runner._attempt_dir = runner.output_dir / "attempts" / args["attempt_id"]
    runner._attempt_dir.mkdir(parents=True)
    runner._preflight = SimpleNamespace(
        key_observation_id=args["before_capture_id"])
    runner._key_evidence_dir = args["before_bundle"]
    execution = tmp_path / "lift_execution.json"
    execution.write_text(json.dumps({
        "schema": "precision_insertion_lift_execution_v1",
        "attempt_id": args["attempt_id"],
        "candidate_id": args["candidate_id"],
        "trajectory_complete": True, "force_abort": False,
        "completed_at_s": args["lift_completed_at_s"],
    }), encoding="utf-8")
    result = runner.prepare_raw_lift_label(
        after_raw_evidence_dir=args["after_bundle"],
        joint_sample=args["joint_sample"], backend=backend,
        lift_execution_log_path=execution,
        decision_timestamp_s=args["decision_timestamp_s"],
        max_phase_skew_s=args["max_phase_skew_s"],
        max_lift_observation_gap_s=args["max_lift_observation_gap_s"],
        max_arm_hand_skew_s=args["max_arm_hand_skew_s"],
        max_hand_command_error_raw=args["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=args["max_arm_velocity_rad_s"])
    assert result.grasp_success is True
    assert runner.active_attempt.labels["grasp_success"] is True
    assert (runner.active_attempt.events[-1]["evidence_refs"][
        "raw_lift_visual"].endswith("/report.json"))
    assert runner.current_decision().action == "held_relation_evidence_required"
