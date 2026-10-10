"""Insertion VLM labels must remain tied to exact pre/post camera pixels."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.insertion_checkpoint import (  # noqa: E402
    FinalInsertionCapture, assess_insertion_checkpoint,
    verify_final_insertion_capture, verify_insertion_checkpoint,
    verify_preinsert_raw_capture, write_final_insertion_capture,
    write_insertion_checkpoint, write_preinsert_raw_capture,
)
from precision_insertion.key_perception import (  # noqa: E402
    admit_held_key_capture, write_key_capture_artifacts,
)
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion import session_runner  # noqa: E402
from test_key_perception import (  # noqa: E402
    CAMERAS, SelectorStub, _capture, _session,
)


class FakeBackend:
    model = "fake-vlm"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def infer(self, images, prompt):
        self.calls.append((images, prompt))
        return json.dumps(self.answer)


def _setup(tmp_path, *, answer=None, metric_overrides=None):
    mode = select_mode("square", 1.5)
    pose = np.eye(4)
    base = _capture({camera: pose for camera in CAMERAS}, tmp_path)
    before = replace(
        base, frame_timestamps_s={"cam_a": 102.0, "cam_b": 102.005},
        frame_evidence={
            camera: {**base.frame_evidence[camera],
                     "timestamp_s": 102.0 + 0.005 * i}
            for i, camera in enumerate(CAMERAS)},
        capture_id="held_preinsert_1", request_id=61)
    admitted = admit_held_key_capture(
        capture=before, init_orchestrator=SelectorStub(mode.key_object),
        mode=mode, shared_root=tmp_path,
        calibration=_session(tmp_path, mode, socket_x=0.0),
        calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0,
        held_pose_prior_world=pose, held_pose_prior_timestamp_s=102.002,
        held_pose_prior_source="measured_wrist_plus_observed_held_relation",
        maximum_held_prior_center_error_mm=2.0,
        maximum_held_prior_angle_error_deg=5.0,
        maximum_held_prior_time_skew_s=0.01,
        minimum_held_refinement_iou=0.5)
    preinsert = write_key_capture_artifacts(
        before, admitted, tmp_path / "preinsert")
    image = np.ones((24, 32, 3), dtype=np.uint8) * 128
    images = {camera: image.copy() for camera in CAMERAS}
    frame_ids = {"cam_a": 41, "cam_b": 42}
    rows = {
        camera: {"frame_id": frame_ids[camera],
                 "image_sha256": image_sha256(image),
                 "timestamp_s": 103.0 + 0.005 * i,
                 "max_error_s": 0.001,
                 "timestamp_method": "hardware_exposure",
                 "clock_domain": "unix_utc"}
        for i, camera in enumerate(CAMERAS)}
    capture = FinalInsertionCapture(
        "final_1", 62, images, frame_ids,
        {"request_id": 62, "source": "camera_acquisition", "frames": rows})
    final = write_final_insertion_capture(capture, tmp_path / "final")
    metric = {
        "schema": "precision_insertion_guarded_execution_v1",
        "attempt_id": "trial_1", "candidate_id": "table/0/3",
        "started_at_s": 102.5, "completed_at_s": 102.8,
        "measurement": {
            "key_depth_interval_m": [0.0201, 0.021],
            "key_depth_source": "key_pose_multiview",
            "alignment_within_limits": True,
            "safety_abort": False, "grasp_held": True,
        },
        "source_records": {},
    }
    for name in ("key_depth", "alignment", "force_trace", "grasp_state"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name, "sample": 1}),
                        encoding="utf-8")
        metric["source_records"][name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    metric["measurement"].update(metric_overrides or {})
    metric_path = tmp_path / "guarded_execution.json"
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    backend = FakeBackend(answer or {
        "visual_class": "normal_appearance",
        "evidence_views": list(CAMERAS),
        "evidence": "key appears seated in both views"})
    args = dict(
        attempt_id="trial_1", candidate_id="table/0/3",
        target_depth_m=0.02,
        session_calibration_sha256="0" * 64,
        preinsert_bundle=preinsert, final_bundle=final,
        metric_record_path=metric_path,
        preinsert_reached_at_s=101.9, decision_timestamp_s=103.1,
        backend=backend, max_phase_skew_s=0.02,
        max_preinsert_age_s=1.0, max_final_observation_gap_s=0.5)
    return args, backend


def test_bound_multiview_insertion_success_and_tamper_rejection(tmp_path):
    args, backend = _setup(tmp_path)
    report = assess_insertion_checkpoint(**args)
    assert report["outcome"]["insertion_success"] is True
    assert report["visual"]["raw_answer"]
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 4
    saved = write_insertion_checkpoint(report, tmp_path / "assessment")
    assert verify_insertion_checkpoint(saved)["outcome"]["insertion_success"] is True
    metric = args["metric_record_path"]
    metric.write_text(metric.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="metric record changed"):
        verify_insertion_checkpoint(saved)


def test_vlm_one_view_cannot_produce_positive_label(tmp_path):
    args, _backend = _setup(tmp_path, answer={
        "visual_class": "normal_appearance", "evidence_views": ["cam_a"],
        "evidence": "one view only"})
    report = assess_insertion_checkpoint(**args)
    assert report["effective_vlm_class"] == "unobservable"
    assert report["outcome"]["insertion_success"] is None
    saved = write_insertion_checkpoint(report, tmp_path / "assessment")
    assert verify_insertion_checkpoint(saved)["outcome"]["insertion_success"] is None


def test_short_depth_is_failure_and_changed_final_image_rejected(tmp_path):
    args, _backend = _setup(tmp_path, metric_overrides={
        "key_depth_interval_m": [0.005, 0.008]})
    report = assess_insertion_checkpoint(**args)
    assert report["outcome"]["insertion_success"] is False
    saved = write_insertion_checkpoint(report, tmp_path / "assessment")
    image = args["final_bundle"] / "images" / "cam_a.png"
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="final raw camera PNG changed"):
        verify_final_insertion_capture(args["final_bundle"])
    with pytest.raises(ValueError, match="final raw camera PNG changed"):
        verify_insertion_checkpoint(saved)


def test_rejects_nonbracketing_frames_before_vlm(tmp_path):
    args, backend = _setup(tmp_path)
    with pytest.raises(ValueError, match="do not bracket"):
        assess_insertion_checkpoint(**{
            **args, "preinsert_reached_at_s": 102.6})
    assert backend.calls == []


def test_changed_force_trace_blocks_vlm_and_result(tmp_path):
    args, backend = _setup(tmp_path)
    metric = json.loads(args["metric_record_path"].read_text())
    force = Path(metric["source_records"]["force_trace"]["path"])
    force.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed: force_trace"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def _raw_preinsert_bundle(args, tmp_path):
    images = {
        camera: cv2.imread(str(args["preinsert_bundle"] / "images" /
                                    f"{camera}.png"))
        for camera in CAMERAS}
    ids = {"cam_a": 71, "cam_b": 72}
    rows = {
        camera: {"frame_id": ids[camera],
                 "image_sha256": image_sha256(images[camera]),
                 "timestamp_s": 102.0 + 0.005 * i,
                 "max_error_s": 0.001,
                 "timestamp_method": "hardware_exposure",
                 "clock_domain": "unix_utc"}
        for i, camera in enumerate(CAMERAS)}
    return write_preinsert_raw_capture(
        FinalInsertionCapture(
            "pre_raw_1", 63, images, ids,
            {"request_id": 63, "source": "camera_acquisition",
             "frames": rows}),
        tmp_path / "pre_raw")


def test_raw_preinsert_images_need_no_postgrasp_foundpose(tmp_path):
    args, backend = _setup(tmp_path)
    raw = _raw_preinsert_bundle(args, tmp_path)
    assert verify_preinsert_raw_capture(raw)["phase"] == "preinsert"
    report = assess_insertion_checkpoint(**{
        **args, "preinsert_bundle": raw})
    assert report["preinsert_capture_kind"] == "raw_images"
    assert report["outcome"]["insertion_success"] is True
    assert len(backend.calls[0][0]) == 4
    saved = write_insertion_checkpoint(report, tmp_path / "raw_assessment")
    assert verify_insertion_checkpoint(saved)["outcome"][
        "insertion_success"] is True


@pytest.mark.parametrize("raw_preinsert", [False, True])
def test_session_runner_records_bound_insertion_label(
        monkeypatch, tmp_path, raw_preinsert):
    args, backend = _setup(tmp_path)
    if raw_preinsert:
        args["preinsert_bundle"] = _raw_preinsert_bundle(args, tmp_path)
    mode = select_mode("square", 1.5)
    calibration = _session(tmp_path, mode, socket_x=0.0)
    calibration.record["schema"] = "precision_insertion_session_calibration_v1"
    monkeypatch.setattr(session_runner, "validate_catalog_session",
                        lambda *_args, **_kwargs: None)
    runner = session_runner.SessionRunner(
        mode=mode, calibration=calibration,
        catalog={"shared_root": str(tmp_path), "complete_scan": True},
        shared_root=tmp_path, output_dir=tmp_path / "session",
        max_xy_retries=1)
    attempt = begin_attempt(
        attempt_id="trial_1", mode=mode,
        session_record=calibration.record, candidate_id="table/0/3",
        tabletop_pose_stem="000", xy_offset_socket_m=(0.0, 0.0),
        started_at_s=101.0)
    attempt.record_stage(
        "grasp_success", True, timestamp_s=101.5,
        evidence_refs={"vlm_observation": "saved/lift.json",
                       "key_wrist_check": "saved/lift.json"})
    attempt.record_stage(
        "preinsert_reached", True, timestamp_s=102.02,
        evidence_refs={
            "trajectory": "saved/transfer.json",
            "key_socket_pose": (
                str(args["preinsert_bundle"] / "key_observation.json")
                if not raw_preinsert else "saved/kinematic_key_pose.json"),
            **({"preinsert_image": str(args["preinsert_bundle"] /
                                       "manifest.json")}
               if raw_preinsert else {}),
            "grasp_state": "saved/grip.json",
            "postlift_preflight": "saved/postlift.json"})
    runner._attempt = attempt
    runner._attempt_dir = runner.output_dir / "attempts" / "trial_1"
    runner._attempt_dir.mkdir(parents=True)
    assert runner.current_decision().action == "await_guarded_insertion_and_observation"
    with pytest.raises(FileNotFoundError):
        runner.observe_insertion(
            object(), timestamp_s=103.1,
            evidence_refs={"vlm_observation": "fabricated"},
            checkpoint_path=tmp_path / "not_a_checkpoint.json")
    result = runner.prepare_observed_insertion_label(
        preinsert_bundle=args["preinsert_bundle"],
        final_bundle=args["final_bundle"],
        metric_record_path=args["metric_record_path"], backend=backend,
        decision_timestamp_s=103.1, max_phase_skew_s=0.02,
        max_preinsert_age_s=1.0, max_final_observation_gap_s=0.5)
    assert result["outcome"]["insertion_success"] is True
    assert runner.active_attempt.labels["insertion_success"] is True
    assert runner.current_decision().action == "hold_for_supervised_completion"
    assessment = (runner._attempt_dir / "insertion_assessments" / "000" /
                  "report.json")
    assert verify_insertion_checkpoint(assessment)["outcome"][
        "insertion_success"] is True
