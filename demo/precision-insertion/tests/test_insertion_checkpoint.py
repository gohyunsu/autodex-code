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
from precision_insertion.guarded_contact import (  # noqa: E402
    GuardedContactLimits, GuardedContactSample,
)
from precision_insertion.guarded_trace import (  # noqa: E402
    replay_guarded_contact_trace, write_guarded_contact_trace,
)
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
from precision_insertion.trial_preflight import _canonical_sha256  # noqa: E402
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
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def _setup(tmp_path, *, answer=None, metric_overrides=None,
           session_calibration_sha256="0" * 64):
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
        "session_calibration_sha256": session_calibration_sha256,
        "started_at_s": 102.5, "completed_at_s": 102.8,
        "measurement": {
            "key_depth_interval_m": [0.0201, 0.021],
            "key_depth_source": "key_pose_multiview",
            "alignment_within_limits": True,
            "safety_abort": False, "grasp_held": True,
        },
        "source_records": {},
    }
    metric["measurement"].update(metric_overrides or {})
    source_fields = {
        "key_depth": ("key_depth_interval_m", "key_depth_source"),
        "alignment": ("alignment_within_limits",),
        "grasp_state": ("grasp_held",),
    }
    for name in ("key_depth", "alignment", "grasp_state"):
        raw = tmp_path / f"{name}_raw.json"
        raw.write_text(json.dumps({"synthetic_test_measurement": name}),
                       encoding="utf-8")
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "schema": "precision_insertion_external_metric_claim_v1",
            "source_name": name,
            "attempt_id": metric["attempt_id"],
            "candidate_id": metric["candidate_id"],
            "session_calibration_sha256": session_calibration_sha256,
            "recorded_at_s": 102.7,
            "producer_id": "synthetic_test_producer",
            "source_method": "synthetic_test_not_physical_sensor",
            "measurement": {field: metric["measurement"][field]
                            for field in source_fields[name]},
            "raw_evidence": [{"path": str(raw),
                              "sha256": hashlib.sha256(
                                  raw.read_bytes()).hexdigest()}],
        }), encoding="utf-8")
        metric["source_records"][name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    limits = GuardedContactLimits(
        target_depth_m=.020, max_axial_force_n=10.,
        max_lateral_force_n=5., max_torque_nm=.5,
        max_lateral_error_m=.002, max_axis_tilt_deg=4.,
        max_yaw_error_deg=5., max_sample_age_s=.05,
        max_sample_gap_s=.1, max_duration_s=.5,
        max_depth_step_m=.005, max_depth_regression_m=.0001,
        max_depth_overshoot_m=.0005)
    events = [(
        GuardedContactSample(
            timestamp_s=102.51 + .05 * i,
            nominal_depth_m=.004 * i,
            force_socket_n=(0., 0., 1.),
            moment_socket_nm=(0., 0., .01),
            lateral_error_m=.0005, axis_tilt_deg=1., yaw_error_deg=1.,
            hand_command_tracked=True),
        102.52 + .05 * i)
        for i in range(6)]
    trace = replay_guarded_contact_trace(
        attempt_id="trial_1", candidate_id="table/0/3", family="square",
        session_calibration_sha256=session_calibration_sha256,
        limits=limits, started_at_s=102.5, events=events)
    trace_path = write_guarded_contact_trace(
        trace, tmp_path / "force_trace.json")
    metric["source_records"]["force_trace"] = {
        "path": str(trace_path),
        "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    metric_path = tmp_path / "guarded_execution.json"
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    backend = FakeBackend(answer or {
        "visual_class": "normal_appearance",
        "evidence_views": list(CAMERAS),
        "evidence": "key appears seated in both views"})
    args = dict(
        attempt_id="trial_1", candidate_id="table/0/3",
        target_depth_m=0.02,
        session_calibration_sha256=session_calibration_sha256,
        preinsert_bundle=preinsert, final_bundle=final,
        metric_record_path=metric_path,
        preinsert_reached_at_s=101.9, decision_timestamp_s=103.1,
        backend=backend, max_phase_skew_s=0.02,
        max_preinsert_age_s=1.0, max_final_observation_gap_s=0.5)
    return args, backend


def test_multiview_appearance_cannot_certify_unverified_depth(tmp_path):
    args, backend = _setup(tmp_path)
    report = assess_insertion_checkpoint(**args)
    assert report["outcome"]["insertion_success"] is None
    assert report["key_depth_admissibility"]["status"] == "not_admitted"
    assert report["key_depth_admissibility"]["raw_key_depth_interval_m"] == [
        0.0201, 0.021]
    assert report["visual"]["raw_answer"]
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 4
    saved = write_insertion_checkpoint(report, tmp_path / "assessment")
    assert verify_insertion_checkpoint(saved)["outcome"]["insertion_success"] is None
    metric = args["metric_record_path"]
    metric.write_text(metric.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="metric record changed"):
        verify_insertion_checkpoint(saved)


def test_saved_local_vlm_json_fence_replays_at_insertion_checkpoint(tmp_path):
    answer = {"visual_class": "normal_appearance",
              "evidence_views": list(CAMERAS),
              "evidence": "key is visible in both views"}
    args, _backend = _setup(
        tmp_path, answer="```json\n" + json.dumps(answer) + "\n```")
    result = assess_insertion_checkpoint(**args)
    assert result["visual"]["parsed"] == answer
    saved = write_insertion_checkpoint(result, tmp_path / "local_insertion")
    replayed = verify_insertion_checkpoint(saved)
    assert replayed["visual"]["parsed"] == answer
    assert replayed["outcome"]["insertion_success"] is None


def test_vlm_one_view_cannot_produce_positive_label(tmp_path):
    args, _backend = _setup(tmp_path, answer={
        "visual_class": "normal_appearance", "evidence_views": ["cam_a"],
        "evidence": "one view only"})
    report = assess_insertion_checkpoint(**args)
    assert report["effective_vlm_class"] == "unobservable"
    assert report["outcome"]["insertion_success"] is None
    saved = write_insertion_checkpoint(report, tmp_path / "assessment")
    assert verify_insertion_checkpoint(saved)["outcome"]["insertion_success"] is None


def test_unverified_short_depth_cannot_drive_failure_or_success(tmp_path):
    args, _backend = _setup(tmp_path, metric_overrides={
        "key_depth_interval_m": [0.005, 0.008]})
    report = assess_insertion_checkpoint(**args)
    assert report["outcome"]["insertion_success"] is None
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


def test_depth_summary_cannot_disagree_with_hashed_producer_claim(tmp_path):
    args, backend = _setup(tmp_path)
    metric_path = args["metric_record_path"]
    metric = json.loads(metric_path.read_text())
    metric["measurement"]["key_depth_interval_m"] = [0.03, 0.031]
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="key_depth claim differs"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def test_changed_raw_metric_source_blocks_even_if_claim_hash_is_unchanged(
        tmp_path):
    args, backend = _setup(tmp_path)
    metric = json.loads(args["metric_record_path"].read_text())
    claim_path = Path(metric["source_records"]["alignment"]["path"])
    claim = json.loads(claim_path.read_text())
    raw_path = Path(claim["raw_evidence"][0]["path"])
    raw_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment raw evidence changed"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def test_guarded_metric_cannot_claim_no_abort_against_replayed_trace(tmp_path):
    args, backend = _setup(tmp_path)
    metric_path = args["metric_record_path"]
    metric = json.loads(metric_path.read_text())
    metric["measurement"]["safety_abort"] = True
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with replayed contact trace"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def test_guarded_metric_cannot_accept_a_fake_trace_summary(tmp_path):
    args, backend = _setup(tmp_path)
    metric_path = args["metric_record_path"]
    metric = json.loads(metric_path.read_text())
    source = metric["source_records"]["force_trace"]
    path = Path(source["path"])
    trace = json.loads(path.read_text())
    trace["safety_abort"] = True
    path.write_text(json.dumps(trace), encoding="utf-8")
    source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from replay"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def test_guarded_metric_is_bound_to_session_and_20mm_target(tmp_path):
    args, backend = _setup(tmp_path)
    with pytest.raises(ValueError, match="20 mm"):
        assess_insertion_checkpoint(**{**args, "target_depth_m": .010})
    metric_path = args["metric_record_path"]
    metric = json.loads(metric_path.read_text())
    metric["session_calibration_sha256"] = "f" * 64
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from this attempt"):
        assess_insertion_checkpoint(**args)
    assert backend.calls == []


def test_guarded_trace_cannot_be_reused_from_another_session(tmp_path):
    args, backend = _setup(tmp_path)
    metric_path = args["metric_record_path"]
    metric = json.loads(metric_path.read_text())
    source = metric["source_records"]["force_trace"]
    path = Path(source["path"])
    trace = json.loads(path.read_text())
    trace["session_calibration_sha256"] = "f" * 64
    path.write_text(json.dumps(trace), encoding="utf-8")
    source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with replayed contact trace"):
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
    assert report["outcome"]["insertion_success"] is None
    assert len(backend.calls[0][0]) == 4
    saved = write_insertion_checkpoint(report, tmp_path / "raw_assessment")
    assert verify_insertion_checkpoint(saved)["outcome"][
        "insertion_success"] is None


@pytest.mark.parametrize("raw_preinsert", [False, True])
def test_session_runner_records_bound_insertion_label(
        monkeypatch, tmp_path, raw_preinsert):
    mode = select_mode("square", 1.5)
    calibration = _session(tmp_path, mode, socket_x=0.0)
    calibration.record["schema"] = "precision_insertion_session_calibration_v1"
    args, backend = _setup(
        tmp_path,
        session_calibration_sha256=_canonical_sha256(calibration.record))
    if raw_preinsert:
        args["preinsert_bundle"] = _raw_preinsert_bundle(args, tmp_path)
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
    assert result["outcome"]["insertion_success"] is None
    assert runner.active_attempt.labels["insertion_success"] is None
    assert runner.current_decision().action == "stop_for_review"
    assessment = (runner._attempt_dir / "insertion_assessments" / "000" /
                  "report.json")
    assert verify_insertion_checkpoint(assessment)["outcome"][
        "insertion_success"] is None
