"""A VLM overlay alone cannot create the observed preinsert arrival label."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.held_relation import HeldRelation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.observer import resize_classification_images  # noqa: E402
from precision_insertion.observer import LabeledFrame  # noqa: E402
from precision_insertion.postlift_preflight import PostLiftPreflight  # noqa: E402
from precision_insertion.preinsert_checkpoint import (  # noqa: E402
    _focus_temporal_frames, _replay_temporal_crops,
    assess_preinsert_checkpoint, verify_preinsert_checkpoint,
    write_preinsert_checkpoint,
)
from precision_insertion import preinsert_checkpoint  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from precision_insertion.runtime_evidence_cache import live_evidence_cache  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from test_key_perception import CAMERAS, _session  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def test_temporal_vlm_focus_replays_saved_pixels_and_includes_moved_socket(
        tmp_path, monkeypatch):
    from PIL import Image
    from precision_insertion import lift_visible_grip

    def geometry(q):
        transform = np.eye(4)
        transform[0, 3] = q[0]
        return transform, np.array([0., 0., 0., 1.])

    def pair_geometry(after_lift_q, preinsert_q, _urdf):
        return {"after_lift": geometry(after_lift_q),
                "preinsert": geometry(preinsert_q)}

    def hand_roi(*, image_size, camera, calibration, T_robot_hand,
                 focus_local):
        origin = 20 if T_robot_hand[0, 3] == 0 else 80
        return (origin, origin, origin + 64, origin + 64), [origin + 32.] * 2

    monkeypatch.setattr(lift_visible_grip, "_hand_foci_for_pair", pair_geometry)
    monkeypatch.setattr(lift_visible_grip, "_hand_roi", hand_roi)
    monkeypatch.setattr(lift_visible_grip, "HAND_CROP_SIDE", 64)
    image = Image.new("RGB", (256, 256), "white")
    frames = [LabeledFrame("cam_a", phase, index + 1., image)
              for index, phase in enumerate(("after_lift", "preinsert"))]
    focused, record = _focus_temporal_frames(
        frames, calibration={}, after_lift_q=[0], preinsert_q=[1],
        robot_urdf=tmp_path / "unused.urdf",
        socket_bboxes={"cam_a": {
            "projected_hull_bbox_px": [150, 150, 180, 180]}})
    pre_roi = record["per_camera"]["cam_a"]["preinsert"][
        "roi_xyxy_full_px"]
    assert pre_roi[0] <= 150 < 180 <= pre_roi[2]
    assert pre_roi[1] <= 150 < 180 <= pre_roi[3]
    _, wide_record = _focus_temporal_frames(
        frames, calibration={}, after_lift_q=[0], preinsert_q=[1],
        robot_urdf=tmp_path / "unused.urdf",
        socket_bboxes={"cam_a": {
            "projected_hull_bbox_px": [0, 0, 250, 250]}})
    assert wide_record["per_camera"]["cam_a"]["preinsert"][
        "roi_xyxy_full_px"] == [0, 0, 256, 256]
    assert [frame.image.tobytes() for frame in focused] == [
        frame.image.tobytes() for frame in _replay_temporal_crops(frames, record)]
    changed = [frames[0], replace(frames[1], image=Image.new(
        "RGB", (256, 256), "black"))]
    with pytest.raises(ValueError, match="crop pixels changed"):
        _replay_temporal_crops(changed, record)


class _Robot:
    def __init__(self):
        self.actuated_joints = [SimpleNamespace(name=f"fr3_joint{i}")
                                for i in range(7)] + [
            SimpleNamespace(name=f"right_joint{i}") for i in range(6)]
        self.base_link = "fr3_link0"
        self.scene = trimesh.Scene()
        self.scene.add_geometry(trimesh.creation.box((0.1, 0.1, 0.1)),
                                geom_name="fr3_link0_visual")
        self.scene.add_geometry(trimesh.creation.box((0.01, 0.01, 0.02)),
                                geom_name="right_index_visual")

    def update_cfg(self, _cfg):
        pass

    def get_transform(self, source, target):
        assert (source, target) == ("base_link", self.base_link)
        T = np.eye(4)
        T[0, 3] = 0.1
        return T


class _Renderer:
    def __init__(self, meshes, names, labels, intrinsics, extrinsics, H, W):
        self.serials = sorted(intrinsics)
        self.n_links = len(names)
        self.color_lut = torch.zeros((len(names) + 1, 3))
        self.alpha_lut = torch.zeros((len(names) + 1, 1))
        assert len(meshes) == len(names)

    def render(self, poses, images):
        return [image.copy() for image in images]


class _VLM:
    def __init__(self, category="coarse_match", *, fenced=False):
        self.category = category
        self.fenced = fenced

    def infer(self, images, prompt):
        if "RAW after-lift" in prompt:
            assert len(images) == 2 and "RAW preinsert" in prompt
        else:
            assert len(images) == 1  # historical retry diagnostic
        camera = "cam_a" if "Use only camera ID cam_a" in prompt else "cam_b"
        answer = json.dumps({
            "class": self.category,
            "evidence_views": ([] if self.category == "unobservable"
                               else [camera]),
            "evidence": ("key is visible by hand" if self.category !=
                         "unobservable" else "hidden"),
        })
        return "```json\n" + answer + "\n```" if self.fenced else answer


def _setup(tmp_path):
    mode = select_mode("square", 1.5)
    calibration = _session(tmp_path, mode)
    calibration.record["schema"] = "precision_insertion_session_calibration_v1"
    for object_id in (mode.key_object, mode.socket_object):
        path = (tmp_path / "object_processing" / object_id / "raw_mesh" /
                f"{object_id}.obj")
        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box((0.02, 0.02, 0.04)).export(path)
    session_hash = hashlib.sha256(json.dumps(
        calibration.record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    attempt = begin_attempt(
        attempt_id="trial_1", mode=mode, session_record=calibration.record,
        candidate_id="table/0/3", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=10.0)
    attempt.record_stage(
        "grasp_success", True, timestamp_s=10.4,
        evidence_refs={"vlm_observation": "saved/lift.json",
                       "key_wrist_check": "saved/key_wrist.json"})
    T = np.eye(4)
    target = T.copy()
    target[0, 3] = 0.1
    relation = HeldRelation(T, T, np.zeros(3), 0.0, 0.0, "identity")
    postlift = PostLiftPreflight(
        "sampled_postlift_preflight_pass", "trial_1", ("table", "0", "3"),
        session_hash, "1" * 64, "key_1", "held_1", 10.5, 10.5,
        np.zeros(13), None, T, T, relation, None,
        SimpleNamespace(T_key_hand=T, T_robot_hand_preinsert=target),
        SimpleNamespace(sampled_planning_pass=True))
    postlift_file = tmp_path / "postlift.json"
    postlift_file.write_text(json.dumps({
        "status": postlift.status, "attempt_id": postlift.attempt_id,
        "candidate_key": list(postlift.candidate_key),
        "session_calibration_sha256": session_hash,
        "observed_held_relation": {"T_key_hand": T.tolist()},
        "targets": {"T_robot_hand_preinsert": target.tolist()},
    }), encoding="utf-8")
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    lifted = image.copy()
    lifted[:, :, 1] = 80
    lift_ids = {"cam_a": 21, "cam_b": 22}
    lift_evidence = {
        camera: {"frame_id": lift_ids[camera],
                 "image_sha256": image_sha256(lifted),
                 "timestamp_s": 10.55 + i * 0.005,
                 "max_error_s": 0.001,
                 "timestamp_method": "hardware_exposure",
                 "clock_domain": "unix_utc"}
        for i, camera in enumerate(CAMERAS)}
    after_lift_bundle = write_raw_camera_capture(
        RawCameraCapture(
            "lift_1", 49, {camera: lifted.copy() for camera in CAMERAS},
            lift_ids, {"request_id": 49, "source": "camera_acquisition",
                       "frames": lift_evidence}),
        tmp_path / "after_lift", phase="after_lift")
    frame_ids = {"cam_a": 31, "cam_b": 32}
    evidence = {
        camera: {"frame_id": frame_ids[camera],
                 "image_sha256": image_sha256(image),
                 "timestamp_s": 11.0 + i * 0.005,
                 "max_error_s": 0.001,
                 "timestamp_method": "hardware_exposure",
                 "clock_domain": "unix_utc"}
        for i, camera in enumerate(CAMERAS)}
    bundle = write_raw_camera_capture(
        RawCameraCapture(
            "hold_1", 50, {camera: image.copy() for camera in CAMERAS},
            frame_ids, {"request_id": 50, "source": "camera_acquisition",
                        "frames": evidence}),
        tmp_path / "preinsert", phase="preinsert")
    source_records = {}
    for name in ("trajectory_feedback", "safety", "grasp_state"):
        source = tmp_path / f"{name}.json"
        source.write_text(json.dumps({"source": name}), encoding="utf-8")
        source_records[name] = {"path": str(source),
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    transfer = tmp_path / "transfer.json"
    transfer.write_text(json.dumps({
        "schema": "precision_insertion_transfer_execution_v1",
        "attempt_id": "trial_1", "candidate_id": "table/0/3",
        "started_at_s": 10.6, "completed_at_s": 10.8,
        "measurement": {"trajectory_complete": True,
                        "safety_abort": False, "grasp_held": True},
        "source_records": source_records,
    }), encoding="utf-8")
    raw = np.zeros(6)
    q = np.zeros(13)
    q[7:] = convert_inspire_raw(raw[None, :])[0]
    measured = LiveRobotState(
        q, np.zeros(7), 11.0025, 11.0025, 50.0, 11.0025,
        raw, raw.copy(), 0.0, np.zeros(6))
    return dict(
        attempt=attempt, postlift=postlift,
        postlift_report_path=postlift_file, calibration=calibration,
        shared_root=tmp_path, mode=mode, raw_bundle=bundle,
        after_lift_raw_bundle=after_lift_bundle,
        transfer_execution_path=transfer, joint_sample=measured,
        backend=_VLM(), max_capture_skew_s=0.02,
        max_joint_frame_skew_s=0.02,
        max_arm_hand_skew_s=0.02,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05,
        renderer_factory=_Renderer, robot_loader=lambda _path: _Robot())


def test_raw_preinsert_vlm_starts_before_diagnostic_overlay(tmp_path):
    vlm_started = Event()
    overlay_started = Event()

    class VLM(_VLM):
        independent_remote_inference = True

        def infer(self, images, prompt):
            vlm_started.set()
            assert overlay_started.wait(5), "overlay should run with raw VLM"
            return super().infer(images, prompt)

    class Renderer(_Renderer):
        def __init__(self, *args):
            assert vlm_started.wait(5), "raw VLM should run during overlay work"
            overlay_started.set()
            super().__init__(*args)

    args = _setup(tmp_path)
    args.update(backend=VLM(), renderer_factory=Renderer)
    assert assess_preinsert_checkpoint(**args).preinsert_reached is True


def test_temporal_preinsert_qwen_input_needs_no_gpu_overlay(tmp_path):
    class UnexpectedRenderer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("raw temporal VLM input must not render")

    args = _setup(tmp_path)
    checkpoint = assess_preinsert_checkpoint(**{
        **args, "render_audit_overlay": False,
        "renderer_factory": UnexpectedRenderer})
    assert checkpoint.preinsert_reached is True
    assert checkpoint.comparison.to_record()["schema"] == (
        "precision_insertion_held_scene_comparison_v2")
    saved = write_preinsert_checkpoint(checkpoint, tmp_path / "raw_arrival")
    assert not (saved.parent / "overlays").exists()
    record = verify_preinsert_checkpoint(saved)
    assert record["preinsert_reached"] is True
    assert "overlay_png_sha256" not in record


def test_measured_preinsert_arrival_needs_multiview_visible_key(tmp_path):
    args = _setup(tmp_path)
    report = assess_preinsert_checkpoint(**args)
    assert report.preinsert_reached is True
    assert report.reason == "measured_arrival_and_visible_held_key"
    assert report.to_record()["robot_ready"] is False
    assert len(report.to_record()["comparison"]["views"]) == 2
    assert all(len(row["image_order"]) == 2 for row in
               report.to_record()["visual"]["per_view"])
    saved = write_preinsert_checkpoint(report, tmp_path / "arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True
    original_report = saved.read_bytes()
    wrong_consensus = json.loads(original_report)
    wrong_consensus["visual"]["status"] = "unknown"
    saved.write_text(json.dumps(wrong_consensus), encoding="utf-8")
    with pytest.raises(ValueError, match="consensus differs"):
        verify_preinsert_checkpoint(saved)
    wrong_label = json.loads(original_report)
    wrong_label["preinsert_reached"] = False
    saved.write_text(json.dumps(wrong_label), encoding="utf-8")
    with pytest.raises(ValueError, match="label contradicts"):
        verify_preinsert_checkpoint(saved)
    saved.write_bytes(original_report)


    overlay = tmp_path / "arrival" / "overlays" / "cam_a.png"
    overlay.write_bytes(overlay.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="pixel evidence changed"):
        verify_preinsert_checkpoint(saved)
    saved.write_bytes(original_report)
    lifted_png = args["after_lift_raw_bundle"] / "images" / "cam_a.png"
    original_lifted_png = lifted_png.read_bytes()
    lifted_png.write_bytes(original_lifted_png + b"tampered")
    with pytest.raises(ValueError, match="raw camera PNG changed"):
        verify_preinsert_checkpoint(saved)
    lifted_png.write_bytes(original_lifted_png)
    assert assess_preinsert_checkpoint(
        **{**args, "backend": _VLM("unobservable")}).preinsert_reached is None


def test_live_preinsert_reuses_producer_result_and_rechecks_changed_report(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path)
    with live_evidence_cache():
        saved = write_preinsert_checkpoint(
            assess_preinsert_checkpoint(**args), tmp_path / "cached_arrival")
        original = saved.read_bytes()

        def unexpected_capture_read(*_args, **_kwargs):
            raise AssertionError("preinsert sources were read again")

        monkeypatch.setattr(preinsert_checkpoint, "verify_raw_camera_capture",
                            unexpected_capture_read)
        verified = verify_preinsert_checkpoint(saved)
        assert verified["preinsert_reached"] is True
        assert verify_preinsert_checkpoint(saved) is verified
        saved.write_bytes(original + b" ")
        with pytest.raises(AssertionError, match="read again"):
            verify_preinsert_checkpoint(saved)
        saved.write_bytes(original)
    with pytest.raises(AssertionError, match="read again"):
        verify_preinsert_checkpoint(saved)


def test_resized_preinsert_vlm_inputs_replay_from_saved_raw_frames(tmp_path):
    class ResizedVLM(_VLM):
        def prepare_classification_images(self, images):
            return resize_classification_images(images, (16, 12))

    args = _setup(tmp_path)
    result = assess_preinsert_checkpoint(**{**args, "backend": ResizedVLM()})
    saved = write_preinsert_checkpoint(result, tmp_path / "resized_arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True
    changed = json.loads(saved.read_text(encoding="utf-8"))
    changed["visual"]["per_view"][0]["input_transform"]["images"][0][
        "input_rgb_sha256"] = "0" * 64
    saved.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="replay"):
        verify_preinsert_checkpoint(saved)


def test_preinsert_accepts_late_new_capture_with_aligned_feedback(tmp_path):
    args = _setup(tmp_path)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    ids = {"cam_a": 41, "cam_b": 42}
    rows = {
        camera: {
            "frame_id": ids[camera], "image_sha256": image_sha256(image),
            "timestamp_s": 211.0 + index * .005, "max_error_s": .001,
            "timestamp_method": "hardware_exposure", "clock_domain": "unix_utc",
        }
        for index, camera in enumerate(CAMERAS)
    }
    args["raw_bundle"] = write_raw_camera_capture(
        RawCameraCapture("late_hold", 51,
                         {camera: image.copy() for camera in CAMERAS}, ids,
                         {"request_id": 51, "source": "camera_acquisition",
                          "frames": rows}),
        tmp_path / "late_preinsert", phase="preinsert")
    args["joint_sample"] = replace(
        args["joint_sample"], sample_timestamp_s=211.0025,
        arm_timestamp_s=211.0025, hand_timestamp_s=211.0025)
    report = assess_preinsert_checkpoint(**args)
    assert report.preinsert_reached is True
    saved = write_preinsert_checkpoint(report, tmp_path / "late_arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True


def test_preinsert_replays_local_fenced_answer_and_rejects_changed_label(
        tmp_path):
    args = _setup(tmp_path)
    report = assess_preinsert_checkpoint(
        **{**args, "backend": _VLM(fenced=True)})
    saved = write_preinsert_checkpoint(report, tmp_path / "fenced_arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True
    altered = json.loads(saved.read_text(encoding="utf-8"))
    altered["visual"]["per_view"][0]["parsed"]["class"] = "slip_or_miss"
    altered["visual"]["status"] = "unknown"
    altered["visual"]["supporting_cameras"] = []
    altered["preinsert_reached"] = None
    altered["reason"] = "preinsert_evidence_incomplete_or_occluded"
    saved.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from replay"):
        verify_preinsert_checkpoint(saved)


def test_preinsert_vlm_vetoes_arrival_and_wrist_error_is_diagnostic(tmp_path):
    args = _setup(tmp_path)
    assert assess_preinsert_checkpoint(
        **{**args, "backend": _VLM("slip_or_miss")}).preinsert_reached is False
    target = args["postlift"].targets.T_robot_hand_preinsert.copy()
    target[0, 3] += 0.01
    args["postlift"].targets.T_robot_hand_preinsert[:] = target
    saved = json.loads(args["postlift_report_path"].read_text())
    saved["targets"]["T_robot_hand_preinsert"] = target.tolist()
    args["postlift_report_path"].write_text(json.dumps(saved), encoding="utf-8")
    report = assess_preinsert_checkpoint(**args)
    assert report.preinsert_reached is True
    assert report.hand_translation_error_m > .005


def test_same_trial_gross_xy_misalignment_remains_a_held_arrival(tmp_path):
    args = _setup(tmp_path)
    args.update(backend=_VLM("gross_misalignment"), label_mode="vlm_only")
    assert assess_preinsert_checkpoint(**args).preinsert_reached is False
    report = assess_preinsert_checkpoint(
        **args, allow_held_misalignment=True)
    assert report.preinsert_reached is True
    assert report.reason == "held_arrival_xy_refinement_pending"
    assert report.to_record()["vlm_label"]["value"] is False
    saved = write_preinsert_checkpoint(report, tmp_path / "alignment_pending")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True
    assert assess_preinsert_checkpoint(
        **{**args, "backend": _VLM("slip_or_miss")},
        allow_held_misalignment=True).preinsert_reached is False


def test_preinsert_rejects_stale_or_tampered_external_evidence(tmp_path):
    args = _setup(tmp_path)
    altered = json.loads(args["transfer_execution_path"].read_text())
    altered["completed_at_s"] = 11.1
    args["transfer_execution_path"].write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="do not follow"):
        assess_preinsert_checkpoint(**args)
    altered["completed_at_s"] = 10.8
    args["transfer_execution_path"].write_text(json.dumps(altered), encoding="utf-8")
    source = Path(altered["source_records"]["safety"]["path"])
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="producer record changed"):
        assess_preinsert_checkpoint(**args)


def test_preinsert_rejects_unbound_postlift_plan(tmp_path):
    args = _setup(tmp_path)
    args["postlift_report_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="saved post-lift plan"):
        assess_preinsert_checkpoint(**args)


def test_preinsert_pair_is_bound_to_successful_lift_checkpoint(
        tmp_path, monkeypatch):
    args = _setup(tmp_path)
    lift_path = tmp_path / "lift_checkpoint.json"
    lift_path.write_text('{"saved":true}', encoding="utf-8")
    lift_record = {
        "schema": "precision_insertion_raw_lift_checkpoint_v1",
        "attempt_id": args["attempt"].attempt_id,
        "candidate_id": args["attempt"].candidate_id,
        "grasp_success": True,
        "after_bundle": str(args["after_lift_raw_bundle"]),
    }
    monkeypatch.setattr(preinsert_checkpoint, "verify_lift_checkpoint",
                        lambda _path: lift_record)
    report = assess_preinsert_checkpoint(
        **args, lift_checkpoint_report_path=lift_path)
    saved = write_preinsert_checkpoint(report, tmp_path / "bound_arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True

    lift_record["attempt_id"] = "some_other_trial"
    with pytest.raises(ValueError, match="binding changed"):
        verify_preinsert_checkpoint(saved)
    with pytest.raises(ValueError, match="differs from this attempt"):
        assess_preinsert_checkpoint(
            **args, lift_checkpoint_report_path=lift_path)
