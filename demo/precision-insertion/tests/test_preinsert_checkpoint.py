"""A VLM overlay alone cannot create the observed preinsert arrival label."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
from precision_insertion.postlift_preflight import PostLiftPreflight  # noqa: E402
from precision_insertion.preinsert_checkpoint import (  # noqa: E402
    assess_preinsert_checkpoint, verify_preinsert_checkpoint,
    write_preinsert_checkpoint,
)
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from precision_insertion.records import begin_attempt  # noqa: E402
from test_key_perception import CAMERAS, _session  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


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
    def __init__(self, category="coarse_match"):
        self.category = category

    def infer(self, images, prompt):
        camera = "cam_a" if "Use only camera ID cam_a" in prompt else "cam_b"
        return json.dumps({
            "class": self.category,
            "evidence_views": ([] if self.category == "unobservable"
                               else [camera]),
            "evidence": ("key is visible by hand" if self.category !=
                         "unobservable" else "hidden"),
        })


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
        transfer_execution_path=transfer, joint_sample=measured,
        backend=_VLM(), max_capture_skew_s=0.02,
        max_joint_frame_skew_s=0.02,
        max_transfer_observation_gap_s=0.5,
        max_hand_translation_error_m=0.005,
        max_hand_rotation_error_deg=5.0,
        max_arm_hand_skew_s=0.02,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05,
        renderer_factory=_Renderer, robot_loader=lambda _path: _Robot())


def test_measured_preinsert_arrival_needs_multiview_visible_key(tmp_path):
    args = _setup(tmp_path)
    report = assess_preinsert_checkpoint(**args)
    assert report.preinsert_reached is True
    assert report.reason == "measured_arrival_and_visible_held_key"
    assert report.to_record()["robot_ready"] is False
    assert len(report.to_record()["comparison"]["views"]) == 2
    saved = write_preinsert_checkpoint(report, tmp_path / "arrival")
    assert verify_preinsert_checkpoint(saved)["preinsert_reached"] is True
    overlay = tmp_path / "arrival" / "overlays" / "cam_a.png"
    overlay.write_bytes(overlay.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="pixel evidence changed"):
        verify_preinsert_checkpoint(saved)
    assert assess_preinsert_checkpoint(
        **{**args, "backend": _VLM("unobservable")}).preinsert_reached is None


def test_preinsert_vlm_or_measured_pose_can_veto_arrival(tmp_path):
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
    assert report.preinsert_reached is False
    assert report.reason == "measured_hold_pose_outside_limits"


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
