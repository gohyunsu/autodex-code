"""One fresh key capture must precede measured-state v8 preflight."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion import live_trial_start as start  # noqa: E402


def _setup(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    shared = tmp_path / "shared"
    paths = AssetPaths(shared, mode)
    for path in (paths.raw_mesh(mode.key_object),
                 paths.foundpose_repre(mode.key_object)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test asset")
    source = tmp_path / "captures"
    source.mkdir()
    events = []
    runner = object.__new__(SessionRunner)
    runner.mode = mode
    runner.shared_root = shared
    runner.calibration = object()
    done = [False]
    runner.current_decision = lambda: SimpleNamespace(
        action="plan_pickup" if done[0] else "capture_fresh_key")

    def preflight_next_key(**kwargs):
        events.append("v8_preflight")
        assert kwargs["measured_start_state"] is state
        assert kwargs["start_q_acquisition_timestamp_s"] == 100.01
        assert kwargs["key_evidence_dir"] == tmp_path / "key_evidence"
        assert kwargs["max_candidate_attempts"] == 3
        done[0] = True
        return SimpleNamespace(status="sampled_trial_preflight_pass")

    runner.preflight_next_key = preflight_next_key

    class Init:
        obj_name = None
        intrinsics_undist = {"cam_a": np.eye(3), "cam_b": np.eye(3)}
        extrinsics = {"cam_a": np.eye(4), "cam_b": np.eye(4)}

        def init_object(self, **kwargs):
            events.append("init_key_foundpose")
            assert kwargs["obj_name"] == mode.key_object
            assert kwargs["mesh_path"] == str(paths.raw_mesh(mode.key_object))
            assert kwargs["load_silhouette"] is False
            self.obj_name = kwargs["obj_name"]

    init = Init()
    def validate_calibration(_calibration, **kwargs):
        events.append("verify_frozen_camera_calibration")
        assert kwargs["calibrated_camera_ids"] == {"cam_a", "cam_b"}

    monkeypatch.setattr(
        start, "validate_session_camera_calibration", validate_calibration)
    capture = SimpleNamespace(capture_id="key_001", request_id=33)

    def collect(**kwargs):
        events.append("capture_key")
        assert init.obj_name == mode.key_object
        assert kwargs["key_prompt"] == "blue insertion key on the board"
        return capture

    def aligned(**kwargs):
        events.append("key_state_alignment")
        assert kwargs["state_timestamp_s"] == 100.01

    observation = SimpleNamespace(
        capture_id="key_001", request_id=33,
        require_state_alignment=aligned)

    def admit(**kwargs):
        events.append("admit_multiview_key")
        assert kwargs["capture"] is capture
        assert kwargs["calibration"] is runner.calibration
        return observation

    def save(raw, admitted, path):
        events.append("save_key_evidence")
        assert raw is capture and admitted is observation
        path.mkdir()
        return path

    def verify(path):
        events.append("verify_key_evidence")
        assert path == tmp_path / "key_evidence"
        return {"capture_id": "key_001", "request_id": 33}

    class State:
        full_q = np.zeros(13)
        sample_timestamp_s = 100.01

        def validate(self, **kwargs):
            events.append("validate_measured_state")
            assert kwargs["max_arm_hand_skew_s"] == .02

    state = State()
    monkeypatch.setattr(start, "LiveRobotState", State)
    for name, func in (
            ("collect_key_capture", collect),
            ("admit_key_capture", admit),
            ("write_key_capture_artifacts", save),
            ("verify_key_capture_artifacts", verify)):
        monkeypatch.setattr(start, name, func)

    def read_state():
        events.append("read_measured_state")
        return state

    limits = start.KeyTrialCaptureLimits(
        SocketViewLimits(50, .5, 10, 2, .02),
        2., 5., .1, 2, .5, .02, 30., .05, .04)
    args = dict(
        runner=runner, planner=object(), init_orchestrator=init,
        acquisition_metadata_for_request=lambda _request: {},
        read_measured_state=read_state,
        capture_root=source, key_evidence_dir=tmp_path / "key_evidence",
        capture_id="key_001",
        key_prompt="blue insertion key on the board",
        calibrated_camera_ids={"cam_a", "cam_b"},
        intrinsics_full={"cam_a": {}, "cam_b": {}},
        extrinsics_full={"cam_a": {}, "cam_b": {}},
        image_hw=(1536, 2048), capture_limits=limits,
        path_limits=SimpleNamespace(validate=lambda: None),
        max_pose_error_deg=5., axial_waypoint_step_m=.002,
        timeout_s=5., max_candidate_attempts=3)
    return args, events, paths


def test_key_capture_state_and_v8_preflight_order(tmp_path, monkeypatch):
    args, events, _paths = _setup(tmp_path, monkeypatch)
    prepared = start.capture_and_preflight_next_key(**args)
    assert events == [
        "init_key_foundpose", "verify_frozen_camera_calibration",
        "capture_key", "admit_multiview_key",
        "save_key_evidence", "verify_key_evidence", "read_measured_state",
        "validate_measured_state", "key_state_alignment", "v8_preflight"]
    assert prepared.observation.capture_id == "key_001"
    assert prepared.preflight.status == "sampled_trial_preflight_pass"
    assert prepared.next_decision.action == "plan_pickup"
    assert prepared.robot_ready is False


def test_key_capture_rejects_wrong_session_stage_before_camera(
        tmp_path, monkeypatch):
    args, events, _paths = _setup(tmp_path, monkeypatch)
    args["runner"].current_decision = lambda: SimpleNamespace(
        action="await_guarded_insertion_and_observation")
    with pytest.raises(ValueError, match="cannot capture a new key"):
        start.capture_and_preflight_next_key(**args)
    assert events == []


def test_key_capture_rejects_missing_foundpose_before_camera(
        tmp_path, monkeypatch):
    args, events, paths = _setup(tmp_path, monkeypatch)
    paths.foundpose_repre(args["runner"].mode.key_object).unlink()
    with pytest.raises(FileNotFoundError, match="FoundPose representation"):
        start.capture_and_preflight_next_key(**args)
    assert events == []


def test_changed_camera_calibration_rejects_before_capture(
        tmp_path, monkeypatch):
    args, events, _paths = _setup(tmp_path, monkeypatch)
    def changed(*_args, **_kwargs):
        raise ValueError("live camera calibration changed")

    monkeypatch.setattr(start, "validate_session_camera_calibration", changed)
    with pytest.raises(ValueError, match="calibration changed"):
        start.capture_and_preflight_next_key(**args)
    assert events == ["init_key_foundpose"]


def test_key_capture_keeps_evidence_but_does_not_plan_without_measured_state(
        tmp_path, monkeypatch):
    args, events, _paths = _setup(tmp_path, monkeypatch)
    args["read_measured_state"] = lambda: None
    with pytest.raises(TypeError, match="measured FR3/Inspire feedback"):
        start.capture_and_preflight_next_key(**args)
    assert (tmp_path / "key_evidence").is_dir()
    assert "v8_preflight" not in events
