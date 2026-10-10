"""A fresh key observation must precede and bind the v8 trial preflight."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion import live_key_trial as trial  # noqa: E402


def _measured_state(timestamp=100.005):
    raw = np.full(6, 500., dtype=float)
    q = np.concatenate((np.zeros(7), convert_inspire_raw(raw[None, :])[0]))
    return LiveRobotState(
        q, np.zeros(7), timestamp, timestamp, 500., timestamp,
        raw.copy(), raw.copy(), 0., np.zeros(6))


def _setup(tmp_path, monkeypatch, *, state_timestamp=100.005):
    shared = tmp_path / "shared"
    mode = select_mode("square", 1.5)
    mesh = (shared / "object_processing" / mode.key_object / "raw_mesh" /
            f"{mode.key_object}.obj")
    repre = (shared / "AutoDex/foundpose_assets" / mode.key_object /
             "object_repre/v1" / mode.key_object / "1/repre.pth")
    for path in (mesh, repre):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test asset")
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    events = []
    runner = object.__new__(SessionRunner)
    runner.mode = mode
    runner.shared_root = shared
    runner.calibration = object()
    runner.current_decision = lambda: SimpleNamespace(action="capture_fresh_key")

    def preflight_next_key(**kwargs):
        events.append("preflight")
        assert kwargs["key_observation"].capture_id == "key_001"
        assert np.array_equal(kwargs["live_start_q"], _measured_state().full_q)
        assert kwargs["start_q_acquisition_timestamp_s"] == state_timestamp
        assert kwargs["key_evidence_dir"] == tmp_path / "key_evidence"
        return SimpleNamespace(status="sampled_planning_pass")

    runner.preflight_next_key = preflight_next_key

    class Init:
        obj_name = None

        def init_object(self, **kwargs):
            events.append("init_key")
            assert kwargs["obj_name"] == mode.key_object
            assert kwargs["mesh_path"] == str(mesh)
            assert kwargs["assets_root"] == str(repre.parents[4])
            assert kwargs["load_silhouette"] is True
            self.obj_name = mode.key_object

    init = Init()
    capture = SimpleNamespace(capture_id="key_001", request_id=25)
    observation = KeyPoseObservation(
        "key_001", 25, mode.key_object, mode.family, np.eye(4), "cam_a",
        100.005, (100., 100.01), {}, {}, {}, {}, capture_root / "key_001",
    )

    def collect(**kwargs):
        events.append("capture")
        assert init.obj_name == mode.key_object
        assert kwargs["key_prompt"] == "blue precision key, excluding red socket"
        return capture

    def admit(**kwargs):
        events.append("admit")
        assert kwargs["calibration"] is runner.calibration
        return observation

    def write(_capture, _observation, output):
        events.append("save_key_evidence")
        output.mkdir()

    def verify(output):
        events.append("verify_key_evidence")
        assert output.is_dir()
        return {}

    for name, value in (("collect_key_capture", collect),
                        ("admit_key_capture", admit),
                        ("write_key_capture_artifacts", write),
                        ("verify_key_capture_artifacts", verify)):
        monkeypatch.setattr(trial, name, value)

    def state_provider(received):
        events.append("measured_state_at_exposure")
        assert received is capture
        return _measured_state(state_timestamp)

    arguments = {
        "runner": runner, "init_orchestrator": init,
        "acquisition_metadata_for_request": lambda _request: {},
        "state_at_capture": state_provider,
        "capture_root": capture_root,
        "key_evidence_dir": tmp_path / "key_evidence",
        "capture_id": "key_001",
        "calibrated_camera_ids": {"cam_a", "cam_b"},
        "intrinsics_full": {"cam_a": {}, "cam_b": {}},
        "extrinsics_full": {"cam_a": {}, "cam_b": {}},
        "image_hw": (1536, 2048),
        "key_prompt": "blue precision key, excluding red socket",
        "view_limits": SocketViewLimits(50, .5, 10, 2, .02),
        "maximum_multiview_center_error_mm": 3.,
        "maximum_multiview_angle_error_deg": 5.,
        "maximum_socket_mask_overlap_fraction": .05,
        "socket_projection_dilation_px": 4,
        "max_arm_hand_skew_s": .02,
        "max_hand_command_error_raw": 20.,
        "max_arm_velocity_rad_s": .1,
        "max_key_state_skew_s": .05,
        "planner": object(), "limits": object(),
        "max_pose_error_deg": 5.,
        "axial_waypoint_step_m": .002,
        "capture_timeout_s": 15.,
        "image_write_timeout_s": 5.,
    }
    return arguments, events, repre


def test_live_key_capture_state_and_preflight_order(tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    result = trial.prepare_next_live_key(**arguments)
    assert events == ["init_key", "capture", "admit", "save_key_evidence",
                      "verify_key_evidence", "measured_state_at_exposure",
                      "preflight"]
    assert result.observation.capture_id == "key_001"
    assert result.preflight.status == "sampled_planning_pass"


def test_missing_canonical_foundpose_blocks_before_camera(tmp_path, monkeypatch):
    arguments, events, repre = _setup(tmp_path, monkeypatch)
    repre.unlink()
    with pytest.raises(FileNotFoundError, match="key FoundPose representation"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


def test_stale_robot_feedback_cannot_enter_preflight(tmp_path, monkeypatch):
    arguments, events, _repre = _setup(
        tmp_path, monkeypatch, state_timestamp=101.)
    with pytest.raises(ValueError, match="not aligned to every key view"):
        trial.prepare_next_live_key(**arguments)
    assert events[-1] == "measured_state_at_exposure"
    assert arguments["key_evidence_dir"].is_dir()


def test_not_ready_session_and_existing_evidence_block_capture(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    arguments["runner"].current_decision = lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment")
    with pytest.raises(ValueError, match="not ready for a fresh key capture"):
        trial.prepare_next_live_key(**arguments)
    assert events == []
    arguments["runner"].current_decision = lambda: SimpleNamespace(
        action="capture_fresh_key")
    arguments["key_evidence_dir"].mkdir()
    with pytest.raises(FileExistsError, match="key evidence already exists"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


def test_runner_owned_planning_inputs_cannot_override_new_capture(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    arguments["planning_options"] = {"attempted": ()}
    with pytest.raises(ValueError, match="runner-owned planning options"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


def test_invalid_commissioning_threshold_blocks_before_camera(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    arguments["axial_waypoint_step_m"] = .010
    with pytest.raises(ValueError, match="may not exceed 5 mm"):
        trial.prepare_next_live_key(**arguments)
    assert events == []
