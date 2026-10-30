"""A fresh key observation must precede and bind the v8 trial preflight."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_capture import KeyCaptureInput  # noqa: E402
from precision_insertion.initial_visual import drain_pending_initial_visuals  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.clear_view import ClearViewForPerception  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.session_key_baseline import SessionKeyBaseline  # noqa: E402
from precision_insertion.registered_key_capture import RegisteredKeyAssistance  # noqa: E402
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
        assert isinstance(kwargs["measured_start_state"], LiveRobotState)
        assert kwargs["measured_state_limits"] == {
            "max_arm_hand_skew_s": .02,
            "max_hand_command_error_raw": 20.,
            "max_arm_velocity_rad_s": .1,
        }
        assert kwargs["key_evidence_dir"] == tmp_path / "key_evidence"
        return SimpleNamespace(status="sampled_planning_pass")

    runner.preflight_next_key = preflight_next_key

    class Init:
        obj_name = None
        intrinsics_undist = {"cam_a": np.eye(3), "cam_b": np.eye(3)}
        extrinsics = {"cam_a": np.eye(4), "cam_b": np.eye(4)}

        def init_object(self, **kwargs):
            events.append("init_key")
            assert kwargs["obj_name"] == mode.key_object
            assert kwargs["mesh_path"] == str(mesh)
            assert kwargs["assets_root"] == str(repre.parents[4])
            assert kwargs["load_silhouette"] is False
            self.obj_name = mode.key_object

    init = Init()

    def verify_frozen_camera(_calibration, **kwargs):
        events.append("verify_frozen_camera")
        assert kwargs["calibrated_camera_ids"] == {"cam_a", "cam_b"}

    monkeypatch.setattr(
        trial, "validate_session_camera_calibration", verify_frozen_camera)
    capture = SimpleNamespace(capture_id="key_001", request_id=25)
    observation = KeyPoseObservation(
        "key_001", 25, mode.key_object, mode.family, np.eye(4), "cam_a",
        100.005, (100., 100.01), {}, {}, {}, {}, capture_root / "key_001",
    )

    def collect(**kwargs):
        events.append("capture")
        assert init.obj_name == mode.key_object
        assert kwargs["key_prompt"] == "white precision key, excluding white socket"
        return capture

    def admit(**kwargs):
        events.append("admit")
        assert kwargs["calibration"] is runner.calibration
        assert kwargs["minimum_refinement_iou"] == 0.5
        return observation

    def write(_capture, _observation, output):
        events.append("save_key_evidence")
        output.mkdir()

    for name, value in (("collect_key_capture", collect),
                        ("admit_key_capture", admit),
                        ("write_key_capture_artifacts", write)):
        monkeypatch.setattr(trial, name, value)

    def prepare_renderer(**kwargs):
        events.append("compatible_renderer")
        assert kwargs["init_orchestrator"] is init
        assert kwargs["object_name"] == mode.key_object
        assert kwargs["raw_mesh"] == mesh

    monkeypatch.setattr(trial, "prepare_key_silhouette", prepare_renderer)

    def state_provider(received):
        events.append("measured_state_at_exposure")
        assert received is observation
        return _measured_state(state_timestamp)

    arguments = {
        "runner": runner, "init_orchestrator": init,
        "acquisition_metadata_for_request": lambda _request: {},
        "state_for_observation": state_provider,
        "capture_root": capture_root,
        "key_evidence_dir": tmp_path / "key_evidence",
        "capture_id": "key_001",
        "calibrated_camera_ids": {"cam_a", "cam_b"},
        "intrinsics_full": {"cam_a": {}, "cam_b": {}},
        "extrinsics_full": {"cam_a": {}, "cam_b": {}},
        "image_hw": (1536, 2048),
        "key_prompt": "white precision key, excluding white socket",
        "view_limits": SocketViewLimits(50, .5, 10, 2, .02),
        "maximum_multiview_center_error_mm": 3.,
        "maximum_multiview_angle_error_deg": 5.,
        "maximum_socket_mask_overlap_fraction": .05,
        "socket_projection_dilation_px": 4,
        "minimum_refinement_iou": .5,
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
    result = trial.prepare_next_live_key(
        **arguments, before_preflight=lambda: events.append("planner_ready"))
    assert events == ["compatible_renderer", "init_key", "verify_frozen_camera",
                      "capture", "admit", "save_key_evidence",
                      "measured_state_at_exposure",
                      "planner_ready",
                      "preflight"]
    assert result.observation.capture_id == "key_001"
    assert result.preflight.status == "sampled_planning_pass"


def test_planner_warmup_error_is_seen_after_capture_before_preflight(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)

    def failed_warmup():
        events.append("planner_ready")
        raise RuntimeError("fixed world warmup failed")

    with pytest.raises(RuntimeError, match="fixed world warmup failed"):
        trial.prepare_next_live_key(
            **arguments, before_preflight=failed_warmup)
    assert events[-2:] == ["measured_state_at_exposure", "planner_ready"]
    assert "preflight" not in events


def test_initial_visual_capture_overlaps_key_candidate_planning(
        tmp_path, monkeypatch):
    arguments, events, _ = _setup(tmp_path, monkeypatch)
    visual_started = threading.Event()
    planning_done = threading.Event()
    allow_save = threading.Event()
    def pose_collect(**kwargs):
        assert kwargs["pose_only"] is True
        events.append("capture")
        return KeyCaptureInput(
            "key_001", 25, "white precision key", {}, {}, {}, {},
            "camera_acquisition", {}, tmp_path / "unused")

    def save_pose(_capture, _observation, output):
        events.append("save_pose")
        output.mkdir()

    def capture_visual():
        visual_started.set()
        assert planning_done.wait(2)
        return object()

    def save_visual(output, _capture):
        events.append("save_visual_started")
        assert allow_save.wait(2)
        events.append("save_visual_done")
        return output / "initial_tabletop"

    def plan(**_kwargs):
        assert visual_started.wait(2)
        events.append("plan_while_visual_busy")
        planning_done.set()
        return SimpleNamespace(status="sampled_planning_pass")

    monkeypatch.setattr(trial, "collect_key_capture", pose_collect)
    monkeypatch.setattr(trial, "write_key_pose_artifacts", save_pose)
    monkeypatch.setattr(trial, "write_initial_visual_capture", save_visual)
    arguments["runner"].preflight_next_key = plan
    result = trial.prepare_next_live_key(
        **arguments, capture_initial_visual=capture_visual)
    assert result.preflight.status == "sampled_planning_pass"
    assert "save_visual_done" not in events
    allow_save.set()
    drain_pending_initial_visuals()
    assert events.index("plan_while_visual_busy") < events.index(
        "save_visual_done")


def test_pose_only_key_capture_retries_socket_mask_as_key(tmp_path, monkeypatch):
    arguments, events, _ = _setup(tmp_path, monkeypatch)
    prompts = []
    failed = []
    def collect(**kwargs):
        prompts.append(kwargs["key_prompt"])
        return KeyCaptureInput(
            "key_001", 25 + len(prompts), kwargs["key_prompt"],
            {}, {}, {}, {}, "camera_acquisition", {}, tmp_path / "unused")
    original_admit = trial.admit_key_capture
    def admit(**kwargs):
        if len(prompts) == 1:
            raise ValueError(
                "too few key views remain after fixed-socket mask exclusion")
        return original_admit(**kwargs)
    # _setup's admission stub expects no actual SAM/FoundPose data.
    monkeypatch.setattr(trial, "collect_key_capture", collect)
    monkeypatch.setattr(trial, "admit_key_capture", admit)
    monkeypatch.setattr(trial, "write_failed_key_pose_capture",
                        lambda _capture, path, reason: failed.append(
                            (path, reason)))
    monkeypatch.setattr(trial, "write_key_pose_artifacts",
                        lambda _capture, _observation, path: path.mkdir())
    key_args = {name: arguments[name] for name in (
        "runner", "init_orchestrator", "acquisition_metadata_for_request",
        "capture_root", "capture_id",
        "calibrated_camera_ids", "intrinsics_full", "extrinsics_full",
        "image_hw", "key_prompt", "view_limits",
        "maximum_multiview_center_error_mm",
        "maximum_multiview_angle_error_deg",
        "maximum_socket_mask_overlap_fraction", "socket_projection_dilation_px",
        "minimum_refinement_iou", "capture_timeout_s", "image_write_timeout_s")
        if name in arguments}
    key_args["output"] = arguments["key_evidence_dir"]
    key_args["pose_only"] = True
    key_args["silhouette_iterations"] = 100
    key_args["silhouette_loss_threshold"] = 0.003
    key_args["request_id_factory"] = None
    key_args["registered_key_assistance"] = None
    trial._capture_admitted_key(**key_args)
    assert prompts == ["white precision key, excluding white socket",
                       "small white solid peg"]
    assert len(failed) == 1
    assert "fixed-socket mask exclusion" in failed[0][1]


def test_reset_landing_reuses_same_camera_key_capture_without_grasp_preflight(
        tmp_path, monkeypatch):
    arguments, events, _ = _setup(tmp_path, monkeypatch)
    def pose_collect(**kwargs):
        events.append("capture")
        assert kwargs["pose_only"] is True
        return KeyCaptureInput(
            "key_001", 25, "white precision key", {}, {}, {}, {},
            "camera_acquisition", {}, tmp_path / "unused")

    def save_pose(_capture, _observation, output):
        events.append("save_pose")
        output.mkdir()

    monkeypatch.setattr(trial, "collect_key_capture", pose_collect)
    monkeypatch.setattr(trial, "write_key_pose_artifacts", save_pose)
    runner = arguments["runner"]
    runner.current_decision = lambda: SimpleNamespace(
        action="recover_held_key_then_reobserve")
    clear = ClearViewForPerception(
        SimpleNamespace(home=lambda **_kwargs: None,
                        arm_dof=7, hand_name="inspire"),
        lambda: _measured_state(), .02, 20., .1, 1.)
    clear.before_key_capture = lambda: events.append("clear_view_before")
    clear.verify_state = lambda state, **_kwargs: (
        events.append("clear_view_at_exposure") or state)
    omitted = {
        "planner", "limits", "max_pose_error_deg", "axial_waypoint_step_m",
        "max_arm_hand_skew_s", "max_hand_command_error_raw",
        "max_arm_velocity_rad_s",
    }
    reset_args = {key: value for key, value in arguments.items()
                  if key not in omitted}
    reset_args["clear_view"] = clear
    result = trial.capture_returned_key_for_reset(**reset_args)
    assert result.observation.capture_id == "key_001"
    assert result.key_evidence_dir == tmp_path / "key_evidence"
    assert events == [
        "clear_view_before", "compatible_renderer", "init_key",
        "verify_frozen_camera", "capture", "admit", "save_pose",
        "measured_state_at_exposure",
        "clear_view_at_exposure",
    ]


def test_reset_key_capture_does_not_start_before_recovery_decision(
        tmp_path, monkeypatch):
    arguments, events, _ = _setup(tmp_path, monkeypatch)
    clear = ClearViewForPerception(
        SimpleNamespace(home=lambda **_kwargs: None,
                        arm_dof=7, hand_name="inspire"),
        lambda: _measured_state(), .02, 20., .1, 1.)
    omitted = {
        "planner", "limits", "max_pose_error_deg", "axial_waypoint_step_m",
        "max_arm_hand_skew_s", "max_hand_command_error_raw",
        "max_arm_velocity_rad_s",
    }
    reset_args = {key: value for key, value in arguments.items()
                  if key not in omitted}
    with pytest.raises(ValueError, match="active held recovery"):
        trial.capture_returned_key_for_reset(**reset_args, clear_view=clear)
    assert events == []


def test_optional_registered_roi_replaces_only_perception_outputs(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    runner = arguments["runner"]
    runner.output_dir = tmp_path / "run"
    runner.output_dir.mkdir()
    baseline_dir = tmp_path / "frozen_session"
    baseline_dir.mkdir()
    manifest_hash = "a" * 64
    (runner.output_dir / "source_evidence_binding.json").write_text(
        json.dumps({
            "schema": "precision_insertion_session_source_binding_v1",
            "session_evidence_dir": str(baseline_dir),
            "session_manifest_sha256": manifest_hash,
        }), encoding="utf-8")
    baseline = SessionKeyBaseline(
        baseline_dir, "socket_001", 12, manifest_hash, {}, {}, {})
    model = SimpleNamespace(obj_name=runner.mode.key_object)
    assistance = RegisteredKeyAssistance(
        baseline, object(), "9", None, object(), model,
        tmp_path / "board.json", "b" * 64)
    arguments["registered_key_assistance"] = assistance
    report = {"schema": "test_derivation"}

    def derive(**kwargs):
        events.append("registered_roi_on_same_capture")
        assert kwargs["capture"].request_id == 25
        assert kwargs["assistance"] is assistance
        assert kwargs["key_object"] == runner.mode.key_object
        return kwargs["capture"], report

    def write(_capture, _observation, output, *, derivation_report):
        events.append("save_key_evidence")
        assert derivation_report is report
        output.mkdir()

    monkeypatch.setattr(trial, "derive_registered_key_capture", derive)
    monkeypatch.setattr(
        trial, "write_registered_key_diagnostic",
        lambda **kwargs: events.append("save_roi_diagnostic") or
        tmp_path / "roi_diagnostic")
    monkeypatch.setattr(
        trial, "verify_registered_key_diagnostic",
        lambda _path: events.append("verify_roi_diagnostic") or {})
    monkeypatch.setattr(trial, "write_key_capture_artifacts", write)
    trial.prepare_next_live_key(**arguments)
    assert events == [
        "compatible_renderer", "init_key", "verify_frozen_camera",
        "capture", "registered_roi_on_same_capture",
        "save_roi_diagnostic", "verify_roi_diagnostic", "admit",
        "save_key_evidence",
        "measured_state_at_exposure", "preflight",
    ]


def test_registered_roi_from_another_session_blocks_camera(tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    runner = arguments["runner"]
    runner.output_dir = tmp_path / "run"
    runner.output_dir.mkdir()
    (runner.output_dir / "source_evidence_binding.json").write_text(
        json.dumps({
            "schema": "precision_insertion_session_source_binding_v1",
            "session_evidence_dir": str(tmp_path / "frozen_session"),
            "session_manifest_sha256": "a" * 64,
        }), encoding="utf-8")
    baseline = SessionKeyBaseline(
        tmp_path / "other_session", "socket_001", 12,
        "a" * 64, {}, {}, {})
    assistance = RegisteredKeyAssistance(
        baseline, object(), "9", None, object(),
        SimpleNamespace(obj_name=runner.mode.key_object),
        tmp_path / "board.json", "b" * 64)
    arguments["registered_key_assistance"] = assistance
    with pytest.raises(ValueError, match="differs from opened key session"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


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
    arguments["axial_waypoint_step_m"] = .002
    arguments["minimum_refinement_iou"] = 1.01
    with pytest.raises(ValueError, match="key mask, silhouette"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


def test_silhouette_failure_blocks_camera_and_daemon_init(tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)

    def fail(**_kwargs):
        raise RuntimeError("local silhouette renderer unavailable")

    monkeypatch.setattr(trial, "prepare_key_silhouette", fail)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        trial.prepare_next_live_key(**arguments)
    assert events == []


def test_changed_camera_calibration_rejects_before_key_capture(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)

    def changed(_calibration, **_kwargs):
        raise ValueError("live camera calibration changed")

    monkeypatch.setattr(trial, "validate_session_camera_calibration", changed)
    with pytest.raises(ValueError, match="camera calibration changed"):
        trial.prepare_next_live_key(**arguments)
    assert events == ["compatible_renderer", "init_key"]
