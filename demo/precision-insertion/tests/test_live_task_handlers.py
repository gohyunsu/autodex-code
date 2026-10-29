"""One selected grasp proceeds through observed hold and 20 mm verdict."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_task_handlers  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def _state():
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=10., arm_timestamp_s=10.,
        arm_robot_uptime_s=1., hand_timestamp_s=10.,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


def _capture(label):
    return RawCameraCapture(
        capture_id=label, request_id=10,
        images_bgr={camera: np.zeros((8, 8, 3), dtype=np.uint8)
                    for camera in ("front", "side")},
        frame_ids={"front": 10, "side": 10}, acquisition_metadata={})


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.active_attempt = SimpleNamespace(
            attempt_id="attempt_001", candidate_id="table/0/3")
        self.decision = SessionDecision(
            "transfer_execution_gate_required", "bounded_plan", "table/0/3")
        self._postlift_report_path = root / "postlift.json"
        self._preinsert_report_path = root / "preinsert.json"
        self._lift_checkpoint = SimpleNamespace(
            after_bundle=root / "attempts" / "attempt_001" /
            "raw_after_lift" / "lift_001")
        self.initial_key_evidence_dir = root / "initial_key"
        self.session_sha256 = "a" * 64
        self.events = []

    def current_decision(self):
        return self.decision

    def prepare_observed_preinsert_label(self, **kwargs):
        self.events.append(("assess_hold", kwargs))
        return SimpleNamespace(preinsert_reached=True)

    def observe_stage(self, stage, status, **kwargs):
        self.events.append(("label_hold", stage, status, kwargs))
        self.decision = SessionDecision(
            "await_guarded_insertion_and_observation", "hold_observed",
            "table/0/3")

    def prepare_guarded_axial_handoff(self, **kwargs):
        self.events.append(("handoff", kwargs))
        return self.output_dir / "handoff.json"

    def prepare_released_insertion_label(self, **kwargs):
        self.events.append(("assess_insertion", kwargs))
        self.decision = SessionDecision("session_success", "observed_20mm", "table/0/3")


def _handler(tmp_path, enabled=True):
    runner = Runner(tmp_path)
    first = _capture("preinsert_001")
    final = _capture("final_001")
    bound = live_task_handlers.LiveTaskHandlers(
        live=SimpleNamespace(
            opened=SimpleNamespace(runner=runner),
            cameras=SimpleNamespace(launch=SimpleNamespace(
                camera_ids={"front", "side"})),
            clear_view=SimpleNamespace(before_key_capture=lambda: None)),
        transfer_adapter=object(), insertion_adapter=object(),
        read_state=_state, state_for_capture=lambda _capture: _state(),
        capture_preinsert=lambda *_args: first,
        capture_final=lambda *_args: final,
        backend=object(), motion_interlock=lambda: True,
        transfer_limits=object(), insertion_limits=object(),
        contact_limits=object(),
        transfer_commissioning_path=tmp_path / "transfer_commissioning.json",
        insertion_watchdog_path=tmp_path / "watchdog.json",
        insertion_contact_path=tmp_path / "contact.json",
        preinsert_parameters={}, handoff_parameters={},
        insertion_parameters={},
        final_parameters={},
        release_and_clear=lambda *_args: tmp_path / "release.json",
        enable_robot_motion=enabled,
        decision_time=lambda: 20.)
    return bound, runner, first, final


def test_transfer_hold_then_guarded_20mm_and_vlm_label(tmp_path, monkeypatch):
    bound, runner, first, final = _handler(tmp_path)
    actions = []

    class Release:
        def prepare(self, _live, _runner, _decision, handoff):
            assert handoff == tmp_path / "handoff.json"
            actions.append(("prepare_release",))

        def __call__(self, *_args):
            actions.append(("release_and_clear",))
            return tmp_path / "release.json"

    bound.release_and_clear = Release()
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_held_transfer",
        lambda **kwargs: actions.append(("transfer", kwargs)))
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_guarded_insertion",
        lambda **kwargs: (actions.append(("insertion", kwargs)) or {
            "safety_abort": False, "grasp_held": True,
            "metric_record": {"path": str(tmp_path / "metric.json")}}))
    monkeypatch.setattr(
        live_task_handlers, "write_raw_camera_capture",
        lambda capture, output, **kwargs:
        actions.append(("write", capture, output, kwargs["phase"])))
    monkeypatch.setattr(
        live_task_handlers, "verify_post_insertion_release",
        lambda *args, **kwargs: {})
    bound.handlers()["transfer_execution_gate_required"](
        runner, runner.current_decision())
    assert runner.current_decision().action == "await_guarded_insertion_and_observation"
    assert [row[0] for row in runner.events] == ["assess_hold", "label_hold"]
    assert runner.events[0][1]["manual_annotation"] is None
    assert runner.events[1][1:3] == ("preinsert_reached", True)
    assert actions[1][1] is first and actions[1][3] == "preinsert"
    bound.handlers()["await_guarded_insertion_and_observation"](
        runner, runner.current_decision())
    assert [row[0] for row in runner.events] == [
        "assess_hold", "label_hold", "handoff", "assess_insertion"]
    assert [row[0] for row in actions] == [
        "transfer", "write", "prepare_release", "insertion",
        "release_and_clear", "write"]
    assert actions[5][1] is final and actions[5][3] == "final_clear_view"
    assert runner.events[-1][1]["metric_record_path"] == tmp_path / "metric.json"
    assert runner.events[-1][1]["initial_key_bundle"] == tmp_path / "initial_key"
    capture_root = tmp_path / "attempts" / "attempt_001" / "raw_captures"
    assert runner.events[-1][1]["preinsert_bundle"] == (
        capture_root / "preinsert" / "preinsert_001")
    assert runner.events[-1][1]["final_bundle"] == (
        capture_root / "final_clear_view" / "final_001")
    assert runner.current_decision().action == "session_success"


def test_preinsert_refs_use_canonical_episode_paths(tmp_path, monkeypatch):
    episode = tmp_path / "attempt_001"
    episode.mkdir()
    run = tmp_path / "run"
    (run / "attempts").mkdir(parents=True)
    (run / "attempts" / "attempt_001").symlink_to(episode,
                                                   target_is_directory=True)
    bound, runner, *_ = _handler(run)
    monkeypatch.setattr(live_task_handlers, "execute_bound_held_transfer",
                        lambda **_kwargs: None)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound.transfer_and_observe(runner, runner.current_decision())
    assessment = runner.events[0][1]
    label_refs = runner.events[1][3]["evidence_refs"]
    canonical_transfer = episode / "transfer_execution.json"
    canonical_raw = (episode / "raw_captures" / "preinsert" /
                     "preinsert_001")
    assert assessment["transfer_execution_path"] == canonical_transfer
    assert assessment["raw_bundle"] == canonical_raw
    assert label_refs["trajectory"] == str(canonical_transfer)
    assert label_refs["preinsert_image"] == str(canonical_raw / "manifest.json")


def test_transfer_reuses_only_same_attempt_lift_pixels(tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    lift = _capture("lift_001")
    bound.after_lift_capture_provider = lambda: ("attempt_001", lift)
    monkeypatch.setattr(live_task_handlers, "execute_bound_held_transfer",
                        lambda **_kwargs: None)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound.transfer_and_observe(runner, runner.current_decision())
    assert runner.events[0][1]["after_lift_live_capture"] is lift

    bound, runner, *_ = _handler(tmp_path / "next")
    bound.after_lift_capture_provider = lambda: ("previous_attempt", lift)
    bound.transfer_and_observe(runner, runner.current_decision())
    assert runner.events[0][1]["after_lift_live_capture"] is None


@pytest.mark.parametrize("second,predicted", [
    ([.0005, 0.], False),
    (None, True),
])
def test_same_trial_xy_reobserves_and_replans_from_final_hold(
        tmp_path, monkeypatch, second, predicted):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .001
    observations = iter(([.004, 0.], second))
    seen = []

    def measure(**kwargs):
        seen.append(("measure", kwargs["raw_bundle"],
                     tuple(kwargs["nominal_key_xy_shift_socket_m"])))
        correction = next(observations)
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction is not None
                               else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        seen.append(("shift", kwargs["increment_socket_m"]))
        path = runner._attempt_dir / "shift.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"measured_increment_socket_m":
                                    [.0035, 0.],
                                    "measured_axis_increment_socket_m": 0.}),
                        encoding="utf-8")
        return path

    def handoff(**kwargs):
        seen.append(("handoff", kwargs))
        return tmp_path / "corrected_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert [row[0] for row in seen] == [
        "measure", "shift", "measure", "handoff"]
    assert seen[0][2] == (0., 0.)
    np.testing.assert_allclose(seen[2][2], [.0035, 0.])
    np.testing.assert_allclose(seen[1][1], [.004, 0.])
    np.testing.assert_allclose(seen[-1][1]["total_xy_m"], [.0035, 0.])
    np.testing.assert_allclose(
        seen[-1][1]["residual_xy_m"],
        [-.0005, 0.])
    assert seen[-1][1]["residual_source"] == (
        "last_metric_plus_completed_held_motion" if predicted else
        "current_metric_multiview")
    assert bound._same_trial_handoff == tmp_path / "corrected_handoff.json"


def test_corrected_plan_failure_records_stop_without_insertion(tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .005
    failure = runner._attempt_dir / "guarded_axial_handoffs/000/plan_failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"status":"corrected_20mm_endpoint_rejected"}\n',
                       encoding="utf-8")
    recorded = []
    runner.record_failure = lambda code, **kwargs: recorded.append((code, kwargs))
    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy",
                        lambda **_kwargs: {"alignment": {
                            "status": "diagnostic_metric_xy_correction",
                            "inlier_cameras": ["front", "side"],
                            "xy_correction_socket_m": [.001, 0.],
                        }})
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            live_task_handlers.CorrectedAxialPlanUnavailable(
                                failure, "corrected_20mm_endpoint_rejected")))
    bound._refine_current_hold(runner)
    assert bound._same_trial_handoff is None
    assert recorded == [("preinsert_path_unavailable", {
        "timestamp_s": 20.,
        "evidence_refs": {
            "corrected_axial_plan_failure": str(failure),
            "preinsert_snapshot": str(bound._preinsert_bundle / "manifest.json"),
        },
    })]


def test_unobserved_first_xy_still_replans_axial_from_measured_hold(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .0025
    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy",
                        lambda **_kwargs: {"alignment": {
                            "status": "abstain", "inlier_cameras": [],
                            "xy_correction_socket_m": None}})
    captured = {}

    def handoff(**kwargs):
        captured.update(kwargs)
        return tmp_path / "measured_handoff.json"

    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    bound._refine_current_hold(runner)
    assert bound._same_trial_handoff == tmp_path / "measured_handoff.json"
    assert captured["total_xy_m"] == [0., 0.]
    assert captured["residual_xy_m"] == [0., 0.]
    assert captured["residual_source"] == "nominal_unobserved"


@pytest.mark.parametrize("raised_correction", [None, [.001, 0.]])
def test_unobserved_xy_can_raise_reobserve_and_return_before_axial(
        tmp_path, monkeypatch, raised_correction):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .0025
    bound.same_trial_observation_raise_m = .04
    seen = []
    corrections = iter((None, raised_correction))

    def measure(**kwargs):
        seen.append(("measure", kwargs["nominal_key_axis_raise_socket_m"]))
        correction = next(corrections)
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        seen.append(("motion", kwargs["axis_increment_socket_m"]))
        path = (runner._attempt_dir / "shift" /
                f'{kwargs["index"]:03d}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "measured_increment_socket_m": [0., 0.],
            "measured_axis_increment_socket_m":
                kwargs["axis_increment_socket_m"],
        }), encoding="utf-8")
        return path

    def handoff(**kwargs):
        seen.append(("handoff", kwargs))
        return tmp_path / "raised_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert seen[:4] == [
        ("measure", 0.), ("motion", .04),
        ("measure", .04), ("motion", -.04)]
    assert seen[-1][0] == "handoff"
    assert len(seen[-1][1]["shifts"]) == 2
    assert seen[-1][1]["residual_source"] == (
        "nominal_unobserved" if raised_correction is None else
        "current_metric_multiview")
    assert bound._same_trial_handoff == tmp_path / "raised_handoff.json"


def test_unobserved_xy_can_rotate_once_after_raise_then_return(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .0025
    bound.same_trial_observation_raise_m = .04
    bound.same_trial_observation_yaw_rad = np.deg2rad(150.)
    actions = []

    def measure(**kwargs):
        actions.append(("observe", kwargs["nominal_key_axis_raise_socket_m"]))
        return {"alignment": {
            "status": "abstain", "inlier_cameras": [],
            "xy_correction_socket_m": None}}

    def shift(**kwargs):
        actions.append(("move", kwargs["axis_increment_socket_m"],
                        kwargs.get("yaw_increment_socket_rad", 0.)))
        path = runner._attempt_dir / "shifts" / f'{kwargs["index"]:03d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "measured_increment_socket_m": [0., 0.],
            "measured_axis_increment_socket_m":
                kwargs["axis_increment_socket_m"],
        }), encoding="utf-8")
        return path

    def handoff(**kwargs):
        actions.append(("handoff", kwargs))
        return tmp_path / "yaw_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert [row[0] for row in actions] == [
        "observe", "move", "observe", "move", "observe", "move", "handoff"]
    assert actions[1] == ("move", .04, 0.)
    assert actions[3] == ("move", 0., np.deg2rad(150.))
    assert actions[5] == ("move", -.04, 0.)
    assert actions[-1][1]["residual_source"] == "nominal_unobserved"
    assert len(actions[-1][1]["observations"]) == 3
    assert len(actions[-1][1]["shifts"]) == 3


@pytest.mark.parametrize("first_lateral_plan_fails", [False, True])
def test_temporary_lateral_view_is_removed_before_insertion(
        tmp_path, monkeypatch, first_lateral_plan_fails):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .0025
    bound.same_trial_observation_raise_m = .04
    bound.same_trial_observation_lateral_m = .03
    corrections = iter((None, None, [-.03, .03], None))
    actions = []

    def measure(**kwargs):
        correction = next(corrections)
        actions.append(("observe", tuple(kwargs["nominal_key_xy_shift_socket_m"]),
                        kwargs["nominal_key_axis_raise_socket_m"]))
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        xy = np.asarray(kwargs["increment_socket_m"], float)
        axis = kwargs.get("axis_increment_socket_m", 0.)
        actions.append(("move", tuple(xy), axis))
        if (first_lateral_plan_fails and
                kwargs.get("action_tag") == "lateral_0"):
            raise live_task_handlers.HeldXYPlanUnavailable(
                tmp_path / "first_lateral_plan.json", "unreachable")
        path = runner._attempt_dir / "shifts" / f'{kwargs["index"]:03d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "measured_increment_socket_m": xy.tolist(),
            "measured_axis_increment_socket_m": axis,
        }), encoding="utf-8")
        return path

    def handoff(**kwargs):
        actions.append(("handoff", kwargs))
        return tmp_path / "lateral_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    monkeypatch.setattr(live_task_handlers, "rank_lateral_observation_offsets",
                        lambda **_kwargs: (
                            [{"offset_socket_m": [.03, 0.]},
                             {"offset_socket_m": [.03, -.03]}]
                            if first_lateral_plan_fails else
                            [{"offset_socket_m": [.03, -.03]}]))
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert actions[0] == ("observe", (0., 0.), 0.)
    assert actions[1] == ("move", (0., 0.), .04)
    assert ("move", (.03, -.03), 0.) in actions
    assert ("observe", (.03, -.03), .04) in actions
    assert ("move", (-.03, .03), 0.) in actions
    assert len([row for row in actions if row[0] == "observe"]) == 3
    assert actions[-2] == ("move", (0., 0.), -.04)
    assert actions[-1][1]["total_xy_m"] == [0., 0.]
    assert actions[-1][1]["residual_xy_m"] == [0., 0.]


def test_lateral_metric_correction_uses_returned_hold_without_extra_inference(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .0025
    bound.same_trial_observation_raise_m = .04
    bound.same_trial_observation_lateral_m = .03
    corrections = iter((None, None, [-.038, .03], None))
    moves = []
    handoff = {}

    def measure(**_kwargs):
        correction = next(corrections)
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        xy = np.asarray(kwargs["increment_socket_m"], float)
        axis = kwargs.get("axis_increment_socket_m", 0.)
        moves.append((xy, axis, kwargs.get("action_tag", "")))
        path = runner._attempt_dir / "shifts" / (
            f'{kwargs["index"]:03d}_{kwargs.get("action_tag", "")}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "measured_increment_socket_m": xy.tolist(),
            "measured_axis_increment_socket_m": axis,
        }), encoding="utf-8")
        return path

    def save_handoff(**kwargs):
        handoff.update(kwargs)
        return tmp_path / "lateral_corrected_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        save_handoff)
    monkeypatch.setattr(live_task_handlers, "rank_lateral_observation_offsets",
                        lambda **_kwargs: [{"offset_socket_m": [.03, -.03]}])
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert len(moves) == 4
    np.testing.assert_allclose(moves[3][0], [-.008, 0.], atol=1e-12)
    assert moves[3][1] == -.04
    assert moves[3][2] == "post_lateral_correction_xy_and_axis_return"
    np.testing.assert_allclose(handoff["total_xy_m"], [-.008, 0.], atol=1e-12)
    np.testing.assert_allclose(handoff["residual_xy_m"], [0., 0.], atol=1e-12)


def test_raised_xy_correction_reobserves_then_returns_for_axial(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .001
    bound.same_trial_observation_raise_m = .04
    bound.same_trial_max_shifts = 1
    corrections = iter((None, [.004, 0.], [.0005, 0.]))
    motions = []
    measured_heights = []
    handed = {}

    def measure(**kwargs):
        measured_heights.append(kwargs["nominal_key_axis_raise_socket_m"])
        correction = next(corrections)
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        motions.append(kwargs)
        path = (runner._attempt_dir / "shift" /
                f'{kwargs["index"]:03d}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        measured_xy = ([.0035, 0.] if np.linalg.norm(
            kwargs["increment_socket_m"]) > 0. else [0., 0.])
        measured_axis = kwargs.get("axis_increment_socket_m", 0.)
        path.write_text(json.dumps({
            "measured_increment_socket_m": measured_xy,
            "measured_axis_increment_socket_m": measured_axis,
        }), encoding="utf-8")
        return path

    def handoff(**kwargs):
        handed.update(kwargs)
        return tmp_path / "raised_corrected_handoff.json"

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        handoff)
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert measured_heights == [0., .04, 0.]
    assert [row.get("axis_increment_socket_m", 0.) for row in motions] == [
        .04, -.04]
    np.testing.assert_allclose(motions[1]["increment_socket_m"], [.004, 0.])
    np.testing.assert_allclose(handed["total_xy_m"], [.0035, 0.])
    np.testing.assert_allclose(handed["residual_xy_m"], [-.0005, 0.])
    assert len(handed["observations"]) == 3
    assert len(handed["shifts"]) == 2
    assert bound._same_trial_handoff == tmp_path / "raised_corrected_handoff.json"


def test_combined_correction_plan_can_use_horizontal_route(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "first_preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .001
    bound.same_trial_observation_raise_m = .04
    corrections = iter((None, [.004, 0.], None))
    moves = []
    handed = {}

    def measure(**_kwargs):
        correction = next(corrections)
        return {"alignment": {
            "status": ("diagnostic_metric_xy_correction" if correction
                       is not None else "abstain"),
            "inlier_cameras": (["front", "side"] if correction else []),
            "xy_correction_socket_m": correction,
        }}

    def shift(**kwargs):
        axis = kwargs.get("axis_increment_socket_m", 0.)
        moves.append((kwargs.get("action_tag", ""), axis))
        if kwargs.get("action_tag") == "xy_and_axis_return":
            raise live_task_handlers.HeldXYPlanUnavailable(
                tmp_path / "combined_plan_failure.json", "unreachable")
        path = runner._attempt_dir / "shifts" / (
            f'{kwargs["index"]:03d}_{kwargs.get("action_tag", "")}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "measured_increment_socket_m": list(kwargs["increment_socket_m"]),
            "measured_axis_increment_socket_m": axis,
        }), encoding="utf-8")
        return path

    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy", measure)
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift", shift)
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        lambda **kwargs: (handed.update(kwargs) or
                                          tmp_path / "handoff.json"))
    monkeypatch.setattr(live_task_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    bound._refine_current_hold(runner)
    assert moves == [("", .04), ("xy_and_axis_return", -.04),
                     ("xy_only", 0.), ("", -.04)]
    np.testing.assert_allclose(handed["total_xy_m"], [.004, 0.])
    np.testing.assert_allclose(handed["residual_xy_m"], [0., 0.])


def test_held_xy_plan_failure_records_recoverable_stop(tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._attempt_dir = tmp_path / "attempts" / runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "preinsert"
    bound.planner = object()
    bound.path_limits = object()
    bound.alignment_limits = object()
    bound.same_trial_tip_height_error_m = .01
    bound.same_trial_xy_tolerance_m = .001
    failure = runner._attempt_dir / "same_trial_xy/plan_failures/000.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"status":"held_xy_pose_unreachable"}\n',
                       encoding="utf-8")
    recorded = []
    runner.record_failure = lambda code, **kwargs: recorded.append((code, kwargs))
    monkeypatch.setattr(live_task_handlers, "measure_preinsert_xy",
                        lambda **_kwargs: {"alignment": {
                            "status": "diagnostic_metric_xy_correction",
                            "inlier_cameras": ["front", "side"],
                            "xy_correction_socket_m": [.004, 0.],
                        }})
    monkeypatch.setattr(live_task_handlers, "plan_and_execute_shift",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            live_task_handlers.HeldXYPlanUnavailable(
                                failure, "held_xy_pose_unreachable")))
    monkeypatch.setattr(live_task_handlers, "plan_corrected_axial_handoff",
                        lambda **_kwargs: pytest.fail("axial plan after XY failure"))
    bound._refine_current_hold(runner)
    assert recorded == [("preinsert_path_unavailable", {
        "timestamp_s": 20.,
        "evidence_refs": {
            "corrected_xy_plan_failure": str(failure),
            "preinsert_snapshot": str(bound._preinsert_bundle / "manifest.json"),
        },
    })]


@pytest.mark.parametrize("result", [
    {"safety_abort": True, "grasp_held": True},
    {"safety_abort": False, "grasp_held": False},
    {"safety_abort": False, "grasp_held": None},
])
def test_guarded_abort_or_lost_key_never_releases(tmp_path, monkeypatch, result):
    bound, runner, *_ = _handler(tmp_path)
    actions = []
    runner.decision = SessionDecision(
        "await_guarded_insertion_and_observation", "hold_observed", "table/0/3")
    bound._preinsert_attempt_id = runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "preinsert_bundle"
    bound.release_and_clear = lambda *_args: actions.append("release")
    bound.capture_final = lambda *_args: actions.append("capture")
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_guarded_insertion",
        lambda **kwargs: {**result, "metric_record": {
            "path": str(tmp_path / "metric.json")}})
    with pytest.raises(RuntimeError, match="supervised recovery"):
        bound.handlers()["await_guarded_insertion_and_observation"](
            runner, runner.current_decision())
    assert actions == []
    assert [row[0] for row in runner.events] == ["handoff"]


def test_stock_force_stop_records_failure_then_reuses_ascent_for_return(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner.decision = SessionDecision(
        "await_guarded_insertion_and_observation", "hold_observed", "table/0/3")
    bound._preinsert_attempt_id = runner.active_attempt.attempt_id
    bound._preinsert_bundle = tmp_path / "preinsert_bundle"
    release_calls = []
    bound.release_and_clear = lambda *_: release_calls.append("release")
    metric = tmp_path / "metric.json"
    withdrawal = tmp_path / "withdrawal.json"
    monkeypatch.setattr(
        live_task_handlers, "execute_bound_guarded_insertion",
        lambda **_: {"safety_abort": True, "grasp_held": True,
                    "retreat_completed": True,
                    "withdrawal_record_path": str(withdrawal),
                    "metric_record": {"path": str(metric)}})
    runner.observe_stock_force_stop = lambda **kwargs: runner.events.append(
        ("force_stop", kwargs))
    runner.record_failure = lambda *args, **kwargs: runner.events.append(
        ("failure", args, kwargs))
    bound.insert_and_observe(runner, runner.current_decision())
    assert [row[0] for row in runner.events] == [
        "handoff", "force_stop", "failure"]
    assert bound.withdrawn_for_return()[2] == withdrawal
    assert runner.events[-1][1] == ("force_abort",)
    assert release_calls == []


def test_motion_disabled_before_transfer_or_insertion(tmp_path):
    bound, runner, *_ = _handler(tmp_path, enabled=False)
    with pytest.raises(PermissionError, match="not enabled"):
        bound.transfer_and_observe(runner, runner.current_decision())
    assert runner.events == []
