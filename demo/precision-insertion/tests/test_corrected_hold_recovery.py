"""A corrected-plan failure must return the held key before opening it."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import corrected_hold_recovery as recovery  # noqa: E402
from precision_insertion.held_reverse_route import HeldReverseRoute  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402


def _save_archive(root, name, attempt_id):
    root.mkdir(parents=True)
    archive = root / "planned_trajectories.npz"
    np.savez_compressed(archive, **{name: np.zeros((2, 13))})
    report = root / "report.json"
    report.write_text(json.dumps({
        "attempt_id": attempt_id,
        "planned_trajectories": archive.name,
        "planned_trajectories_sha256": hashlib.sha256(
            archive.read_bytes()).hexdigest()}))
    return report


def _state():
    return LiveRobotState(
        np.zeros(13), np.zeros(7), 20., 20., 10., 20.,
        np.zeros(6), np.zeros(6), 0., np.zeros(6))


@pytest.mark.parametrize("failure_kind", ["axial", "xy"])
def test_corrected_return_uses_attempt_sources_and_opens_after_held_motion(
        tmp_path, monkeypatch, failure_kind):
    attempt_id = "attempt_123"
    failure_dir = (tmp_path / "guarded_axial_handoffs" / "000"
                   if failure_kind == "axial" else
                   tmp_path / "same_trial_xy" / "plan_failures")
    failure_dir.mkdir(parents=True)
    failure_path = failure_dir / ("plan_failure.json" if failure_kind ==
                                  "axial" else "000.json")
    failure_path.write_text(json.dumps({
        "schema": f"precision_insertion_same_trial_{failure_kind}_plan_failure_v1",
        "attempt_id": attempt_id,
        "measured_start": {"full_q": [0.] * 13},
        "targets": {"T_robot_key_preinsert": np.eye(4).tolist()}}))
    lift = _save_archive(tmp_path / "measured_lift_preflights" / "000",
                         "lift", attempt_id)
    transfer = _save_archive(tmp_path / "postlift_preflights" / "000",
                             "transfer", attempt_id)
    decision = SimpleNamespace(action="recover_corrected_hold_then_reobserve")
    attempt = SimpleNamespace(
        attempt_id=attempt_id, candidate_id="table/0/1",
        failure_code="preinsert_path_unavailable",
        events=[{"evidence_refs": {
            f"corrected_{failure_kind}_plan_failure": str(failure_path)}}])
    runner = SimpleNamespace(
        active_attempt=attempt, current_decision=lambda: decision,
        _attempt_dir=tmp_path, _measured_lift_report_path=lift,
        _postlift_report_path=transfer,
        calibration=SimpleNamespace(record={"c2r": np.eye(4).tolist()}),
        _preflight=SimpleNamespace(
            key_pose_world=np.eye(4),
            trial_scene={"mesh": {"target": {"pose": [0.] * 7}}}),
        shared_root=tmp_path, mode=object())
    events = []
    executor = SimpleNamespace(
        joint_vmax=1.2, traj_dt=.01,
        release=lambda *_args, **_kwargs: events.append("open"),
        stop_and_acknowledge=lambda: {"acknowledged": True})
    planner = SimpleNamespace(fk_wrist=lambda _q: np.eye(4))
    monkeypatch.setattr(recovery, "is_demo_completion_checked_executor",
                        lambda _: True)
    def fake_plan(**kwargs):
        assert kwargs["executed_xy_trajectories"] == []
        assert kwargs["executed_lift_trajectory"].shape == (2, 13)
        assert kwargs["executed_transfer_trajectory"].shape == (2, 13)
        events.append("plan")
        return HeldReverseRoute(
            np.zeros((2, 13)), np.eye(4), {}, np.zeros((2, 13)),
            {"held_sample_count": 2})
    monkeypatch.setattr(recovery, "plan_held_reverse_route", fake_plan)
    clear = SimpleNamespace(
        _follow_autodex=lambda _path: events.append("follow"),
        _verify_held_endpoint=lambda *_args, **_kwargs: _state(),
        wait_for_open_hand=lambda *_args, **_kwargs: _state(),
        verify_state=lambda state, **_kwargs: state)
    bound = recovery.StockCorrectedHoldRecovery(
        executor=executor, planner=planner, read_state=_state,
        limits=object(), motion_interlock=lambda: True)
    output = bound(SimpleNamespace(clear_view=clear), runner, decision)
    assert events == ["plan", "follow", "open", "follow"]
    assert json.loads(output.read_text())["clear_view_reached"] is True
