"""Failed insertion reset needs a bound plan, motion log and new key image."""

from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.failed_return_evidence import (  # noqa: E402
    PHASES, TRANSFER_RETURN_PHASES, SOURCES,
    verify_failed_return_execution, write_failed_return_plan,
)
from precision_insertion.repose_preflight import ReposeHeldPreflight  # noqa: E402
from precision_insertion.repose_release import ReposeReleasePreflight  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion import stock_failed_return  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from autodex.utils.robot_config import FR3_CLEAR_VIEW  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, transfer_origin=False, stock_force=False,
             plan_dir_name="return_plan"):
    attempt = "attempt_1"
    candidate = "table/0/3"
    session = "a" * 64
    catalog = "b" * 64
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text(json.dumps({
        "schema": "precision_insertion_guarded_withdrawal_v1",
        "attempt_id": attempt, "candidate_id": candidate,
        "status": "withdrawn_to_preinsert_hold", "completed_at_s": 4.0,
        "key_still_held": True, "safety_abort": stock_force,
        "source": ("stock_franka_force_stop_follow" if stock_force else
                   "commissioned_guarded_controller"),
    }), encoding="utf-8")
    transfer = tmp_path / "transfer_execution.json"
    transfer.write_text(json.dumps({
        "schema": "precision_insertion_transfer_execution_v1",
        "attempt_id": attempt, "candidate_id": candidate,
        "completed_at_s": 4.0,
        "measurement": {"trajectory_complete": True,
                        "safety_abort": False, "grasp_held": True},
    }), encoding="utf-8")
    postlift = tmp_path / "postlift.json"
    postlift.write_text(json.dumps({
        "attempt_id": attempt, "status": "sampled_postlift_preflight_pass",
        "session_calibration_sha256": session, "catalog_sha256": catalog,
    }), encoding="utf-8")
    q = np.zeros(13)
    raw = np.full(6, 500.)
    held_hand = convert_inspire_raw(raw[None])[0]
    q[7:] = held_hand
    withdrawn_state = LiveRobotState(
        q.copy(), np.zeros(7), 4.1, 4.1, 100., 4.1,
        raw.copy(), raw.copy(), 0., np.zeros(6))
    mid = q.copy()
    mid[0] += .1
    low = mid.copy()
    low[0] += .1
    release_low = low.copy()
    release_low[7:] = 0
    high = release_low.copy()
    high[0] += .1
    parked = np.r_[FR3_CLEAR_VIEW, np.zeros(6)]
    held = ReposeHeldPreflight(
        "sampled_held_path_pass_release_unplanned", np.vstack([q, q]),
        np.vstack([q, mid]), np.vstack([mid, low]),
        {"sampled_clear": True}, (), np.eye(4), .04, held_hand, "measured")
    release = ReposeReleasePreflight(
        "nominal_release_exit_path_pass_drop_unobserved",
        np.vstack([release_low, high]), np.vstack([high, parked]),
        {"sampled_clear": True}, (), np.zeros(6), parked[:7])
    report = write_failed_return_plan(
        output_dir=tmp_path / plan_dir_name, attempt_id=attempt,
        candidate_id=candidate, session_calibration_sha256=session,
        catalog_sha256=catalog,
        **({"transfer_evidence_path": transfer} if transfer_origin else
           {"withdrawal_evidence_path": withdrawal}),
        postlift_report_path=postlift, withdrawn_state=withdrawn_state,
        held=held, release=release)
    sources = {}
    for name in SOURCES:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name}), encoding="utf-8")
        sources[name] = {"path": str(path), "sha256": _sha(path)}
    phases = []
    phases_expected = TRANSFER_RETURN_PHASES if transfer_origin else PHASES
    for i, name in enumerate(phases_expected):
        started = ((4.2 + i) if transfer_origin else
                   (3.0 if i == 0 else 3.2 + i))
        phase = {
            "name": name, "started_at_s": started,
            "completed_at_s": started + 1.0,
            "complete": True,
            "safety_abort": False,
        }
        if name in {"axial_withdrawal", "held_transfer", "held_descent"}:
            phase["grasp_held"] = True
        if name == "release_open":
            phase["hand_open_feedback"] = True
        phases.append(phase)
    record = {
        "schema": "precision_insertion_failed_return_execution_v1",
        "return_origin": "transfer" if transfer_origin else "insertion",
        "source": "commissioned_external_controller",
        "attempt_id": attempt, "candidate_id": candidate,
        "session_calibration_sha256": session, "catalog_sha256": catalog,
        "safety_abort": False, "grip_loss_before_release": False,
        "plan_report": {"path": str(report), "sha256": _sha(report)},
        "phases": phases, "source_records": sources,
        "hand_open_final": True, "arm_clear_view_final": True,
    }
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(record), encoding="utf-8")
    args = dict(path=execution, attempt_id=attempt, candidate_id=candidate,
                attempt_started_at_s=2.0,
                session_calibration_sha256=session, catalog_sha256=catalog)
    return report, execution, record, args


def test_bound_failed_return_execution_is_admitted(tmp_path):
    report, _, _, args = _fixture(tmp_path)
    result = verify_failed_return_execution(**args)
    assert result["release_completed_at_s"] == 7.2
    assert result["exit_completed_at_s"] == 9.2
    assert result["plan_report"]["sha256"] == _sha(report)


def test_stock_force_ascent_is_admitted_as_held_return_start(tmp_path):
    report, _, _, args = _fixture(tmp_path, stock_force=True)
    assert verify_failed_return_execution(**args)["plan_report"]["path"] == (
        str(report))


def test_stock_return_recorder_binds_measured_waypoints_to_force_stop(
        tmp_path, monkeypatch):
    report, _, _, args = _fixture(
        tmp_path, stock_force=True, plan_dir_name="failed_return_preflight")
    with np.load(report.parent / "planned_trajectories.npz") as paths:
        planned = [
            np.asarray(json.loads(report.read_text())["held_start_joint_state"][
                "full_q"]),
            paths["held_transfer"][-1].copy(),
            paths["held_descent"][-1].copy(),
            paths["post_release_lift"][0].copy(),
            paths["post_release_lift"][-1].copy(),
            paths["post_release_retract"][-1].copy(),
        ]
    samples = [LiveRobotState(
        q.copy(), np.zeros(7), 4.1 + index, 4.1 + index, 100.,
        4.1 + index, np.zeros(6), np.zeros(6), 0., np.zeros(6))
        for index, q in enumerate(planned)]
    decision = SimpleNamespace(action="recover_held_key_then_reobserve")
    runner = SimpleNamespace(
        _attempt_dir=tmp_path,
        active_attempt=SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/3",
            started_at_s=2.0),
        session_sha256="a" * 64, catalog_sha256="b" * 64,
        current_decision=lambda: decision)
    calls = []

    def run(*, evidence_sink):
        calls.append("motion")
        evidence_sink.extend(
            {"name": name, "state": state.to_record()}
            for name, state in zip(
                stock_failed_return.STOCK_CHECKPOINTS, samples))
        return samples[-1]

    executor = SimpleNamespace(stop_and_acknowledge=lambda: {
        "acknowledged": True})
    monkeypatch.setattr(stock_failed_return,
                        "is_demo_completion_checked_executor", lambda _: True)
    saved = stock_failed_return.StockFailedReturnRecorder(executor)(
        run, report, runner, decision)
    assert calls == ["motion"]
    assert verify_failed_return_execution(**{**args, "path": saved})[
        "release_completed_at_s"] == samples[3].sample_timestamp_s
    record = json.loads(saved.read_text())
    assert record["source"] == "stock_autodex_franka_executor"
    with pytest.raises(FileExistsError):
        stock_failed_return.StockFailedReturnRecorder(executor)(
            run, report, runner, decision)


def test_stock_return_rejects_changed_withdrawal_before_motion(
        tmp_path, monkeypatch):
    report, _, _, _ = _fixture(
        tmp_path, stock_force=True, plan_dir_name="failed_return_preflight")
    source = Path(json.loads(report.read_text())["sources"]["withdrawal"]["path"])
    source.write_text("{}", encoding="utf-8")
    decision = SimpleNamespace(action="recover_held_key_then_reobserve")
    runner = SimpleNamespace(
        _attempt_dir=tmp_path,
        active_attempt=SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/3",
            started_at_s=2.0),
        session_sha256="a" * 64, catalog_sha256="b" * 64,
        current_decision=lambda: decision)
    monkeypatch.setattr(stock_failed_return,
                        "is_demo_completion_checked_executor", lambda _: True)
    with pytest.raises(ValueError, match="sources changed"):
        stock_failed_return.StockFailedReturnRecorder(object())(
            lambda **_: pytest.fail("tampered plan commanded motion"),
            report, runner, decision)
    assert not (tmp_path / "failed_return_stock_started.json").exists()


def test_completed_held_transfer_uses_same_return_without_fake_withdrawal(tmp_path):
    report, _, _, args = _fixture(tmp_path, transfer_origin=True)
    plan = json.loads(report.read_text(encoding="utf-8"))
    assert plan["return_origin"] == "transfer"
    assert set(plan["sources"]) == {"transfer_execution", "postlift_report"}
    assert verify_failed_return_execution(**args)["exit_completed_at_s"] == 9.2


def test_transfer_return_rejects_lost_grasp(tmp_path):
    report, _, _, args = _fixture(tmp_path, transfer_origin=True)
    plan = json.loads(report.read_text(encoding="utf-8"))
    transfer = Path(plan["sources"]["transfer_execution"]["path"])
    transfer_row = json.loads(transfer.read_text(encoding="utf-8"))
    transfer_row["measurement"]["grasp_held"] = False
    transfer.write_text(json.dumps(transfer_row), encoding="utf-8")
    with pytest.raises(ValueError, match="plan source changed"):
        verify_failed_return_execution(**args)


@pytest.mark.parametrize(("change", "message"), [
    (lambda row: row["phases"].pop(), "ordered execution phase"),
    (lambda row: row["phases"][2].update(grasp_held=False), "not completed"),
    (lambda row: row["phases"][3].update(hand_open_feedback=False), "not completed"),
    (lambda row: row.update(arm_clear_view_final=False), "clear view"),
    (lambda row: row.update(candidate_id="other"), "not bound"),
])
def test_false_failed_return_execution_is_rejected(tmp_path, change, message):
    _, path, record, args = _fixture(tmp_path)
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_failed_return_execution(**args)


def test_changed_return_plan_or_withdrawal_is_rejected(tmp_path):
    report, _, _, args = _fixture(tmp_path)
    report.write_text(report.read_text(encoding="utf-8") + " ",
                      encoding="utf-8")
    with pytest.raises(ValueError, match="plan report changed"):
        verify_failed_return_execution(**args)


def test_mutated_guarded_withdrawal_invalidates_saved_return(tmp_path):
    report, _, _, args = _fixture(tmp_path)
    record = json.loads(report.read_text(encoding="utf-8"))
    withdrawal = Path(record["sources"]["withdrawal"]["path"])
    withdrawal.write_text(withdrawal.read_text(encoding="utf-8") + " ",
                          encoding="utf-8")
    with pytest.raises(ValueError, match="plan source changed"):
        verify_failed_return_execution(**args)
