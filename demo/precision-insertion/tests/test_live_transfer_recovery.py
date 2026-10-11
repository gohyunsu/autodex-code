"""A misaligned but still-held transfer may return; a suspected slip may not."""

from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_transfer_recovery  # noqa: E402
from precision_insertion.live_transfer_recovery import (  # noqa: E402
    LiveTransferRecoveryHandlers,
)
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def test_automatic_return_eligibility_requires_visible_rigid_hold(tmp_path):
    sources = {}
    for name in ("trajectory_feedback", "safety", "grasp_state"):
        source = tmp_path / f"{name}.json"
        source.write_text("{}", encoding="utf-8")
        sources[name] = {"path": str(source),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (tmp_path / "transfer_execution.json").write_text(json.dumps({
        "schema": "precision_insertion_transfer_execution_v1",
        "attempt_id": "trial", "candidate_id": "table/0/3",
        "started_at_s": 1.0, "completed_at_s": 2.0,
        "measurement": {"trajectory_complete": True,
                        "safety_abort": False, "grasp_held": True},
        "source_records": sources,
    }), encoding="utf-8")
    checkpoint = SimpleNamespace(
        preinsert_reached=False, reason="measured_hold_pose_outside_limits",
        visual=SimpleNamespace(status="coarse_match"))
    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(attempt_id="trial",
                                       candidate_id="table/0/3"),
        _attempt_dir=tmp_path, _preinsert_checkpoint=checkpoint)
    handler = LiveTransferRecoveryHandlers(
        live=object(), recover_to_tabletop=lambda *_: None,
        state_for_observation=lambda *_: None, key_prompt="key",
        capture_parameters={}, landing_parameters={})
    assert handler._completed_held_transfer(runner)
    checkpoint.visual.status = "slip_or_miss"
    assert not handler._completed_held_transfer(runner)
    checkpoint.visual.status = "gross_misalignment"
    checkpoint.reason = "multiview_visible_preinsert_failure"
    assert handler._completed_held_transfer(runner)
    checkpoint.visual.status = "unknown"
    assert not handler._completed_held_transfer(runner)


def test_grasp_miss_recaptures_and_restores_before_closing_attempt(
        tmp_path, monkeypatch):
    events = []
    decision = SessionDecision(
        "recover_key_then_reobserve", "grasp_miss_restore_original_pose",
        "table/0/3")
    attempt = SimpleNamespace(
        attempt_id="trial", candidate_id="table/0/3",
        labels={"grasp_success": False},
        events=[{"timestamp_s": 1.0}])
    runner = SimpleNamespace(
        active_attempt=attempt, _attempt_dir=tmp_path,
        output_dir=tmp_path,
        current_decision=lambda: decision,
        observe_reset_landing=lambda **kwargs: events.append(
            ("record_reset", kwargs["restoration_evidence_paths"])))
    clear_view = SimpleNamespace(
        prepare_failed_grasp_for_capture=lambda **kwargs: (
            events.append("autodex_clear_view") or
            SimpleNamespace(sample_timestamp_s=2.0,
                            to_record=lambda: {"sample_timestamp_s": 2.0})),
        before_key_capture=lambda: events.append("before_capture"))
    launch = SimpleNamespace(
        capture_root=tmp_path, camera_ids=("cam0",),
        intrinsics_full={}, extrinsics_full={}, image_hw=(10, 10),
        view_limits=object())
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner), clear_view=clear_view,
        cameras=SimpleNamespace(
            launch=launch, init_orchestrator=object(),
            acquisition_metadata_for_request=lambda *_: {}))
    selected = SimpleNamespace(preflight=SimpleNamespace(
        pickup_plan=object(), trial_scene={"mesh": {}},
        selected_candidate_key=("table", "0", "3")))
    monkeypatch.setattr(
        live_transfer_recovery, "capture_returned_key_for_reset",
        lambda **_kwargs: (
            events.append("fresh_key_capture") or SimpleNamespace(
                observation=SimpleNamespace(acquisition_interval_s=(3.0, 3.1)),
                key_evidence_dir=tmp_path / "capture")))
    restoration = tmp_path / "restoration.json"
    restoration.write_text("{}", encoding="utf-8")

    def restore(_runner, _decision, landed, _launch):
        events.append("restore_original")
        return landed, (restoration,)

    handler = LiveTransferRecoveryHandlers(
        live=live, recover_to_tabletop=lambda *_: (_ for _ in ()).throw(
            AssertionError("a judged miss needs AutoDex clear view")),
        state_for_observation=lambda *_: object(), key_prompt="blue key",
        capture_parameters={}, landing_parameters={}, planner=object(),
        key_planning=SimpleNamespace(latest=selected),
        restore_original=restore)
    handler.recover(runner, decision)
    assert events == [
        "autodex_clear_view", "before_capture", "fresh_key_capture",
        "restore_original", ("record_reset", (restoration,))]
    log = json.loads((tmp_path / "missed_grasp_recovery.json").read_text())
    assert log["schema"] == "precision_insertion_missed_grasp_clear_view_v1"
    assert log["completed_at_s"] == 2.0
