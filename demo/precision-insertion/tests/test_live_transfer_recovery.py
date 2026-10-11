"""A misaligned but still-held transfer may return; a suspected slip may not."""

from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.live_transfer_recovery import (  # noqa: E402
    LiveTransferRecoveryHandlers,
)


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
    assert not handler._completed_held_transfer(runner)
