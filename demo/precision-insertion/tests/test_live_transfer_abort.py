"""A stopped transfer reaches supervised recovery without inventing a hold."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_task_handlers  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


@pytest.mark.parametrize("acknowledged", [True, False])
def test_transfer_abort_routes_only_after_controller_stop(tmp_path, monkeypatch,
                                                           acknowledged):
    decision = SessionDecision("transfer_execution_gate_required", "ready",
                               "table/0/3")
    attempt = SimpleNamespace(attempt_id="trial", candidate_id="table/0/3")
    observed = []
    runner = SimpleNamespace(
        current_decision=lambda: decision, active_attempt=attempt,
        output_dir=tmp_path,
        observe_stage=lambda *args, **kwargs: observed.append((args, kwargs)))
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))
    handler = live_task_handlers.LiveTaskHandlers(
        live=live, transfer_adapter=object(), insertion_adapter=object(),
        read_state=lambda: object(), state_for_capture=lambda _: object(),
        capture_preinsert=lambda *_: None, capture_final=lambda *_: None,
        backend=object(), motion_interlock=lambda: True,
        transfer_limits=object(), insertion_limits=object(),
        contact_limits=object(), transfer_commissioning_path=tmp_path,
        insertion_watchdog_path=tmp_path, insertion_contact_path=tmp_path,
        preinsert_parameters={}, handoff_parameters={},
        insertion_parameters={}, final_parameters={},
        enable_robot_motion=True, decision_time=lambda: 10.0)

    def failed_transfer(**_kwargs):
        failure = tmp_path / "attempts" / "trial" / "transfer_failure.json"
        failure.parent.mkdir(parents=True)
        failure.write_text(json.dumps({
            "schema": "precision_insertion_transfer_execution_v1",
            "attempt_id": "trial", "candidate_id": "table/0/3",
            "status": "execution_or_feedback_failed",
            "stop_acknowledgement": {"acknowledged": acknowledged},
        }), encoding="utf-8")
        raise RuntimeError("controller aborted")

    monkeypatch.setattr(live_task_handlers, "execute_bound_held_transfer",
                        failed_transfer)
    if acknowledged:
        handler.transfer_and_observe(runner, decision)
        assert observed[0][0] == ("preinsert_reached", False)
        assert observed[0][1]["evidence_refs"]["transfer_failure"].endswith(
            "transfer_failure.json")
    else:
        with pytest.raises(RuntimeError, match="controller aborted"):
            handler.transfer_and_observe(runner, decision)
        assert not observed
