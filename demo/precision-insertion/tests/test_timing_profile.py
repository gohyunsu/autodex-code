"""Run-level monotonic timing uses stock AutoDex trace without robot access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from threading import Barrier

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.timing_profile import (  # noqa: E402
    profile_run, summarize_events, timed, timing_directory,
)
from precision_insertion.view_parallel import map_view_calls  # noqa: E402
import report_timing  # noqa: E402


def test_profile_groups_nested_spans_without_double_counting(tmp_path):
    output = tmp_path / "session"
    with profile_run(output):
        with timed("live.session", phase="session"):
            with timed("socket.capture", phase="socket", capture_index=0):
                with timed("vlm.infer", phase="vlm", camera_id="cam-1",
                           stage="lift", image_count=2):
                    pass
    directory = timing_directory(output)
    summary = json.loads((directory / "timing_summary.json").read_text())
    assert (directory / "events.jsonl").is_file()
    assert (directory / "manifest.json").is_file()
    assert summary["unfinished_span_count"] == 0
    assert summary["groups"]["vlm.infer"]["count"] == 1
    assert summary["groups"]["vlm.infer"]["occupied_s"] == (
        summary["groups"]["vlm.infer"]["inclusive_s"])
    assert summary["groups"]["vlm.infer"]["overlap_s"] == 0
    assert summary["groups"]["live.session"]["self_s"] <= (
        summary["groups"]["live.session"]["inclusive_s"])
    assert next(row for row in summary["spans"] if row["name"] ==
                "socket.capture")["attributes"]["capture_index"] == 0
    session = next(row for row in summary["spans"]
                   if row["name"] == "live.session")
    assert summary["span_envelope_s"] >= session["duration_s"]
    assert summary["phases"]["session"]["occupied_s"] == pytest.approx(
        session["duration_s"])
    assert summary["kinds"]["computation"]["count"] == 3
    assert summary["root_span_ids"] == [session["span_id"]]
    assert summarize_events(directory / "events.jsonl") == summary
    assert not output.exists()


def test_summary_groups_attempts_and_ordered_actions(tmp_path):
    output = tmp_path / "attempt_run"
    with profile_run(output):
        with timed("action.await_lift_observation", phase="trial",
                   kind="action", attempt_id="attempt_7",
                   candidate_id="table/0/20195"):
            with timed("trial.lift_motion", phase="trial", kind="motion",
                       attempt_id="attempt_7", iteration=0):
                time.sleep(.001)

    summary = json.loads((timing_directory(output) /
                          "timing_summary.json").read_text())
    attempt = summary["attempts"]["attempt_7"]
    assert attempt["candidate_ids"] == ["table/0/20195"]
    assert attempt["count"] == 2
    assert attempt["occupied_s"] > 0
    assert attempt["phases"]["trial"]["count"] == 2
    action = summary["ordered_actions"][0]
    assert action["name"] == "action.await_lift_observation"
    assert action["attempt_id"] == "attempt_7"
    assert action["candidate_id"] == "table/0/20195"
    assert action["human_wait_s"] == 0
    assert action["automatic_elapsed_s"] == pytest.approx(
        action["inclusive_s"])
    child = next(row for row in summary["spans"]
                 if row["name"] == "trial.lift_motion")
    assert child["effective_attempt_id"] == "attempt_7"
    assert child["effective_candidate_id"] == "table/0/20195"
    assert child["attributes"]["iteration"] == 0


def test_summary_separates_human_wait_from_automatic_action_time(tmp_path):
    output = tmp_path / "human_wait"
    with profile_run(output):
        with timed("live.session", phase="session"):
            with timed("action.await_lift_observation", phase="trial",
                       kind="action", attempt_id="attempt_8"):
                with timed("trial.lift_motion", phase="trial", kind="motion"):
                    time.sleep(.002)
                with timed("label.manual_review_wait", phase="trial",
                           kind="human"):
                    time.sleep(.003)

    summary = json.loads((timing_directory(output) /
                          "timing_summary.json").read_text())
    action = summary["ordered_actions"][0]
    assert action["human_wait_s"] > 0
    assert action["automatic_elapsed_s"] > 0
    assert action["human_wait_s"] + action["automatic_elapsed_s"] == \
        pytest.approx(action["inclusive_s"])
    assert summary["human_wait_occupied_s"] > 0
    assert summary["human_wait_cumulative_s"] >= \
        summary["human_wait_occupied_s"]
    assert (summary["human_wait_occupied_s"] +
            summary["span_envelope_non_human_s"] ==
            pytest.approx(summary["span_envelope_s"]))


def test_parallel_view_spans_retain_parent_and_union_overlap(tmp_path):
    class ParallelBackend:
        max_parallel_views = 2

    gate = Barrier(2)

    def run_view(camera):
        with timed("vlm.infer", phase="vlm", camera_id=camera):
            gate.wait(timeout=5)
            time.sleep(.03)
        return camera

    output = tmp_path / "parallel"
    with profile_run(output):
        with timed("trial.final_label", phase="trial"):
            assert map_view_calls(ParallelBackend(), ["front", "side"],
                                  run_view) == ["front", "side"]
    summary = json.loads((timing_directory(output) /
                          "timing_summary.json").read_text())
    parent = next(row for row in summary["spans"]
                  if row["name"] == "trial.final_label")
    batch = next(row for row in summary["spans"]
                 if row["name"] == "vlm.view_batch")
    infer = [row for row in summary["spans"] if row["name"] == "vlm.infer"]
    assert len(infer) == 2
    assert batch["parent_id"] == parent["span_id"]
    assert all(row["parent_id"] == batch["span_id"] for row in infer)
    assert summary["vlm_inference_overlap_s"] > 0
    assert summary["groups"]["vlm.infer"]["occupied_s"] < (
        summary["groups"]["vlm.infer"]["inclusive_s"])
    assert summary["groups"]["vlm.infer"]["overlap_s"] > 0
    assert summary["kinds"]["computation"]["overlap_s"] > 0
    assert batch["child_overlap_s"] > 0
    assert batch["child_cumulative_s"] > batch["child_occupied_s"]
    assert parent["self_s"] < parent["duration_s"]


def test_report_can_show_nested_elapsed_and_parallel_work(tmp_path, capsys):
    output = tmp_path / "hierarchy"
    with profile_run(output):
        with timed("live.session", phase="session"):
            with timed("socket.body_fit", phase="socket"):
                with timed("socket.body_multiview_median", phase="socket"):
                    pass
    assert report_timing.main([str(timing_directory(output)), "--tree"]) == 0
    printed = capsys.readouterr().out
    assert "occupied" in printed
    assert "overlap" in printed
    assert "Hierarchy (inclusive" in printed
    assert "    socket.body_multiview_median" in printed
    assert "child work" in printed


def test_profile_keeps_failure_trace(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="capture failed"):
        with profile_run(output):
            with timed("socket.capture", phase="socket"):
                raise RuntimeError("capture failed")
    directory = timing_directory(output)
    summary = json.loads((directory / "timing_summary.json").read_text())
    assert summary["groups"]["socket.capture"]["failures"] == 1
    failed = next(row for row in summary["spans"]
                  if row["name"] == "socket.capture")
    assert failed["exception"] == "RuntimeError('capture failed')"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failure"


def test_failed_action_summary_preserves_root_cause(tmp_path, capsys):
    output = tmp_path / "failed_action"
    with pytest.raises(ValueError, match="reset proposal scene changed"):
        with profile_run(output):
            with timed("action.recover_held_key_then_reobserve",
                       phase="trial", kind="action",
                       attempt_id="attempt_9", candidate_id="table/0/20195"):
                raise ValueError("reset proposal scene changed")
    summary = json.loads((timing_directory(output) /
                          "timing_summary.json").read_text())
    action = summary["ordered_actions"][0]
    assert action["outcome"] == "failure"
    assert action["exception"] == (
        "ValueError('reset proposal scene changed')")
    assert report_timing.main([str(timing_directory(output)), "--tree"]) == 0
    printed = capsys.readouterr().out
    assert "Failure causes (innermost first)" in printed
    assert "reset proposal scene changed" in printed


def test_profile_manifest_records_reproducible_nonsecret_configuration(
        tmp_path, monkeypatch):
    policy = tmp_path / "stage_policy.json"
    policy.write_text('{"post_lift_settle_s": 2.0}\n', encoding="utf-8")
    monkeypatch.setenv("PRECISION_STAGE_POLICY", str(policy))
    monkeypatch.setenv("PRECISION_FINAL_VLM_VIEW_COUNT", "3")
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-recorded")

    with profile_run(tmp_path / "run"):
        pass

    manifest = json.loads(
        (timing_directory(tmp_path / "run") / "manifest.json").read_text(
            encoding="utf-8"))
    config = manifest["runtime_configuration"]
    assert config["environment"]["PRECISION_FINAL_VLM_VIEW_COUNT"] == "3"
    assert "GEMINI_API_KEY" not in config["environment"]
    assert config["input_files"]["PRECISION_STAGE_POLICY"]["sha256"] == (
        hashlib.sha256(policy.read_bytes()).hexdigest())
    assert config["scope"] == (
        "allowlisted_nonsecret_configuration_no_hardware_access")


def test_profile_can_record_nonexception_terminal_status(tmp_path):
    output = tmp_path / "not_yet_done"
    with profile_run(output) as profile:
        profile.outcome = "needs_handler"
    manifest = json.loads((timing_directory(output) / "manifest.json").read_text())
    assert manifest["status"] == "needs_handler"


def test_profile_can_account_for_python_bootstrap_before_trace(tmp_path):
    output = tmp_path / "boot_timing"
    process_started_ns = time.perf_counter_ns()
    time.sleep(.002)
    with profile_run(output, process_started_ns=process_started_ns):
        with timed("live.session", phase="session"):
            time.sleep(.001)
    summary = json.loads((timing_directory(output) /
                          "timing_summary.json").read_text())
    assert summary["python_pre_profile_s"] >= .001
    assert summary["profile_body_s"] >= .001
    assert summary["python_process_to_profile_body_end_s"] == pytest.approx(
        summary["python_pre_profile_s"] + summary["profile_body_s"])


def test_profile_rejects_invalid_python_bootstrap_marker(tmp_path):
    with pytest.raises(ValueError, match="must precede"):
        with profile_run(tmp_path / "bad", process_started_ns=0):
            pass


def test_timed_without_live_profile_is_noop():
    with timed("standalone", phase="test"):
        pass


def test_local_trace_flushes_to_nas_only_after_run(tmp_path, monkeypatch):
    local = tmp_path / "local"
    output = tmp_path / "nas" / "session_001"
    monkeypatch.setenv("PRECISION_TIMING_LOCAL_ROOT", str(local))
    with profile_run(output):
        assert not timing_directory(output).exists()
        with timed("planning.v8_pickup", phase="planning", candidate_id="table/0/1"):
            pass
    assert (local / "session_001.timing" / "events.jsonl").is_file()
    summary = json.loads((timing_directory(output) / "timing_summary.json").read_text())
    assert summary["groups"]["planning.v8_pickup"]["count"] == 1
