"""Multiple run traces aggregate occupied wall time without double counting."""

from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.timing_profile import profile_run, timed  # noqa: E402
from summarize_timing_runs import summarize  # noqa: E402


def _run(root: Path, name: str, candidate: str) -> None:
    with profile_run(root / name):
        with timed("live.session", phase="session"):
            with timed("action.await_lift_observation", phase="trial",
                       kind="action", attempt_id=f"attempt_{name}",
                       candidate_id=candidate):
                with timed("trial.lift_motion", phase="trial", kind="motion"):
                    time.sleep(.001)
                with timed("label.manual_review_wait", phase="trial",
                           kind="human"):
                    time.sleep(.001)


def test_multi_run_timing_reports_median_and_human_wait(tmp_path):
    _run(tmp_path, "run_a", "table/0/20195")
    _run(tmp_path, "run_b", "table/0/20195")
    result = summarize(tmp_path, candidate_id="table/0/20195")
    assert result["discovered_run_count"] == 2
    assert result["complete_run_count"] == 2
    assert result["statistics_run_count"] == 2
    assert result["overall"]["span_envelope_s"]["median"] > 0
    assert result["overall"]["human_wait_occupied_s"]["median"] > 0
    assert result["operations"]["trial.lift_motion"]["present_runs"] == 2
    assert result["actions"]["action.await_lift_observation"][
        "automatic_s"]["median"] > 0


def test_multi_run_timing_candidate_filter(tmp_path):
    _run(tmp_path, "run_a", "table/0/20195")
    _run(tmp_path, "run_b", "table/0/20193")
    result = summarize(tmp_path, candidate_id="table/0/20195")
    assert result["discovered_run_count"] == 1
    assert result["runs"][0]["candidate_ids"] == ["table/0/20195"]


def test_multi_run_timing_can_filter_exact_runtime_configuration(tmp_path):
    _run(tmp_path, "run_a", "table/0/20195")
    _run(tmp_path, "run_b", "table/0/20195")
    manifest = tmp_path / "run_b.timing" / "manifest.json"
    value = json.loads(manifest.read_text())
    value["runtime_configuration"] = {
        "schema": "test_runtime_v1", "mode": "closed_loop"}
    manifest.write_text(json.dumps(value))
    all_runs = summarize(tmp_path)
    assert all_runs["mixed_runtime_configurations"] is True
    configured = next(row for row in all_runs["runs"]
                      if row["run_id"] == "run_b")
    filtered = summarize(
        tmp_path, runtime_configuration_sha256=
        configured["runtime_configuration_sha256"])
    assert filtered["discovered_run_count"] == 1
    assert filtered["runs"][0]["run_id"] == "run_b"
