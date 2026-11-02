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
    assert result["recorded_runtime_configuration_count"] == 2
    assert result["unrecorded_runtime_configuration_count"] == 0
    assert result["recorded_runtime_configuration_identity_count"] == 1
    assert result["runtime_configuration_comparison_available"] is True
    assert result["runtime_configuration_mixing_status"] == "known_single"
    assert result["statistics_pool_is_controlled_condition"] is True
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
    legacy_manifest = tmp_path / "run_a.timing" / "manifest.json"
    legacy_value = json.loads(legacy_manifest.read_text())
    legacy_value.pop("runtime_configuration")
    legacy_manifest.write_text(json.dumps(legacy_value))
    all_runs = summarize(tmp_path)
    assert all_runs["mixed_runtime_configurations"] is True
    assert all_runs["recorded_runtime_configuration_count"] == 1
    assert all_runs["unrecorded_runtime_configuration_count"] == 1
    assert all_runs["recorded_runtime_configuration_identity_count"] == 1
    assert all_runs["runtime_configuration_comparison_available"] is False
    assert all_runs["runtime_configuration_mixing_status"] == (
        "partially_recorded")
    assert all_runs["statistics_pool_is_controlled_condition"] is False
    configured = next(row for row in all_runs["runs"]
                      if row["run_id"] == "run_b")
    filtered = summarize(
        tmp_path, runtime_configuration_sha256=
        configured["runtime_configuration_sha256"])
    assert filtered["discovered_run_count"] == 1
    assert filtered["runs"][0]["run_id"] == "run_b"
    assert filtered["runtime_configuration_comparison_available"] is True


def test_multi_run_timing_marks_all_legacy_configuration_unknown(tmp_path):
    _run(tmp_path, "run_a", "table/0/20195")
    _run(tmp_path, "run_b", "table/0/20195")
    for manifest in tmp_path.glob("*.timing/manifest.json"):
        value = json.loads(manifest.read_text())
        value.pop("runtime_configuration")
        manifest.write_text(json.dumps(value))

    result = summarize(tmp_path)
    assert result["mixed_runtime_configurations"] is False
    assert result["recorded_runtime_configuration_count"] == 0
    assert result["unrecorded_runtime_configuration_count"] == 2
    assert result["recorded_runtime_configuration_identity_count"] == 0
    assert result["runtime_configuration_comparison_available"] is False
    assert result["runtime_configuration_mixing_status"] == "unknown"
    assert result["statistics_runtime_configurations"][
        "mixing_status"] == "unknown"
    assert result["statistics_pool_is_controlled_condition"] is False


def test_multi_run_timing_separates_discovered_from_statistics_config(
        tmp_path):
    _run(tmp_path, "complete", "table/0/20195")
    # This trace is discoverable but excluded from default statistics. Its
    # different configuration must not make the complete-run pool look mixed.
    _run(tmp_path, "incomplete", "table/0/20195")
    events = tmp_path / "incomplete.timing" / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    rows = [row for row in rows if not (
        row.get("edge") == "end" and row.get("name") == "live.session")]
    events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "incomplete.timing" / "manifest.json"
    value = json.loads(manifest.read_text())
    value["runtime_configuration"] = {
        "schema": "test_runtime_v1", "mode": "different"}
    manifest.write_text(json.dumps(value))

    result = summarize(tmp_path)
    assert result["discovered_run_count"] == 2
    assert result["statistics_run_count"] == 1
    assert result["discovered_runtime_configurations"][
        "mixing_status"] == "known_mixed"
    assert result["statistics_runtime_configurations"][
        "mixing_status"] == "known_single"
    assert result["statistics_pool_is_controlled_condition"] is True
