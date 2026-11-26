import json
from pathlib import Path

from jsonschema import Draft202012Validator

from precision_insertion.undergraduate_evidence import compile_continuity_audit


SCHEMA = json.loads((Path(__file__).resolve().parents[1] /
                     "config/mvgotrack_continuity_audit.schema.json").read_text())


def _event(span, edge, name, attempt, utc_ns, *, outcome=None):
    row = {
        "span_id": span, "edge": edge, "name": name,
        "attempt_id": attempt, "utc_ns": utc_ns,
    }
    if outcome is not None:
        row["outcome"] = outcome
    return row


def _write_case(tmp_path, *, calibration="calib:a", missing_phase=None):
    attempt = "attempt_1"
    events = []
    names = {
        "pickup": "trial.pickup_squeeze",
        "lift": "trial.lift_motion",
        "transfer": "trial.transfer_motion",
        "precontact_hold": "trial.same_trial_alignment",
    }
    for index, (phase, name) in enumerate(names.items()):
        if phase == missing_phase:
            continue
        start = 100 + index * 100
        events.extend([
            _event(f"p{index}", "start", name, attempt, start),
            _event(f"p{index}", "end", name, attempt, start + 80,
                   outcome="success"),
        ])
    timing = tmp_path / "events.jsonl"
    timing.write_text("".join(json.dumps(row) + "\n" for row in events))
    stream = tmp_path / "mvgotrack_pose_stream.jsonl"
    rows = []
    for seq, received in enumerate((120, 220, 320, 420), start=1):
        rows.append({
            "schema": "forgehand.tracked_pose.v1", "seq": seq,
            "object": "key", "valid": True,
            "calib_session": calibration,
            "received_unix_ns": received,
        })
    stream.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return attempt, stream, timing


def test_compiler_passes_native_valid_pose_coverage_in_all_phases(tmp_path):
    attempt, stream, timing = _write_case(tmp_path)

    result = compile_continuity_audit(
        run_id="run_1", pose_stream=stream, timing_events=timing,
        attempt_id=attempt)

    assert result["verdict"] == "pass"
    assert result["phase_sample_counts"] == {
        "pickup": 1, "lift": 1, "transfer": 1, "precontact_hold": 1,
    }
    assert result["thresholds"] == {
        "score": None, "pose_jump": None, "workspace": None,
        "maximum_interframe_gap": None,
    }
    Draft202012Validator(SCHEMA).validate(result)


def test_compiler_does_not_infer_missing_motion_phase(tmp_path):
    attempt, stream, timing = _write_case(tmp_path, missing_phase="lift")

    result = compile_continuity_audit(
        run_id="run_2", pose_stream=stream, timing_events=timing,
        attempt_id=attempt)

    assert result["verdict"] == "fail"
    assert result["reason"] == "missing_or_ambiguous_phase:lift"
    Draft202012Validator(SCHEMA).validate(result)


def test_compiler_requires_one_calibration_identity(tmp_path):
    attempt, stream, timing = _write_case(tmp_path)
    rows = [json.loads(line) for line in stream.read_text().splitlines()]
    rows[-1]["calib_session"] = "calib:b"
    stream.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = compile_continuity_audit(
        run_id="run_3", pose_stream=stream, timing_events=timing,
        attempt_id=attempt)

    assert result["verdict"] == "fail"
    assert result["calibration_sessions"] == ["calib:a", "calib:b"]
