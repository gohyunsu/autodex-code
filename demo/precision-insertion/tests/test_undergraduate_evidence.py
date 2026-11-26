import json
from pathlib import Path

from precision_insertion.undergraduate_evidence import build_report


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _base_run(tmp_path: Path, run_id: str) -> tuple[Path, Path]:
    root = tmp_path / "runs"
    control = root / "_control" / run_id
    recording = root / "2026/11/03" / run_id
    video = recording / "video/camera.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write(control / f"launch_{run_id}.json", {
        "evidence_dir": str(control / "session_evidence"),
    })
    _write(root / "_recording_receipts" / run_id / "recording_ref.json", {
        "schema": "precision_insertion_zerodex_recording_receipt_v1",
        "pipeline_exit_code": 0,
        "recording_stop_reason": "release_and_clear_completed",
        "recording": {"video": "camera.avi", "status": "completed"},
    })
    return root, control


def test_video_only_is_not_continuous_tracking_evidence(tmp_path):
    run_id = "20261103_video_only"
    root, _ = _base_run(tmp_path, run_id)

    report = build_report(root)
    run = report["runs"][0]

    assert run["u1_visual_commissioning"]["complete"] is False
    assert run["u1_visual_commissioning"]["checks"]["episode_video"] is True
    assert report["summary"]["u1_complete_count"] == 0


def test_same_episode_tracker_bundle_still_needs_continuity_audit(tmp_path):
    run_id = "20261103_tracked"
    root, control = _base_run(tmp_path, run_id)
    tracker = control / "session_evidence/mvgotrack/init_capture"
    _write(tracker / "initialization.json", {
        "schema": "precision_insertion_mvgotrack_initialization_v1",
        "pose_source": "accepted_sam3_foundpose_tabletop",
    })
    _write(tracker / "first_pose.json", {"seq": 1})
    (tracker.parent / "mvgotrack_pose_trace.jsonl").write_text(
        '{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")

    report = build_report(root)

    assert report["summary"]["u1_complete_runs"] == []
    result = report["runs"][0]["u1_visual_commissioning"]
    assert result["artifact_bundle_complete"] is True
    assert result["complete"] is False


def test_episode_owned_continuity_audit_completes_u1(tmp_path):
    run_id = "20261103_audited"
    root, control = _base_run(tmp_path, run_id)
    tracker = control / "session_evidence/mvgotrack/init_capture"
    _write(tracker / "initialization.json", {
        "schema": "precision_insertion_mvgotrack_initialization_v1",
    })
    _write(tracker / "first_pose.json", {"seq": 1})
    (tracker.parent / "mvgotrack_pose_trace.jsonl").write_text(
        '{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
    _write(tracker.parent / "mvgotrack_continuity_audit.json", {
        "schema": "precision_insertion_mvgotrack_continuity_audit_v1",
        "run_id": run_id,
        "interval": "pickup_start_to_precontact_hold",
        "verdict": "pass",
        "first_seq": 1,
        "last_seq": 10,
        "sample_count": 10,
        "sequence_strictly_increasing": True,
        "calibration_sessions": ["calib:test"],
        "objects": ["precision_key_cylinder_r15_h80"],
        "phase_sample_counts": {
            "pickup": 2, "lift": 3, "transfer": 3, "precontact_hold": 2,
        },
        "interval_unix_ns": {
            "pickup_start": 100, "precontact_hold": 200,
        },
        "evidence_refs": ["mvgotrack_pose_trace.jsonl", "video/camera.avi"],
    })

    report = build_report(root)

    assert report["summary"]["u1_complete_runs"] == [run_id]


def test_continuity_audit_without_all_motion_phases_does_not_pass(tmp_path):
    run_id = "20261103_missing_phase"
    root, control = _base_run(tmp_path, run_id)
    tracker = control / "session_evidence/mvgotrack/init_capture"
    _write(tracker / "initialization.json", {
        "schema": "precision_insertion_mvgotrack_initialization_v1",
    })
    _write(tracker / "first_pose.json", {"seq": 1})
    (tracker.parent / "mvgotrack_pose_trace.jsonl").write_text(
        '{"seq": 1}\n{"seq": 2}\n', encoding="utf-8")
    _write(tracker.parent / "mvgotrack_continuity_audit.json", {
        "schema": "precision_insertion_mvgotrack_continuity_audit_v1",
        "run_id": run_id,
        "interval": "pickup_start_to_precontact_hold",
        "verdict": "pass",
        "first_seq": 1,
        "last_seq": 10,
        "sample_count": 10,
        "phase_sample_counts": {
            "pickup": 2, "lift": 3, "transfer": 3,
        },
        "interval_unix_ns": {
            "pickup_start": 100, "precontact_hold": 200,
        },
        "evidence_refs": ["mvgotrack_pose_trace.jsonl"],
    })

    result = build_report(root)["runs"][0]["u1_visual_commissioning"]

    assert result["artifact_bundle_complete"] is True
    assert result["checks"]["pickup_to_precontact_continuity_audit"] is False
    assert result["complete"] is False



def test_unrelated_jsonl_is_not_mistaken_for_tracker_trace(tmp_path):
    run_id = "20261103_unrelated"
    root, control = _base_run(tmp_path, run_id)
    tracker = control / "session_evidence/mvgotrack/init_capture"
    _write(tracker / "initialization.json", {
        "schema": "precision_insertion_mvgotrack_initialization_v1",
    })
    _write(tracker / "first_pose.json", {"seq": 1})
    (control / "session_evidence/timing_trace.jsonl").write_text(
        "{}\n", encoding="utf-8")

    run = build_report(root)["runs"][0]

    assert run["artifacts"]["tracker_pose_stream"] == []
    assert run["u1_visual_commissioning"]["complete"] is False
