"""A saved reset landing may not be credited from an arbitrary release file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.repose_execution_evidence import (  # noqa: E402
    PHASES, verify_repose_execution_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    report_dir = tmp_path / "preflight"
    report_dir.mkdir()
    scene = report_dir / "trial_scene.json"
    scene.write_text('{"mesh": {"fixture_socket": {}}}\n', encoding="utf-8")
    trajectory = report_dir / "planned_trajectories.npz"
    trajectory.write_bytes(b"test archive bytes")
    inputs = {}
    for name in ("session", "catalog", "key_pose_world", "live_start_q",
                 "limits"):
        path = tmp_path / f"{name}.input"
        path.write_text(name, encoding="utf-8")
        inputs[name] = {"path": str(path), "sha256": _sha(path)}
    report = report_dir / "report.json"
    report.write_text(json.dumps({
        "status": "nominal_reset_preflight_pass_drop_unobserved",
        "selected_seed": {"seed_id": "191"}, "robot_ready": False,
        "artifacts": {
            "trial_scene": scene.name, "trial_scene_sha256": _sha(scene),
            "planned_trajectories": trajectory.name,
            "planned_trajectories_sha256": _sha(trajectory),
            "input_files": inputs,
        },
    }), encoding="utf-8")
    sources = {}
    for name in ("trajectory_feedback", "safety", "grasp_state",
                 "hand_feedback"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name}), encoding="utf-8")
        sources[name] = {"path": str(path), "sha256": _sha(path)}
    phases = []
    for index, name in enumerate(PHASES):
        row = {"name": name, "started_at_s": 2.0 + index,
               "completed_at_s": 3.0 + index, "complete": True,
               "safety_abort": False}
        if name in {"held_lift", "held_transfer", "held_descent"}:
            row["grasp_held"] = True
        if name == "release_open":
            row["hand_open_feedback"] = True
        phases.append(row)
    record = {
        "schema": "precision_insertion_repose_execution_evidence_v1",
        "source": "commissioned_external_controller",
        "attempt_id": "reset_1", "preflight_report_sha256": _sha(report),
        "planned_trajectories_sha256": _sha(trajectory),
        "session_calibration_sha256": "frozen-session-hash",
        "selected_seed": {"seed_id": "191"},
        "safety_abort": False, "grip_loss_before_release": False,
        "phases": phases, "source_records": sources,
    }
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(record), encoding="utf-8")
    kwargs = dict(
        path=execution, attempt_id="reset_1", attempt_started_at_s=1.0,
        preflight_report_path=report,
        preflight_report_sha256=_sha(report),
        session_calibration_sha256="frozen-session-hash",
        selected_seed={"seed_id": "191"})
    return record, kwargs


def test_repose_execution_requires_complete_bound_sequence(tmp_path):
    record, kwargs = _fixture(tmp_path)
    checked = verify_repose_execution_evidence(**kwargs)
    assert checked["release_completed_at_s"] == 7.0
    assert checked["exit_completed_at_s"] == 9.0
    assert checked["sha256"] == _sha(kwargs["path"])
    assert len(record["phases"]) == len(PHASES)


@pytest.mark.parametrize(("change", "message"), [
    (lambda row: row.update(attempt_id="other"), "bound"),
    (lambda row: row.update(safety_abort=True), "bound"),
    (lambda row: row["phases"].pop(), "ordered phase"),
    (lambda row: row["phases"][2].update(grasp_held=False), "lost"),
    (lambda row: row["phases"][4].update(hand_open_feedback=False),
     "open-hand"),
    (lambda row: row["phases"][5].update(started_at_s=6.5),
     "not safely completed"),
    (lambda row: row["source_records"].pop("safety"), "source records"),
])
def test_repose_execution_rejects_false_or_incomplete_claims(
        tmp_path, change, message):
    record, kwargs = _fixture(tmp_path)
    change(record)
    kwargs["path"].write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_repose_execution_evidence(**kwargs)


def test_repose_execution_rejects_changed_producer_and_plan(tmp_path):
    record, kwargs = _fixture(tmp_path)
    source = Path(record["source_records"]["hand_feedback"]["path"])
    source.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source is missing or changed"):
        verify_repose_execution_evidence(**kwargs)
    source.write_text('{"source": "hand_feedback"}', encoding="utf-8")
    scene = kwargs["preflight_report_path"].parent / "trial_scene.json"
    scene.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scene changed"):
        verify_repose_execution_evidence(**kwargs)
