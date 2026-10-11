"""An exposed-key report needs a matching reviewed held-out error audit."""

import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import depth_admission  # noqa: E402
from precision_insertion.exposed_depth_eval import (  # noqa: E402
    evaluate_exposed_depth_manifest,
)
from test_exposed_depth_eval import _sample  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path, monkeypatch):
    inputs, manifest, _, _ = _sample(tmp_path, monkeypatch)
    evaluation = evaluate_exposed_depth_manifest(
        manifest, mode=inputs["mode"], shared_root=tmp_path,
        calibration=inputs["calibration"])
    audit = tmp_path / "heldout_audit.json"
    audit.write_text(json.dumps(evaluation), encoding="utf-8")
    report_path = tmp_path / "depth_estimate/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mode = inputs["mode"]
    review = {
        "schema": "precision_insertion_exposed_depth_commissioning_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "camera_calibration_sha256": report["source"][
            "camera_calibration_sha256"],
        "task_geometry_sha256": report["task_geometry_sha256"],
        "vlm_model_id": report["vlm_observations"][0]["backend_model"],
        "limits": report["diagnostic"]["limits"],
        "reviewed_by": "synthetic_test_reviewer",
        "minimum_heldout_trials": 1,
        "heldout_evaluation": {"path": str(audit), "sha256": _sha(audit)},
    }
    commissioned = tmp_path / "commissioning.json"
    commissioned.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(depth_admission, "load_session_calibration",
                        lambda *_args, **_kwargs: inputs["calibration"])
    kwargs = dict(
        depth_report_path=report_path, commissioning_path=commissioned,
        attempt_id=inputs["attempt_id"],
        candidate_id=inputs["candidate_id"],
        session_calibration_sha256=report["source"][
            "session_calibration_sha256"],
        final_bundle=inputs["final_bundle"],
        guarded_completed_at_s=inputs["completed_at_s"],
        decision_timestamp_s=inputs["decision_timestamp_s"])
    return kwargs, review, commissioned


def test_replayed_visible_depth_can_be_admitted_with_reviewed_audit(
        tmp_path, monkeypatch):
    kwargs, _, _ = _case(tmp_path, monkeypatch)
    interval, evidence = depth_admission.admit_exposed_depth(**kwargs)
    assert interval[0] >= 0.020
    assert evidence["status"] == "admitted_exposed_length_cad"
    assert evidence["key_depth_interval_m"] == list(interval)


def test_depth_admission_rejects_wrong_camera_or_changed_audit(
        tmp_path, monkeypatch):
    kwargs, review, commissioned = _case(tmp_path, monkeypatch)
    review["camera_calibration_sha256"] = "0" * 64
    commissioned.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        depth_admission.admit_exposed_depth(**kwargs)
    review["camera_calibration_sha256"] = json.loads(
        kwargs["depth_report_path"].read_text())["source"][
            "camera_calibration_sha256"]
    commissioned.write_text(json.dumps(review), encoding="utf-8")
    Path(review["heldout_evaluation"]["path"]).write_text("{}",
                                                         encoding="utf-8")
    with pytest.raises(ValueError, match="audit changed"):
        depth_admission.admit_exposed_depth(**kwargs)
