"""The square XY retry needs a new observed hold before axial contact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.observed_retry_arrival_checkpoint import (  # noqa: E402
    assess_observed_retry_arrival, write_observed_retry_arrival,
)
from precision_insertion.retry_preflight import XYRetryPreflight  # noqa: E402
from test_preinsert_checkpoint import _setup  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retry_inputs(tmp_path):
    args = _setup(tmp_path)
    attempt = args["attempt"]
    attempt.labels["preinsert_reached"] = True
    attempt.labels["insertion_success"] = False
    attempt._pending_retry = True
    attempt.xy_offset_socket_m = (0.001, 0.0)
    attempt.events.append({"stage": "xy_retry", "value": "x_plus_1mm"})
    old = args.pop("postlift")
    args.pop("postlift_report_path")
    target = old.targets.T_robot_hand_preinsert.copy()
    targets = SimpleNamespace(
        xy_offset_socket_m=(0.001, 0.0),
        T_robot_hand_preinsert=target,
        to_record=lambda: {"xy_offset_socket_m": [0.001, 0.0],
                           "T_robot_hand_preinsert": target.tolist()})
    trajectory = np.zeros((2, 13))
    planning = SimpleNamespace(
        sampled_planning_pass=True, transfer_trajectory=trajectory,
        to_record=lambda: {"sampled_planning_pass": True})
    preflight = XYRetryPreflight(
        "sampled_retry_preflight_pass", ("table", "0", "3"),
        "x_plus_1mm", targets, planning, {}, np.zeros(13),
        np.eye(4), 10.5, 10.5)
    folder = tmp_path / "xy_preflight"
    folder.mkdir()
    archive = folder / "planned_trajectories.npz"
    np.savez_compressed(archive, transfer=trajectory)
    report = folder / "report.json"
    report.write_text(json.dumps({
        **preflight.to_record(),
        "planned_trajectories": archive.name,
        "planned_trajectories_sha256": _sha(archive),
    }), encoding="utf-8")
    transfer = args["transfer_execution_path"]
    record = json.loads(transfer.read_text(encoding="utf-8"))
    record.update({
        "schema": "precision_insertion_observed_xy_transfer_execution_v1",
        "preflight_report_path": str(report),
        "preflight_report_sha256": _sha(report),
        "trajectory_sha256": _sha(archive),
    })
    transfer.write_text(json.dumps(record), encoding="utf-8")
    return {**args, "preflight": preflight,
            "preflight_report_path": report}


def test_observed_retry_arrival_uses_fresh_camera_and_saved_xy_plan(tmp_path):
    args = _retry_inputs(tmp_path)
    arrival = assess_observed_retry_arrival(**args)
    assert arrival.preinsert_reached is True
    assert arrival.to_record()["retry_arrived"] is True
    assert "preinsert_reached" not in arrival.to_record()
    saved = write_observed_retry_arrival(arrival, tmp_path / "arrival")
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert report["schema"] == "precision_insertion_observed_xy_retry_arrival_v1"
    assert sorted(report["overlay_png_sha256"]) == ["cam_a", "cam_b"]


def test_observed_retry_arrival_never_promotes_an_occluded_key(tmp_path):
    args = _retry_inputs(tmp_path)
    from test_preinsert_checkpoint import _VLM
    args["backend"] = _VLM("unobservable")
    assert assess_observed_retry_arrival(**args).preinsert_reached is None
    transfer = args["transfer_execution_path"]
    row = json.loads(transfer.read_text(encoding="utf-8"))
    row["completed_at_s"] = 11.1
    transfer.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="do not follow"):
        assess_observed_retry_arrival(**args)
