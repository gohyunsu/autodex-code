"""Post-shift planning has a measured transfer start, not an axial start."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import postshift_path_handoff as handoff  # noqa: E402
from precision_insertion import postshift_insertion  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from test_postshift_insertion import _setup  # noqa: E402


def _case(tmp_path, monkeypatch):
    args, _ = _setup(tmp_path, monkeypatch)
    result = postshift_insertion.plan_postshift_insertion_preflight(**args)
    checkpoint = args["checkpoint"]
    start = checkpoint.joint_sample.full_q.copy()
    hold = start.copy()
    hold[0] += .0001
    end = hold.copy()
    end[0] += .0001
    pre = np.eye(4)
    pre[2, 3] = .05
    final = pre.copy()
    final[2, 3] -= .03  # 10 mm approach clearance + 20 mm insertion
    mode = args["mode"]
    target_record = {
        "schema": "precision_insertion_rigid_targets_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object,
                 "target_depth_m": mode.target_depth_m},
        "T_robot_hand_preinsert": pre.tolist(),
        "T_robot_hand_verification": final.tolist(),
        "insertion_axis_robot": [0., 0., -1.],
        "preinsert_clearance_m": .01,
    }
    geometry = AssetPaths(tmp_path, mode).task_geometry
    targets = SimpleNamespace(
        to_record=lambda: target_record,
        T_robot_hand_preinsert=pre,
        task_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())
    planning = replace(
        result.planning,
        transfer_trajectory=np.stack([start, hold]),
        axial_trajectory=np.stack([hold, end]),
        sampled_held_path_audit={"sampled_clear": True},
        axial_waypoint_count=1,
        held_hand_q=start[7:], held_hand_source="measured",
        planner_query_records=(
            {"stage": "transfer", "success": True},
            {"stage": "axial_waypoint", "index": 1, "success": True}),
    )
    result = replace(result, planning=planning, targets=targets)
    report = postshift_insertion.write_postshift_insertion_preflight(
        result, tmp_path / "postshift_20mm")
    # This unit isolates the additional handoff contract. The upstream
    # source/CAD/archive replay has its own tests in test_postshift_insertion.
    monkeypatch.setattr(
        handoff, "verify_postshift_insertion_preflight",
        lambda path, **_kwargs: json.loads(Path(path).read_text()))
    measured = replace(
        checkpoint.joint_sample,
        sample_timestamp_s=100.4, arm_timestamp_s=100.4,
        hand_timestamp_s=100.4, arm_robot_uptime_s=10.4)
    kwargs = dict(
        preflight_report_path=report, expected=result,
        checkpoint=checkpoint, shift_plan=args["shift_plan"],
        mode=mode, shared_root=tmp_path, calibration=args["calibration"],
        measured_start=measured, decision_timestamp_s=100.45,
        max_state_age_s=.1, max_start_joint_error_rad=.001,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01)
    return kwargs


def test_handoff_preserves_transfer_before_axial(tmp_path, monkeypatch):
    kwargs = _case(tmp_path, monkeypatch)
    path = handoff.prepare_postshift_path_handoff(
        output_dir=tmp_path / "handoff", **kwargs)
    record = handoff.verify_postshift_path_handoff(
        path, expected=kwargs["expected"],
        checkpoint=kwargs["checkpoint"], shift_plan=kwargs["shift_plan"],
        mode=kwargs["mode"], shared_root=tmp_path,
        calibration=kwargs["calibration"])
    assert record["transfer_required"] is True
    assert record["transfer_start_q"] != record["axial_start_q"]
    assert record["transfer_end_q"] == record["axial_start_q"]
    assert record["robot_ready"] is False
    assert record["target_depth_m"] == .02
    with pytest.raises(FileExistsError):
        handoff.prepare_postshift_path_handoff(
            output_dir=tmp_path / "handoff", **kwargs)


def test_handoff_rejects_stale_state_and_swapped_axial(tmp_path, monkeypatch):
    kwargs = _case(tmp_path, monkeypatch)
    stale = replace(kwargs["measured_start"],
                    sample_timestamp_s=100.3, arm_timestamp_s=100.3,
                    hand_timestamp_s=100.3)
    with pytest.raises(ValueError, match="fresh measured hold"):
        handoff.prepare_postshift_path_handoff(
            output_dir=tmp_path / "stale",
            **{**kwargs, "measured_start": stale})
    path = handoff.prepare_postshift_path_handoff(
        output_dir=tmp_path / "handoff", **kwargs)
    record = json.loads(path.read_text())
    record["axial_start_q"][0] += .001
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs from replay"):
        handoff.verify_postshift_path_handoff(
            path, expected=kwargs["expected"],
            checkpoint=kwargs["checkpoint"], shift_plan=kwargs["shift_plan"],
            mode=kwargs["mode"], shared_root=tmp_path,
            calibration=kwargs["calibration"])


def test_handoff_rejects_non_20mm_targets(tmp_path, monkeypatch):
    kwargs = _case(tmp_path, monkeypatch)
    target_record = kwargs["expected"].targets.to_record().copy()
    final = np.asarray(target_record["T_robot_hand_verification"])
    final[2, 3] += .001
    target_record["T_robot_hand_verification"] = final.tolist()
    wrong = replace(kwargs["expected"],
                    targets=SimpleNamespace(to_record=lambda: target_record))
    report = postshift_insertion.write_postshift_insertion_preflight(
        wrong, tmp_path / "wrong_20mm")
    with pytest.raises(ValueError, match="do not describe 20 mm"):
        handoff.prepare_postshift_path_handoff(
            output_dir=tmp_path / "wrong_handoff",
            **{**kwargs, "expected": wrong,
               "preflight_report_path": report})
