from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.mvgotrack import (  # noqa: E402
    MVGoTrackPose, symmetry_reduced_alignment,
)
from precision_insertion import same_trial_xy  # noqa: E402
from precision_insertion import mvgotrack_runtime  # noqa: E402


def _assets(root: Path):
    info = root / "key" / "processed_data" / "info"
    info.mkdir(parents=True)
    (info / "symmetry.json").write_text(json.dumps({
        "type": "Dinf", "center": [0., 0., .04],
        "axes": [{"axis": [0., 0., 1.], "fold": "inf"},
                 {"axis": [1., 0., 0.], "fold": 2}],
    }))


def _alignment(root: Path, tracked: np.ndarray):
    nominal = np.eye(4)
    nominal[:3, :3] = np.diag([1., -1., -1.])
    nominal[:3, 3] = [.003, -.004, .1]
    return symmetry_reduced_alignment(
        tracked_pose_world=tracked, T_world_robot=np.eye(4),
        T_robot_socket=np.eye(4), T_robot_key_nominal=nominal,
        object_root=root, key_object="key",
        task_geometry={"key_frame": {
            "tip_z_m": .08, "insertion_axis": [0., 0., 1.]}})


def test_dinf_reduction_discards_axial_yaw(tmp_path):
    _assets(tmp_path)
    tracked = np.eye(4)
    tracked[:3, :3] = np.diag([1., -1., -1.])
    tracked[:3, 3] = [.003, -.004, .1]
    yaw = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                    [0., 0., 1., 0.], [0., 0., 0., 1.]])
    first = _alignment(tmp_path, tracked)
    second = _alignment(tmp_path, tracked @ yaw)
    np.testing.assert_allclose(first["tip_socket_m"], [.003, -.004, .02])
    np.testing.assert_allclose(first["xy_correction_socket_m"], [-.003, .004])
    np.testing.assert_allclose(second["tip_socket_m"], first["tip_socket_m"])
    np.testing.assert_allclose(
        second["insertion_axis_socket"], first["insertion_axis_socket"])
    assert first["symmetry_reduction"]["axial_yaw_used"] is False


def test_dinf_end_exchange_preserves_physical_tip(tmp_path):
    _assets(tmp_path)
    first_pose = np.eye(4)
    first_pose[:3, :3] = np.diag([1., -1., -1.])
    first_pose[:3, 3] = [.003, -.004, .1]
    flip = np.eye(4)
    flip[:3, :3] = np.diag([1., -1., -1.])
    center = first_pose[:3, :3] @ np.array([0., 0., .04]) + first_pose[:3, 3]
    exchanged = first_pose @ flip
    exchanged[:3, 3] = center - exchanged[:3, :3] @ np.array([0., 0., .04])
    first = _alignment(tmp_path, first_pose)
    second = _alignment(tmp_path, exchanged)
    np.testing.assert_allclose(second["tip_socket_m"], first["tip_socket_m"])
    np.testing.assert_allclose(
        second["insertion_axis_socket"], first["insertion_axis_socket"])
    assert second["symmetry_reduction"]["end_exchange_applied"] is True


def test_tracker_message_uses_native_valid_without_score_gate():
    row = MVGoTrackPose.from_message({
        "schema": "forgehand.tracked_pose.v1", "seq": 4,
        "object": "key", "frame": "world", "T": np.eye(4).tolist(),
        "valid": True, "status": "ok", "score": -100.,
        "frame_ids": {"a": 1, "b": 2, "c": 3, "d": 4},
        "stamp_capture": 1., "stamp_publish": 2.,
    })
    assert row.valid and row.score == -100.


def test_same_trial_measurement_accepts_live_tracker_without_images(
        tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "manifest.json").write_text("{}")
    report = tmp_path / "postlift.json"
    report.write_text("{}")
    geometry = tmp_path / "task_geometry.json"
    geometry.write_text("{}")
    monkeypatch.setattr(
        same_trial_xy, "AssetPaths",
        lambda *_args, **_kwargs: SimpleNamespace(task_geometry=geometry))
    alignment = {
        "status": "diagnostic_metric_xy_correction",
        "tracker_pose": {"frame_ids": {"a": 1, "b": 2}},
    }
    tracker = SimpleNamespace(measure=lambda **_kwargs: alignment)
    runner = SimpleNamespace(
        calibration=SimpleNamespace(record={"camera_calibration": {}}),
        shared_root=tmp_path, mode=object(),
        active_attempt=SimpleNamespace(attempt_id="attempt", candidate_id="candidate"),
        _postlift_report_path=report)
    record = same_trial_xy.measure_preinsert_xy(
        runner=runner, raw_bundle=raw, backend=None,
        alignment_limits=object(), index=0,
        nominal_tip_height_error_m=.01,
        xy_estimator="mvgotrack", mvgotrack=tracker)
    assert record["alignment"] is alignment
    assert record["camera_input_policy"] == {
        "method": "continuous_tracker_frames", "camera_ids": ["a", "b"],
        "scope": "no_socket_residual_camera_selection"}
    assert record["vlm_observations"] == []


def test_runtime_measurement_waits_for_frame_after_stopped_hold(monkeypatch):
    calls = []
    pose = SimpleNamespace(seq=13)

    class Subscriber:
        def newest_seq(self):
            return 12

        def wait_for_valid(self, **kwargs):
            calls.append(kwargs)
            return pose

    runner = object()
    runtime = mvgotrack_runtime.MVGoTrackRuntime(
        mvgotrack_runtime.MVGoTrackRuntimeConfig(
            endpoint="tcp://127.0.0.1:5591", ssh_host="capture4",
            ssh_port=77, remote_manager=Path("/manager"),
            remote_session_root=Path("/sessions"), timeout_s=4.),
        runner=runner)
    runtime._subscriber = Subscriber()
    runtime._last_seq = 8
    monkeypatch.setattr(
        mvgotrack_runtime, "alignment_from_tracker",
        lambda **kwargs: {"runner": kwargs["runner"],
                          "pose": kwargs["pose"]})
    result = runtime.measure(runner=runner)
    assert calls == [{"after_seq": 12, "timeout_s": 4.}]
    assert result == {"runner": runner, "pose": pose}
