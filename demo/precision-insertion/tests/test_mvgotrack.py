from pathlib import Path
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.mvgotrack import (  # noqa: E402
    MVGoTrackPose, symmetry_reduced_alignment,
)
from precision_insertion import same_trial_xy  # noqa: E402
from precision_insertion import mvgotrack_runtime  # noqa: E402
from precision_insertion.live_key_trial import (  # noqa: E402
    ObservedLiveKey, PreparedLiveKeyTrial,
)


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


def test_runtime_initializes_capture4_from_accepted_foundpose_before_measurement(
        tmp_path, monkeypatch):
    mesh = tmp_path / "key.obj"
    mesh.write_text("v 0 0 0\n")
    runner = SimpleNamespace(
        output_dir=tmp_path / "run", shared_root=tmp_path, mode=object())
    observation = SimpleNamespace(
        family="cylinder", capture_id="key_capture_7", key_object="key",
        pose_world=np.diag([1., -1., -1., 1.]),
        source_capture_dir=tmp_path / "source_capture")
    prepared = SimpleNamespace(observation=observation)
    events = []

    class Pose:
        seq = 22

        @staticmethod
        def to_record():
            return {"schema": "forgehand.tracked_pose.v1", "seq": 22}

    class Subscriber:
        def __init__(self, endpoint, expected_object):
            events.append(("subscriber", endpoint, expected_object))
            self.closed = False

        @staticmethod
        def newest_seq():
            return 21

        def wait_for_valid(self, **kwargs):
            events.append(("wait", kwargs))
            return Pose()

        def close(self):
            self.closed = True
            events.append(("subscriber_close",))

    monkeypatch.setattr(mvgotrack_runtime, "MVGoTrackSubscriber", Subscriber)
    monkeypatch.setattr(
        mvgotrack_runtime, "AssetPaths",
        lambda *_args, **_kwargs: SimpleNamespace(
            raw_mesh=lambda object_name: mesh))
    runtime = mvgotrack_runtime.MVGoTrackRuntime(
        mvgotrack_runtime.MVGoTrackRuntimeConfig(
            endpoint="tcp://capture4:5591", ssh_host="capture4",
            ssh_port=77, remote_manager=Path("/manager"),
            remote_session_root=Path("/sessions"), timeout_s=4.),
        runner=runner)

    def fake_ssh(*args, check=True):
        events.append(("ssh", args, check))
        return SimpleNamespace(stdout="tracker ready\n")

    def fake_scp(source, destination):
        events.append(("scp", Path(source).name, destination))

    monkeypatch.setattr(runtime, "_ssh", fake_ssh)
    monkeypatch.setattr(runtime, "_scp", fake_scp)
    runtime.initialize(prepared)

    local = runner.output_dir / "mvgotrack" / "init_key_capture_7"
    np.testing.assert_allclose(
        np.loadtxt(local / "T_world_key.txt"), observation.pose_world)
    metadata = json.loads((local / "initialization.json").read_text())
    assert metadata["pose_source"] == "accepted_sam3_foundpose_tabletop"
    assert metadata["key_object"] == "key"
    assert (local / "key.obj").read_bytes() == mesh.read_bytes()
    assert json.loads((local / "first_pose.json").read_text())["seq"] == 22
    assert events[0] == ("subscriber", "tcp://capture4:5591", "key")
    assert ("wait", {"after_seq": 21, "timeout_s": 4.}) in events
    start = [row for row in events
             if row[0] == "ssh" and "start-or-reinit" in row[1]]
    assert len(start) == 1
    assert start[0][1][0:2] == ("/manager", "start-or-reinit")
    assert runtime._initialized_capture_id == "key_capture_7"
    assert runtime._last_seq == 22

    runtime.close()
    assert any(row[0] == "ssh" and row[1][1] == "stop" for row in events)
    assert ("subscriber_close",) in events


def test_runtime_refuses_non_cylinder_initializer(tmp_path):
    runtime = mvgotrack_runtime.MVGoTrackRuntime(
        mvgotrack_runtime.MVGoTrackRuntimeConfig(
            endpoint="tcp://127.0.0.1:5591", ssh_host="capture4",
            ssh_port=77, remote_manager=Path("/manager"),
            remote_session_root=Path("/sessions"), timeout_s=4.),
        runner=SimpleNamespace(output_dir=tmp_path))
    prepared = SimpleNamespace(observation=SimpleNamespace(family="square"))
    with pytest.raises(ValueError, match="cylindrical key"):
        runtime.initialize(prepared)


def test_runtime_overlaps_tracker_initialization_with_candidate_planning(
        tmp_path, monkeypatch):
    runtime = mvgotrack_runtime.MVGoTrackRuntime(
        mvgotrack_runtime.MVGoTrackRuntimeConfig(
            endpoint="tcp://127.0.0.1:5591", ssh_host="capture4",
            ssh_port=77, remote_manager=Path("/manager"),
            remote_session_root=Path("/sessions"), timeout_s=4.),
        runner=SimpleNamespace(output_dir=tmp_path))
    observation = SimpleNamespace(capture_id="key_capture_8")
    state = SimpleNamespace()
    observed = ObservedLiveKey(observation, state, tmp_path / "key")
    prepared = PreparedLiveKeyTrial(
        observation, state, SimpleNamespace(status="sampled_planning_pass"),
        tmp_path / "key")
    started = threading.Event()
    allow_first_pose = threading.Event()
    completed = threading.Event()

    def initialize(value):
        assert value is observed
        started.set()
        assert allow_first_pose.wait(2.)
        runtime._initialized_capture_id = value.observation.capture_id
        completed.set()

    monkeypatch.setattr(runtime, "initialize", initialize)
    runtime.begin_initialize(observed)
    assert started.wait(2.)
    assert not completed.is_set()
    # Candidate planning may run here while capture4 waits for its first pose.
    planning_result = "sampled_planning_pass"
    assert planning_result == prepared.preflight.status
    allow_first_pose.set()
    runtime.await_initialized(prepared)
    assert completed.is_set()
    runtime.close()
