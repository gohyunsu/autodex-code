from pathlib import Path
import importlib.util
import json
from types import SimpleNamespace

import pytest


CAMERAS = ("26053248", "25322639", "25305462", "25322642")


def _manager():
    path = Path(__file__).resolve().parents[1] / "capture4/manage_mvgotrack.py"
    spec = importlib.util.spec_from_file_location("capture4_mvgotrack_manager", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibration(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "intrinsics.json").write_text(
        json.dumps({camera: [camera, "intrinsic"]
                    for camera in CAMERAS}) + "\n")
    (root / "extrinsics.json").write_text(
        json.dumps({camera: [camera, "extrinsic"]
                    for camera in CAMERAS}) + "\n")
    return root


def test_capture4_command_names_exact_camera_world(tmp_path, monkeypatch):
    manager = _manager()
    remote = tmp_path / "remote"
    _calibration(remote / "cam_param")
    monkeypatch.setattr(manager, "ROOT", remote)
    args = SimpleNamespace(
        endpoint="tcp://capture4:5591", mesh=tmp_path / "key.obj",
        object="key", init_pose=tmp_path / "pose.txt",
        session=tmp_path / "session")

    command = manager._tracker_command(args)
    index = command.index("--calib-session")
    assert command[index + 1] == manager.calibration_identity(
        remote / "cam_param")


def test_main_and_capture4_calibration_identities_are_identical(tmp_path):
    manager = _manager()
    from precision_insertion.mvgotrack_runtime import calibration_identity

    calibration = _calibration(tmp_path / "calibration")
    assert manager.calibration_identity(calibration) == calibration_identity(
        calibration)


def test_runtime_rejects_only_a_different_camera_world(tmp_path):
    from precision_insertion.mvgotrack_runtime import (
        MVGoTrackRuntime, MVGoTrackRuntimeConfig,
        require_matching_camera_world,
    )

    calibration = _calibration(tmp_path / "calibration")
    config = MVGoTrackRuntimeConfig(
        endpoint="tcp://capture4:5591", ssh_host="capture4", ssh_port=77,
        remote_manager=Path("/manager"), remote_session_root=Path("/sessions"),
        timeout_s=4.0, calibration_dir=calibration)
    runtime = MVGoTrackRuntime(config, runner=object())
    assert runtime.config.expected_calibration_session.startswith(
        "precision_calib_v1:")

    changed = tmp_path / "changed"
    _calibration(changed)
    values = json.loads((changed / "extrinsics.json").read_text())
    values[CAMERAS[0]] = [CAMERAS[0], "changed"]
    (changed / "extrinsics.json").write_text(
        json.dumps(values) + "\n")
    other = MVGoTrackRuntimeConfig(
        endpoint="tcp://capture4:5591", ssh_host="capture4", ssh_port=77,
        remote_manager=Path("/manager"), remote_session_root=Path("/sessions"),
        timeout_s=4.0, calibration_dir=changed)
    assert other.expected_calibration_session != config.expected_calibration_session
    require_matching_camera_world(
        config.expected_calibration_session,
        config.expected_calibration_session)
    with pytest.raises(RuntimeError, match="camera world differs"):
        require_matching_camera_world(
            config.expected_calibration_session,
            other.expected_calibration_session)


def test_missing_calibration_file_is_not_a_camera_world(tmp_path):
    from precision_insertion.mvgotrack_runtime import MVGoTrackRuntimeConfig

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "intrinsics.json").write_text(
        json.dumps({camera: [camera] for camera in CAMERAS}) + "\n")
    config = MVGoTrackRuntimeConfig(
        endpoint="tcp://capture4:5591", ssh_host="capture4", ssh_port=77,
        remote_manager=Path("/manager"), remote_session_root=Path("/sessions"),
        timeout_s=4.0, calibration_dir=calibration)
    with pytest.raises(FileNotFoundError):
        config.validate()


def test_unused_camera_and_json_format_do_not_change_tracker_world(tmp_path):
    manager = _manager()
    from precision_insertion.mvgotrack_runtime import calibration_identity

    calibration = _calibration(tmp_path / "calibration")
    baseline = calibration_identity(calibration, CAMERAS)
    for name in ("intrinsics.json", "extrinsics.json"):
        values = json.loads((calibration / name).read_text())
        values["unused_camera"] = {"formatting": "irrelevant"}
        (calibration / name).write_text(
            json.dumps(values, indent=4, sort_keys=False) + "\n")

    assert calibration_identity(calibration, CAMERAS) == baseline
    assert manager.calibration_identity(calibration) == baseline
