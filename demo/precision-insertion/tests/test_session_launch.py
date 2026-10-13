"""Live startup must validate complete camera inputs before any connection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline  # noqa: E402
from precision_insertion import session_launch as launch  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def _config(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    mode = select_mode("square", 1.5)
    paths = AssetPaths(shared, mode)
    for path in (paths.raw_mesh(mode.socket_object),
                 paths.socket_collision_mesh,
                 paths.foundpose_repre(mode.socket_object)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test asset", encoding="utf-8")
    capture = tmp_path / "capture"
    capture.mkdir()
    camera_dir = tmp_path / "cam_param"
    camera_dir.mkdir()
    intr = {serial: {
        "original_intrinsics": np.eye(3).tolist(),
        "intrinsics_undistort": np.eye(3).tolist(),
        "dist_params": [0, 0, 0, 0, 0], "height": 480, "width": 640,
    } for serial in ("cam_a", "cam_b")}
    extr = {serial: np.eye(4)[:3].tolist() for serial in intr}
    (camera_dir / "intrinsics.json").write_text(json.dumps(intr), encoding="utf-8")
    (camera_dir / "extrinsics.json").write_text(json.dumps(extr), encoding="utf-8")
    c2r = tmp_path / "C2R.npy"
    np.save(c2r, np.eye(4))
    base = tmp_path / "base_scene.json"
    base.write_text(json.dumps({"mesh": {}, "cuboid": {}}), encoding="utf-8")
    timing = {}
    for serial in intr:
        path = tmp_path / f"{serial}_time.json"
        path.write_text("{}", encoding="utf-8")
        timing[serial] = str(path)
    record = {
        "schema": "precision_insertion_live_session_launch_v1",
        "mode": {"family": "square", "gap_mm": 1.5},
        "shared_root": str(shared), "capture_root": str(capture),
        "evidence_dir": str(tmp_path / "new_session"),
        "pc_list": ["capture1"], "capture_ips": ["192.0.2.1"],
        "camera_ids": ["cam_a", "cam_b"],
        "calib_dir": str(camera_dir),
        "camera_time_calibrations": timing,
        "c2r_npy": str(c2r), "base_scene_json": str(base),
        "view_limits": {
            "minimum_mask_pixels": 50,
            "minimum_foundpose_quality": .5,
            "minimum_foundpose_inliers": 10,
            "minimum_border_clearance_px": 2,
            "maximum_capture_skew_s": .02,
        },
        "socket_prompt": "fixed white socket",
        "socket_capture_count": 2,
        "board_timeout_s": 5., "socket_timeout_s": 15.,
        "image_write_timeout_s": 5.,
        "max_socket_translation_mm": 1.,
        "max_socket_angle_deg": 1.,
        "minimum_socket_refined_iou": .8,
    }
    config = tmp_path / "launch.json"
    config.write_text(json.dumps(record), encoding="utf-8")
    return config, record


def _stub_time(monkeypatch):
    monkeypatch.setattr(launch.CameraTimeCalibration, "load", classmethod(
        lambda _cls, path: SimpleNamespace(
            calibration_id=Path(path).stem, camera_serial=Path(path).stem[:5],
            valid_until_utc_s=time.time() + 3600)))


def test_check_only_loads_stock_calibration_and_does_not_connect(
        tmp_path, monkeypatch, capsys):
    config, _record = _config(tmp_path)
    _stub_time(monkeypatch)
    monkeypatch.setattr(launch, "capture_session", lambda _config: pytest.fail(
        "check-only must not contact cameras"))
    assert run_pipeline.main(["start-session", "--config", str(config)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["camera_count"] == 2
    assert output["robot_ready"] is False
    loaded = launch.load_session_launch(config)
    assert loaded.image_hw == (480, 640)
    assert set(loaded.intrinsics_full) == {"cam_a", "cam_b"}
    np.testing.assert_allclose(loaded.extrinsics_full["cam_a"], np.eye(4))


def test_static_preflight_launch_cannot_open_camera_capture(tmp_path, monkeypatch):
    config, record = _config(tmp_path)
    _stub_time(monkeypatch)
    record["execution_scope"] = "static_preflight_only"
    config.write_text(json.dumps(record), encoding="utf-8")
    loaded = launch.load_session_launch(config)
    assert loaded.readiness()["execution_scope"] == "static_preflight_only"
    with pytest.raises(PermissionError, match="static preflight"):
        launch.capture_session(loaded)
    with pytest.raises(PermissionError, match="static preflight"):
        with launch.open_session_cameras(loaded):
            pytest.fail("static launch opened cameras")
    diagnostic = replace(launch.load_session_launch(
        config, evidence_dir_override=tmp_path / "new_camera_qa"),
        execution_scope="camera_only_diagnostic")
    assert diagnostic.evidence_dir == tmp_path / "new_camera_qa"
    assert diagnostic.readiness()["execution_scope"] == "camera_only_diagnostic"
    with pytest.raises(PermissionError, match="static preflight"):
        launch.capture_session(diagnostic)


def test_missing_one_camera_time_or_existing_output_stops_before_capture(
        tmp_path, monkeypatch):
    config, record = _config(tmp_path)
    _stub_time(monkeypatch)
    record["camera_time_calibrations"].pop("cam_b")
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="every active camera"):
        launch.load_session_launch(config)
    record["camera_time_calibrations"]["cam_b"] = str(
        tmp_path / "cam_b_time.json")
    Path(record["evidence_dir"]).mkdir()
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        launch.load_session_launch(config)


def test_bounded_request_launch_needs_no_camera_time_files(tmp_path):
    config, record = _config(tmp_path)
    record.pop("camera_time_calibrations")
    record["camera_time_mode"] = "bounded_request_interval"
    config.write_text(json.dumps(record), encoding="utf-8")
    loaded = launch.load_session_launch(config)
    assert loaded.camera_times == {}
    assert loaded.readiness()["camera_time_mode"] == (
        "bounded_request_interval")


def test_stock_socket_perception_mode_is_explicit_in_launch(tmp_path):
    config, record = _config(tmp_path)
    record.pop("camera_time_calibrations")
    record["camera_time_mode"] = "bounded_request_interval"
    record["socket_perception_mode"] = "ignore_sil_loss"
    config.write_text(json.dumps(record), encoding="utf-8")
    loaded = launch.load_session_launch(config)
    assert loaded.socket_perception_mode == "ignore_sil_loss"
    assert loaded.readiness()["socket_perception_mode"] == "ignore_sil_loss"
    record["socket_perception_mode"] = "quality"
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="socket perception mode"):
        launch.load_session_launch(config)


def test_board_9_partial_corner_contract_is_explicit(tmp_path):
    config, record = _config(tmp_path)
    record["board_id"] = "9"
    record["minimum_board_corners"] = 36
    record["board_capture_phase"] = "socket_present_partial"
    record["camera_time_mode"] = "bounded_request_interval"
    record.pop("camera_time_calibrations")
    config.write_text(json.dumps(record), encoding="utf-8")
    loaded = launch.load_session_launch(config)
    assert loaded.readiness()["board_id"] == "9"
    assert loaded.minimum_board_corners == 36
    record["board_id"] = "11"
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum_board_corners"):
        launch.load_session_launch(config)


def test_placeholder_capture_address_fails_file_preflight(tmp_path, monkeypatch):
    config, record = _config(tmp_path)
    _stub_time(monkeypatch)
    record["capture_ips"] = ["REPLACE_WITH_CAPTURE_PC_IP"]
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="literal IP"):
        launch.load_session_launch(config)


def test_capture_closes_all_camera_connections_even_on_failure(
        tmp_path, monkeypatch):
    config, _record = _config(tmp_path)
    _stub_time(monkeypatch)
    loaded = launch.load_session_launch(config)
    events = []

    class Snapshot:
        def __init__(self, *_args):
            events.append("snapshot_open")

        def close(self):
            events.append("snapshot_close")

    class Init:
        def __init__(self, *_args):
            events.append("init_open")

        def close(self):
            events.append("init_close")

    class Tap:
        def __init__(self, *_args):
            events.append("tap_open")
            self.buffer = object()

        def close(self):
            events.append("tap_close")

    monkeypatch.setattr(
        "autodex.perception.snapshot_orchestrator.SnapshotOrchestrator",
        Snapshot)
    monkeypatch.setattr(
        "autodex.perception.init_orchestrator.InitOrchestrator", Init)
    monkeypatch.setattr(launch, "SnapshotMetadataTap", Tap)
    monkeypatch.setattr(launch, "PrecisionInitOrchestrator",
                        lambda stock, _registry, **_kwargs: stock)
    monkeypatch.setattr(launch, "AcquisitionTimeProvider",
                        lambda _registry, _times, **_kwargs: object())
    monkeypatch.setattr(launch, "start_precision_session",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("camera capture failed")))
    with pytest.raises(RuntimeError, match="camera capture failed"):
        launch.capture_session(loaded)
    assert events == ["snapshot_open", "tap_open", "init_open",
                      "init_close", "tap_close", "snapshot_close"]


def test_open_camera_context_survives_socket_start_until_trial_is_done(
        tmp_path, monkeypatch):
    config, _record = _config(tmp_path)
    _stub_time(monkeypatch)
    loaded = launch.load_session_launch(config)
    events = []

    class Snapshot:
        def __init__(self, *_args):
            events.append("snapshot_open")

        def close(self):
            events.append("snapshot_close")

    class Init:
        def __init__(self, *_args, **kwargs):
            assert kwargs == {"port_mask": 15006, "port_pose": 15007,
                              "port_cmd": 16893}
            events.append("init_open")

        def close(self):
            events.append("init_close")

    class Tap:
        def __init__(self, *_args):
            events.append("tap_open")
            self.buffer = object()

        def close(self):
            events.append("tap_close")

    monkeypatch.setattr(
        "autodex.perception.snapshot_orchestrator.SnapshotOrchestrator",
        Snapshot)
    monkeypatch.setattr(
        "autodex.perception.init_orchestrator.InitOrchestrator", Init)
    monkeypatch.setattr(launch, "SnapshotMetadataTap", Tap)
    monkeypatch.setattr(launch, "PrecisionInitOrchestrator",
                        lambda stock, _registry, **_kwargs: stock)
    monkeypatch.setattr(launch, "AcquisitionTimeProvider",
                        lambda _registry, _times, **_kwargs: object())

    def start(**kwargs):
        events.append("socket_frozen")
        assert isinstance(kwargs["init_orchestrator"], Init)
        return object()

    monkeypatch.setattr(launch, "start_precision_session", start)
    with launch.open_session_cameras(
            loaded, init_ports=(15006, 15007, 16893)) as cameras:
        assert cameras.start() is not None
        assert isinstance(cameras.init_orchestrator, Init)
        assert events == ["snapshot_open", "tap_open", "init_open",
                          "socket_frozen"]
    assert events == ["snapshot_open", "tap_open", "init_open",
                      "socket_frozen", "init_close", "tap_close",
                      "snapshot_close"]


def test_alternate_init_ports_validate_before_camera_connection(tmp_path, monkeypatch):
    config, _record = _config(tmp_path)
    _stub_time(monkeypatch)
    loaded = launch.load_session_launch(config)
    for ports in ((15006, 15006, 16893), (15006, 15007, 65536),
                  (15006, 15007)):
        with pytest.raises(ValueError, match="distinct TCP port"):
            with launch.open_session_cameras(loaded, init_ports=ports):
                pytest.fail("invalid ports opened a camera")
