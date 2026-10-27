"""A long-lived camera session must close transports and never start a robot."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_orchestration as orchestration  # noqa: E402
from precision_insertion.clear_view import ClearViewForPerception  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from autodex.utils.robot_config import FR3_CLEAR_VIEW  # noqa: E402


def _clear_view(events):
    class Executor:
        arm_dof = 7
        hand_name = "inspire"

        def home(self, *, clear_view):
            assert clear_view
            events.append("home_clear_view")

    def read_state():
        now = time.time()
        raw = np.full(6, 1000., dtype=float)
        return LiveRobotState(
            np.r_[FR3_CLEAR_VIEW, np.zeros(6)], np.zeros(7), now, now,
            10., now, raw, raw.copy(), 0., np.zeros(6))

    return ClearViewForPerception(
        Executor(), read_state, .1, 20., .05, 1.)


def _patch_camera_lifetime(monkeypatch, events):
    launch = SimpleNamespace(
        mode=SimpleNamespace(family="cylinder",
                             socket_object="precision_socket_cylinder_gap_20mm"),
        shared_root=Path("/shared"),
        pc_list=("capture1", "capture2"))
    monkeypatch.setattr(orchestration, "load_session_launch",
                        lambda _path: launch)
    monkeypatch.setattr(orchestration, "preflight_endpoint_catalog_file",
                        lambda **_kwargs: events.append("catalog_preflight"))

    @contextmanager
    def stream(pc_list, *, name, acquisition_mode):
        assert pc_list == launch.pc_list
        assert name == "precision_insertion_live"
        assert acquisition_mode == "acquire"
        events.append("stream_open")
        try:
            yield object()
        finally:
            events.append("stream_close")

    monkeypatch.setattr(orchestration, "open_autodex_camera_stream", stream)

    def warm_socket(*, init_orchestrator, object_name, raw_mesh):
        assert init_orchestrator == "same_init"
        assert object_name == launch.mode.socket_object
        assert raw_mesh == (Path("/shared/object_processing") / object_name /
                            "raw_mesh" / f"{object_name}.obj")
        events.append("socket_renderer_warmup")

    monkeypatch.setattr(orchestration, "prepare_precision_silhouette",
                        warm_socket)

    @contextmanager
    def cameras(received, *, init_ports=None):
        assert received is launch
        if init_ports is not None:
            events.append(("init_ports", init_ports))
        events.append("camera_open")
        try:
            yield SimpleNamespace(
                init_orchestrator="same_init",
                acquisition_metadata_for_request="same_provider",
                start=lambda: (events.append("socket_frozen") or
                               SimpleNamespace(evidence_dir=Path("/evidence"))))
        finally:
            events.append("camera_close")

    monkeypatch.setattr(orchestration, "open_session_cameras", cameras)
    return launch


def test_live_session_keeps_same_camera_transport_for_next_key(
        tmp_path, monkeypatch):
    events = []
    launch = _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    opened = SimpleNamespace(runner="same_runner")

    def open_ledger(**kwargs):
        events.append("ledger_open")
        assert kwargs == {
            "mode": launch.mode, "shared_root": Path("/shared"),
            "evidence_dir": Path("/evidence"), "catalog_path": catalog,
            "output_dir": tmp_path / "new_run", "max_xy_retries": 2,
            "xy_update_weight": 1.0, "catalog_snapshot": None,
            "producer_manifest": None, "calibration_snapshot": None,
            "initial_preferred_candidate": None,
            "upright_cylinder_only": True,
        }
        return opened

    monkeypatch.setattr(orchestration, "open_verified_session", open_ledger)
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=2, clear_view=_clear_view(events)) as live:
        assert events == ["catalog_preflight", "home_clear_view", "camera_open",
                          "socket_renderer_warmup", "stream_open",
                          "socket_frozen", "ledger_open"]
        assert live.opened is opened
        assert live.cameras.init_orchestrator == "same_init"
        assert live.cameras.acquisition_metadata_for_request == "same_provider"
    assert events[-2:] == ["stream_close", "camera_close"]


def test_board_constrained_socket_prewarms_key_not_unused_socket_renderer(
        tmp_path, monkeypatch):
    events = []
    launch = _patch_camera_lifetime(monkeypatch, events)
    launch.mode.key_object = "precision_key_cylinder_r15_h80"
    launch.socket_perception_mode = "board_constrained_body"
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def warm_key(*, init_orchestrator, object_name, raw_mesh):
        assert init_orchestrator == "same_init"
        assert object_name == launch.mode.key_object
        assert raw_mesh == (Path("/shared/object_processing") / object_name /
                            "raw_mesh" / f"{object_name}.obj")
        events.append("key_renderer_warmup")

    monkeypatch.setattr(orchestration, "prepare_precision_silhouette", warm_key)
    monkeypatch.setattr(
        orchestration, "open_verified_session",
        lambda **_kwargs: SimpleNamespace(runner="same_runner"))
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=None, clear_view=_clear_view(events)):
        pass
    assert events.index("key_renderer_warmup") < events.index("stream_open")
    assert "socket_renderer_warmup" not in events


def test_live_session_reuses_already_validated_launch(
        tmp_path, monkeypatch):
    events = []
    launch = _patch_camera_lifetime(monkeypatch, events)
    monkeypatch.setattr(
        orchestration, "load_session_launch",
        lambda _path: pytest.fail("live launch was parsed twice"))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        orchestration, "open_verified_session",
        lambda **_kwargs: SimpleNamespace(runner="same_runner"))
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog,
            run_output_dir=tmp_path / "new_run", max_xy_retries=2,
            clear_view=_clear_view(events), launch_snapshot=launch):
        pass
    assert events.count("catalog_preflight") == 1


@pytest.mark.parametrize("board_free", (False, True))
def test_reused_socket_evidence_skips_live_board_and_socket_capture(
        tmp_path, monkeypatch, board_free):
    events = []
    launch = _patch_camera_lifetime(monkeypatch, events)
    launch.mode.key_object = "precision_key_cylinder_r15_h80"
    launch.camera_ids = ("cam_a",)
    launch.intrinsics_full = {"cam_a": {"K_undist": np.eye(3).tolist()}}
    launch.extrinsics_full = {"cam_a": np.eye(4)}
    launch.c2r = np.eye(4)
    launch.scene_mode = "board_free" if board_free else "charuco_board"
    launch.board_id = None if board_free else "9"
    launch.table_surface_z_m = .04 if board_free else None
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    saved = tmp_path / "saved_socket"
    saved.mkdir()
    manifest = {"socket_request_ids": [12],
                "socket_capture_ids": ["socket_000"]}
    if not board_free:
        manifest["board_request_id"] = 11
    calibration = SimpleNamespace(record=(
        {"scene_mode": "board_free", "support_surface": {
            "source": "configured_table_surface", "table_surface_z_m": .04}}
        if board_free else {"board": {"board": "9"}}))
    monkeypatch.setattr(orchestration, "verify_session_evidence_bundle",
                        lambda path: events.append("saved_evidence_verified") or manifest)
    monkeypatch.setattr(orchestration, "load_session_calibration",
                        lambda *args, **kwargs:
                        events.append("saved_calibration_loaded") or calibration)
    monkeypatch.setattr(orchestration, "validate_session_camera_calibration",
                        lambda *args, **kwargs:
                        events.append("saved_camera_calibration_matched"))
    monkeypatch.setattr(orchestration, "prepare_precision_silhouette",
                        lambda **kwargs: events.append("key_renderer_warmup"))
    monkeypatch.setattr(orchestration, "open_verified_session",
                        lambda **kwargs: events.append("ledger_open") or
                        SimpleNamespace(runner="same_runner"))
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=None, clear_view=_clear_view(events),
            reuse_session_evidence_dir=saved) as live:
        assert live.started.evidence_dir == saved
        assert live.started.socket_request_ids == (12,)
        assert live.started.board_request_id == (None if board_free else 11)
    assert events.index("saved_evidence_verified") < events.index("home_clear_view")
    assert "key_renderer_warmup" in events
    assert "socket_frozen" not in events


def test_live_session_routes_precision_init_ports_before_socket_capture(
        tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(orchestration, "open_verified_session", lambda **_kwargs:
                        SimpleNamespace(runner="same_runner"))
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=None, clear_view=_clear_view(events),
            init_ports=(15006, 15007, 16893)):
        pass
    assert events.index(("init_ports", (15006, 15007, 16893))) < events.index(
        "socket_frozen")


@pytest.mark.parametrize("reason", ("missing_catalog", "existing_run",
                                    "negative_retries"))
def test_live_session_static_failures_do_not_connect(
        tmp_path, monkeypatch, reason):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "new_run"
    retries = 1
    if reason != "missing_catalog":
        catalog.write_text("{}", encoding="utf-8")
    if reason == "existing_run":
        output.mkdir()
    if reason == "negative_retries":
        retries = -1
    with pytest.raises((FileNotFoundError, FileExistsError, ValueError)):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog, run_output_dir=output,
                max_xy_retries=retries, clear_view=_clear_view(events)):
            pytest.fail("static failure must not yield a live session")
    assert events == []


def test_failed_session_catalog_binding_closes_camera_but_keeps_capture(
        tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def reject(**_kwargs):
        events.append("catalog_rejected")
        raise ValueError("catalog differs from socket session")

    monkeypatch.setattr(orchestration, "open_verified_session", reject)
    with pytest.raises(ValueError, match="catalog differs"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog, run_output_dir=tmp_path / "run",
                max_xy_retries=1, clear_view=_clear_view(events)):
            pytest.fail("catalog rejection must not yield")
    assert events == ["catalog_preflight", "home_clear_view", "camera_open",
                      "socket_renderer_warmup", "stream_open", "socket_frozen",
                      "catalog_rejected", "stream_close", "camera_close"]


def test_stale_catalog_preflight_does_not_connect_cameras(tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def stale(**kwargs):
        assert kwargs == {
            "mode": SimpleNamespace(
                family="cylinder",
                socket_object="precision_socket_cylinder_gap_20mm"),
            "shared_root": Path("/shared"),
            "catalog_path": catalog,
        }
        raise ValueError("v8 catalog is stale for pose 000")

    monkeypatch.setattr(orchestration, "preflight_endpoint_catalog_file", stale)
    with pytest.raises(ValueError, match="stale for pose 000"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog,
                run_output_dir=tmp_path / "run",
                max_xy_retries=1, clear_view=_clear_view(events)):
            pytest.fail("stale catalogue must not yield a live session")
    assert events == []


def test_initial_clear_view_mismatch_prevents_camera_capture(tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    guard = _clear_view(events)
    guard.read_state = lambda: None
    with pytest.raises(TypeError, match="measured LiveRobotState"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog,
                run_output_dir=tmp_path / "run", max_xy_retries=1,
                clear_view=guard):
            pytest.fail("invalid clear-view feedback must prevent capture")
    assert events == ["catalog_preflight", "home_clear_view"]
