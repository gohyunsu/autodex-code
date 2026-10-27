"""The capture4 diagnostic must follow the live perception path without motion."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_zerodex_camera_only as camera_only  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def test_camera_only_reuses_socket_key_and_catalog_without_robot(tmp_path, monkeypatch):
    events = []
    mode = select_mode("cylinder", 20)
    shared = tmp_path / "shared"
    shared.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    evidence = tmp_path / "run" / "socket_evidence"
    calibration = SimpleNamespace()
    launch = SimpleNamespace(
        mode=mode, socket_perception_mode="board_constrained_body",
        pc_list=("capture4",), capture_ips=("192.0.2.1",),
        shared_root=shared, capture_root=capture,
        camera_acquisition_mode="stream", camera_ids={"a", "b"},
        intrinsics_full={"a": {}, "b": {}},
        extrinsics_full={"a": {}, "b": {}},
        image_hw=(1536, 2048), view_limits=SimpleNamespace(),
    )

    def load(_path, *, evidence_dir_override):
        assert evidence_dir_override == evidence
        events.append("launch")
        return launch

    def catalog(**_kwargs):
        events.append("catalog")
        return {"candidates": []}

    cameras = SimpleNamespace(
        init_orchestrator=object(),
        acquisition_metadata_for_request=lambda _id: {},
        start=lambda: (
            events.append("socket") or SimpleNamespace(
                evidence_dir=evidence, producer_manifest={},
                calibration=calibration)),
    )

    @contextmanager
    def open_cameras(_launch, *, init_ports):
        assert init_ports == (15006, 15007, 16893)
        events.append("camera_open")
        yield cameras
        events.append("camera_close")

    @contextmanager
    def open_stream(_pcs, **kwargs):
        assert kwargs["acquisition_mode"] == "stream"
        events.append("stream_open")
        yield object()
        events.append("stream_close")

    def capture_key(**kwargs):
        events.append("key")
        assert kwargs["silhouette_iterations"] == 100
        assert kwargs["silhouette_loss_threshold"] == 0.003
        assert "max_key_state_skew_s" not in kwargs
        assert kwargs["pose_only"] is False
        return SimpleNamespace(
            pose_world=np.eye(4), request_id=42, selected_camera_id="a")

    monkeypatch.setattr(camera_only, "load_session_launch", load)
    monkeypatch.setattr(camera_only, "preflight_endpoint_catalog_file", catalog)
    monkeypatch.setattr(camera_only, "open_session_cameras", open_cameras)
    monkeypatch.setattr(camera_only, "open_autodex_camera_stream", open_stream)
    monkeypatch.setattr(camera_only, "prepare_precision_silhouette",
                        lambda **_kwargs: events.append("prewarm"))
    monkeypatch.setattr(camera_only, "open_verified_session",
                        lambda **_kwargs: (
                            events.append("ledger") or SimpleNamespace(runner=object())))
    monkeypatch.setattr(camera_only, "_capture_admitted_key", capture_key)
    monkeypatch.setattr(camera_only, "build_trial_scene_from_session",
                        lambda **_kwargs: (
                            events.append("scene") or {
                                "mesh": {"target": {"pose": [0, 0, 0, 1, 0, 0, 0]}}}))
    monkeypatch.setattr(camera_only, "classify_key_tabletop_pose",
                        lambda **_kwargs: (
                            events.append("classify") or {"stem": "001"}))
    monkeypatch.setattr(camera_only, "select_pose_candidates",
                        lambda *_args, **_kwargs: (
                            events.append("select") or {
                                "status": "candidates_available",
                                "candidates": [{"key": ["table", "0", "1"]}]}))
    monkeypatch.setattr(camera_only, "_key_capture", lambda: {
        "max_key_state_skew_s": 5., "capture_timeout_s": 120.,
        "image_write_timeout_s": 90.,
    })
    monkeypatch.setattr(camera_only, "profile_run", lambda _path: nullcontext())
    monkeypatch.setattr(camera_only, "timed",
                        lambda *_args, **_kwargs: nullcontext())

    report = camera_only.run_camera_only(
        launch_config=tmp_path / "launch.json",
        endpoint_catalog=tmp_path / "catalog.json",
        output_dir=tmp_path / "run",
        wait_for_key=lambda: events.append("key_placed"),
    )
    assert events == [
        "launch", "catalog", "camera_open", "prewarm", "stream_open",
        "socket", "ledger", "key_placed", "key", "stream_close",
        "camera_close", "scene", "classify", "select",
    ]
    assert report["endpoint_eligible_candidate_count"] == 1
    assert report["robot_ready"] is False
    assert (tmp_path / "run" / "report.json").is_file()
