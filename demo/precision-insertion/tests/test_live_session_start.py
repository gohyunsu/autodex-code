"""The live session sequence must bind board, socket and one frozen world."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion import live_session_start as start  # noqa: E402


def _setup(tmp_path, monkeypatch, request_ids=(11, 12, 13)):
    shared = tmp_path / "shared"
    mode = select_mode("cylinder", 20)
    object_dir = shared / "object_processing" / mode.socket_object
    mesh = object_dir / "raw_mesh" / f"{mode.socket_object}.obj"
    collision = object_dir / "processed_data/mesh/static_collision.obj"
    repre = (shared / "AutoDex/foundpose_assets" / mode.socket_object /
             "object_repre/v1" / mode.socket_object / "1/repre.pth")
    for path in (mesh, collision, repre):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test asset")
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    events = []

    class Init:
        obj_name = None

        def init_object(self, **kwargs):
            events.append("socket_init")
            assert kwargs["obj_name"] == mode.socket_object
            assert kwargs["mesh_path"] == str(mesh)
            assert kwargs["assets_root"] == str(repre.parents[4])
            assert kwargs["load_silhouette"] is False
            self.obj_name = kwargs["obj_name"]

    init = Init()

    def board(**kwargs):
        events.append("board")
        assert init.obj_name is None
        return SimpleNamespace(
            request_id=kwargs["request_id_factory"](),
            images_bgr={"cam_a": np.zeros((8, 8, 3), dtype=np.uint8),
                        "cam_b": np.zeros((8, 8, 3), dtype=np.uint8)},
            frame_timestamps_s={"cam_a": 100.0, "cam_b": 100.001},
            frame_timestamp_source="camera_acquisition",
            frame_evidence={"cam_a": {}, "cam_b": {}},
        )

    def socket(**kwargs):
        events.append(kwargs["capture_id"])
        assert init.obj_name == mode.socket_object
        assert kwargs["capture_root"] == capture_root
        return SimpleNamespace(request_id=kwargs["request_id_factory"]())

    frozen = SimpleNamespace(record={"schema": "test"})

    def bootstrap(**kwargs):
        events.append("freeze_world")
        assert kwargs["board_request_id"] == 11
        assert [row.request_id for row in kwargs["socket_captures"]] == [12, 13]
        assert kwargs["socket_collision_mesh"] == collision
        assert kwargs["object_root"] == shared / "object_processing"
        return SimpleNamespace(calibration=frozen)

    def write(_bootstrap, output):
        events.append("save")
        output.mkdir()
        return output

    def verify(_output):
        events.append("verify")
        return {"all_frames_bound_to_acquisition_evidence": True}

    def reload(_path, **kwargs):
        events.append("reload_frozen_world")
        assert kwargs["mode"] == mode
        return frozen

    for name, value in (("collect_board_snapshot", board),
                        ("collect_socket_capture", socket),
                        ("bootstrap_session", bootstrap),
                        ("write_session_bootstrap_artifacts", write),
                        ("verify_session_evidence_bundle", verify),
                        ("load_session_calibration", reload)):
        monkeypatch.setattr(start, name, value)
    identifiers = iter(request_ids)
    arguments = {
        "mode": mode, "shared_root": shared,
        "snapshot_orchestrator": object(), "init_orchestrator": init,
        "acquisition_metadata_for_request": lambda _request: {},
        "capture_root": capture_root, "evidence_dir": tmp_path / "session",
        "calibrated_camera_ids": {"cam_a", "cam_b"},
        "intrinsics_full": {"cam_a": {}, "cam_b": {}},
        "extrinsics_full": {"cam_a": {}, "cam_b": {}},
        "image_hw": (1536, 2048), "c2r": np.eye(4),
        "base_scene": {"mesh": {}, "cuboid": {}},
        "view_limits": SocketViewLimits(50, .5, 10, 2, .02),
        "socket_prompt": "fixed red cylindrical socket", "socket_capture_count": 2,
        "board_timeout_s": 5., "socket_timeout_s": 15.,
        "max_socket_translation_mm": 1., "max_socket_angle_deg": 1.,
        "request_id_factory": lambda: next(identifiers),
    }
    return arguments, events, repre


def test_session_start_orders_board_before_socket_and_reloads_frozen_world(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    result = start.start_precision_session(**arguments)
    assert events == ["board", "socket_init", "socket_000", "socket_001",
                      "freeze_world", "save", "verify", "reload_frozen_world"]
    assert result.board_request_id == 11
    assert result.socket_request_ids == (12, 13)
    assert result.evidence_dir == arguments["evidence_dir"]
    assert result.calibration.record["schema"] == "test"


def test_missing_canonical_socket_repre_rejects_before_any_capture(
        tmp_path, monkeypatch):
    arguments, events, repre = _setup(tmp_path, monkeypatch)
    repre.unlink()
    with pytest.raises(FileNotFoundError, match="FoundPose representation"):
        start.start_precision_session(**arguments)
    assert events == []


def test_duplicate_request_or_existing_output_never_reuses_a_session(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(
        tmp_path, monkeypatch, request_ids=(11, 12, 12))
    with pytest.raises(ValueError, match="unique positive int31"):
        start.start_precision_session(**arguments)
    assert events == ["board", "socket_init", "socket_000", "socket_001"]
    events.clear()
    arguments["evidence_dir"].mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        start.start_precision_session(**arguments)
    assert events == []


def test_session_rejects_relative_capture_root_and_bad_camera_set(
        tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)
    arguments["capture_root"] = Path("captures")
    with pytest.raises(ValueError, match="absolute shared directory"):
        start.start_precision_session(**arguments)
    arguments["capture_root"] = tmp_path / "captures"
    arguments["calibrated_camera_ids"] = {"cam_a", "cam_b", "cam_c"}
    with pytest.raises(ValueError, match="active camera IDs"):
        start.start_precision_session(**arguments)
    assert events == []


def test_session_rejects_unbound_saved_frame_evidence(tmp_path, monkeypatch):
    arguments, events, _repre = _setup(tmp_path, monkeypatch)

    def unbound(_output):
        events.append("verify")
        return {"all_frames_bound_to_acquisition_evidence": False}

    monkeypatch.setattr(start, "verify_session_evidence_bundle", unbound)
    with pytest.raises(ValueError, match="acquisition-bound camera frames"):
        start.start_precision_session(**arguments)
    assert events == ["board", "socket_init", "socket_000", "socket_001",
                      "freeze_world", "save", "verify"]
    assert arguments["evidence_dir"].is_dir()
