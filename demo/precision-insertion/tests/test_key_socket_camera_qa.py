"""Camera-only key QA saves real frame/mask bindings without robot state."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import key_socket_camera_qa as qa  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.live_capture import KeyCaptureInput  # noqa: E402


def _capture(tmp_path):
    capture_dir = tmp_path / "capture"
    images_dir = capture_dir / "images"
    images_dir.mkdir(parents=True)
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=bool)
    mask[3:8, 5:11] = True
    views = {}
    masks = {}
    poses = {}
    evidence = {}
    for camera, ok in (("front", True), ("side", False)):
        assert cv2.imwrite(str(images_dir / f"{camera}.png"), image)
        views[camera] = image.copy()
        masks[camera] = {
            "mask": mask.copy() if ok else None,
            "frame_id": 17, "image_sha256": image_sha256(image)}
        poses[camera] = {
            "ok": ok, "frame_id": 17,
            "image_sha256": image_sha256(image),
            **({"pose_world": np.eye(4)} if ok else {})}
        evidence[camera] = {"frame_id": 17,
                            "image_sha256": image_sha256(image),
                            "timestamp_s": 100.0}
    capture = KeyCaptureInput(
        "key_socket_001", 51, qa.DEFAULT_KEY_PROMPT,
        views, masks, poses, {camera: 100. for camera in views},
        "camera_acquisition", evidence, capture_dir)
    session = tmp_path / "session"
    session.mkdir()
    (session / "session_calibration.json").write_text(
        "{}", encoding="utf-8")
    return capture, session


def test_camera_qa_keeps_failed_pose_and_zero_mask_view(tmp_path):
    capture, session = _capture(tmp_path)
    path = qa.write_camera_only_key_qa(
        capture=capture, evidence_dir=session,
        session_calibration_path=session / "session_calibration.json",
        expected_camera_ids={"front", "side", "top"})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["camera_count"] == 2
    assert record["missing_camera_ids"] == ["top"]
    assert record["robot_ready"] is False
    assert record["views"]["front"]["sam"]["mask_pixels"] == 30
    assert record["views"]["side"]["sam"]["mask_pixels"] == 0
    assert record["views"]["side"]["foundpose"]["ok"] is False
    image_ref = record["views"]["front"]["raw_image"]
    assert hashlib.sha256(Path(image_ref["path"]).read_bytes()).hexdigest() == (
        image_ref["sha256"])
    assert qa.verify_camera_only_key_qa(path)["capture_id"] == "key_socket_001"
    mask_path = Path(record["views"]["side"]["sam"]["mask_png"]["path"])
    mask_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pixels changed"):
        qa.verify_camera_only_key_qa(path)
    with pytest.raises(FileExistsError):
        qa.write_camera_only_key_qa(
            capture=capture, evidence_dir=session,
            session_calibration_path=session / "session_calibration.json",
            expected_camera_ids={"front", "side", "top"})


def test_camera_qa_waits_for_key_after_socket_and_closes_transport(
        tmp_path, monkeypatch):
    capture, session = _capture(tmp_path)
    events = []
    launch = SimpleNamespace(
        execution_scope="live", mode=SimpleNamespace(family="cylinder",
                                                     key_object="white_key"),
        shared_root=tmp_path, capture_root=tmp_path,
        camera_ids={"front", "side"},
        intrinsics_full={}, extrinsics_full={}, image_hw=(16, 20),
        socket_timeout_s=5., image_write_timeout_s=5.)
    monkeypatch.setattr(qa, "SessionLaunch", SimpleNamespace)

    class Init:
        def init_object(self, **kwargs):
            events.append("init_key")

    init = Init()

    @contextmanager
    def cameras(_launch, *, init_ports):
        assert init_ports == (15006, 15007, 16893)
        events.append("open_cameras")
        try:
            yield SimpleNamespace(
                init_orchestrator=init,
                acquisition_metadata_for_request=lambda _: {},
                start=lambda **_: events.append("socket_frozen") or
                SimpleNamespace(evidence_dir=session))
        finally:
            events.append("close_cameras")

    monkeypatch.setattr(qa, "open_session_cameras", cameras)
    monkeypatch.setattr(qa, "AssetPaths", lambda *_: SimpleNamespace(
        raw_mesh=lambda _: tmp_path / "key.obj",
        foundpose_repre=lambda _: tmp_path / "repre.pth",
        foundpose_assets_root=lambda _: tmp_path))
    (tmp_path / "key.obj").write_text("# key", encoding="utf-8")
    (tmp_path / "repre.pth").write_bytes(b"model")
    monkeypatch.setattr(qa, "prepare_key_silhouette",
                        lambda **_: events.append("silhouette"))
    monkeypatch.setattr(qa, "collect_key_capture",
                        lambda **_: events.append("capture_key") or capture)
    monkeypatch.setattr(qa, "write_camera_only_key_qa",
                        lambda **_: events.append("save_qa") or
                        session / "report.json")
    result = qa.run_camera_only_key_socket_qa(
        launch=launch,
        init_ports=(15006, 15007, 16893),
        confirm_key_placement=lambda: events.append("place_key") or True)
    assert result == session / "report.json"
    assert events == ["open_cameras", "socket_frozen", "place_key",
                      "silhouette", "init_key", "capture_key", "save_qa",
                      "close_cameras"]
