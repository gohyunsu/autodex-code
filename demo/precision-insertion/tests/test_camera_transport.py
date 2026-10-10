"""Precision camera metadata survives transport without stock-file edits."""

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_transport import (  # noqa: E402
    FrameIdentityRegistry, PrecisionInitOrchestrator, ProvenancePublisher,
    ProvenanceSnapshotAdapter, RecordingReader,
    SnapshotMetadataBuffer, decode_precision_mask, decode_precision_pose,
    install_init_provenance,
)
from precision_insertion.frame_provenance import image_sha256  # noqa: E402


class _Reader:
    camera_names = ["cam"]

    def __init__(self, image, fid=11):
        self.image = image
        self.fid = fid

    def wait_for_new_frames(self, **_kwargs):
        return {"cam": (self.image.copy(), self.fid)}


class _Publisher:
    def __init__(self):
        self.published = []

    def send_data(self, metadata, blobs):
        self.published.append((metadata, blobs))


def _owner(image):
    h, w = image.shape[:2]
    mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32),
                            np.arange(h, dtype=np.float32))
    return SimpleNamespace(
        pub_mask=_Publisher(), pub_pose=_Publisher(),
        undistort_maps={"cam": (mapx, mapy)})


def test_recording_reader_binds_exact_undistorted_pixels_and_monotonic_fid():
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    owner = _owner(image)
    install_init_provenance(owner)
    owner._precision_active_request_id = 42
    reader = RecordingReader(_Reader(image), owner)
    frames = reader.wait_for_new_frames()
    assert frames["cam"][1] == 11
    assert owner._precision_frames[(42, "cam")]["image_sha256"] == (
        image_sha256(image))
    owner.pub_mask.send_data([{"req_id": 42, "serial": "cam"}], [b"png"])
    owner.pub_pose.send_data([{"req_id": 42, "serial": "cam"}], [b"pose"])
    mask_meta = owner.pub_mask._publisher.published[0][0][0]
    pose_meta = owner.pub_pose._publisher.published[0][0][0]
    assert mask_meta["fid"] == pose_meta["fid"] == 11
    assert mask_meta["image_sha256"] == pose_meta["image_sha256"]
    with pytest.raises(ValueError, match="did not advance"):
        reader.wait_for_new_frames()


def test_publisher_rejects_unbound_or_preexisting_precision_metadata():
    owner = SimpleNamespace(_precision_frames={(7, "cam"): {
        "frame_id": 3, "image_sha256": "a" * 64,
        "image_space": "autodex_undistorted_full_frame"}})
    pub = ProvenancePublisher(_Publisher(), owner)
    with pytest.raises(ValueError, match="no bound source"):
        pub.send_data([{"req_id": 8, "serial": "cam"}], [b"x"])
    with pytest.raises(ValueError, match="unexpectedly set"):
        pub.send_data([{"req_id": 7, "serial": "cam", "fid": 9}], [b"x"])


def test_robot_side_mask_pose_callbacks_preserve_same_image_identity():
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:9, 5:12] = 255
    ok, png = cv2.imencode(".png", mask)
    assert ok
    common = {"req_id": 42, "serial": "cam", "fid": 11,
              "image_sha256": "a" * 64,
              "image_space": "autodex_undistorted_full_frame"}
    req, serial, row = decode_precision_mask(
        {**common, "h": 24, "w": 32}, png.tobytes())
    assert (req, serial, row["frame_id"]) == (42, "cam", 11)
    assert row["mask"].sum() == 35
    pose = np.eye(4, dtype=np.float64)
    req, serial, row = decode_precision_pose(
        {**common, "ok": True, "quality": .9, "inliers": 10},
        pose.tobytes())
    assert (req, serial, row["frame_id"]) == (42, "cam", 11)
    assert np.array_equal(row["pose_world"], pose)
    with pytest.raises(ValueError, match="bound camera frame"):
        decode_precision_pose({"req_id": 42, "serial": "cam", "ok": True},
                              pose.tobytes())


def test_board_metadata_tap_matches_exact_jpeg_not_later_frame():
    image = np.full((24, 32, 3), 10, dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", image)
    assert ok
    jpeg = jpg.tobytes()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    metadata = SnapshotMetadataBuffer()
    metadata.put({"req_id": 42, "serial": "cam", "fid": 11}, jpeg)
    identities = FrameIdentityRegistry()

    class _Snapshot:
        def snap(self, **kwargs):
            return {"cam": {"jpeg": jpeg, "image": decoded}}, {
                "request_id": kwargs["request_id"]}

    adapter = ProvenanceSnapshotAdapter(_Snapshot(), metadata, identities)
    payload, timing = adapter.snap(decode=True, request_id=42,
                                   timeout_s=.1)
    assert payload["cam"]["frame_id"] == 11
    assert timing["request_id"] == 42
    assert metadata.get(42) == {}
    assert identities.get(42)["cam"]["image_sha256"] == image_sha256(decoded)
    metadata.put({"req_id": 43, "serial": "cam", "fid": 12}, b"other JPEG")
    with pytest.raises(ValueError, match="differs from frame-ID metadata"):
        adapter.snap(decode=True, request_id=43, timeout_s=.1)


def test_robot_orchestrator_callback_keeps_fid_and_hash():
    class _Buffer:
        def __init__(self):
            self.values = {}

        def put(self, request, serial, row):
            self.values[(request, serial)] = row

    stock = SimpleNamespace(
        _mask_thread=SimpleNamespace(on_message=None),
        _pose_thread=SimpleNamespace(on_message=None),
        mask_buf=_Buffer(), pose_buf=_Buffer(), obj_name="key")
    identities = FrameIdentityRegistry()
    wrapper = PrecisionInitOrchestrator(stock, identities)
    mask = np.zeros((4, 5), dtype=np.uint8)
    ok, png = cv2.imencode(".png", mask)
    assert ok
    metadata = {"req_id": 7, "serial": "cam", "fid": 12,
                "image_sha256": "f" * 64,
                "image_space": "autodex_undistorted_full_frame"}
    stock._mask_thread.on_message({**metadata, "h": 4, "w": 5}, png.tobytes())
    stock._pose_thread.on_message({**metadata, "ok": False}, b"")
    assert wrapper.obj_name == "key"
    assert stock.mask_buf.values[(7, "cam")]["frame_id"] == 12
    assert stock.pose_buf.values[(7, "cam")]["image_sha256"] == "f" * 64
    stock.collect_payloads = lambda **_kwargs: (
        {"cam": stock.mask_buf.values[(7, "cam")]},
        {"cam": stock.pose_buf.values[(7, "cam")]},
        {"request_id": 7})
    wrapper.collect_payloads()
    assert identities.get(7)["cam"]["frame_id"] == 12
