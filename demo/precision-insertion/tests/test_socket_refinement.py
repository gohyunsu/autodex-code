"""A stock socket refinement records all views and can use stock fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SocketObservation  # noqa: E402
from precision_insertion.perception_evidence import SocketCaptureEvidence  # noqa: E402
from precision_insertion.session_bootstrap import SocketCaptureInput  # noqa: E402
from precision_insertion.socket_refinement import refine_socket_capture  # noqa: E402


def _setup(tmp_path, monkeypatch, *, per_view=None, stock_result=None):
    raw_mesh = tmp_path / "socket.obj"
    raw_mesh.write_text("v 0 0 0\n", encoding="utf-8")
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:18, 9:24] = True
    images = {camera: np.zeros((24, 32, 3), dtype=np.uint8)
              for camera in ("cam_a", "cam_b")}
    capture = SocketCaptureInput(
        "socket_1", 101, "fixed socket", images,
        {camera: {"mask": mask.copy()} for camera in images},
        {camera: {"ok": True, "pose_world": np.eye(4)}
         for camera in images},
        {"cam_a": 1.0, "cam_b": 1.005}, "camera_acquisition")
    admission = SocketCaptureEvidence(
        "socket_1", tuple(SocketObservation(
            "socket_1", camera, 1.0, np.eye(4), "camera_acquisition")
            for camera in images), {}, "camera_acquisition")
    selection = {"best_serial": "cam_a", "n_candidates": 2, "n_masks": 2,
                 "sil_loss": .0005}
    calls = []

    class Stock:
        obj_name = "socket"
        H = 24
        W = 32
        intrinsics_undist = {camera: np.eye(3) for camera in images}
        extrinsics = {camera: np.eye(4) for camera in images}
        _sil = SimpleNamespace(
            _obj_name="socket",
            _precision_mesh_sha256=hashlib.sha256(
                raw_mesh.read_bytes()).hexdigest(),
            glctx=object(), mesh_tensors=object())

        def refine_from_payloads(self, masks, poses, **kwargs):
            calls.append(kwargs)
            return ((np.eye(4), selection) if stock_result is None
                    else stock_result)

    stock = Stock()
    monkeypatch.setattr(
        "autodex.perception.pose_select.compute_cross_view_iou",
        lambda *_args: (.9, per_view or {"cam_a": .91, "cam_b": .89}))
    return capture, admission, stock, raw_mesh, calls


def test_stock_refinement_is_bound_to_mesh_capture_and_all_views(
        tmp_path, monkeypatch):
    capture, admission, stock, raw_mesh, calls = _setup(tmp_path, monkeypatch)
    row = refine_socket_capture(
        capture=capture, admission=admission, init_orchestrator=stock,
        socket_object="socket", raw_mesh=raw_mesh, minimum_refined_iou=.8)
    assert row.capture_id == "socket_1" and row.request_id == 101
    assert row.accepted_camera_ids == ("cam_a", "cam_b")
    assert row.per_view_iou == {"cam_a": .91, "cam_b": .89}
    assert row.stock_selection["post_refinement_mean_iou"] == .9
    assert calls[0]["selection_mode"] == "iou"
    assert calls[0]["subset_serials"] == ["cam_a", "cam_b"]


def test_uses_stock_silhouette_when_one_view_has_low_iou(
        tmp_path, monkeypatch):
    capture, admission, stock, raw_mesh, _ = _setup(
        tmp_path, monkeypatch, per_view={"cam_a": .99, "cam_b": .72})
    row = refine_socket_capture(
        capture=capture, admission=admission, init_orchestrator=stock,
        socket_object="socket", raw_mesh=raw_mesh, minimum_refined_iou=.8)
    assert row.refinement_gate == "stock_silhouette"
    assert row.per_view_iou == {"cam_a": .99, "cam_b": .72}
    assert row.stock_selection["post_refinement_iou_support_camera_ids"] == [
        "cam_a"]


def test_stock_ignore_sil_loss_retains_refinement_and_records_policy(
        tmp_path, monkeypatch):
    capture, admission, stock, raw_mesh, calls = _setup(
        tmp_path, monkeypatch, per_view={"cam_a": .92, "cam_b": .72},
        stock_result=(np.eye(4), {
            "best_serial": "cam_a", "n_candidates": 2, "n_masks": 2,
            "sil_loss": .0043}))
    row = refine_socket_capture(
        capture=capture, admission=admission, init_orchestrator=stock,
        socket_object="socket", raw_mesh=raw_mesh, minimum_refined_iou=.8,
        perception_mode="ignore_sil_loss")
    assert calls[0]["sil_loss_threshold"] == float("inf")
    assert calls[0]["selection_mode"] == "iou"
    assert row.refinement_gate == "stock_ignore_sil_loss"
    assert row.stock_selection["sil_loss"] == .0043
    assert row.stock_selection["silhouette_loss_threshold"] is None
    assert row.stock_selection["silhouette_loss_gate_applied"] is False
    assert row.stock_selection["perception_mode"] == "ignore_sil_loss"


def test_rejects_stock_failure_or_changed_renderer_mesh(tmp_path, monkeypatch):
    capture, admission, stock, raw_mesh, _ = _setup(
        tmp_path, monkeypatch, stock_result=(None, {"sil_reject": True}))
    with pytest.raises(ValueError, match="stock socket"):
        refine_socket_capture(
            capture=capture, admission=admission, init_orchestrator=stock,
            socket_object="socket", raw_mesh=raw_mesh, minimum_refined_iou=.8)
    raw_mesh.write_text("v 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact raw mesh"):
        refine_socket_capture(
            capture=capture, admission=admission, init_orchestrator=stock,
            socket_object="socket", raw_mesh=raw_mesh, minimum_refined_iou=.8)
