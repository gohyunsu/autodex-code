"""Socket-only geometry: mixed SAM regions cannot tilt the fixed fixture."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SocketObservation  # noqa: E402
from precision_insertion.board_socket_fit import fit_board_socket_body  # noqa: E402
from precision_insertion.perception_evidence import SocketCaptureEvidence  # noqa: E402
from precision_insertion.session_bootstrap import SocketCaptureInput  # noqa: E402
from precision_insertion.socket_refinement import refine_socket_capture  # noqa: E402


def _camera(eye: np.ndarray) -> np.ndarray:
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    E = np.eye(4)
    E[:3, :3] = np.stack((right, down, forward))
    E[:3, 3] = -E[:3, :3] @ eye
    return E


def _disk_mask(E: np.ndarray, K: np.ndarray, radius: float,
               bottom: float, top: float) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    points = np.array([[radius * np.cos(a), radius * np.sin(a), z, 1.0]
                       for z in (bottom, top) for a in angle])
    xyz = (E @ points.T)[:3].T
    pixel = (K @ xyz.T).T
    pixel = np.rint(pixel[:, :2] / pixel[:, 2, None]).astype(np.int32)
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pixel), 1)
    return mask.astype(bool)


@pytest.mark.parametrize("limit", [0.8, 0.0])
def test_board_socket_fit_keeps_axis_upright_with_one_flange_mask(
        tmp_path, limit):
    mesh = tmp_path / "socket.obj"
    mesh.write_text("\n".join(
        f"v {r * np.cos(a)} {r * np.sin(a)} {z}"
        for r, z in ((.06, 0), (.06, .005), (.04, .005), (.04, .055))
        for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)) + "\n")
    K = np.array([[850, 0, 400], [0, 850, 300], [0, 0, 1.0]])
    eyes = ((.24, -.28, .42), (-.30, -.20, .40),
            (.27, .30, .44), (-.29, .25, .41))
    cameras = {f"cam_{i}": _camera(np.array(eye))
               for i, eye in enumerate(eyes)}
    masks = {camera: _disk_mask(E, K, .04, .005, .055)
             for camera, E in cameras.items()}
    masks["cam_3"] = _disk_mask(cameras["cam_3"], K, .06, 0, .005)
    board = np.diag([1.0, -1.0, -1.0, 1.0])
    capture = SocketCaptureInput(
        "socket_000", 42, "white socket", {},
        {camera: {"mask": mask} for camera, mask in masks.items()},
        {camera: {"ok": False}
         for camera in masks},
        {camera: 10.0 for camera in masks}, "camera_acquisition")
    admission = SocketCaptureEvidence(
        "socket_000", tuple(SocketObservation(
            "socket_000", camera, 10.0, None, "camera_acquisition")
            for camera in masks), {}, "camera_acquisition")
    stock = SimpleNamespace(
        obj_name="socket", H=600, W=800,
        intrinsics_undist={camera: K for camera in masks},
        extrinsics=cameras, _sil=None)
    row = refine_socket_capture(
        capture=capture, admission=admission, init_orchestrator=stock,
        socket_object="socket", raw_mesh=mesh, minimum_refined_iou=limit,
        perception_mode="board_constrained_body",
        board_pose_robot=board, c2r=np.eye(4))
    assert row.refinement_gate == "board_constrained_body"
    assert np.allclose(row.pose_world[:3, 2], [0, 0, 1], atol=1e-6)
    assert np.linalg.norm(row.pose_world[:3, 3]) < .003
    assert np.mean(list(row.per_view_iou.values())) > .8
    assert row.stock_selection["sil_skipped"] is True
    assert row.per_view_iou["cam_3"] > .98
    diagnostic = row.stock_selection["body_fit"]
    assert diagnostic["per_view_selected_region"] == {
        "cam_0": "wall", "cam_1": "wall", "cam_2": "wall",
        "cam_3": "flange"}
    assert diagnostic["maximum_pairwise_center_disagreement_mm"] > 0
    assert set(diagnostic["per_view_center_residual_mm"]) == set(cameras)


def test_two_flange_only_views_use_flange_height_not_wall_midpoint(tmp_path):
    mesh = tmp_path / "socket.obj"
    mesh.write_text("\n".join(
        f"v {r * np.cos(a)} {r * np.sin(a)} {z}"
        for r, z in ((.06, 0), (.06, .005), (.04, .005), (.04, .055))
        for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)) + "\n")
    K = np.array([[850, 0, 400], [0, 850, 300], [0, 0, 1.0]])
    cameras = {
        "left": _camera(np.array([.24, -.28, .42])),
        "right": _camera(np.array([-.30, -.20, .40])),
    }
    masks = {name: _disk_mask(E, K, .06, 0, .005)
             for name, E in cameras.items()}
    pose, diagnostic = fit_board_socket_body(
        masks=masks, intrinsics_undist={name: K for name in cameras},
        extrinsics_camera_world=cameras, c2r=np.eye(4),
        board_pose_robot=np.diag([1.0, -1.0, -1.0, 1.0]),
        raw_mesh=mesh)
    assert np.linalg.norm(pose[:3, 3]) < .002
    assert diagnostic["per_view_selected_region"] == {
        "left": "flange", "right": "flange"}
    assert diagnostic["maximum_pairwise_center_disagreement_mm"] < 2
    assert min(diagnostic["per_view_iou"].values()) > .98


def test_whole_body_masks_select_both_socket_regions(tmp_path):
    mesh = tmp_path / "socket.obj"
    mesh.write_text("\n".join(
        f"v {r * np.cos(a)} {r * np.sin(a)} {z}"
        for r, z in ((.06, 0), (.06, .005), (.04, .005), (.04, .055))
        for a in np.linspace(0, 2 * np.pi, 36, endpoint=False)) + "\n")
    K = np.array([[850, 0, 400], [0, 850, 300], [0, 0, 1.0]])
    cameras = {
        "left": _camera(np.array([.24, -.28, .42])),
        "right": _camera(np.array([-.30, -.20, .40])),
    }
    masks = {}
    for name, E in cameras.items():
        wall = _disk_mask(E, K, .04, .005, .055)
        flange = _disk_mask(E, K, .06, 0, .005)
        masks[name] = wall | flange
    pose, diagnostic = fit_board_socket_body(
        masks=masks, intrinsics_undist={name: K for name in cameras},
        extrinsics_camera_world=cameras, c2r=np.eye(4),
        board_pose_robot=np.diag([1.0, -1.0, -1.0, 1.0]),
        raw_mesh=mesh)
    assert np.linalg.norm(pose[:3, 3]) < .002
    assert diagnostic["per_view_selected_region"] == {
        "left": "whole_body", "right": "whole_body"}
    assert min(diagnostic["per_view_iou"].values()) > .9


def test_board_socket_fit_rejects_noncanonical_intrinsics(tmp_path):
    mesh = tmp_path / "socket.obj"
    mesh.write_text("v .04 0 .005\nv -.04 0 .055\n")
    K = np.array([[850, 0, 400], [0, 850, 300], [0, .1, 1.0]])
    with pytest.raises(ValueError, match="intrinsics"):
        fit_board_socket_body(
            masks={"cam": np.ones((8, 8), dtype=bool)},
            intrinsics_undist={"cam": K},
            extrinsics_camera_world={"cam": np.eye(4)},
            c2r=np.eye(4),
            board_pose_robot=np.diag([1.0, -1.0, -1.0, 1.0]),
            raw_mesh=mesh)
