"""Saved stock-refinement replay must bind every source before CUDA work."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_foundpose_stock_refinement as replay  # noqa: E402


def _report_and_stubs(monkeypatch):
    fields = {
        "image_path": "/shot/cam.jpg",
        "image_sha256": "jpeg_hash",
        "mask_path": "/mask.png",
        "mask_sha256": "mask_hash",
        "shot_json_sha256": "shot_hash",
        "intrinsics_sha256": "intr_hash",
        "extrinsics_sha256": "extr_hash",
        "condition_id": "square/socket_only",
    }
    report = {
        "schema": "precision_insertion_saved_foundpose_probe_v2",
        "robot_ready": False, "family": "square", "gap_mm": 1.5,
        "part": "socket", "object": "precision_socket_unified",
        "shot_dir": "/shot", "condition_id": "square/socket_only",
        "mesh_sha256": "mesh_hash", "repre_sha256": "repre_hash",
        "views": [dict(fields, camera_id=serial,
                       status="pose_estimated_not_validated",
                       pose_world_m=np.eye(4).tolist())
                  for serial in ("cam0", "cam1")],
    }

    class Paths:
        def __init__(self, *_args):
            pass

        def raw_mesh(self, _name):
            return Path("/tmp/mesh.obj")

        def foundpose_repre(self, _name):
            return Path("/tmp/repre.pth")

    monkeypatch.setattr(replay, "AssetPaths", Paths)
    monkeypatch.setattr(replay, "_hash_file", lambda path:
                        "mesh_hash" if path.suffix == ".obj" else "repre_hash")
    monkeypatch.setattr(replay, "_load_view", lambda _shot, _calib, _camera,
                        _mask: dict(fields))
    return report


def test_bound_views_accept_only_matching_source_and_pose(monkeypatch):
    report = _report_and_stubs(monkeypatch)
    mesh, views, selection = replay._load_bound_views(
        report, shared_root=Path("/tmp"), calib_dir=Path("/tmp"))
    assert mesh == Path("/tmp/mesh.obj")
    assert list(views) == ["cam0", "cam1"]
    assert np.array_equal(views["cam0"]["pose_world"], np.eye(4))
    assert selection["object"] == "precision_socket_unified"


def test_bound_views_reject_changed_mask_hash(monkeypatch):
    report = _report_and_stubs(monkeypatch)
    report["views"][1]["mask_sha256"] = "changed"
    with pytest.raises(ValueError, match="mask_sha256"):
        replay._load_bound_views(report, shared_root=Path("/tmp"),
                                 calib_dir=Path("/tmp"))


def test_bound_views_reject_swapped_object_or_duplicate_camera(monkeypatch):
    report = _report_and_stubs(monkeypatch)
    report["object"] = "precision_socket_cylinder_gap_15mm"
    with pytest.raises(ValueError, match="report object"):
        replay._load_bound_views(report, shared_root=Path("/tmp"),
                                 calib_dir=Path("/tmp"))
    report["object"] = "precision_socket_unified"
    report["views"][1]["camera_id"] = "cam0"
    with pytest.raises(ValueError, match="duplicate camera"):
        replay._load_bound_views(report, shared_root=Path("/tmp"),
                                 calib_dir=Path("/tmp"))


def test_bound_views_reject_nonready_or_invalid_pose(monkeypatch):
    report = _report_and_stubs(monkeypatch)
    report["robot_ready"] = True
    with pytest.raises(ValueError, match="non-ready"):
        replay._load_bound_views(report, shared_root=Path("/tmp"),
                                 calib_dir=Path("/tmp"))
    report["robot_ready"] = False
    report["views"][0]["pose_world_m"][0][0] = 2.0
    with pytest.raises(ValueError, match="not orthonormal"):
        replay._load_bound_views(report, shared_root=Path("/tmp"),
                                 calib_dir=Path("/tmp"))
