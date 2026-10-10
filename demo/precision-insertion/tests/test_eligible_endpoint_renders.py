"""No diagnostic seed or stale/failed v8 row can become a success image."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.candidates import build_endpoint_catalog  # noqa: E402
from render_eligible_grasp_endpoints import (  # noqa: E402
    _write_bundle, eligible_rows, render_catalog,
)
from precision_insertion.endpoint import nominal_inspire_hold_poses  # noqa: E402
from test_candidates import _candidate_fixture  # noqa: E402


def test_only_complete_pose_verified_rows_renderable(tmp_path):
    mode, _, candidates, screen = _candidate_fixture(tmp_path)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    actual_mode, rows = eligible_rows(catalog)
    assert actual_mode == mode
    assert [row["key"] for row in rows] == [
        ["table", "0", "1"], ["table", "1", "3"]]
    assert all(row["key"] != ["table", "0", "2"] for row in rows)

    pilot = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, max_candidates=1, screen=screen)
    with pytest.raises(ValueError, match="complete v8 scan"):
        eligible_rows(pilot)
    (candidates[0] / "sim_eval.json").write_text('{"success": false}')
    with pytest.raises(ValueError, match="stale"):
        eligible_rows(catalog)


def test_zero_eligible_catalog_has_manifest_but_no_success_images(tmp_path):
    mode, _, _, _ = _candidate_fixture(tmp_path)

    def reject_all(**_kwargs):
        return {"endpoint_pass": False, "input_sha256": {}}

    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=reject_all)
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(catalog), encoding="utf-8")
    output = tmp_path / "render_output"
    manifest = render_catalog(catalog_path=source, output_root=output)
    assert manifest["eligible_count"] == 0
    assert manifest["rendered_grasp_count"] == 0
    assert list(output.rglob("*.png")) == []
    assert json.loads((output / "manifest.json").read_text())["robot_ready"] is False
    with pytest.raises(FileExistsError):
        render_catalog(catalog_path=source, output_root=output)


def test_fresh_endpoint_failure_prevents_any_image(tmp_path, monkeypatch):
    mode, paths, _, screen = _candidate_fixture(tmp_path)
    paths.task_geometry.write_text("{}")
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=screen)
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(catalog), encoding="utf-8")
    blender = tmp_path / "blender"
    blender.touch()
    monkeypatch.setattr("render_eligible_grasp_endpoints._load_mesh",
                        lambda _path: object())
    monkeypatch.setattr("render_eligible_grasp_endpoints.validate_task_geometry",
                        lambda _geometry, _mode: None)
    monkeypatch.setattr("render_eligible_grasp_endpoints.screen_grasp_endpoint",
                        lambda **_kwargs: {"endpoint_pass": False})
    output = tmp_path / "render_output"
    with pytest.raises(ValueError, match="current 20 mm screen failed"):
        render_catalog(catalog_path=source, output_root=output,
                       blender=blender)
    assert not output.exists()


def test_bundle_keeps_key_hand_transform_fixed_at_twenty_mm(
    tmp_path, monkeypatch,
):
    mode, _, candidates, _ = _candidate_fixture(tmp_path)
    candidate = candidates[0]
    pre = np.load(candidate / "pregrasp_pose.npy")
    grasp = np.load(candidate / "grasp_pose.npy")
    hold = nominal_inspire_hold_poses(pre, grasp)["mujoco_squeeze"]

    class FakeMesh:
        def export(self, path):
            Path(path).write_bytes(b"mock mesh")

    monkeypatch.setattr("render_eligible_grasp_endpoints._hand_link_meshes",
                        lambda _urdf, q: {"right_link": FakeMesh()}
                        if np.allclose(q, hold) else None)
    T_socket_key = np.eye(4)
    T_socket_key[2, 3] = 0.02
    T_key_hand = np.eye(4)
    T_key_hand[0, 3] = 0.03
    endpoint = {
        "T_socket_key_tested": T_socket_key.tolist(),
        "T_key_hand": T_key_hand.tolist(),
        "hold_pose_screens": {"mujoco_squeeze": {"hand_q": hold.tolist()}},
    }
    output = tmp_path / "bundle"
    output.mkdir()
    path = _write_bundle(
        output=output, shared_root=tmp_path, mode=mode,
        candidate=candidate, endpoint=endpoint, hold_name="mujoco_squeeze",
        key_mesh_path=tmp_path / "key.ply", socket_mesh_path=tmp_path / "socket.ply",
    )
    with np.load(path, allow_pickle=False) as bundle:
        key_pose = bundle["object_poses"][0]
        hand_pose = bundle["robot_geometry_transforms"][0, 0]
    assert np.allclose(np.linalg.inv(key_pose) @ hand_pose, T_key_hand,
                       atol=1e-6)
    assert key_pose[2, 3] == pytest.approx(0.06)
