"""Actual mesh transforms are assembled without inventing an observed key."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.held_scene_overlay import (  # noqa: E402
    build_held_scene_comparison, build_held_scene_prediction,
    render_held_scene_overlays,
)
from precision_insertion.observer import observe_preinsert_hold_views  # noqa: E402


class _Robot:
    def __init__(self):
        self.actuated_joints = [SimpleNamespace(name=f"fr3_joint{i}")
                                for i in range(7)] + [
            SimpleNamespace(name=f"right_joint{i}") for i in range(6)]
        self.base_link = "fr3_link0"
        self.scene = trimesh.Scene()
        self.scene.add_geometry(trimesh.creation.box((0.1, 0.1, 0.1)),
                                geom_name="fr3_link0_visual")
        self.scene.add_geometry(trimesh.creation.box((0.02, 0.02, 0.05)),
                                geom_name="right_index_visual")
        self.cfg = None

    def update_cfg(self, cfg):
        self.cfg = cfg

    def get_transform(self, from_link, to_link):
        assert from_link == "base_link" and to_link == self.base_link
        pose = np.eye(4)
        pose[0, 3] = 0.1
        return pose


def _prediction(tmp_path):
    mode = select_mode("square", 1.5)
    for name in (mode.key_object, mode.socket_object):
        path = tmp_path / "object_processing" / name / "raw_mesh" / f"{name}.obj"
        path.parent.mkdir(parents=True)
        trimesh.creation.box((0.02, 0.02, 0.05)).export(path)
    relation = np.eye(4)
    relation[0, 3] = 0.02
    return build_held_scene_prediction(
        shared_root=tmp_path, mode=mode, full_q_measured=np.zeros(13),
        T_robot_socket_frozen=np.eye(4), T_key_hand_hypothesis=relation,
        relation_source="v8_nominal_diagnostic",
        robot_loader=lambda _path: _Robot())


def test_full_robot_key_socket_prediction_is_explicitly_not_observation(tmp_path):
    prediction = _prediction(tmp_path)
    assert prediction.names[-2:] == ("predicted_key", "frozen_socket")
    assert "fr3_link0_visual" in prediction.names
    assert "right_index_visual" in prediction.names
    assert prediction.T_robot_key_predicted[0, 3] == pytest.approx(0.08)
    assert prediction.to_record()["robot_ready"] is False
    assert "rendered_hypothesis" in prediction.to_record()["scope"]


def test_existing_depth_renderer_adapter_keeps_raw_pixels_and_colors_objects(tmp_path):
    prediction = _prediction(tmp_path)
    frame = np.full((32, 48, 3), 20, dtype=np.uint8)
    frames = {"cam_b": frame.copy(), "cam_a": frame.copy()}
    K = np.array([[100, 0, 24], [0, 100, 16], [0, 0, 1]], dtype=float)
    created = []

    class FakeRenderer:
        def __init__(self, meshes, names, labels, intrinsics, extrinsics, H, W):
            self.serials = sorted(intrinsics)
            self.n_links = len(names)
            self.color_lut = torch.zeros((len(names) + 1, 3))
            self.alpha_lut = torch.zeros((len(names) + 1, 1))
            self.names = names
            self.meshes = meshes
            self.poses = None
            created.append(self)
            assert (H, W) == (32, 48)
            assert len(meshes) == len(names)
            assert all(ext.shape == (3, 4) for ext in extrinsics.values())

        def render(self, poses, images):
            self.poses = poses
            # Even a renderer that modifies its inputs must not corrupt the
            # saved raw image used for a paired VLM comparison.
            images[0][:] = 99
            return [image.copy() for image in images]

    overlays = render_held_scene_overlays(
        prediction=prediction, frames_bgr=frames,
        intrinsics_undistorted={serial: K for serial in frames},
        T_camera_robot={serial: np.eye(4) for serial in frames},
        renderer_factory=FakeRenderer)
    assert set(overlays) == set(frames)
    assert np.array_equal(frames["cam_a"], frame)
    assert np.array_equal(frames["cam_b"], frame)
    assert len(created[0].poses) == len(prediction.names)
    key_index = prediction.names.index("predicted_key") + 1
    socket_index = prediction.names.index("frozen_socket") + 1
    assert not torch.equal(created[0].color_lut[key_index],
                           created[0].color_lut[socket_index])


def test_bad_relation_source_or_camera_set_fails_closed(tmp_path):
    mode = select_mode("square", 1.5)
    with pytest.raises(ValueError, match="relation source"):
        build_held_scene_prediction(
            shared_root=tmp_path, mode=mode, full_q_measured=np.zeros(13),
            T_robot_socket_frozen=np.eye(4),
            T_key_hand_hypothesis=np.eye(4), relation_source="squeeze_is_actual")
    prediction = _prediction(tmp_path)
    with pytest.raises(ValueError, match="identical IDs"):
        render_held_scene_overlays(
            prediction=prediction,
            frames_bgr={"cam": np.zeros((8, 8, 3), dtype=np.uint8)},
            intrinsics_undistorted={}, T_camera_robot={})


def test_paired_mesh_comparison_feeds_per_view_preinsert_vlm(tmp_path):
    prediction = _prediction(tmp_path)
    frames = {camera: np.full((32, 48, 3), 20, dtype=np.uint8)
              for camera in ("front", "side")}
    K = np.array([[100, 0, 24], [0, 100, 16], [0, 0, 1]], dtype=float)

    class FakeRenderer:
        def __init__(self, meshes, names, labels, intrinsic, extrinsic, H, W):
            self.serials = sorted(intrinsic)
            self.n_links = len(meshes)
            self.color_lut = torch.zeros((len(names) + 1, 3))
            self.alpha_lut = torch.zeros((len(names) + 1, 1))

        def render(self, poses, images):
            return [np.full_like(image, 80) for image in images]

    comparison = build_held_scene_comparison(
        prediction=prediction, frames_bgr=frames,
        frame_timestamps_s={"front": 10.0, "side": 10.01},
        intrinsics_undistorted={camera: K for camera in frames},
        T_camera_robot={camera: np.eye(4) for camera in frames},
        renderer_factory=FakeRenderer)
    assert len(comparison.views) == 2
    assert comparison.views[0].raw.getpixel((0, 0)) == (20, 20, 20)
    assert comparison.views[0].predicted_overlay.getpixel((0, 0)) == (80, 80, 80)
    assert len(comparison.to_record()["views"]["front"]["raw_image_sha256"]) == 64
    assert comparison.to_record()["prediction"]["robot_ready"] is False

    class FakeVLM:
        def infer(self, images, prompt):
            camera = "front" if "Use only camera ID front" in prompt else "side"
            return json.dumps({
                "class": "coarse_match", "evidence_views": [camera],
                "evidence": "visible key moves with hand",
            })

    visual = observe_preinsert_hold_views(
        FakeVLM(), comparison.views, max_capture_skew_s=0.02)
    assert visual.status == "coarse_match"
    assert visual.to_record()["robot_ready"] is False
