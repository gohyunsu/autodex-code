"""Square v8 relocation is source-bound, non-overwriting and replayable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_square_tabletop_handoff import SCHEMA, _sha  # noqa: E402
from rehydrate_square_tabletop_handoff import (  # noqa: E402
    CANDIDATE_PARENT, FIXTURE_PARENT, KEY, ROBOT_URDF, SCENE_PARENT,
    SOCKET, _json_bytes, rehydrate,
)


def _write(root: Path, relative: Path, content: bytes | dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(content) if isinstance(content, dict) else content)
    return path


def _geometry(origin: Path) -> dict:
    identity = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                [0., 0., 1., 0.], [0., 0., 0., 1.]]
    entry = [[1., 0., 0., 0.], [0., -1., 0., 0.],
             [0., 0., -1., .1], [0., 0., 0., 1.]]
    target = [[1., 0., 0., 0.], [0., -1., 0., 0.],
              [0., 0., -1., .08], [0., 0., 0., 1.]]
    return {
        "units": "m", "socket_pose_object": SOCKET,
        "socket_pose_mesh": str(origin / "object_processing" / SOCKET /
                                "raw_mesh" / f"{SOCKET}.obj"),
        "T_socket_pose_object": identity,
        "T_socket_key_entry": entry,
        "T_socket_key_verification": target,
        "verification_insertion_depth_m": .02,
        "insertion_direction_socket": [0., 0., -1.],
        "key_frame": {"insertion_axis": [0., 0., 1.]},
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    origin = tmp_path / "old_shared"
    target = tmp_path / "recipient_shared"
    bundle = tmp_path / "bundle"
    payload = bundle / "payload/shared_data"
    target.mkdir(parents=True)
    robot_bytes = b"<robot name='fr3_inspire'/>"
    _write(target, ROBOT_URDF, robot_bytes)
    _write(payload, Path("object_processing") / KEY /
           "processed_data/mesh/simplified.obj", b"key cad")
    _write(payload, Path("object_processing") / SOCKET /
           "processed_data/mesh/static_collision.obj", b"socket cad")
    _write(payload, FIXTURE_PARENT / "socket_shared_bore_1p5.obj",
           b"socket cad")
    socket_root = origin / "object_processing" / SOCKET
    foundpose = (origin / "AutoDex/foundpose_assets" / SOCKET /
                 "object_repre/v1" / SOCKET / "1/repre.pth")
    _write(payload, FIXTURE_PARENT / "task_geometry.json", _geometry(origin))
    _write(payload, FIXTURE_PARENT / "fixture_pose.template.json", {
        "pose_object_mesh": str(socket_root / "raw_mesh" / f"{SOCKET}.obj"),
        "pose_object_frame_contract": str(socket_root /
                                          "processed_data/info/frame_contract.json"),
        "pose_estimator_asset": str(foundpose),
    })
    _write(payload, FIXTURE_PARENT / "pose_measurement_asset.json", {
        "pose_object": {
            "object_root": str(socket_root),
            "raw_mesh": str(socket_root / "raw_mesh" / f"{SOCKET}.obj"),
            "processed_mesh": str(socket_root /
                                  "processed_data/mesh/simplified.obj"),
            "static_collision_mesh": str(socket_root /
                                         "processed_data/mesh/static_collision.obj"),
            "static_urdf": str(socket_root /
                               "processed_data/urdf/socket_static_exact.urdf"),
            "frame_contract": str(socket_root /
                                  "processed_data/info/frame_contract.json"),
            "foundpose_representation": str(foundpose),
        },
    })
    for i in range(5):
        _write(payload, SCENE_PARENT / f"{i}.json", {
            "scene": {"mesh": {"target": {
                "file_path": str(origin / "object_processing" / KEY /
                                 "processed_data/mesh/simplified.obj"),
                "urdf_path": str(origin / "object_processing" / KEY /
                                 "processed_data/urdf/coacd.urdf"),
            }}}, "meta": {"pose_idx": f"{i:03d}"},
        })
    source_scene = _sha(payload / SCENE_PARENT / "4.json")
    ids = ["35", "70", "79", "95", "99", "104", "5102"]
    for candidate in ids:
        _write(payload, CANDIDATE_PARENT / "table/4" / candidate /
               "simulation_validation.json", {
                   "schema": "precision_insertion_square_tabletop_simulation_v1",
                   "source_scene_sha256": source_scene,
                   "status": "passed", "robot_ready": False,
               })
    hashes = {str(path.relative_to(bundle)): _sha(path)
              for path in sorted(bundle.rglob("*")) if path.is_file()}
    _write(bundle, Path("MANIFEST.json"), {
        "schema": SCHEMA, "origin_shared_root": str(origin),
        "source_robot_urdf_sha256": hashlib.sha256(robot_bytes).hexdigest(),
        "candidate_ids": [f"table/4/{candidate}" for candidate in ids],
        "file_sha256": hashes, "robot_ready": False,
    })
    return bundle, origin, target, ids


def test_relocate_square_bundle_and_preserve_original_provenance(tmp_path):
    bundle, origin, target, ids = _bundle(tmp_path)
    dry = rehydrate(bundle_root=bundle, target_shared_root=target)
    assert dry["installed"] is False
    assert dry["files_to_install"] == dry["payload_files"]
    assert not (target / SCENE_PARENT).exists()
    installed = rehydrate(bundle_root=bundle, target_shared_root=target,
                          install=True)
    assert installed["installed"] is True
    assert installed["robot_ready"] is False
    scene = target / SCENE_PARENT / "4.json"
    assert str(target) in scene.read_text()
    assert str(origin) not in scene.read_text()
    validation = json.loads((target / CANDIDATE_PARENT / "table/4" / ids[0] /
                             "simulation_validation.json").read_text())
    assert validation["source_scene_sha256"] == _sha(scene)
    assert validation["relocation"]["source_scene_sha256"] == _sha(
        bundle / "payload/shared_data" / SCENE_PARENT / "4.json")
    assert validation["relocation"]["scope"] == (
        "absolute_scene_paths_only_not_new_simulation")
    again = rehydrate(bundle_root=bundle, target_shared_root=target)
    assert again["files_to_install"] == 0
    assert again["identical_existing_files"] == installed["payload_files"]


def test_relocate_square_rejects_conflict_before_any_other_write(tmp_path):
    bundle, _, target, _ = _bundle(tmp_path)
    _write(target, SCENE_PARENT / "4.json", b"different")
    with pytest.raises(ValueError, match="conflicts"):
        rehydrate(bundle_root=bundle, target_shared_root=target, install=True)
    assert not (target / SCENE_PARENT / "0.json").exists()


def test_relocate_square_rejects_bundle_tamper_and_wrong_robot(tmp_path):
    bundle, _, target, _ = _bundle(tmp_path)
    (bundle / "payload/shared_data" / SCENE_PARENT / "4.json").write_text(
        "changed", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        rehydrate(bundle_root=bundle, target_shared_root=target, install=True)
    assert not (target / SCENE_PARENT).exists()
    bundle, _, target, _ = _bundle(tmp_path / "fresh")
    (target / ROBOT_URDF).write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="URDF"):
        rehydrate(bundle_root=bundle, target_shared_root=target, install=True)
    assert not (target / SCENE_PARENT).exists()
