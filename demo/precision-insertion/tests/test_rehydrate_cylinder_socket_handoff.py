"""Relocated socket CAD must retain geometry while changing only host paths."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import (  # noqa: E402
    CYLINDER_RADIAL_GAPS_MM, select_mode,
)
from precision_insertion.endpoint import validate_task_geometry  # noqa: E402
from precision_insertion.fixture_contract import (  # noqa: E402
    validate_cylinder_socket_fixture,
)
from rehydrate_cylinder_socket_handoff import rehydrate  # noqa: E402


def _save(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _json(path: Path, value: dict) -> None:
    _save(path, json.dumps(value, allow_nan=False))


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "extracted"
    origin = tmp_path / "historical_shared_data"
    target = tmp_path / "new_shared_data"
    rotation = [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]

    def pose(z: float) -> list[list[float]]:
        return [rotation[0] + [0.], rotation[1] + [0.],
                rotation[2] + [z], [0., 0., 0., 1.]]

    identity = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                [0., 0., 1., 0.], [0., 0., 0., 1.]]
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        name = mode.socket_object
        obj_rel = Path("object_processing") / name
        fixture_rel = Path("AutoDex/precision_insertion/fixtures") / name
        raw_rel = obj_rel / "raw_mesh" / f"{name}.obj"
        frame_rel = obj_rel / "processed_data/info/frame_contract.json"
        collision_rel = obj_rel / "processed_data/mesh/static_collision.obj"
        _save(source / raw_rel, f"o {name}\nv 0 0 0\n")
        _json(source / frame_rel, {
            "T_socket_raw_mesh": identity, "rim_z_m": .055})
        _save(source / collision_rel, f"o {name}_collision\nv 0 0 0\n")
        _save(source / fixture_rel / "static_collision.obj",
              (source / collision_rel).read_text())
        _save(source / fixture_rel / "material.mtl", "newmtl fixture\n")
        _json(source / fixture_rel / "pose_measurement_asset.json",
              {"pose_object": name, "output_scope": "session_only"})
        _json(source / fixture_rel / "task_geometry.json", {
            "units": "m", "socket_pose_object": name,
            "key_object": mode.key_object,
            "socket_mesh": "static_collision.obj",
            "socket_pose_mesh": str(origin / raw_rel),
            "socket_rim_z_m": .055,
            "T_socket_pose_object": identity,
            "T_socket_key_entry": pose(.135),
            "T_socket_key_verification": pose(.115),
            "verification_insertion_depth_m": .02,
            "insertion_direction_socket": [0., 0., -1.],
            "key_frame": {"insertion_axis": [0., 0., 1.]},
        })
        _json(source / fixture_rel / "fixture_pose.template.json", {
            "calibrated": False, "T_robot_socket": None,
            "pose_object": name,
            "pose_object_mesh": str(origin / raw_rel),
            "pose_object_frame_contract": str(origin / frame_rel),
            "pose_estimator_asset": str(
                origin / "AutoDex/foundpose_assets" / name /
                "object_repre/v1" / name / "1/repre.pth"),
            "T_socket_pose_object": identity,
        })
    return source, origin, target


def _fixture(root: Path, gap: int) -> Path:
    return (root / "AutoDex/precision_insertion/fixtures" /
            select_mode("cylinder", gap).socket_object)


def test_dry_run_then_installs_six_rebound_fixtures(tmp_path):
    source, origin, target = _source(tmp_path)
    preview = rehydrate(
        source_shared_root=source, source_origin_shared_root=origin,
        target_shared_root=target)
    assert preview["installed"] is False
    assert preview["files_to_install"] == 48
    assert not target.exists()

    report = rehydrate(
        source_shared_root=source, source_origin_shared_root=origin,
        target_shared_root=target, install=True)
    assert report["installed"] is True
    assert report["robot_ready"] is False
    assert len(report["sockets"]) == 6
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        original = _fixture(source, gap)
        received = _fixture(target, gap)
        source_geometry = json.loads((original / "task_geometry.json").read_text())
        geometry = json.loads((received / "task_geometry.json").read_text())
        expected_mesh = (target / "object_processing" / mode.socket_object /
                         "raw_mesh" / f"{mode.socket_object}.obj")
        assert geometry.pop("socket_pose_mesh") == str(expected_mesh)
        source_geometry.pop("socket_pose_mesh")
        assert geometry == source_geometry  # including 20 mm insertion transforms
        validate_task_geometry(geometry | {"socket_pose_mesh": str(expected_mesh)}, mode)
        template = json.loads((received / "fixture_pose.template.json").read_text())
        assert template["pose_object_mesh"] == str(expected_mesh)
        assert template["calibrated"] is False
        assert not Path(template["pose_estimator_asset"]).exists()
        assert (received / "static_collision.obj").read_bytes() == (
            original / "static_collision.obj").read_bytes()
        assert validate_cylinder_socket_fixture(
            shared_root=target, mode=mode)["paths_bound_to_shared_root"] is True
        assert (target / "AutoDex/precision_insertion/cylindrical/"
                "socket_fixture_handoff_relocation/original_templates" /
                mode.socket_object / "task_geometry.json").read_bytes() == (
                    original / "task_geometry.json").read_bytes()
    with pytest.raises(FileExistsError, match="report already exists"):
        rehydrate(source_shared_root=source, source_origin_shared_root=origin,
                  target_shared_root=target, install=True)


def test_changed_recipient_fails_before_any_new_copy(tmp_path):
    source, origin, target = _source(tmp_path)
    last = select_mode("cylinder", 20).socket_object
    mismatch = target / "object_processing" / last / "raw_mesh" / f"{last}.obj"
    _save(mismatch, "different CAD")
    with pytest.raises(ValueError, match="recipient socket asset changed"):
        rehydrate(source_shared_root=source, source_origin_shared_root=origin,
                  target_shared_root=target, install=True)
    assert not _fixture(target, 1).exists()
    assert not (target / "AutoDex/precision_insertion/cylindrical/"
                "socket_fixture_handoff_relocation").exists()


def test_source_bound_template_requires_explicit_archive(tmp_path):
    source, origin, target = _source(tmp_path)
    stale = _fixture(target, 1) / "task_geometry.json"
    stale.parent.mkdir(parents=True)
    shutil.copyfile(_fixture(source, 1) / "task_geometry.json", stale)
    with pytest.raises(ValueError, match="recipient fixture differs"):
        rehydrate(source_shared_root=source, source_origin_shared_root=origin,
                  target_shared_root=target, install=True)
    result = rehydrate(
        source_shared_root=source, source_origin_shared_root=origin,
        target_shared_root=target, install=True,
        archive_source_bound_templates=True)
    assert result["templates_to_archive"] == 1
    archive = (target / "AutoDex/precision_insertion/cylindrical/"
               "socket_fixture_handoff_relocation/archived_source_bound" /
               select_mode("cylinder", 1).socket_object / "task_geometry.json")
    assert archive.read_bytes() == (_fixture(source, 1) /
                                    "task_geometry.json").read_bytes()
    assert json.loads(stale.read_text())["socket_pose_mesh"].startswith(str(target))


def test_unexpected_historical_path_or_geometry_is_rejected(tmp_path):
    source, origin, target = _source(tmp_path)
    task = _fixture(source, 3) / "task_geometry.json"
    geometry = json.loads(task.read_text())
    geometry["socket_pose_mesh"] = "/unrelated/object.obj"
    _json(task, geometry)
    with pytest.raises(ValueError, match="unexpected historical path"):
        rehydrate(source_shared_root=source, source_origin_shared_root=origin,
                  target_shared_root=target, install=True)
    assert not target.exists()


def test_stale_template_and_collision_mismatch_fail_fixture_gate(tmp_path):
    source, origin, target = _source(tmp_path)
    rehydrate(source_shared_root=source, source_origin_shared_root=origin,
              target_shared_root=target, install=True)
    mode = select_mode("cylinder", 20)
    template_path = _fixture(target, 20) / "fixture_pose.template.json"
    template = json.loads(template_path.read_text())
    template["pose_object_mesh"] = str(origin / "old_socket.obj")
    _json(template_path, template)
    with pytest.raises(ValueError, match="template is stale"):
        validate_cylinder_socket_fixture(shared_root=target, mode=mode)
    template["pose_object_mesh"] = str(
        target / "object_processing" / mode.socket_object /
        "raw_mesh" / f"{mode.socket_object}.obj")
    _json(template_path, template)
    _save(_fixture(target, 20) / "static_collision.obj", "wrong collision")
    with pytest.raises(ValueError, match="collision meshes differ"):
        validate_cylinder_socket_fixture(shared_root=target, mode=mode)
    shutil.copyfile(_fixture(source, 20) / "static_collision.obj",
                    _fixture(target, 20) / "static_collision.obj")
    task_path = _fixture(target, 20) / "task_geometry.json"
    task = json.loads(task_path.read_text())
    task["socket_pose_mesh"] = str(origin / "old_socket.obj")
    _json(task_path, task)
    with pytest.raises(ValueError, match="stale socket mesh path"):
        validate_cylinder_socket_fixture(shared_root=target, mode=mode)
