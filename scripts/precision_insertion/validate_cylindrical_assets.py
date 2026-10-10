#!/usr/bin/env python3
"""Validate generated cylindrical AutoDex geometry and symmetry contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from build_cylindrical_assets import (
    KEY_HEIGHT_M,
    KEY_OBJECT,
    KEY_PROXY_OBJECT,
    PREINSERT_CLEARANCE_M,
    SOCKET_BORE_DEPTH_M,
    SOCKET_SPECS,
    VERIFICATION_DEPTH_M,
)


def validate(shared_root: Path, *, require_learned: bool = False) -> int:
    shared_root = shared_root.expanduser().resolve()
    object_root = shared_root / "object_processing"
    project = shared_root / "AutoDex"
    failures: list[str] = []
    blockers: list[str] = []
    manifest_path = project / "precision_insertion" / "cylindrical" / "asset_manifest.json"
    family_path = project / "precision_insertion" / "cylindrical" / "socket_family.json"
    if not manifest_path.is_file() or not family_path.is_file():
        failures.append("cylinder asset and socket-family manifests are required")
        manifest = {}
        family = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = json.loads(family_path.read_text(encoding="utf-8"))
        if manifest.get("key", {}).get("object") != KEY_OBJECT:
            failures.append("manifest key object ID mismatch")
        if family.get("key_object") != KEY_OBJECT:
            failures.append("socket family key object ID mismatch")
        if family.get("clearance_definition") != "radial one-sided nominal CAD gap":
            failures.append("socket gap naming must explicitly mean radial clearance")

    def verify_source(record: dict, expected_name: str) -> None:
        source = Path(record.get("source", ""))
        if not source.is_file():
            source = (Path(__file__).resolve().parents[2] / "assets" /
                      "precision_insertion" / "cylindrical_fixed_key" /
                      "source" / source.name)
        if record.get("object") != expected_name or not source.is_file():
            failures.append(f"{expected_name}: source manifest ID/path mismatch")
            return
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != record.get("source_sha256"):
            failures.append(f"{expected_name}: source STL SHA-256 mismatch")

    if manifest:
        verify_source(manifest.get("key", {}), KEY_OBJECT)

    key = object_root / KEY_OBJECT
    key_required = [
        key / "raw_mesh" / f"{KEY_OBJECT}.obj",
        key / "processed_data" / "mesh" / "simplified.obj",
        key / "processed_data" / "mesh" / "contact_allowed.obj",
        key / "processed_data" / "mesh" / "contact_forbidden.obj",
        key / "processed_data" / "info" / "symmetry.json",
        key / "processed_data" / "info" / "tabletop" / "000.npy",
        key / "processed_data" / "info" / "tabletop" / "001.npy",
        object_root / KEY_PROXY_OBJECT / "processed_data" / "mesh" / "simplified.obj",
    ]
    failures.extend(f"missing: {p}" for p in key_required if not p.is_file())
    key_mesh_path = key / "processed_data" / "mesh" / "simplified.obj"
    allowed_path = key / "processed_data" / "mesh" / "contact_allowed.obj"
    forbidden_path = key / "processed_data" / "mesh" / "contact_forbidden.obj"
    if all(path.is_file() for path in (key_mesh_path, allowed_path, forbidden_path)):
        key_mesh = trimesh.load(key_mesh_path, force="mesh", process=False)
        allowed_mesh = trimesh.load(allowed_path, force="mesh", process=False)
        forbidden_mesh = trimesh.load(forbidden_path, force="mesh", process=False)
        if not key_mesh.is_watertight:
            failures.append("processed full key mesh must be watertight")
        if len(allowed_mesh.faces) + len(forbidden_mesh.faces) != len(key_mesh.faces):
            failures.append("contact partition does not cover processed key faces")
        triangles = allowed_mesh.vertices[allowed_mesh.faces]
        side = np.abs(allowed_mesh.face_normals[:, 2]) < 0.05
        rear = allowed_mesh.face_normals[:, 2] < -0.95
        if not np.any(side) or not np.any(rear):
            failures.append("allowed contacts must contain both grip side and rear cap")
        if np.any(side & (triangles[:, :, 2].max(axis=1) > 0.025 + 1e-8)):
            failures.append("allowed side extends beyond 25 mm grip zone")
        if np.any(~(side | rear)):
            failures.append("allowed mesh contains a forbidden or artificial face")
        if not np.isclose(allowed_mesh.bounds[1, 2], 0.025):
            failures.append("allowed contact side must reach the 25 mm boundary")
        policy_path = key / "processed_data" / "info" / "contact_regions.json"
        if policy_path.is_file():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy["allowed"]["face_count"] != len(allowed_mesh.faces):
                failures.append("allowed contact face count mismatch")
            if policy["forbidden"]["face_count"] != len(forbidden_mesh.faces):
                failures.append("forbidden contact face count mismatch")
    proxy_info_path = (object_root / KEY_PROXY_OBJECT / "processed_data" /
                       "info" / "simplified.json")
    proxy_symmetry_path = (object_root / KEY_PROXY_OBJECT / "processed_data" /
                           "info" / "symmetry.json")
    if proxy_info_path.is_file():
        proxy_info = json.loads(proxy_info_path.read_text(encoding="utf-8"))
        if not np.allclose(proxy_info.get("obb", []), [0.03, 0.03, KEY_HEIGHT_M]):
            failures.append("proposal proxy OBB must retain full-key dimensions")
    if proxy_symmetry_path.is_file():
        proxy_symmetry = json.loads(proxy_symmetry_path.read_text(encoding="utf-8"))
        if proxy_symmetry.get("type") != "none" or proxy_symmetry.get("axes"):
            failures.append("proposal-only proxy must not declare full-key Dinf symmetry")
    for object_name in (KEY_OBJECT, KEY_PROXY_OBJECT):
        scene_root = project / "scene" / "inspire" / object_name / "table"
        for scene_id in (0, 1):
            scene_path = scene_root / f"{scene_id}.json"
            if not scene_path.is_file():
                failures.append(f"{object_name}: tabletop scene {scene_id} missing")
                continue
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            target = scene["scene"]["mesh"]["target"]
            if not Path(target["file_path"]).is_file() or not Path(target["urdf_path"]).is_file():
                failures.append(f"{object_name}: scene {scene_id} references missing mesh/URDF")
    tabletop_end = key / "processed_data" / "info" / "tabletop" / "000.npy"
    if tabletop_end.is_file():
        pose = np.load(tabletop_end)
        grasp_end_z = float((pose @ np.array([0.0, 0.0, 0.0, 1.0]))[2])
        insertion_end_z = float(
            (pose @ np.array([0.0, 0.0, KEY_HEIGHT_M, 1.0]))[2])
        if not (np.isclose(grasp_end_z, KEY_HEIGHT_M) and
                np.isclose(insertion_end_z, 0.0)):
            failures.append(
                "end-down tabletop pose must expose z=0 grasp end and place "
                "z=80 mm insertion end on the table"
            )
    tabletop_side = key / "processed_data" / "info" / "tabletop" / "001.npy"
    if tabletop_side.is_file() and key_mesh_path.is_file():
        side_pose = np.load(tabletop_side)
        key_mesh = trimesh.load(key_mesh_path, force="mesh", process=False)
        world_vertices = (side_pose[:3, :3] @ key_mesh.vertices.T).T + side_pose[:3, 3]
        if not np.isclose(world_vertices[:, 2].min(), 0.0, atol=1e-6):
            failures.append("side-down key tabletop pose must touch, not penetrate, table")
    symmetry_path = key / "processed_data" / "info" / "symmetry.json"
    if symmetry_path.is_file():
        symmetry = json.loads(symmetry_path.read_text(encoding="utf-8"))
        axes = symmetry.get("axes", [])
        if symmetry.get("type") != "Dinf":
            failures.append("key symmetry must be Dinf")
        if not any(a.get("axis") == [0.0, 0.0, 1.0] and a.get("fold") == "inf" for a in axes):
            failures.append("key must declare continuous local-z symmetry")
        if sum(a.get("fold") == 2 for a in axes) < 2:
            failures.append("key must declare end-exchange flip symmetry")

    for object_name, gap_mm, bore_radius_mm, _source in SOCKET_SPECS:
        if manifest:
            matching = [item for item in manifest.get("sockets", [])
                        if item.get("object") == object_name]
            if len(matching) != 1:
                failures.append(f"{object_name}: expected exactly one manifest entry")
            else:
                verify_source(matching[0], object_name)
                expected_fixture = (project / "precision_insertion" / "fixtures" /
                                    object_name / "task_geometry.json")
                if Path(matching[0].get("task_geometry", "")) != expected_fixture:
                    failures.append(f"{object_name}: manifest fixture path is not canonical")
                if matching[0].get("radial_clearance_mm") != gap_mm:
                    failures.append(f"{object_name}: manifest radial gap mismatch")
        root = object_root / object_name
        required = [
            root / "raw_mesh" / f"{object_name}.obj",
            root / "processed_data" / "mesh" / "simplified.obj",
            root / "processed_data" / "mesh" / "static_collision.obj",
            root / "processed_data" / "info" / "symmetry.json",
            root / "processed_data" / "info" / "frame_contract.json",
            project / "precision_insertion" / "fixtures" /
            object_name / "task_geometry.json",
            project / "precision_insertion" / "fixtures" /
            object_name / "fixture_pose.template.json",
            project / "precision_insertion" / "fixtures" /
            object_name / "pose_measurement_asset.json",
        ]
        failures.extend(f"missing: {p}" for p in required if not p.is_file())
        collision_path = required[2]
        if collision_path.is_file():
            collision = trimesh.load(collision_path, force="mesh", process=False)
            outer_radius = (bore_radius_mm + 5) * 1e-3
            inner_radius = bore_radius_mm * 1e-3
            expected_volume = np.pi * (0.060**2 * 0.005 +
                                       (outer_radius**2 - inner_radius**2) * 0.050)
            if not collision.is_watertight or not np.isclose(
                    collision.volume, expected_volume, rtol=2e-4):
                failures.append(f"{object_name}: static collision mesh does not preserve open bore")
        sym_path = root / "processed_data" / "info" / "symmetry.json"
        if sym_path.is_file():
            sym = json.loads(sym_path.read_text(encoding="utf-8"))
            if sym.get("type") != "Cinf" or len(sym.get("axes", [])) != 1:
                failures.append(f"{object_name}: socket symmetry must be Cinf only")
        task_path = required[-3]
        if task_path.is_file():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("schema_version") != 2:
                failures.append(f"{object_name}: task geometry schema must match square fixture")
            if task.get("radial_clearance_mm") != gap_mm:
                failures.append(f"{object_name}: radial clearance mismatch")
            if task.get("socket_bore_radius_m") != bore_radius_mm * 1e-3:
                failures.append(f"{object_name}: bore radius mismatch")
            entry = np.asarray(task["T_socket_key_entry"], dtype=float)
            verify = np.asarray(task["T_socket_key_verification"], dtype=float)
            preinsert = np.asarray(task["T_socket_key_preinsert"], dtype=float)
            seated = np.asarray(task["T_socket_key_seated"], dtype=float)
            if not np.isclose(entry[2, 3] - verify[2, 3], VERIFICATION_DEPTH_M):
                failures.append(f"{object_name}: verification stroke is not 20 mm")
            if not np.isclose(entry[2, 3] - seated[2, 3], SOCKET_BORE_DEPTH_M):
                failures.append(f"{object_name}: seated stroke does not equal bore depth")
            if not np.isclose(preinsert[2, 3] - entry[2, 3], PREINSERT_CLEARANCE_M):
                failures.append(f"{object_name}: pre-insertion clearance mismatch")
            tip_entry = (entry @ np.array([0.0, 0.0, KEY_HEIGHT_M, 1.0]))[2]
            if not np.isclose(tip_entry, 0.055):
                failures.append(f"{object_name}: key tip is not on the rim at entry")
        fixture_template_path = required[-2]
        if fixture_template_path.is_file():
            template = json.loads(fixture_template_path.read_text(encoding="utf-8"))
            if template.get("calibrated") is not False or template.get("T_robot_socket") is not None:
                failures.append(f"{object_name}: fixture template must not claim calibration")

        repre = (
            project / "foundpose_assets" / object_name / "object_repre" /
            "v1" / object_name / "1" / "repre.pth"
        )
        if not repre.is_file():
            blockers.append(f"{object_name}: FoundPose repre.pth required")

    key_repre = (
        project / "foundpose_assets" / KEY_OBJECT / "object_repre" /
        "v1" / KEY_OBJECT / "1" / "repre.pth"
    )
    if not key_repre.is_file():
        blockers.append(f"{KEY_OBJECT}: FoundPose repre.pth required")
    candidate_root = project / "candidates" / "inspire" / "v8" / KEY_OBJECT
    if not any(candidate_root.rglob("wrist_se3.npy")):
        blockers.append(f"{KEY_OBJECT}: no generated Inspire grasp candidates")
    if manifest and len(manifest.get("sockets", [])) != len(SOCKET_SPECS):
        failures.append("manifest must list exactly six cylindrical sockets")
    if family and len(family.get("sockets", [])) != len(SOCKET_SPECS):
        failures.append("socket family must list exactly six cylindrical sockets")
    if family:
        family_by_name = {item.get("object"): item
                          for item in family.get("sockets", [])}
        for object_name, gap_mm, _bore_radius_mm, _source in SOCKET_SPECS:
            row = family_by_name.get(object_name)
            expected = (project / "precision_insertion" / "fixtures" /
                        object_name / "task_geometry.json")
            if row is None or row.get("radial_clearance_mm") != gap_mm or Path(
                    row.get("task_geometry", "")) != expected:
                failures.append(f"{object_name}: socket family path/gap mismatch")

    if failures:
        print("CYLINDRICAL ASSET VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("CYLINDRICAL GEOMETRY VALIDATION PASSED")
    for blocker in blockers:
        print(f"BLOCKED RUNTIME: {blocker}")
    return 1 if require_learned and blockers else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--require-learned", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.shared_root, require_learned=args.require_learned))


if __name__ == "__main__":
    main()
