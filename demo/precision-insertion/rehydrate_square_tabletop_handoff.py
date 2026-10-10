#!/usr/bin/env python3
"""Safely relocate the verified square v8 bundle to another shared-data root.

The default is read-only.  --install only creates missing files; conflicting
recipient files are never overwritten.  Source simulation evidence remains in
the bundle, while recipient scene hashes are explicitly rebound in copies of
the seven promoted validation records.  This does not certify robot execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from export_square_tabletop_handoff import _sha, verify  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.endpoint import validate_task_geometry  # noqa: E402


KEY = "precision_key_1p5mm"
SOCKET = "precision_socket_unified"
SCENE_PARENT = Path("AutoDex/scene/inspire") / KEY / "table"
CANDIDATE_PARENT = Path("AutoDex/candidates/inspire/v8") / KEY
FIXTURE_PARENT = Path("AutoDex/precision_insertion/fixtures/unified_socket")
ROBOT_URDF = Path(
    "AutoDex/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_object(data: bytes, name: str) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {name}")
    return value


def _rebind(value: dict, path: tuple[str, ...], relative: Path, *,
            origin: Path, target: Path) -> None:
    node = value
    for key in path[:-1]:
        node = node[key]
        if not isinstance(node, dict):
            raise ValueError(f"not an object at {'.'.join(path[:-1])}")
    field = path[-1]
    if node.get(field) != str(origin / relative):
        raise ValueError(f"unexpected source path at {'.'.join(path)}")
    node[field] = str(target / relative)


def _has_unbound_origin(value, *, origin: Path, target: Path) -> bool:
    if isinstance(value, str):
        source_path = value == str(origin) or value.startswith(str(origin) + "/")
        recipient_path = value == str(target) or value.startswith(str(target) + "/")
        return source_path and not recipient_path
    if isinstance(value, dict):
        return any(_has_unbound_origin(item, origin=origin, target=target)
                   for item in value.values())
    if isinstance(value, list):
        return any(_has_unbound_origin(item, origin=origin, target=target)
                   for item in value)
    return False


def _relocate_json(relative: Path, source: bytes, *, origin: Path,
                   target: Path) -> bytes:
    if origin == target:
        return source
    value = _read_object(source, str(relative))
    key_root = Path("object_processing") / KEY
    socket_root = Path("object_processing") / SOCKET
    foundpose = (Path("AutoDex/foundpose_assets") / SOCKET /
                 "object_repre/v1" / SOCKET / "1/repre.pth")
    if relative.parent == SCENE_PARENT and relative.name in {
            f"{i}.json" for i in range(5)}:
        _rebind(value, ("scene", "mesh", "target", "file_path"),
                key_root / "processed_data/mesh/simplified.obj",
                origin=origin, target=target)
        _rebind(value, ("scene", "mesh", "target", "urdf_path"),
                key_root / "processed_data/urdf/coacd.urdf",
                origin=origin, target=target)
    elif relative == FIXTURE_PARENT / "task_geometry.json":
        _rebind(value, ("socket_pose_mesh",),
                socket_root / "raw_mesh" / f"{SOCKET}.obj",
                origin=origin, target=target)
        validate_task_geometry(value, select_mode("square", 1.5))
    elif relative == FIXTURE_PARENT / "fixture_pose.template.json":
        for path, subpath in (
            (("pose_object_mesh",), socket_root / "raw_mesh" /
             f"{SOCKET}.obj"),
            (("pose_object_frame_contract",), socket_root /
             "processed_data/info/frame_contract.json"),
            (("pose_estimator_asset",), foundpose),
        ):
            _rebind(value, path, subpath, origin=origin, target=target)
    elif relative == FIXTURE_PARENT / "pose_measurement_asset.json":
        for field, subpath in (
            ("object_root", socket_root),
            ("raw_mesh", socket_root / "raw_mesh" / f"{SOCKET}.obj"),
            ("processed_mesh", socket_root /
             "processed_data/mesh/simplified.obj"),
            ("static_collision_mesh", socket_root /
             "processed_data/mesh/static_collision.obj"),
            ("static_urdf", socket_root /
             "processed_data/urdf/socket_static_exact.urdf"),
            ("frame_contract", socket_root /
             "processed_data/info/frame_contract.json"),
            ("foundpose_representation", foundpose),
        ):
            _rebind(value, ("pose_object", field), subpath,
                    origin=origin, target=target)
    else:
        raise ValueError(f"not an approved relocatable JSON: {relative}")
    if _has_unbound_origin(value, origin=origin, target=target):
        raise ValueError(f"unbound source-root path remains in {relative}")
    return _json_bytes(value)


def _allowed(relative: Path) -> bool:
    parts = relative.parts
    return (
        (len(parts) >= 3 and parts[:2] == ("object_processing", KEY)) or
        (len(parts) >= 3 and parts[:2] == ("object_processing", SOCKET)) or
        (relative.parent == SCENE_PARENT and relative.name in {
            f"{i}.json" for i in range(5)}) or
        (len(parts) > len(CANDIDATE_PARENT.parts) and
         parts[:len(CANDIDATE_PARENT.parts)] == CANDIDATE_PARENT.parts) or
        (relative.parent == FIXTURE_PARENT and relative.name in {
            "task_geometry.json", "fixture_pose.template.json",
            "pose_measurement_asset.json", "socket_shared_bore_1p5.obj",
            "material.mtl"})
    )


def _check_target_path(root: Path, relative: Path) -> Path:
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"recipient symlink is not allowed: {current}")
        if (index < len(relative.parts) - 1 and current.exists() and
                not current.is_dir()):
            raise ValueError(f"recipient parent is not a directory: {current}")
    return current


def _create_complete_file(path: Path, content: bytes) -> None:
    """Link a complete temporary file into place without replacing a target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".precision_rehydrate.",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rehydrate(*, bundle_root: Path, target_shared_root: Path,
              install: bool = False) -> dict:
    """Verify the complete payload, preflight every target, then copy missing files."""
    bundle = Path(bundle_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    manifest = verify(bundle)
    origin_name = manifest.get("origin_shared_root")
    if not isinstance(origin_name, str) or not Path(origin_name).is_absolute():
        raise ValueError("handoff lacks an absolute origin shared-data root")
    origin = Path(origin_name)
    if not target.is_dir():
        raise FileNotFoundError(f"recipient shared-data root is missing: {target}")
    # The stock AutoDex content tree may itself be a read-only NAS symlink.
    # It is only read here; destination payload paths are checked separately.
    urdf = target / ROBOT_URDF
    if (not urdf.is_file() or
            _sha(urdf) != manifest.get("source_robot_urdf_sha256")):
        raise ValueError("recipient Franka/Inspire URDF differs from source")
    payload = bundle / "payload/shared_data"
    if not payload.is_dir():
        raise FileNotFoundError(payload)
    files = sorted(path for path in payload.rglob("*") if path.is_file())
    relatives = {path.relative_to(payload) for path in files}
    expected_scenes = {SCENE_PARENT / f"{i}.json" for i in range(5)}
    if not expected_scenes <= relatives or any(not _allowed(p) for p in relatives):
        raise ValueError("handoff payload has missing scenes or unexpected paths")
    collision = (payload / FIXTURE_PARENT / "socket_shared_bore_1p5.obj")
    object_collision = (payload / "object_processing" / SOCKET /
                        "processed_data/mesh/static_collision.obj")
    if (not collision.is_file() or not object_collision.is_file() or
            _sha(collision) != _sha(object_collision)):
        raise ValueError("square fixture and object collision meshes differ")
    source_geometry = _read_object(
        (payload / FIXTURE_PARENT / "task_geometry.json").read_bytes(),
        "source task geometry")
    validate_task_geometry(source_geometry, select_mode("square", 1.5))
    manifest_digest = _sha(bundle / "MANIFEST.json")
    scene4_source = (payload / SCENE_PARENT / "4.json").read_bytes()
    scene4 = _relocate_json(SCENE_PARENT / "4.json", scene4_source,
                            origin=origin, target=target)
    candidate_ids = manifest.get("candidate_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != 7:
        raise ValueError("square handoff must contain seven promoted candidates")
    for candidate in candidate_ids:
        parts = str(candidate).split("/")
        if len(parts) != 3 or parts[:2] != ["table", "4"]:
            raise ValueError(f"unexpected promoted candidate: {candidate}")
        if CANDIDATE_PARENT.joinpath(*parts, "simulation_validation.json") not in relatives:
            raise ValueError(f"missing promoted simulation validation: {candidate}")

    planned: list[tuple[Path, bytes]] = []
    same = 0
    for source in files:
        relative = source.relative_to(payload)
        destination = _check_target_path(target, relative)
        content = source.read_bytes()
        if relative in expected_scenes or relative in {
            FIXTURE_PARENT / "task_geometry.json",
            FIXTURE_PARENT / "fixture_pose.template.json",
            FIXTURE_PARENT / "pose_measurement_asset.json",
        }:
            content = _relocate_json(relative, content,
                                     origin=origin, target=target)
        elif (origin != target and relative.parent.parent ==
              CANDIDATE_PARENT / "table/4" and
              relative.name == "simulation_validation.json"):
            value = _read_object(content, str(relative))
            if (value.get("source_scene_sha256") != _sha_bytes(scene4_source)
                    or "relocation" in value):
                raise ValueError(f"candidate scene provenance differs: {relative}")
            value["source_scene_sha256"] = _sha_bytes(scene4)
            value["relocation"] = {
                "schema": "precision_insertion_square_scene_relocation_v1",
                "source_scene_sha256": _sha_bytes(scene4_source),
                "recipient_scene_sha256": _sha_bytes(scene4),
                "handoff_manifest_sha256": manifest_digest,
                "recipient_shared_root": str(target),
                "scope": "absolute_scene_paths_only_not_new_simulation",
            }
            content = _json_bytes(value)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                raise ValueError(f"recipient file conflicts with handoff: {destination}")
            same += 1
        else:
            planned.append((destination, content))
    report = {
        "schema": "precision_insertion_square_tabletop_relocation_v1",
        "origin_shared_root": str(origin),
        "recipient_shared_root": str(target),
        "handoff_manifest_sha256": manifest_digest,
        "payload_files": len(files), "identical_existing_files": same,
        "files_to_install": len(planned), "installed": bool(install),
        "robot_ready": False,
        "next_step": "rebuild full endpoint catalog at recipient root",
    }
    if not install:
        return report
    for path, content in planned:
        _create_complete_file(path, content)
    validate_task_geometry(
        _read_object((target / FIXTURE_PARENT / "task_geometry.json").read_bytes(),
                     "recipient task geometry"), select_mode("square", 1.5))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="copy missing files; default is read-only preflight")
    args = parser.parse_args()
    report = rehydrate(bundle_root=args.bundle_root,
                       target_shared_root=args.target_shared_root,
                       install=args.install)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
