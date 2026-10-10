#!/usr/bin/env python3
"""Install cylinder socket assets from a handoff at a new shared-data root.

Only host-specific paths in the two fixture JSONs are rebound. CAD geometry,
numeric insertion transforms and all object-processing files are preserved.
The default CLI is a read-only preflight; --install makes non-overwriting
copies and writes a provenance report. No FoundPose representation is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.config import (  # noqa: E402
    CYLINDER_RADIAL_GAPS_MM, select_mode,
)
from precision_insertion.endpoint import validate_task_geometry  # noqa: E402
from precision_insertion.fixture_contract import (  # noqa: E402
    validate_cylinder_socket_fixture,
)


_FIXTURE_FILES = (
    "static_collision.obj", "material.mtl", "pose_measurement_asset.json",
    "task_geometry.json", "fixture_pose.template.json",
)
_REBOUND_FIELDS = {
    "task_geometry.json": ("socket_pose_mesh",),
    "fixture_pose.template.json": (
        "pose_object_mesh", "pose_object_frame_contract", "pose_estimator_asset"),
}
_REPORT_REL = Path(
    "AutoDex/precision_insertion/cylindrical/socket_fixture_handoff_relocation")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _socket_paths(root: Path, name: str) -> dict[str, Path]:
    obj = root / "object_processing" / name
    return {
        "mesh": obj / "raw_mesh" / f"{name}.obj",
        "frame_contract": obj / "processed_data/info/frame_contract.json",
        "foundpose": (root / "AutoDex/foundpose_assets" / name /
                      "object_repre/v1" / name / "1/repre.pth"),
        "collision": obj / "processed_data/mesh/static_collision.obj",
    }


def _rebound_json(
    *, source_file: Path, name: str, origin: Path, target: Path,
) -> dict:
    value = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"fixture JSON must be an object: {source_file}")
    fields = _REBOUND_FIELDS[source_file.name]
    source_paths = _socket_paths(origin, name)
    target_paths = _socket_paths(target, name)
    references = {
        "socket_pose_mesh": "mesh", "pose_object_mesh": "mesh",
        "pose_object_frame_contract": "frame_contract",
        "pose_estimator_asset": "foundpose",
    }
    for field in fields:
        kind = references[field]
        if value.get(field) != str(source_paths[kind]):
            raise ValueError(
                f"unexpected historical path in {source_file}: {field}")
        value[field] = str(target_paths[kind])
    return value


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"handoff must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError(f"handoff asset directory is empty: {root}")
    return files


def rehydrate(
    *, source_shared_root: Path, target_shared_root: Path,
    source_origin_shared_root: Path | None = None,
    install: bool = False, archive_source_bound_templates: bool = False,
) -> dict:
    """Preflight all six sockets before optionally installing any files.

    Existing identical files are kept. Existing *source-bound* fixture JSONs
    require an explicit archive option; any other disagreement fails closed.
    The report directory is a non-overwriting completion marker.
    """
    source = Path(source_shared_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    origin = (source if source_origin_shared_root is None else
              Path(source_origin_shared_root).expanduser().resolve())
    if source == target or not source.is_dir():
        raise ValueError("source must exist and differ from target shared root")
    report_dir = target / _REPORT_REL
    if report_dir.exists():
        raise FileExistsError(f"relocation report already exists: {report_dir}")

    writes: list[tuple[Path, bytes]] = []
    archives: list[tuple[Path, Path]] = []
    original_templates: list[tuple[Path, Path]] = []
    records = []
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        name = mode.socket_object
        object_src = source / "object_processing" / name
        object_dst = target / "object_processing" / name
        fixture_rel = Path("AutoDex/precision_insertion/fixtures") / name
        fixture_src = source / fixture_rel
        fixture_dst = target / fixture_rel
        paths = _socket_paths(source, name)
        for kind in ("mesh", "frame_contract", "collision"):
            if not paths[kind].is_file():
                raise FileNotFoundError(paths[kind])
        for filename in _FIXTURE_FILES:
            if (fixture_src / filename).is_symlink():
                raise ValueError(f"handoff must not contain symlinks: {fixture_src / filename}")
            if not (fixture_src / filename).is_file():
                raise FileNotFoundError(fixture_src / filename)
        if _sha(fixture_src / "static_collision.obj") != _sha(paths["collision"]):
            raise ValueError(f"fixture and object collision CAD differ: {name}")

        source_geometry = json.loads(
            (fixture_src / "task_geometry.json").read_text(encoding="utf-8"))
        validate_task_geometry(source_geometry, mode)
        if source_geometry.get("socket_mesh") != "static_collision.obj":
            raise ValueError(f"fixture collision path is not local: {name}")
        pose_template = json.loads(
            (fixture_src / "fixture_pose.template.json").read_text(
                encoding="utf-8"))
        if (pose_template.get("pose_object") != name or
                pose_template.get("calibrated") is not False or
                pose_template.get("T_robot_socket") is not None or
                pose_template.get("T_socket_pose_object") !=
                source_geometry.get("T_socket_pose_object")):
            raise ValueError(f"fixture template is not uncalibrated or frame-matched: {name}")
        measurement = json.loads(
            (fixture_src / "pose_measurement_asset.json").read_text(
                encoding="utf-8"))
        if measurement.get("pose_object") != name:
            raise ValueError(f"fixture measurement identity differs: {name}")

        source_hashes = {}
        for path in _regular_files(object_src):
            relative = path.relative_to(object_src)
            destination = object_dst / relative
            digest = _sha(path)
            source_hashes[str(Path("object_processing") / name / relative)] = digest
            if destination.exists():
                if not destination.is_file() or _sha(destination) != digest:
                    raise ValueError(f"recipient socket asset changed: {destination}")
            else:
                writes.append((destination, path.read_bytes()))

        for filename in _FIXTURE_FILES:
            path = fixture_src / filename
            destination = fixture_dst / filename
            source_hashes[str(fixture_rel / filename)] = _sha(path)
            if filename in _REBOUND_FIELDS:
                rebound = _rebound_json(
                    source_file=path, name=name, origin=origin, target=target)
                if filename == "task_geometry.json":
                    validate_task_geometry(rebound, mode)
                desired = _json_bytes(rebound)
                original = report_dir / "original_templates" / name / filename
                original_templates.append((path, original))
                if destination.exists():
                    if not destination.is_file():
                        raise ValueError(f"recipient fixture is not a file: {destination}")
                    existing = destination.read_bytes()
                    if existing == desired:
                        continue
                    if (archive_source_bound_templates and
                            existing == path.read_bytes()):
                        archive = (report_dir / "archived_source_bound" /
                                   name / filename)
                        archives.append((destination, archive))
                    else:
                        raise ValueError(
                            f"recipient fixture differs from relocated source: {destination}")
                writes.append((destination, desired))
            elif destination.exists():
                if not destination.is_file() or _sha(destination) != _sha(path):
                    raise ValueError(f"recipient fixture asset changed: {destination}")
            else:
                writes.append((destination, path.read_bytes()))
        records.append({
            "gap_mm": gap, "socket_object": name,
            "source_sha256": source_hashes,
            "target_task_geometry_sha256": hashlib.sha256(_json_bytes(
                _rebound_json(source_file=fixture_src / "task_geometry.json",
                              name=name, origin=origin, target=target))).hexdigest(),
            "foundpose_repre_present": _socket_paths(target, name)["foundpose"].is_file(),
        })

    report = {
        "schema": "precision_insertion_cylinder_socket_relocation_v1",
        "source_shared_root": str(source),
        "source_origin_shared_root": str(origin),
        "target_shared_root": str(target),
        "sockets": records,
        "files_to_install": len(writes),
        "templates_to_archive": len(archives),
        "installed": bool(install),
        "scope": "CAD_and_uncalibrated_socket_templates_only",
        "robot_ready": False,
        "remaining": (
            "FoundPose mesh-matched representations, live camera/hand-eye "
            "calibration and fixture repeatability are not established by this tool"),
    }
    if not install:
        return report

    # All content/path comparisons finish before the first target mutation.
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".socket_fixture_relocation.",
                                 dir=report_dir.parent))
    for path, original in original_templates:
        staged = work / original.relative_to(report_dir)
        staged.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, staged.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    for path, archive in archives:
        staged = work / archive.relative_to(report_dir)
        staged.parent.mkdir(parents=True, exist_ok=True)
        path.rename(staged)
    for path, data in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        validate_cylinder_socket_fixture(shared_root=target, mode=mode)
    with (work / "RELOCATION.json").open("xb") as stream:
        stream.write(_json_bytes(report))
    work.rename(report_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shared-root", type=Path, required=True)
    parser.add_argument("--source-origin-shared-root", type=Path,
                        help="root embedded in source fixture JSON; required "
                             "when an archive was extracted elsewhere")
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="copy verified files; default is read-only preflight")
    parser.add_argument("--archive-source-bound-templates", action="store_true",
                        help="archive only byte-identical old fixture JSONs before rebind")
    args = parser.parse_args()
    report = rehydrate(
        source_shared_root=args.source_shared_root,
        source_origin_shared_root=args.source_origin_shared_root,
        target_shared_root=args.target_shared_root,
        install=args.install,
        archive_source_bound_templates=args.archive_source_bound_templates)
    print(json.dumps({
        "installed": report["installed"], "socket_count": len(report["sockets"]),
        "files_to_install": report["files_to_install"],
        "templates_to_archive": report["templates_to_archive"],
        "robot_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
