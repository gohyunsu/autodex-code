#!/usr/bin/env python3
"""Materialize the ZeroDex capture4 precision insertion project on NAS.

The bundle contains tracked source snapshots and the runtime data tree needed
by the six cylindrical socket modes. Its shared_data layout mirrors the
pipeline's AssetPaths contract; run artifacts and videos live outside assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GAPS = (1, 3, 5, 10, 15, 20)
KEY = "precision_key_cylinder_r15_h80"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _copy_tree(source: Path, destination: Path, *, link_dest: Path | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--no-owner", "--no-group", "--no-perms"]
    if link_dest is not None and link_dest.is_dir():
        command.append(f"--link-dest={link_dest}")
    subprocess.run(command + [str(source) + "/", str(destination) + "/"],
                   check=True)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        subprocess.run(["rsync", "-a", "--no-owner", "--no-group",
                        "--no-perms", str(source), str(destination)],
                       check=True)
        return
    try:
        os.link(source, destination)
    except OSError:
        subprocess.run(["rsync", "-a", "--no-owner", "--no-group",
                        "--no-perms", str(source), str(destination)],
                       check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_archive(repo: Path, destination: Path) -> dict:
    commit = _git(repo, "rev-parse", "HEAD")
    archive = destination / f"{repo.name}_{commit[:12]}.tar.gz"
    if not archive.is_file():
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz",
                        "-o", str(archive), "HEAD"], check=True)
    return {"repository": repo.name, "commit": commit,
            "archive": archive.name, "sha256": _sha256(archive)}


def _rig_inventory(nas_root: Path) -> list[dict]:
    rig = nas_root / "rigs" / "zerodex_capture4"
    return [{"path": str(path.relative_to(nas_root)),
             "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(rig.rglob("*")) if path.is_file() and not path.is_symlink()]


def refresh_code(*, nas_root: Path, paradex_repo: Path) -> dict:
    """Update tracked code without retransferring large runtime assets."""
    manifest_path = nas_root / "PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Socket representation TARs are installation inputs. Keep them out of
    # the asset tree scanned by the live pipeline.
    old_packages = nas_root / "assets/packages/foundpose"
    restore_packages = nas_root / "restore/foundpose_packages"
    if old_packages.is_dir():
        restore_packages.parent.mkdir(parents=True, exist_ok=True)
        if restore_packages.exists():
            for package in old_packages.iterdir():
                package.rename(restore_packages / package.name)
            old_packages.rmdir()
        else:
            old_packages.rename(restore_packages)
        (nas_root / "assets/packages").rmdir()
    migrated = list(manifest.get("restore_files", []))
    for row in manifest["assets"]:
        if row["path"].startswith("assets/packages/foundpose/"):
            migrated.append({**row, "path": row["path"].replace(
                "assets/packages/foundpose/", "restore/foundpose_packages/", 1)})
    manifest["restore_files"] = migrated
    manifest["asset_packages"] = [
        {**row, "path": row["path"].replace(
            "assets/packages/foundpose/", "restore/foundpose_packages/", 1)}
        for row in manifest["asset_packages"]]
    old_archives = {row["archive"] for row in manifest["code"]}
    code = nas_root / "code"
    snapshots = [_source_archive(REPO, code),
                 _source_archive(paradex_repo.expanduser().resolve(), code)]
    manifest["code"] = snapshots
    manifest["rig_files"] = _rig_inventory(nas_root)
    foundation = nas_root / "assets/capture4_foundationpose"
    asset_updates = [
        nas_root / "assets/shared_data/AutoDex/foundpose_assets"
        / KEY / name for name in ("summary.json", "reference_intrinsics.json")]
    socket_root = nas_root / "assets/shared_data/AutoDex/foundpose_assets"
    socket_prefixes = tuple(
        f"assets/shared_data/AutoDex/foundpose_assets/precision_socket_cylinder_gap_{gap:02d}mm/"
        for gap in GAPS)
    for gap in GAPS:
        extracted = socket_root / f"precision_socket_cylinder_gap_{gap:02d}mm"
        if extracted.is_dir():
            asset_updates.extend(path for path in sorted(extracted.rglob("*"))
                                 if path.is_file())
    if foundation.is_dir():
        asset_updates.extend(path for path in sorted(foundation.rglob("*"))
                             if path.is_file())
    updates = {str(path.relative_to(nas_root)):
               {"path": str(path.relative_to(nas_root)),
                "bytes": path.stat().st_size, "sha256": _sha256(path)}
               for path in asset_updates if path.is_file()}
    manifest["assets"] = sorted(
        {**{row["path"]: row for row in manifest["assets"]
             if not row["path"].startswith("assets/packages/foundpose/")
             and not row["path"].startswith(socket_prefixes)},
         **updates}.values(),
        key=lambda row: row["path"])
    manifest["asset_file_count"] = len(manifest["assets"])
    _copy_file(HERE / "install_zerodex_nas_project.py",
               nas_root / "install_project.py")
    _copy_file(HERE / "ZERODEX_NAS_PROJECT.md", nas_root / "README.md")
    _copy_file(HERE / "DEVELOPMENT_ARTIFACTS.md",
               nas_root / "development" / "README.md")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    for name in old_archives - {row["archive"] for row in snapshots}:
        (code / name).unlink()
    return {"root": str(nas_root), "code": snapshots,
            "asset_file_count": manifest["asset_file_count"]}


def _package_socket_representations(shared: Path, nas_root: Path) -> list[dict]:
    """Send many small representation files as one file per socket."""
    packages = []
    package_dir = nas_root / "restore" / "foundpose_packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    for gap in GAPS:
        name = f"precision_socket_cylinder_gap_{gap:02d}mm"
        source = shared / "AutoDex" / "foundpose_assets" / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = package_dir / f"{name}.tar"
        print(f"[bundle] packed FoundPose {name}", flush=True)
        with tempfile.TemporaryDirectory(prefix="precision_foundpose_") as temp:
            temporary_tar = Path(temp) / destination.name
            subprocess.run(["tar", "-C", str(source), "-cf",
                            str(temporary_tar), "."], check=True)
            _copy_file(temporary_tar, destination)
        packages.append({
            "path": str(destination.relative_to(nas_root)),
            "extract_to": f"AutoDex/foundpose_assets/{name}",
        })
    return packages


def build(*, source_shared: Path, nas_root: Path, paradex_repo: Path,
          source_stage: Path | None, capture4_host: str | None = None,
          ssh_port: int = 77,
          capture4_source_archive: Path | None = None,
          paradex_system: Path | None = None) -> dict:
    assets = nas_root / "assets" / "shared_data"
    code = nas_root / "code"
    envs = nas_root / "environments"
    for path in (assets, code, envs, nas_root / "runs" / "zerodex_capture4"):
        path.mkdir(parents=True, exist_ok=True)
    rig = nas_root / "rigs" / "zerodex_capture4"
    rig.mkdir(parents=True, exist_ok=True)
    archive = (capture4_source_archive or
               Path.home() / "precision_insertion_project/restore/capture4_source_20261028.tar.gz")
    system = (paradex_system or
              Path.home() / "precision_insertion_project/vendor/paradex/system/current")
    _copy_file(archive, rig / "capture4_source_20261028.tar.gz")
    _copy_tree(system, rig / "main_pc_system/current")
    runtime_link = rig / "runtime"
    if not runtime_link.is_symlink() and not runtime_link.exists():
        runtime_link.symlink_to(
            Path("../../assets/shared_data/AutoDex/precision_insertion/zerodex_capture4"))
    shared = source_shared.expanduser().resolve()
    source_stage = (source_stage.expanduser().resolve()
                    if source_stage is not None else None)
    relative_trees = [
        f"object_processing/{KEY}",
        f"object_processing/{KEY}_grip_proxy",
        f"AutoDex/candidates/inspire/v8/{KEY}",
        f"AutoDex/candidates/inspire/reset_12/{KEY}",
        f"AutoDex/scene/inspire/{KEY}",
        f"AutoDex/foundpose_assets/{KEY}",
        "AutoDex/models/Qwen3-VL-2B-Instruct",
        "AutoDex/precision_insertion/zerodex_capture4/calibration",
        "AutoDex/precision_insertion/zerodex_capture4/robot_deps",
    ]
    for gap in GAPS:
        name = f"precision_socket_cylinder_gap_{gap:02d}mm"
        relative_trees.extend((
            f"object_processing/{name}",
            f"AutoDex/precision_insertion/fixtures/{name}",
        ))
    for relative in relative_trees:
        print(f"[bundle] {relative}", flush=True)
        _copy_tree(shared / relative, assets / relative,
                   link_dest=(source_stage / relative
                              if source_stage is not None else None))
    # The restored daemon links against libfranka, not the incomplete
    # libmsgpack development symlink in the old dependency snapshot.
    for relative in (
        "AutoDex/precision_insertion/zerodex_capture4/robot_deps/prefix/usr/lib/x86_64-linux-gnu/libmsgpackc.so",
        "AutoDex/precision_insertion/zerodex_capture4/robot_deps/prefix/usr/share/doc/libmsgpack-dev/changelog.Debian.gz",
    ):
        (assets / relative).unlink(missing_ok=True)
    packages = _package_socket_representations(shared, nas_root)
    runtime = Path("AutoDex/precision_insertion/zerodex_capture4")
    runtime_files = (
        "base_scene.json", "charuco_board9.json", "clear_view_arm_q.json",
        "franka_runtime.json", "launch.json",
        *(f"recipient_gap{gap}_endpoint_catalog_straight.json" for gap in GAPS),
    )
    for name in runtime_files:
        _copy_file(shared / runtime / name, assets / runtime / name)
    _copy_file(shared / "AutoDex/weights/sam3/sam3.pt",
               assets / "AutoDex/weights/sam3/sam3.pt")
    snapshots = [_source_archive(REPO, code),
                 _source_archive(paradex_repo.expanduser().resolve(), code)]
    (envs / "precision-vlm.pip-freeze.txt").write_text(
        subprocess.check_output(
            [str(Path.home() / ".venvs/precision-vlm/bin/python"),
             "-m", "pip", "freeze"], text=True), encoding="utf-8")
    if capture4_host is not None:
        remote_cache = nas_root / "assets" / "capture4_cache"
        for source, relative in (
            ("/home/capture4/.cache/huggingface/hub/models--facebook--sam3/",
             "huggingface/hub/models--facebook--sam3"),
            ("/home/capture4/.cache/torch/hub/checkpoints/",
             "torch/hub/checkpoints"),
        ):
            target = remote_cache / relative
            target.mkdir(parents=True, exist_ok=True)
            includes = (["--include=dinov2_vits14_reg4_pretrain.pth",
                         "--include=dinov2_vitl14_pretrain.pth", "--exclude=*"]
                        if relative == "torch/hub/checkpoints" else [])
            subprocess.run(
                ["rsync", "-a", "--no-owner", "--no-group", "--no-perms", "-e",
                 f"ssh -p {ssh_port} -o BatchMode=yes", *includes,
                 f"{capture4_host}:{source}", str(target) + "/"],
                check=True)
        foundation = nas_root / "assets/capture4_foundationpose"
        foundation.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
             "-e", f"ssh -p {ssh_port} -o BatchMode=yes",
             f"{capture4_host}:/home/capture4/AutoDex/autodex/perception/thirdparty/weights/foundationpose/",
             str(foundation) + "/"], check=True)
        for name, python in (
            ("capture4-flir", "/home/capture4/anaconda3/envs/flir_env/bin/python"),
            ("capture4-gotrack", "/home/capture4/anaconda3/envs/gotrack_cu128/bin/python"),
        ):
            output = subprocess.check_output(
                ["ssh", "-p", str(ssh_port), "-o", "BatchMode=yes",
                 capture4_host, f"{python} -m pip freeze"], text=True)
            (envs / f"{name}.pip-freeze.txt").write_text(
                output, encoding="utf-8")
    # The standalone installer is usable before extracting either repository.
    _copy_file(HERE / "install_zerodex_nas_project.py",
               nas_root / "install_project.py")
    _copy_file(HERE / "ZERODEX_NAS_PROJECT.md", nas_root / "README.md")
    _copy_file(HERE / "DEVELOPMENT_ARTIFACTS.md",
               nas_root / "development" / "README.md")
    records = []
    for path in sorted((nas_root / "assets").rglob("*")):
        if path.is_file() and not path.is_symlink():
            records.append({"path": str(path.relative_to(nas_root)),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path)})
    manifest = {
        "schema": "precision_insertion_reproducible_project_v1",
        "project": "precision_insertion", "rig": "zerodex_capture4",
        "source_shared_root": str(shared),
        "cylindrical_gap_mm": list(GAPS),
        "rig_runtime": "rigs/zerodex_capture4/runtime",
        "rig_files": _rig_inventory(nas_root),
        "code": snapshots, "assets": records,
        "asset_packages": packages,
        "restore_files": [
            {"path": str(path.relative_to(nas_root)),
             "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted((nas_root / "restore").rglob("*"))
            if path.is_file()],
        "asset_file_count": len(records),
        "runtime_nas_upload": "separate_post_run_command_only",
    }
    (nas_root / "PROJECT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"root": str(nas_root), "asset_file_count": len(records),
            "code": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shared", type=Path,
                        default=Path.home() / "shared_data")
    parser.add_argument("--nas-root", type=Path,
                        default=Path(os.environ.get(
                            "PRECISION_NAS_PROJECT_ROOT",
                            "/mnt/paradex2/hyunsu/precision_insertion")))
    parser.add_argument("--paradex-repo", type=Path,
                        default=Path.home() / "paradex")
    parser.add_argument("--source-stage", type=Path,
                        default=(Path(os.environ["PRECISION_NAS_LINK_STAGE"])
                                 if "PRECISION_NAS_LINK_STAGE" in os.environ
                                 else None))
    parser.add_argument("--capture4-host",
                        default=os.environ.get("PRECISION_CAPTURE4_SSH_HOST",
                                               "capture4@147.46.219.136"))
    parser.add_argument("--ssh-port", type=int,
                        default=int(os.environ.get("PRECISION_CAPTURE4_SSH_PORT", "77")))
    parser.add_argument("--capture4-source-archive", type=Path,
                        help="local captured source snapshot for the namespaced capture4 adapter")
    parser.add_argument("--paradex-system", type=Path,
                        help="local project-owned ParaDex system/current directory")
    parser.add_argument("--refresh-code-only", action="store_true",
                        help="update committed source archives and README without recopying assets")
    args = parser.parse_args()
    if args.refresh_code_only:
        print(json.dumps(refresh_code(
            nas_root=args.nas_root,
            paradex_repo=args.paradex_repo), indent=2))
        return
    print(json.dumps(build(
        source_shared=args.source_shared, nas_root=args.nas_root,
        paradex_repo=args.paradex_repo, source_stage=args.source_stage,
        capture4_host=args.capture4_host, ssh_port=args.ssh_port,
        capture4_source_archive=args.capture4_source_archive,
        paradex_system=args.paradex_system), indent=2))


if __name__ == "__main__":
    main()
