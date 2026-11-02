import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from precision_insertion.project_bundle import verify_project


PREPARE_PATH = Path(__file__).resolve().parents[1] / (
    "prepare_zerodex_nas_project.py")


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "precision_prepare_zerodex_nas_project", PREPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_repo(path: Path, payload: str) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "test"],
                   check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email",
                    "test@example.invalid"], check=True)
    (path / "source.txt").write_text(payload, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"],
                   check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "source"],
                   check=True)
    return path


def _record(root: Path, relative: str) -> dict:
    path = root / relative
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    payloads = {
        "assets/asset.bin": b"asset",
        "restore/package.tar": b"restore",
        "rigs/zerodex_capture4/rig.json": b"rig",
        "code/source.tar.gz": b"source",
        "environments/python.txt": b"packages",
        "install_project.py": b"installer",
        "README.md": b"readme",
        "development/README.md": b"development",
    }
    for relative, data in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    runtime = root / "runtime/active_shared_data"
    runtime.mkdir(parents=True)
    (runtime / "asset.bin").symlink_to("../../assets/asset.bin")
    rig_runtime = root / "rigs/zerodex_capture4/runtime"
    rig_runtime.symlink_to("../../assets")
    manifest = {
        "schema": "precision_insertion_reproducible_project_v2",
        "rig_runtime": "rigs/zerodex_capture4/runtime",
        "assets": [_record(root, "assets/asset.bin")],
        "asset_file_count": 1,
        "asset_packages": [{
            "path": "restore/package.tar",
            "extract_to": "AutoDex/foundpose_assets/socket",
        }],
        "restore_files": [_record(root, "restore/package.tar")],
        "rig_files": [
            _record(root, "rigs/zerodex_capture4/rig.json"),
            {
                "path": "rigs/zerodex_capture4/runtime",
                "kind": "symlink",
                "target": "../../assets",
            },
        ],
        "code": [{
            "repository": "source",
            "archive": "source.tar.gz",
            "commit": "abc",
            "sha256": _record(root, "code/source.tar.gz")["sha256"],
        }],
        "project_files": [
            _record(root, relative) for relative in (
                "environments/python.txt",
                "install_project.py",
                "README.md",
                "development/README.md",
            )
        ],
    }
    (root / "PROJECT_MANIFEST.json").write_text(json.dumps(manifest))
    return root


def test_verify_project_accepts_complete_v2_bundle(tmp_path):
    report = verify_project(
        nas_root=_bundle(tmp_path), verify_sha256=True, repo=None)

    assert report["declared_integrity_ok"] is True
    assert report["inventory_complete"] is True
    assert report["runtime_view"]["ok"] is True
    assert report["reproduction_ready"] is True


def test_verify_project_reports_untracked_and_changed_files(tmp_path):
    root = _bundle(tmp_path)
    (root / "assets/asset.bin").write_bytes(b"changed")
    (root / "code/untracked.tar.gz").write_bytes(b"extra")

    report = verify_project(
        nas_root=root, verify_sha256=False, repo=None)

    assert report["declared_integrity_ok"] is False
    assert report["inventory_complete"] is False
    assert report["declared_failures"][0]["status"] == "size_mismatch"
    assert report["untracked_managed_files"] == {
        "code": ["code/untracked.tar.gz"]}
    assert report["reproduction_ready"] is False


def test_verify_project_marks_v1_inventory_as_incomplete(tmp_path):
    root = _bundle(tmp_path)
    manifest_path = root / "PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "precision_insertion_reproducible_project_v1"
    manifest.pop("project_files")
    manifest_path.write_text(json.dumps(manifest))

    report = verify_project(
        nas_root=root, verify_sha256=False, repo=None)

    assert report["declared_integrity_ok"] is True
    assert report["project_files_manifested"] is False
    assert report["inventory_complete"] is False
    assert report["reproduction_ready"] is False


def test_verify_project_requires_renderer_source_for_zerodex_bundle(tmp_path):
    root = _bundle(tmp_path)
    manifest_path = root / "PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["project"] = "precision_insertion"
    manifest["rig"] = "zerodex_capture4"
    manifest_path.write_text(json.dumps(manifest))

    report = verify_project(
        nas_root=root, verify_sha256=False, repo=None)

    assert report["required_code_repositories"] == [
        "autodex-code", "nvdiffrast", "paradex"]
    assert report["missing_code_repositories"] == [
        "autodex-code", "nvdiffrast", "paradex"]
    assert report["declared_integrity_ok"] is False
    assert report["reproduction_ready"] is False


def test_refresh_code_upgrades_v1_inventory_and_archives_three_git_sources(
        tmp_path, monkeypatch):
    """The read-only NAS audit's exact v1 failure class is repairable.

    Runtime ParaDex may be a metadata-free import tree; refresh_code receives
    the separate Git source checkout and must produce a complete v2 handoff
    without recopying the runtime asset payload.
    """
    prepare = _load_prepare_module()
    sources = tmp_path / "sources"
    main = _git_repo(sources / "autodex-code", "main")
    paradex = _git_repo(sources / "paradex", "camera")
    nvdiffrast = _git_repo(sources / "nvdiffrast", "renderer")
    monkeypatch.setattr(prepare, "REPO", main)

    root = tmp_path / "project"
    asset = root / "assets/asset.bin"
    untracked = root / "assets/new.bin"
    restore = root / "restore/package.tar"
    rig = root / "rigs/zerodex_capture4/rig.json"
    environment = root / "environments/python.txt"
    old_archive = root / "code/old_autodex.tar.gz"
    for path, payload in (
            (asset, b"current-asset"), (untracked, b"new"),
            (restore, b"restore"), (rig, b"rig"),
            (environment, b"environment"), (old_archive, b"old")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    runtime = root / "runtime/active_shared_data"
    runtime.mkdir(parents=True)
    (runtime / "asset.bin").symlink_to("../../assets/asset.bin")
    (root / "rigs/zerodex_capture4/runtime").symlink_to("../../assets")
    manifest = {
        "schema": "precision_insertion_reproducible_project_v1",
        "project": "precision_insertion",
        "rig": "zerodex_capture4",
        "source_shared_root": "/unused/source",
        "rig_runtime": "rigs/zerodex_capture4/runtime",
        # Deliberately stale and incomplete, matching the current NAS audit.
        "assets": [{"path": "assets/asset.bin", "bytes": 1,
                    "sha256": "stale"}],
        "asset_file_count": 1,
        "asset_packages": [{"path": "restore/package.tar",
                            "extract_to": "AutoDex/foundpose_assets/socket"}],
        "restore_files": [_record(root, "restore/package.tar")],
        "code": [{"repository": "autodex-code", "commit": "old",
                  "archive": old_archive.name,
                  "sha256": _record(root, "code/old_autodex.tar.gz")["sha256"]}],
    }
    (root / "PROJECT_MANIFEST.json").write_text(json.dumps(manifest))

    refreshed = prepare.refresh_code(
        nas_root=root, paradex_repo=paradex,
        nvdiffrast_repo=nvdiffrast)
    report = verify_project(nas_root=root, verify_sha256=True, repo=main)
    updated = json.loads((root / "PROJECT_MANIFEST.json").read_text())

    assert refreshed["asset_file_count"] == 2
    assert updated["schema"] == "precision_insertion_reproducible_project_v2"
    assert sorted(row["repository"] for row in updated["code"]) == [
        "autodex-code", "nvdiffrast", "paradex"]
    assert not old_archive.exists()
    assert report["reproduction_ready"] is True
    assert report["untracked_managed_files"] == {}
    assert report["declared_failures"] == []
