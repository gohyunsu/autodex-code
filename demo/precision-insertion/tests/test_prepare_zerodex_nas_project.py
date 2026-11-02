import hashlib
import json
from pathlib import Path

from precision_insertion.project_bundle import verify_project


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
