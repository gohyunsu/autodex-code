"""Inventory and verify the standalone precision-insertion project bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


SUPPORTED_SCHEMAS = {
    "precision_insertion_reproducible_project_v1",
    "precision_insertion_reproducible_project_v2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_record(path: Path, root: Path) -> dict:
    record = {"path": str(path.relative_to(root))}
    if path.is_symlink():
        return {**record, "kind": "symlink", "target": os.readlink(path)}
    return {
        **record,
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_tree(directory: Path, root: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    return [
        inventory_record(path, root)
        for path in sorted(directory.rglob("*"))
        if path.is_file() or path.is_symlink()
    ]


def refresh_inventory(
    records: list[dict], directory: Path, root: Path,
) -> list[dict]:
    """Refresh changed/new records without rereading unchanged large files."""
    previous = {str(row.get("path")): row for row in records}
    refreshed = []
    if not directory.is_dir():
        return refreshed
    for path in sorted(directory.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = str(path.relative_to(root))
        old = previous.get(relative)
        if path.is_symlink():
            refreshed.append(inventory_record(path, root))
        elif (isinstance(old, dict) and old.get("kind", "file") == "file" and
              old.get("bytes") == path.stat().st_size and
              isinstance(old.get("sha256"), str)):
            refreshed.append({**old, "kind": "file"})
        else:
            refreshed.append(inventory_record(path, root))
    return refreshed


def rig_inventory(root: Path) -> list[dict]:
    return inventory_tree(root / "rigs" / "zerodex_capture4", root)


def project_inventory(root: Path) -> list[dict]:
    """Inventory reproduction metadata that is neither an asset nor code."""
    paths = [
        root / "install_project.py",
        root / "README.md",
        root / "development" / "README.md",
    ]
    environments = root / "environments"
    if environments.is_dir():
        paths.extend(path for path in sorted(environments.rglob("*"))
                     if path.is_file() and not path.is_symlink())
    return [inventory_record(path, root) for path in paths
            if path.is_file() or path.is_symlink()]


def _actual_files(root: Path, relative: str) -> set[str]:
    directory = root / relative
    if not directory.is_dir():
        return set()
    return {
        str(path.relative_to(root))
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def _actual_project_files(root: Path) -> set[str]:
    relative_paths = {
        "install_project.py",
        "README.md",
        "development/README.md",
    }
    relative_paths.update(_actual_files(root, "environments"))
    return {relative for relative in relative_paths
            if (root / relative).is_file() or (root / relative).is_symlink()}


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def verify_project(
    *,
    nas_root: Path,
    verify_sha256: bool = False,
    repo: Path | None = None,
) -> dict:
    """Read-only integrity and inventory audit for one project bundle.

    This is deliberately an operator audit, not a live manipulation gate.
    Metadata mode checks paths and declared sizes. SHA256 mode additionally
    rereads every declared byte and is intended for handoff verification.
    """
    root = nas_root.expanduser().resolve()
    manifest_path = root / "PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    groups: dict[str, list[dict]] = {
        "assets": list(manifest.get("assets", [])),
        "restore": list(manifest.get("restore_files", [])),
        "rig": list(manifest.get("rig_files", [])),
        "code": [
            {"path": f"code/{row['archive']}",
             "sha256": row.get("sha256")}
            for row in manifest.get("code", [])
        ],
        "project": list(manifest.get("project_files", [])),
    }
    rows: list[dict] = []
    declared_paths: list[str] = []
    for group, records in groups.items():
        for record in records:
            relative = str(record.get("path", ""))
            declared_paths.append(relative)
            path = root / relative
            kind = str(record.get("kind", "file"))
            row = {"group": group, "path": relative,
                   "kind": kind, "status": "ok"}
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                row["status"] = "invalid_relative_path"
            elif kind == "symlink":
                if not path.is_symlink():
                    row["status"] = "missing_symlink"
                elif os.readlink(path) != record.get("target"):
                    row["status"] = "symlink_target_mismatch"
                    row["target"] = os.readlink(path)
                    row["expected_target"] = record.get("target")
                else:
                    resolved = path.resolve(strict=False)
                    if resolved != root and root not in resolved.parents:
                        row["status"] = "symlink_outside_project"
                    elif not path.exists():
                        row["status"] = "broken_symlink"
            elif not path.is_file():
                row["status"] = "missing"
            elif path.is_symlink() and manifest.get("schema") != (
                    "precision_insertion_reproducible_project_v1"):
                row["status"] = "unexpected_symlink"
            else:
                size = path.stat().st_size
                row["bytes"] = size
                expected_size = record.get("bytes")
                if isinstance(expected_size, int) and size != expected_size:
                    row["status"] = "size_mismatch"
                    row["expected_bytes"] = expected_size
                if verify_sha256 and row["status"] == "ok":
                    expected_hash = record.get("sha256")
                    actual_hash = sha256_file(path)
                    row["sha256"] = actual_hash
                    if not isinstance(expected_hash, str):
                        row["status"] = "missing_declared_sha256"
                    elif actual_hash != expected_hash:
                        row["status"] = "sha256_mismatch"
                        row["expected_sha256"] = expected_hash
            rows.append(row)

    duplicates = sorted(
        path for path in set(declared_paths) if declared_paths.count(path) > 1)
    supported_schema = manifest.get("schema") in SUPPORTED_SCHEMAS
    required_code_repositories = (
        {"autodex-code", "paradex", "nvdiffrast"}
        if (manifest.get("project") == "precision_insertion" and
            manifest.get("rig") == "zerodex_capture4")
        else set()
    )
    declared_code_repositories = {
        str(row.get("repository", "")) for row in manifest.get("code", [])
    }
    missing_code_repositories = sorted(
        required_code_repositories - declared_code_repositories)
    declared_asset_count = manifest.get("asset_file_count")
    asset_count_matches = (
        not isinstance(declared_asset_count, int) or
        declared_asset_count == len(groups["assets"]))
    restore_paths = {str(row.get("path", "")) for row in groups["restore"]}
    package_reference_failures = []
    for package in manifest.get("asset_packages", []):
        relative = str(package.get("path", ""))
        extract_to = str(package.get("extract_to", ""))
        if relative not in restore_paths:
            package_reference_failures.append({
                "path": relative, "status": "package_not_in_restore_inventory"})
        if (not extract_to or extract_to.startswith("/") or
                ".." in Path(extract_to).parts):
            package_reference_failures.append({
                "path": relative, "extract_to": extract_to,
                "status": "invalid_extract_target"})

    actual_by_group = {
        "assets": _actual_files(root, "assets"),
        "restore": _actual_files(root, "restore"),
        "rig": _actual_files(root, "rigs/zerodex_capture4"),
        "code": _actual_files(root, "code"),
        "project": _actual_project_files(root),
    }
    declared_by_group = {
        group: {str(row.get("path", "")) for row in records}
        for group, records in groups.items()
    }
    untracked = {
        group: sorted(actual - declared_by_group[group])
        for group, actual in actual_by_group.items()
        if actual - declared_by_group[group]
    }
    missing_from_inventory = {
        group: sorted(declared_by_group[group] - actual)
        for group, actual in actual_by_group.items()
        if declared_by_group[group] - actual
    }

    runtime_root = root / "runtime" / "active_shared_data"
    runtime_symlinks = [
        path for path in runtime_root.rglob("*") if path.is_symlink()
    ] if runtime_root.is_dir() else []
    broken_runtime_links = [
        str(path.relative_to(root)) for path in runtime_symlinks
        if not path.exists()
    ]
    outside_runtime_links = []
    for path in runtime_symlinks:
        resolved = path.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            outside_runtime_links.append(str(path.relative_to(root)))

    rig_runtime = root / str(manifest.get(
        "rig_runtime", "rigs/zerodex_capture4/runtime"))
    rig_runtime_ok = rig_runtime.is_symlink() and rig_runtime.exists()
    if rig_runtime_ok:
        resolved = rig_runtime.resolve()
        rig_runtime_ok = resolved == root or root in resolved.parents

    declared_failures = [row for row in rows if row["status"] != "ok"]
    inventory_complete = (
        bool(manifest.get("project_files")) and
        not duplicates and not untracked and not missing_from_inventory)
    runtime_view_ok = (
        runtime_root.is_dir() and not broken_runtime_links and
        not outside_runtime_links and rig_runtime_ok)
    checkout = None
    if repo is not None and (repo / ".git").exists():
        checkout_commit = _git_commit(repo)
        archived = next((row for row in manifest.get("code", [])
                         if row.get("repository") == repo.name), None)
        checkout = {
            "repository": repo.name,
            "commit": checkout_commit,
            "archived_commit": (archived or {}).get("commit"),
            "current_commit_archived": (
                archived is not None and archived.get("commit") == checkout_commit),
        }
    integrity_ok = (
        supported_schema and not declared_failures and not duplicates and
        asset_count_matches and not package_reference_failures and
        not missing_code_repositories)
    reproduction_ready = (
        integrity_ok and inventory_complete and runtime_view_ok and
        (checkout is None or checkout["current_commit_archived"]))
    return {
        "schema": "precision_insertion_project_verification_v1",
        "project_root": str(root),
        "manifest": str(manifest_path),
        "manifest_schema": manifest.get("schema"),
        "manifest_schema_supported": supported_schema,
        "mode": "sha256" if verify_sha256 else "metadata",
        "declared_file_count": len(rows),
        "declared_asset_count": declared_asset_count,
        "asset_count_matches": asset_count_matches,
        "declared_failures": declared_failures,
        "duplicate_declared_paths": duplicates,
        "package_reference_failures": package_reference_failures,
        "required_code_repositories": sorted(required_code_repositories),
        "missing_code_repositories": missing_code_repositories,
        "untracked_managed_files": untracked,
        "missing_inventory_files": missing_from_inventory,
        "project_files_manifested": bool(manifest.get("project_files")),
        "inventory_complete": inventory_complete,
        "runtime_view": {
            "root": str(runtime_root),
            "symlink_count": len(runtime_symlinks),
            "broken_symlinks": broken_runtime_links,
            "outside_project_symlinks": outside_runtime_links,
            "rig_runtime_link_ok": rig_runtime_ok,
            "ok": runtime_view_ok,
        },
        "checkout": checkout,
        "declared_integrity_ok": integrity_ok,
        "reproduction_ready": reproduction_ready,
        "scope": "read_only_bundle_audit_not_live_runtime_gate",
    }
