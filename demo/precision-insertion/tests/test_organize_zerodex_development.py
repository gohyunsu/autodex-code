"""Legacy mutable data is relocated without invalidating saved paths."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from organize_zerodex_development import organize_runtime_mutables  # noqa: E402
from link_zerodex_nas_runtime import link_runtime  # noqa: E402


def test_runtime_mutables_move_to_development_with_relative_links(tmp_path):
    stage = tmp_path / "runtime/zerodex_capture4"
    stage.mkdir(parents=True)
    runs = stage / "runs"
    runs.mkdir()
    (runs / "launch.json").write_text("{}\n", encoding="utf-8")
    capture = stage / "socket_000_request_42"
    (capture / "images").mkdir(parents=True)
    (capture / "images/cam.png").write_bytes(b"pixels")
    development = tmp_path / "development/zerodex_capture4"

    first = organize_runtime_mutables(
        stage_root=stage, development_root=development)

    assert first["moved_count"] == 2
    assert runs.is_symlink() and not runs.readlink().is_absolute()
    assert capture.is_symlink() and not capture.readlink().is_absolute()
    assert (runs / "launch.json").is_file()
    assert (capture / "images/cam.png").read_bytes() == b"pixels"
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == (
        "precision_insertion_runtime_mutable_migration_v1")
    assert len(manifest["moved"]) == 2

    second = organize_runtime_mutables(
        stage_root=stage, development_root=development)
    assert second["moved_count"] == 0
    assert second["existing_count"] == 2


def test_runtime_linker_does_not_create_mutable_runs_in_asset_view(tmp_path):
    project = tmp_path / "precision_insertion"
    source = project / "assets/shared_data"
    source.mkdir(parents=True)
    (project / "PROJECT_MANIFEST.json").write_text(
        json.dumps({"source_shared_root": "/source/shared_data"}),
        encoding="utf-8")
    (source / "asset.json").write_text(
        '{"path": "/source/shared_data/object.obj"}\n', encoding="utf-8")

    report = link_runtime(project)

    runtime = Path(report["runtime"])
    assert json.loads((runtime / "asset.json").read_text(encoding="utf-8")) == {
        "path": str(runtime / "object.obj")}
    assert not (runtime / "AutoDex/precision_insertion/zerodex_capture4/runs").exists()
