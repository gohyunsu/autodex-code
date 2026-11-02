"""System-dependent precision project roots are configured once."""

from pathlib import Path

from precision_insertion.project_layout import (
    capture4_project_root, map_main_project_path_to_capture4,
    nas_project_root,
)


def test_environment_selects_both_machine_views(monkeypatch):
    monkeypatch.setenv("PRECISION_NAS_PROJECT_ROOT", "/main/custom")
    monkeypatch.setenv("PRECISION_CAPTURE4_PROJECT_ROOT", "/capture/custom")
    assert nas_project_root() == Path("/main/custom")
    assert capture4_project_root() == Path("/capture/custom")
    assert map_main_project_path_to_capture4(
        "/main/custom/runs/a.png") == Path("/capture/custom/runs/a.png")


def test_unrelated_path_is_not_rewritten():
    source = Path("/tmp/not-the-project/frame.png")
    assert map_main_project_path_to_capture4(
        source, main_root="/main/project",
        capture_root="/capture/project") == source
