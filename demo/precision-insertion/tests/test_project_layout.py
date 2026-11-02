"""System-dependent precision project roots are configured once."""

from pathlib import Path

from precision_insertion.project_layout import (
    capture4_project_root, map_main_project_path_to_capture4,
    nas_project_root, runtime_namespace_report,
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


def test_runtime_namespace_reports_only_nas_paths_outside_hyunsu():
    report = runtime_namespace_report({
        "PRECISION_NAS_MOUNT_ROOT": "/mnt/paradex2",
        "PRECISION_ALLOWED_NAS_ROOT": "/mnt/paradex2/hyunsu",
        "PRECISION_SHARED_ROOT": "/mnt/paradex2/hyunsu/precision_insertion/runtime",
        "PRECISION_RUN_OUTPUT_ROOT": "/mnt/paradex2/another_user/runs",
        "PRECISION_SAM_CHECKPOINT": "/home/hyunsu/cache/sam3.pt",
    })
    assert report is not None
    assert report["all_nas_paths_within_allowed_root"] is False
    assert [row["variable"] for row in report["violations"]] == [
        "PRECISION_RUN_OUTPUT_ROOT"]
    local = next(row for row in report["paths"]
                 if row["variable"] == "PRECISION_SAM_CHECKPOINT")
    assert local["storage"] == "local"
    assert local["within_allowed_nas_root"] is True
