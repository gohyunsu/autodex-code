"""System-dependent precision project roots are configured once."""

from pathlib import Path

from precision_insertion.project_layout import (
    capture4_project_root, map_main_project_path_to_capture4,
    nas_project_root, runtime_json_namespace_report,
    runtime_namespace_report,
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


def test_runtime_json_namespace_audits_nested_absolute_paths(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text(
        '{"shared":"/mnt/paradex2/hyunsu/precision_insertion/assets",'
        '"local":"/home/hyunsu/cache",'
        '"nested":[{"bad":"/mnt/paradex2/another_user/model"}]}',
        encoding="utf-8")
    report = runtime_json_namespace_report(
        {"launch_config": launch}, {
            "PRECISION_NAS_MOUNT_ROOT": "/mnt/paradex2",
            "PRECISION_ALLOWED_NAS_ROOT": "/mnt/paradex2/hyunsu",
        })
    assert report is not None
    assert report["absolute_path_references"] == 3
    assert report["unique_absolute_paths"] == 3
    assert report["references_by_source"] == {"launch_config": 3}
    assert report["all_nas_paths_within_allowed_root"] is False
    assert report["violations"][0]["field"] == "nested[0].bad"


def test_runtime_json_namespace_does_not_read_outside_source(tmp_path):
    mount = tmp_path / "nas"
    allowed = mount / "hyunsu"
    outside = mount / "another_user" / "launch.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("not valid JSON and must not be read", encoding="utf-8")
    report = runtime_json_namespace_report(
        {"launch_config": outside}, {
            "PRECISION_NAS_MOUNT_ROOT": str(mount),
            "PRECISION_ALLOWED_NAS_ROOT": str(allowed),
        })
    assert report is not None
    assert report["absolute_path_references"] == 0
    assert report["references_by_source"] == {"launch_config": 0}
    assert report["violations"][0]["field"] == "$source"
    assert report["violations"][0]["path"] == str(outside)
