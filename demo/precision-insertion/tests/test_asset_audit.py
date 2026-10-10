"""Offline contracts for explicit v8 paths and read-only audit behavior."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import audit_assets  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def _file(path: Path, contents: str = "stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_exact_mode_ids_and_gaps():
    square = select_mode("square", 1.5)
    assert square.key_object == "precision_key_1p5mm"
    assert square.socket_object == "precision_socket_unified"
    assert square.yaw_relevant is True
    cylinder = select_mode("cylinder", 20)
    assert cylinder.key_object == "precision_key_cylinder_r15_h80"
    assert cylinder.socket_object == "precision_socket_cylinder_gap_20mm"
    assert cylinder.yaw_relevant is False
    with pytest.raises(ValueError, match="square gap"):
        select_mode("square", 0.1)
    with pytest.raises(ValueError, match="cylinder radial gap"):
        select_mode("cylinder", 0.5)


def test_generation_markers_are_not_foundpose_or_candidate_assets(tmp_path):
    mode = select_mode("square", 1.5)
    _file(tmp_path / "AutoDex" / "foundpose_assets" / mode.key_object /
          "GENERATION_REQUIRED.json", "{}")
    _file(tmp_path / "AutoDex" / "candidates" / "inspire" / "v8" /
          mode.key_object / "GENERATION_REQUIRED.json", "{}")
    report = audit_assets(tmp_path, mode)
    assert "key_foundpose_repre" in report["missing"]
    assert "inspire_v8_grasp_candidates" in report["missing"]
    assert report["counts"]["inspire_v8_grasp_candidates"] == 0
    assert report["robot_ready"] is False


def test_complete_file_set_still_cannot_claim_robot_ready(tmp_path):
    mode = select_mode("square", 1.5)
    key = mode.key_object
    socket = mode.socket_object
    for name in (key, socket):
        _file(tmp_path / "object_processing" / name / "raw_mesh" /
              f"{name}.obj")
        _file(tmp_path / "AutoDex" / "foundpose_assets" / name /
              "object_repre" / "v1" / name / "1" / "repre.pth")
    _file(tmp_path / "object_processing" / key / "processed_data" /
          "mesh" / "simplified.obj")
    _file(tmp_path / "object_processing" / socket / "processed_data" /
          "mesh" / "static_collision.obj")
    _file(tmp_path / "AutoDex" / "precision_insertion" / "fixtures" /
          "unified_socket" / "task_geometry.json", "{}")
    _file(tmp_path / "AutoDex" / "content" / "assets" / "robot" /
          "fr3_inspire_description" / "fr3_inspire.urdf")
    _file(tmp_path / "object_processing" / key / "processed_data" /
          "info" / "tabletop" / "000.npy")
    _file(tmp_path / "AutoDex" / "scene" / "inspire" / key /
          "table" / "0.json", "{}")
    candidate = (tmp_path / "AutoDex" / "candidates" / "inspire" /
                 "v8" / key / "table" / "0" / "84")
    for name in ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy"):
        _file(candidate / name)
    report = audit_assets(tmp_path, mode)
    assert report["missing"] == []
    assert report["file_inputs_present"] is True
    assert report["robot_ready"] is False
    assert report["counts"]["inspire_v8_grasp_candidates"] == 1


def test_cli_audit_is_read_only_and_reports_missing(tmp_path):
    runner = Path(__file__).resolve().parents[1] / "run_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "audit", "--shared-root", str(tmp_path),
         "--mode", "square", "--gap-mm", "1.5"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    report = json.loads(proc.stdout)
    assert "socket_foundpose_repre" in report["missing"]
    assert list(tmp_path.iterdir()) == []
