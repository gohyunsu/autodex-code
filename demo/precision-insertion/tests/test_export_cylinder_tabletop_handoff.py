"""The NAS addendum copies only promoted grasps; catalogues stay audit-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_cylinder_tabletop_handoff as handoff  # noqa: E402
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM  # noqa: E402


def test_export_preserves_source_and_marks_catalogs_audit_only(tmp_path, monkeypatch):
    key = handoff.KEY
    source = tmp_path / "AutoDex/candidates/inspire/v8" / key
    identifiers = [f"table/0/{i}" for i in range(11)] + ["table/1/0"]
    for identifier in identifiers:
        directory = source / identifier
        directory.mkdir(parents=True)
        (directory / "wrist_se3.npy").write_bytes(identifier.encode())
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps({
        "schema": "precision_insertion_cylinder_tabletop_v8_promotion_v1",
        "robot_ready": False,
        "status": "offline_simulated_candidates_not_physical",
        "candidate_root": str(source), "count": 12,
        "selected_ids": identifiers,
    }))
    audit_dir = tmp_path / "AutoDex/precision_insertion/cylindrical"
    audit_dir.mkdir(parents=True)
    for gap in CYLINDER_RADIAL_GAPS_MM:
        path = audit_dir / handoff.CATALOG_PATTERN.format(gap=int(gap))
        path.write_text(json.dumps({
            "complete_scan": True, "eligible_count": 12,
            "robot_ready": False, "shared_root": str(tmp_path),
        }))

    def select(_catalog, *, expected_mode, tabletop_pose_stem):
        scene = "0" if tabletop_pose_stem == "000" else "1"
        return {"status": "candidates_available", "candidates": [
            {"key": identifier.split("/")} for identifier in identifiers
            if identifier.startswith(f"table/{scene}/")]}

    monkeypatch.setattr(handoff, "select_pose_candidates", select)
    target = tmp_path / "nas_handoff"
    result = handoff.export(
        shared_root=tmp_path, promotion_path=promotion_path,
        output_root=target, code_commit="abcdef123456")
    copied = (target / "payload/shared_data/AutoDex/candidates/inspire/v8" /
              key / "table/1/0/wrist_se3.npy")
    assert copied.read_bytes() == b"table/1/0"
    assert result["robot_ready"] is False
    assert result["catalogs_audit_only"] is True
    assert result["file_sha256"][str(copied.relative_to(target))] == (
        hashlib.sha256(copied.read_bytes()).hexdigest())
    assert "not** be used as live catalogues" in (
        target / "README.md").read_text()
    with pytest.raises(FileExistsError, match="handoff"):
        handoff.export(shared_root=tmp_path, promotion_path=promotion_path,
                       output_root=target, code_commit="abcdef123456")
