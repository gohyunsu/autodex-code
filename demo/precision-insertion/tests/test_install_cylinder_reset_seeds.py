"""Canonical reset installation keeps only source-verified eligible seeds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "install_cylinder_reset_seeds.py"
SPEC = importlib.util.spec_from_file_location("install_cylinder_reset_seeds", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def _fidelity(drift: float, tilt: float) -> dict:
    row = {"center_in_hand_displacement_m": drift,
           "symmetry_reduced_axis_tilt_deg": tilt}
    return {"end_squeeze": row, "end_gravity": row}


def test_installs_only_eligible_seed_without_overwriting(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    source = tmp_path / "staging" / "reset_12"
    key = installer.select_mode("cylinder", 20).key_object
    cell = source / key / "reorient_12" / "0_1"
    for name in ("191", "631"):
        seed = cell / name
        seed.mkdir(parents=True)
        (seed / "sim_eval.json").write_text(name, encoding="utf-8")

    def verify(*, seed_id, **_kwargs):
        return {"fidelity": _fidelity(
            0.002 if seed_id == "191" else 0.015,
            5.0 if seed_id == "191" else 30.0)}

    monkeypatch.setattr(installer, "verify_v8_reset_seed", verify)
    manifest = tmp_path / "installed.json"
    report = installer.install(
        shared_root=shared, source_reset_root=source, gap_mm=20,
        from_pose_stem=0, to_pose_stem=1,
        max_center_drift_m=0.005, max_axis_tilt_deg=10.0,
        output_manifest=manifest)
    assert [row["seed_id"] for row in report["installed"]] == ["191"]
    assert report["fidelity_rejected_seed_ids"] == ["631"]
    assert report["robot_ready"] is False
    assert json.loads(manifest.read_text()) == report
    assert Path(report["installed"][0]["canonical_path"]).is_dir()
    assert not (shared / "AutoDex/candidates/inspire/reset_12" / key /
                "reorient_12/0_1/631").exists()
    with pytest.raises(ValueError, match="new manifest"):
        installer.install(
            shared_root=shared, source_reset_root=source, gap_mm=20,
            from_pose_stem=0, to_pose_stem=1,
            max_center_drift_m=0.005, max_axis_tilt_deg=10.0,
            output_manifest=manifest)


def test_no_eligible_seed_does_not_create_canonical_asset(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    source = tmp_path / "staging" / "reset_12"
    key = installer.select_mode("cylinder", 20).key_object
    seed = source / key / "reorient_12/1_0/631"
    seed.mkdir(parents=True)
    (seed / "sim_eval.json").write_text("631", encoding="utf-8")
    monkeypatch.setattr(
        installer, "verify_v8_reset_seed",
        lambda **_kwargs: {"fidelity": _fidelity(0.015, 30.0)})
    with pytest.raises(ValueError, match="no fidelity-eligible"):
        installer.install(
            shared_root=shared, source_reset_root=source, gap_mm=20,
            from_pose_stem=1, to_pose_stem=0,
            max_center_drift_m=0.005, max_axis_tilt_deg=10.0,
            output_manifest=tmp_path / "rejected.json")
    assert not (shared / "AutoDex/candidates").exists()
