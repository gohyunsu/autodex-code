"""The inventory must point at the catalogs actually installed for each mode."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_offline_asset_profile as profile  # noqa: E402

_catalog_path = profile._catalog_path
_reset_root = profile._reset_root


def test_square_catalog_names_cover_every_gap():
    root = Path("/example/shared_data")
    assert [_catalog_path(root, "square", gap).name for gap in
            (0.3, 0.5, 1.0, 1.5)] == [
                "square_0p3mm_20261011_collision_only.json",
                "square_0p5mm_20261011_collision_only.json",
                "square_1p0mm_20261011_collision_only.json",
                "square_1p5mm_20261011_collision_only.json",
            ]


def test_cylinder_catalog_uses_current_implementation():
    assert _catalog_path(Path("/example/shared_data"), "cylinder", 15).name == (
        "endpoint_catalog_full_length_collision_only_gap_15mm_20261011.json")


def test_reset_staging_roots_match_available_key_modes():
    root = Path("/example/shared_data")
    assert _reset_root(root, "square", 0.3) is None
    assert _reset_root(root, "square", 1.0) is None
    assert _reset_root(root, "square", 0.5).name == (
        "reorient_0p5_curated_5000_20261011")
    assert _reset_root(root, "square", 1.5).name == (
        "reorient_curated_handoff_20261011")
    assert _reset_root(root, "cylinder", 1) == _reset_root(
        root, "cylinder", 20)


def test_inventory_uses_current_catalogs_and_no_fidelity_cutoff(
        tmp_path, monkeypatch):
    selected = []

    def catalog(*, mode, shared_root, catalog_path):
        selected.append((mode.family, mode.gap_mm, catalog_path.name))
        return {"candidates": [{"eligible": True,
                                 "tabletop_pose_stem": "000"}]}

    def resets(*, shared_root, mode, candidate_root, **extra):
        assert not extra  # no arbitrary 5 mm / 10 degree acceptance screen
        return {"directed_pairs": [{
            "from_v8_pose": 0, "to_v8_pose": 1,
            "stable_reset_seed_counts_by_height_cm": {"12": 1}}]}

    monkeypatch.setattr(profile, "audit_assets", lambda *_: {})
    monkeypatch.setattr(profile, "preflight_endpoint_catalog_file", catalog)
    monkeypatch.setattr(profile, "audit_v8_reorient_assets", resets)
    result = profile.build_profile(tmp_path)
    assert result["schema"].endswith("_v2")
    assert len(selected) == 10
    assert all("collision_only" in name for _, _, name in selected)
    assert result["reorient"]["cylinder"]["fidelity_threshold_applied"] is False
    assert result["reorient"]["cylinder"]["directed_cells"][0][
        "mujoco_stable_12cm"] == 1
