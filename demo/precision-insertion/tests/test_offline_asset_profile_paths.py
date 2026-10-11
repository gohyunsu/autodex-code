"""The inventory must point at the catalogs actually installed for each mode."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_offline_asset_profile import _catalog_path, _reset_root  # noqa: E402


def test_square_catalog_names_cover_every_gap():
    root = Path("/example/shared_data")
    assert [_catalog_path(root, "square", gap).name for gap in
            (0.3, 0.5, 1.0, 1.5)] == [
                "square_0p3mm_20261011_fullkey1000_schema3.json",
                "square_0p5mm_20261011_fullkey1000_schema3.json",
                "square_1p0mm_20261011_fullkey1000_schema3.json",
                "square_1p5mm_20261011_fullkey1000_schema3.json",
            ]


def test_cylinder_catalog_uses_current_implementation():
    assert _catalog_path(Path("/example/shared_data"), "cylinder", 15).name == (
        "endpoint_catalog_current_gap_15mm_20261011_stock_schema3.json")


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
