"""The inventory must point at the catalogs actually installed for each mode."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_offline_asset_profile import _catalog_path  # noqa: E402


def test_square_catalog_names_cover_every_gap():
    root = Path("/example/shared_data")
    assert [_catalog_path(root, "square", gap).name for gap in
            (0.3, 0.5, 1.0, 1.5)] == [
                "square_0p3mm_20261011_nominal_r2.json",
                "square_0p5mm_20261011_nominal_r2.json",
                "square_1p0mm_20261011_nominal_r2.json",
                "square_1p5mm_20261011_v8_8_nominal_r2.json",
            ]
