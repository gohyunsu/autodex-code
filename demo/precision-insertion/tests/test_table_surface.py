import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.table_surface import load_table_surface_z  # noqa: E402


def test_load_measured_table_surface(tmp_path):
    path = tmp_path / "table_plane.json"
    path.write_text(json.dumps({
        "schema": "precision_insertion_table_plane_v1",
        "table_surface_z_m": 0.0412,
    }), encoding="utf-8")
    assert load_table_surface_z(path) == pytest.approx(0.0412)


def test_table_surface_rejects_unrelated_json(tmp_path):
    path = tmp_path / "table_plane.json"
    path.write_text('{"table_surface_z_m": 0.04}', encoding="utf-8")
    with pytest.raises(ValueError, match="table-plane"):
        load_table_surface_z(path)
