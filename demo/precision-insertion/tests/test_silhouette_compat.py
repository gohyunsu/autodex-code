"""Solid-MTL fallback keeps exact mesh geometry without editing AutoDex."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.perception.silhouette import SilhouetteOptimizer  # noqa: E402
from precision_insertion import silhouette_compat as compat  # noqa: E402


def _solid_obj(tmp_path: Path) -> Path:
    (tmp_path / "material.mtl").write_text(
        "newmtl precision_part\nKd 0.05 0.22 0.85\nd 1.0\n")
    path = tmp_path / "key.obj"
    path.write_text(
        "mtllib material.mtl\nusemtl precision_part\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


def test_solid_mtl_becomes_vertex_colour_without_mesh_change(tmp_path):
    path = _solid_obj(tmp_path)
    baseline = SilhouetteOptimizer._load_mesh(object.__new__(
        SilhouetteOptimizer), str(path))
    assert isinstance(baseline.visual, trimesh.visual.TextureVisuals)
    assert baseline.visual.material.image is None
    compatible = compat.SolidMtlSilhouetteOptimizer._load_mesh(
        object.__new__(compat.SolidMtlSilhouetteOptimizer), str(path))
    assert isinstance(compatible.visual, trimesh.visual.ColorVisuals)
    assert np.array_equal(compatible.vertices, baseline.vertices)
    assert np.array_equal(compatible.faces, baseline.faces)
    assert compatible.visual.vertex_colors.shape == (len(baseline.vertices), 4)


def test_compatible_renderer_cached_only_for_identical_raw_mesh(tmp_path, monkeypatch):
    raw = _solid_obj(tmp_path)
    constructed = []

    class FakeRenderer:
        def __init__(self, mesh_path, device):
            constructed.append((mesh_path, device))
            self.device = device

    monkeypatch.setattr(compat, "SolidMtlSilhouetteOptimizer", FakeRenderer)
    stock = SimpleNamespace(device="cuda:0", _sil=None)
    proxy = SimpleNamespace(stock=stock)
    first = compat.prepare_key_silhouette(
        init_orchestrator=proxy, object_name="key", raw_mesh=raw)
    assert compat.prepare_key_silhouette(
        init_orchestrator=proxy, object_name="key", raw_mesh=raw) == first
    assert len(constructed) == 1
    raw.write_text(raw.read_text() + "# changed\n")
    second = compat.prepare_key_silhouette(
        init_orchestrator=proxy, object_name="key", raw_mesh=raw)
    assert second != first
    assert len(constructed) == 2


def test_missing_mesh_or_renderer_device_fails_before_install(tmp_path):
    stock = SimpleNamespace(device=None, _sil=None)
    with pytest.raises(FileNotFoundError):
        compat.prepare_key_silhouette(
            init_orchestrator=stock, object_name="key",
            raw_mesh=tmp_path / "absent.obj")
    raw = _solid_obj(tmp_path)
    with pytest.raises(ValueError, match="renderer device"):
        compat.prepare_key_silhouette(
            init_orchestrator=stock, object_name="key", raw_mesh=raw)
    assert stock._sil is None
