"""Demo-only guard for the known vendored cuRobo sample-count typo."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.curobo_compat import (  # noqa: E402
    install_curobo_ik_world_compat, install_curobo_sample_count_compat,
)


class _KnownTypoGraph:
    sample_pts = 17

    def _sample_pts(self, n_samples=None, bounded=False):
        if n_samples is None:
            n_sampels = self.sample_pts
        return n_samples, bounded


class _FixedGraph:
    sample_pts = 17

    def _sample_pts(self, n_samples=None, bounded=False):
        if n_samples is None:
            n_samples = self.sample_pts
        return n_samples, bounded


class _UnknownGraph:
    def _sample_pts(self, n_samples=None):
        return n_samples


class _WorldConfig:
    pass


class _Checker:
    def __init__(self):
        self.received = []

    def load_batch_collision_model(self, world):
        self.received.append(world)


class _KnownIK:
    def __init__(self):
        self.world_coll_checker = _Checker()

    def update_world(self, world):
        self.world_coll_checker.load_batch_collision_model(world)


class _UnknownIK:
    def update_world(self, world):
        return world


def _install_fake_graph(monkeypatch, cls):
    module = ModuleType("curobo.graph.graph_base")
    module.GraphPlanBase = cls
    monkeypatch.setitem(sys.modules, "curobo.graph.graph_base", module)


def _install_fake_ik(monkeypatch, cls):
    geometry = ModuleType("curobo.geom.types")
    geometry.WorldConfig = _WorldConfig
    solver = ModuleType("curobo.wrap.reacher.ik_solver")
    solver.IKSolver = cls
    monkeypatch.setitem(sys.modules, "curobo.geom.types", geometry)
    monkeypatch.setitem(sys.modules, "curobo.wrap.reacher.ik_solver", solver)


def test_known_vendor_typo_uses_default_and_preserves_explicit_count(monkeypatch):
    _install_fake_graph(monkeypatch, _KnownTypoGraph)
    original = _KnownTypoGraph._sample_pts
    try:
        assert install_curobo_sample_count_compat() is True
        graph = _KnownTypoGraph()
        assert graph._sample_pts() == (17, False)
        assert graph._sample_pts(5, bounded=True) == (5, True)
        assert install_curobo_sample_count_compat() is True
    finally:
        _KnownTypoGraph._sample_pts = original


def test_already_fixed_vendor_is_left_unchanged(monkeypatch):
    _install_fake_graph(monkeypatch, _FixedGraph)
    original = _FixedGraph._sample_pts
    assert install_curobo_sample_count_compat() is False
    assert _FixedGraph._sample_pts is original


def test_unknown_vendor_implementation_fails_closed(monkeypatch):
    _install_fake_graph(monkeypatch, _UnknownGraph)
    with pytest.raises(RuntimeError, match="unknown cuRobo"):
        install_curobo_sample_count_compat()


def test_ik_singleton_world_is_wrapped_for_existing_autodex_call(monkeypatch):
    _install_fake_ik(monkeypatch, _KnownIK)
    original = _KnownIK.update_world
    try:
        assert install_curobo_ik_world_compat() is True
        solver = _KnownIK()
        world = _WorldConfig()
        solver.update_world(world)
        solver.update_world([world])
        assert solver.world_coll_checker.received == [[world], [world]]
        assert install_curobo_ik_world_compat() is True
    finally:
        _KnownIK.update_world = original


def test_unknown_ik_world_update_fails_closed(monkeypatch):
    _install_fake_ik(monkeypatch, _UnknownIK)
    with pytest.raises(RuntimeError, match="unknown cuRobo"):
        install_curobo_ik_world_compat()
