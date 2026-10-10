"""Process-local compatibility for the checked-out vendored cuRobo graph.

The current ``GraphPlanBase._sample_pts`` misspells ``n_samples`` when it
fills the default from ``self.sample_pts``.  This sends ``(None,)`` to Torch
and stops the real AutoDex FR3 planner before any candidate is considered.
The original cuRobo and AutoDex files remain unchanged.  Install this narrow
adapter only inside the precision demo and reject unknown implementations.
"""

from __future__ import annotations

from functools import wraps
import inspect


def install_curobo_sample_count_compat() -> bool:
    """Return True when the known vendored typo was corrected in this process."""
    from curobo.graph.graph_base import GraphPlanBase

    original = GraphPlanBase._sample_pts
    if getattr(original, "_precision_sample_count_compat", False):
        return True
    source = inspect.getsource(original)
    if "n_sampels = self.sample_pts" not in source:
        if "n_samples = self.sample_pts" in source:
            return False  # a future cuRobo revision already fixed the typo
        raise RuntimeError("unknown cuRobo graph sample-count implementation")

    @wraps(original)
    def with_default_count(self, n_samples=None, *args, **kwargs):
        if n_samples is None:
            n_samples = self.sample_pts
        return original(self, n_samples, *args, **kwargs)

    with_default_count._precision_sample_count_compat = True
    GraphPlanBase._sample_pts = with_default_count
    return True


def install_curobo_ik_world_compat() -> bool:
    """Accept the singleton world sent by AutoDex's existing IK update calls.

    This vendored ``IKSolver.update_world`` expects a *list* of worlds, while
    AutoDex passes one ``WorldConfig`` on subsequent planning attempts.  The
    first attempt initializes IK instead, so the mismatch only appears later.
    Keep the conversion at the cuRobo boundary and leave both source trees
    unchanged.  Return False if a future cuRobo revision already accepts a
    singleton; refuse to patch an unrecognized implementation.
    """
    from curobo.geom.types import WorldConfig
    from curobo.wrap.reacher.ik_solver import IKSolver

    original = IKSolver.update_world
    if getattr(original, "_precision_singleton_world_compat", False):
        return True
    source = inspect.getsource(original)
    if "self.world_coll_checker.load_batch_collision_model(world)" not in source:
        raise RuntimeError("unknown cuRobo IK world-update implementation")
    if "isinstance(world, WorldConfig)" in source:
        return False

    @wraps(original)
    def with_singleton_list(self, world):
        if isinstance(world, WorldConfig):
            world = [world]
        return original(self, world)

    with_singleton_list._precision_singleton_world_compat = True
    IKSolver.update_world = with_singleton_list
    return True


def install_curobo_planner_compat() -> dict[str, bool]:
    """Install only the two known process-local v8/curobo compatibility fixes."""
    return {
        "sample_count": install_curobo_sample_count_compat(),
        "ik_singleton_world": install_curobo_ik_world_compat(),
    }
