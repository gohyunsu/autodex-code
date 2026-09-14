"""Executor package with lazy hardware/simulation imports.

Small policy modules (for example ``lift_policy``) must remain usable without
loading cuRobo, torch, or a robot SDK.  Keep the historical public executor
names while importing their heavyweight implementations only on demand.
"""

__all__ = ["SimExecutor", "RealExecutor"]


def __getattr__(name):
    if name == "SimExecutor":
        from .sim import SimExecutor
        return SimExecutor
    if name == "RealExecutor":
        from .real import RealExecutor
        return RealExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
