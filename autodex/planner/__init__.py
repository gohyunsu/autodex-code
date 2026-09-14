"""Planning APIs with a CPU-safe differential-IK submodule import path."""
from __future__ import annotations

from .jacobian_stroke import JacobianStrokeOptions, JacobianStrokeResult

_MOTION_GEN_EXPORTS = {
    "ConstrainedPlanResult", "CudaPlanningFault", "GraspPlanner",
    "LiftPreflight", "PlanResult", "raise_cuda_planning_fault",
}

__all__ = sorted(_MOTION_GEN_EXPORTS | {
    "JacobianStrokeOptions", "JacobianStrokeResult",
})


def __getattr__(name: str):
    """Import cuRobo only when a MotionGen-backed symbol is requested."""
    if name not in _MOTION_GEN_EXPORTS:
        raise AttributeError(name)
    import importlib

    _planner = importlib.import_module(".planner", __name__)
    value = getattr(_planner, name)
    globals()[name] = value
    return value
