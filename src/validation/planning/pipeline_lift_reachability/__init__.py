"""Production-path lift reachability validation.

The package keeps the historical IK-only reachability tools untouched.  Its
authoritative success bit comes from :meth:`autodex.planner.GraspPlanner.plan`;
the extra endpoint IK probes are diagnostic-only and never gate that result.
"""

from .analysis import analyze_run
from .core import CandidateCatalogue, evaluate_pose, load_candidate_catalogue

__all__ = [
    "CandidateCatalogue",
    "analyze_run",
    "evaluate_pose",
    "load_candidate_catalogue",
]
