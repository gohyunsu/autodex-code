"""Shared fake continuation for composition tests without a cuRobo GPU."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mock_axis_for_fake_planner(monkeypatch):
    def stroke(planner, start, reference, target, *, axis_robot,
               scene_cfg, expected_travel_m, step_m):
        del scene_cfg
        assert np.allclose(target[:3, 3] - reference[:3, 3],
                           expected_travel_m * axis_robot)
        count = int(np.ceil(expected_travel_m / step_m))
        path = [np.asarray(start).copy()]
        records = []
        for index in range(1, count + 1):
            if getattr(planner, "fail_query", None) == index + 1:
                records.append({"stage": "axial_waypoint", "index": index,
                                "success": False,
                                "failure_code": "mock_unreachable"})
                return SimpleNamespace(success=False, trajectory=None,
                                       step_records=records, validation={})
            end = np.asarray(start).copy()
            end[:3] = (reference[:3, 3] +
                       index / count * expected_travel_m * axis_robot)
            if getattr(planner, "hand_drift_query", None) == index + 1:
                end[7] += .01
            path.extend(np.linspace(path[-1], end, 5)[1:])
            records.append({"stage": "axial_waypoint", "index": index,
                            "success": True})
        return SimpleNamespace(success=True, trajectory=np.asarray(path),
                               step_records=records,
                               validation={"axis_robot": axis_robot.tolist()})
    monkeypatch.setattr("precision_insertion.preflight.plan_jacobian_axis_stroke",
                        stroke)


@pytest.fixture
def mock_curobo_compat_for_composition(monkeypatch):
    """Fake-planner tests exercise orchestration, not the installed cuRobo ABI."""
    for module in ("precision_insertion.trial_preflight",
                   "precision_insertion.measured_lift_preflight",
                   "precision_insertion.repose_transition"):
        monkeypatch.setattr(
            f"{module}.install_curobo_planner_compat", lambda: {})
