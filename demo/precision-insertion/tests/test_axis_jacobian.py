"""Tilted socket-axis continuation, including the dense execution audit."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import axis_jacobian  # noqa: E402


class _Planner:
    _n_arm = 7
    _init_state = np.zeros(13)

    def __init__(self):
        self.world = None

    def _set_motion_world(self, world):
        self.world = world

    def fk_wrist(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        return pose


@pytest.fixture
def solver_stubs(monkeypatch):
    def checked(_planner, q):
        values = np.asarray(q)
        return (np.ones(len(values), dtype=bool), None,
                {"sample_count": len(values)})

    def fk_batch(planner, q):
        return np.stack([planner.fk_wrist(row) for row in q])

    monkeypatch.setattr(axis_jacobian, "_fixed_world_cfg",
                        lambda scene: {"mesh": {
                            key: value for key, value in scene["mesh"].items()
                            if key != "target"}})
    monkeypatch.setattr(axis_jacobian, "_check_state", lambda _p, _q: (True, None))
    monkeypatch.setattr(axis_jacobian, "_check_states_batch", checked)
    monkeypatch.setattr(axis_jacobian, "_fk_batch", fk_batch)
    monkeypatch.setattr(axis_jacobian, "_joint_bounds",
                        lambda _p, _n: (np.full(13, -1.), np.full(13, 1.)))
    monkeypatch.setattr(axis_jacobian, "_retime_c2",
                        lambda nodes, **_kw: (
                            np.asarray(nodes, dtype=np.float32),
                            np.arange(len(nodes), dtype=float) * .01,
                            {"method": "test_dense_reference"}))


def _request():
    planner = _Planner()
    start = np.zeros(13)
    start[2] = .30
    start[7:] = .2
    reference = planner.fk_wrist(start)
    axis = np.array([.6, 0., -.8])
    target = reference.copy()
    target[:3, 3] += .05 * axis
    return planner, start, reference, target, axis


def test_tilted_axis_keeps_continuous_branch_and_fixed_hand(solver_stubs):
    planner, start, reference, target, axis = _request()
    result = axis_jacobian.plan_jacobian_axis_stroke(
        planner, start, reference, target, axis_robot=axis,
        scene_cfg={"mesh": {"fixture_socket": {}, "target": {}}},
        expected_travel_m=.05, step_m=.005)
    assert result.success and len(result.step_records) == 10
    assert planner.world == {"mesh": {"fixture_socket": {}}}
    assert all(row["success"] for row in result.step_records)
    assert np.allclose(result.trajectory[:, 7:], start[7:])
    assert result.validation["max_lateral_error_m"] < .0015
    assert result.validation["final_travel_m"] == pytest.approx(.05, abs=.0015)
    assert np.ptp(result.trajectory[:, 0]) > .025  # tilted, not world-Z


def test_tilted_axis_rejects_bad_request_and_dense_collision(
        solver_stubs, monkeypatch):
    planner, start, reference, target, axis = _request()
    wrong = target.copy()
    wrong[0, 3] += .004
    with pytest.raises(ValueError, match="frozen socket-axis"):
        axis_jacobian.plan_jacobian_axis_stroke(
            planner, start, reference, wrong, axis_robot=axis,
            scene_cfg={"mesh": {}}, expected_travel_m=.05, step_m=.005)

    def collision(_planner, q):
        rows = np.asarray(q)
        feasible = np.ones(len(rows), dtype=bool)
        if len(rows) > 10:
            feasible[len(rows) // 2] = False
        return feasible, "mock collision", {"sample_count": len(rows)}
    monkeypatch.setattr(axis_jacobian, "_check_states_batch", collision)
    result = axis_jacobian.plan_jacobian_axis_stroke(
        planner, start, reference, target, axis_robot=axis,
        scene_cfg={"mesh": {"fixture_socket": {}, "target": {}}},
        expected_travel_m=.05, step_m=.005)
    assert not result.success
    assert result.failure_code == "axis_execution_robot_collision"
