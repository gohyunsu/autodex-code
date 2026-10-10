"""Reject synthetic planner-mode ambiguity before loading any CAD or solver."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_synthetic_full_chain import main  # noqa: E402
from diagnose_synthetic_repose import (  # noqa: E402
    _capture_failed_stroke_state, _diagnose_failed_world_obstacle,
    main as repose_main,
)
from run_pipeline import main as pipeline_main  # noqa: E402


_REQUIRED = [
    "--shared-root", "/unused", "--catalog", "/unused/catalog.json",
    "--mode", "square", "--gap-mm", "1.5", "--pose-stem", "004",
    "--table-z-m", "0.04", "--key-x-m", "0.4", "--key-y-m", "0",
    "--socket-x-m", "0.6", "--socket-y-m", "0",
    "--output-dir", "/unused/new-output",
]

_REPOSE_REQUIRED = [
    "--shared-root", "/unused", "--catalog", "/unused/catalog.json",
    "--reset-candidate-dir", "/unused/reset_12",
    "--mode", "cylinder", "--gap-mm", "1",
    "--from-pose-stem", "000", "--to-pose-stem", "001",
    "--height-cm", "12", "--table-z-m", "0.04",
    "--key-x-m", "0.4", "--key-y-m", "0",
    "--socket-x-m", "0.6", "--socket-y-m", "0",
    "--release-x-m", "0.45", "--release-y-m", "0.08",
    "--board-x-min-m", "0.2", "--board-x-max-m", "0.8",
    "--board-y-min-m", "-0.25", "--board-y-max-m", "0.25",
    "--max-reset-drift-mm", "3", "--max-reset-axis-tilt-deg", "8",
    "--min-rest-socket-clearance-mm", "10",
    "--min-board-edge-clearance-mm", "10",
    "--output-dir", "/unused/new-repose-output",
]


@pytest.mark.parametrize("env_value,declared_mode", [
    ("1", "default"), ("0", "native-locked-experimental"),
])
def test_planner_mode_must_match_environment_before_io(
    monkeypatch, capsys, env_value, declared_mode,
):
    monkeypatch.setenv("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS", env_value)
    with pytest.raises(SystemExit) as error:
        main([*_REQUIRED, "--planner-mode", declared_mode])
    assert error.value.code == 2
    assert "--planner-mode must match" in capsys.readouterr().err


def test_saved_trial_cli_rejects_undeclared_native_mode_before_io(
    monkeypatch, capsys, tmp_path,
):
    monkeypatch.setenv("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS", "1")
    args = [
        "preflight-trial", "--shared-root", str(tmp_path),
        "--mode", "square", "--gap-mm", "1.5",
        "--session", str(tmp_path / "session.json"),
        "--catalog", str(tmp_path / "catalog.json"),
        "--key-pose-world-npy", str(tmp_path / "key.npy"),
        "--key-observation-id", "test", "--key-capture-time-s", "2",
        "--live-start-q-npy", str(tmp_path / "start.npy"),
        "--start-q-time-s", "2", "--max-key-state-skew-s", "0.1",
        "--limits-json", str(tmp_path / "limits.json"),
        "--max-pose-error-deg", "10", "--axial-waypoint-step-mm", "5",
        "--output-dir", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit) as error:
        pipeline_main(args)
    assert error.value.code == 2
    assert "--planner-mode must match" in capsys.readouterr().err


def test_synthetic_repose_rejects_undeclared_native_mode_before_io(
    monkeypatch, capsys,
):
    monkeypatch.setenv("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS", "1")
    with pytest.raises(SystemExit) as error:
        repose_main([*_REPOSE_REQUIRED, "--planner-mode", "default"])
    assert error.value.code == 2
    assert "--planner-mode must match" in capsys.readouterr().err


def test_synthetic_repose_release_options_are_paired_before_io(
    monkeypatch, capsys,
):
    monkeypatch.setenv("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS", "0")
    with pytest.raises(SystemExit) as error:
        repose_main([*_REPOSE_REQUIRED, "--retreat-goal-q-npy",
                     "/unused/retreat.npy"])
    assert error.value.code == 2
    assert "release exit needs both" in capsys.readouterr().err


def test_synthetic_repose_observes_failed_joint_without_changing_verdict(
    monkeypatch,
):
    import autodex.planner.jacobian_stroke as stroke_module

    original = lambda _planner, _q, *_, **__: (
        np.array([True, False]), "world_collision", {})
    monkeypatch.setattr(stroke_module, "_check_states_batch", original)
    q = np.arange(26, dtype=np.float32).reshape(2, 13)
    with _capture_failed_stroke_state() as captured:
        valid, status, _ = stroke_module._check_states_batch(None, q)
    assert valid.tolist() == [True, False]
    assert status == "world_collision"
    assert np.array_equal(captured["q"], q[1])
    assert stroke_module._check_states_batch is original


def test_synthetic_repose_isolates_table_obstacle_without_waiving_it(
    monkeypatch,
):
    import autodex.planner.jacobian_stroke as stroke_module

    class Checker:
        enabled = {"table": True, "fixture_socket": True}

        def get_obstacle_names(self):
            return list(self.enabled)

        def enable_obstacle(self, name, *, enable):
            self.enabled[name] = enable

    class Planner:
        _motion_gen = type("MotionGen", (), {"world_coll_checker": Checker()})()

    planner = Planner()

    def check(active, _q):
        blocked = active._motion_gen.world_coll_checker.enabled["table"]
        return (np.array([not blocked]),
                "world_collision" if blocked else None,
                {"backend": "test"})

    monkeypatch.setattr(stroke_module, "_check_states_batch", check)
    scene = {
        "cuboid": {"table": {"dims": [1., 1., .1],
                             "pose": [0., 0., 0., 1., 0., 0., 0.]}},
        "mesh": {
            "fixture_socket": {"file_path": "/unused/socket.obj",
                               "pose": [0., 0., 0., 1., 0., 0., 0.]},
        },
    }
    result = _diagnose_failed_world_obstacle(planner, scene, np.zeros(13))
    assert result["classification"] == "table_world_obstacle_necessary"
    assert result["variants"]["full"]["feasible"] is False
    assert result["variants"]["socket_only"]["feasible"] is True
    assert planner._motion_gen.world_coll_checker.enabled == {
        "table": True, "fixture_socket": True}
