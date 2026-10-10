"""Reject synthetic planner-mode ambiguity before loading any CAD or solver."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_synthetic_full_chain import main  # noqa: E402
from diagnose_synthetic_repose import main as repose_main  # noqa: E402
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
