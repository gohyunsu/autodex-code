"""The launcher forwards the configured XY gain to the live CLI."""

import os
from pathlib import Path
import subprocess
import sys


def test_run_launcher_forwards_xy_update_weight(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "launch.json").write_text("{}\n", encoding="utf-8")
    (stage / "recipient_gap20_endpoint_catalog.json").write_text(
        "{}\n", encoding="utf-8")
    (stage / "franka_runtime.json").write_text("{}\n", encoding="utf-8")
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_planner.chmod(0o755)
    env = dict(os.environ, PRECISION_REPO_ROOT=str(root),
               PRECISION_STAGE_ROOT=str(stage),
               PRECISION_CHECK_PY=sys.executable,
               PRECISION_PLANNER_PY=str(fake_planner),
               PRECISION_XY_UPDATE_WEIGHT="0.4",
               PRECISION_SAME_TRIAL_XY_TOLERANCE_MM="2.5",
               PRECISION_SAME_TRIAL_OBSERVATION_YAW_DEG="60")
    completed = subprocess.run(
        ["bash", str(demo / "run_robot_pc.sh"), "--run"],
        env=env, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    assert lines[lines.index("--xy-update-weight") + 1] == "0.4"
    assert lines[lines.index("--same-trial-observation-yaw-deg") + 1] == "60"


def test_zerodex_launcher_provides_demo_collision_compat(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("launch.json", "recipient_gap5_endpoint_catalog_straight.json",
                 "franka_runtime.json"):
        (stage / name).write_text("{}\n", encoding="utf-8")
    fake_check = tmp_path / "fake_check"
    fake_check.write_text(
        "#!/bin/sh\nprintf 'PYTHONPATH=%s\\nPREWARM=%s\\nPRELOAD=%s\\nHOST=%s\\nFR3=%s\\n' "
        "\"$PYTHONPATH\" \"$PRECISION_PREWARM_FIXED_WORLD\" "
        "\"$PRECISION_PRELOAD_VLM\" \"$PRECISION_CAPTURE4_SSH_HOST\" "
        "\"$PRECISION_ZERODEX_FR3_FCI_IP\"\n",
        encoding="utf-8")
    fake_check.chmod(0o755)
    env = dict(os.environ, PRECISION_STAGE_ROOT=str(stage),
               PRECISION_CHECK_PY=str(fake_check),
               PRECISION_EXTRA_PYTHONPATH=str(demo / "compat"))
    env.pop("PRECISION_PREWARM_FIXED_WORLD", None)
    env.pop("PRECISION_CAPTURE4_SSH_HOST", None)
    env.pop("PRECISION_ZERODEX_FR3_FCI_IP", None)
    completed = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    expected = f"{demo / 'compat'}:{root}:"
    assert expected in completed.stdout
    assert "PREWARM=1" in completed.stdout
    assert "PRELOAD=1" in completed.stdout
    assert "HOST=capture4@147.46.219.136" in completed.stdout
    assert "FR3=172.16.1.11" in completed.stdout

    env["PRECISION_CAPTURE4_SSH_HOST"] = "capture4@example.test"
    overridden_host = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "HOST=capture4@example.test" in overridden_host.stdout

    env.pop("PRECISION_CAPTURE4_SSH_HOST")
    profile = tmp_path / "alternate_profile.sh"
    profile.write_text(
        "export PRECISION_CAPTURE4_SSH_HOST=capture4@alternate.test\n",
        encoding="utf-8")
    env["PRECISION_SYSTEM_PROFILE"] = str(profile)
    alternate = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "HOST=capture4@alternate.test" in alternate.stdout
    env.pop("PRECISION_SYSTEM_PROFILE")

    env["PRECISION_KEY_ALREADY_PLACED"] = "1"
    already_placed = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "PREWARM=1" in already_placed.stdout

    env["PRECISION_PREWARM_FIXED_WORLD"] = "0"
    overridden = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "PREWARM=0" in overridden.stdout

    env["PRECISION_PRELOAD_VLM"] = "0"
    no_preload = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "PRELOAD=0" in no_preload.stdout


def test_zerodex_run_selects_its_own_robot_factory(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("launch.json", "recipient_gap5_endpoint_catalog_straight.json",
                 "franka_runtime.json"):
        (stage / name).write_text("{}\n", encoding="utf-8")
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text(
        "#!/bin/sh\nprintf 'FACTORY=%s\\n' "
        "\"$PRECISION_INTEGRATION_FACTORY\"\n"
        "printf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_planner.chmod(0o755)
    env = dict(os.environ, PRECISION_REPO_ROOT=str(root),
               PRECISION_STAGE_ROOT=str(stage),
               PRECISION_CHECK_PY=sys.executable,
               PRECISION_PLANNER_PY=str(fake_planner))
    completed = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--plan-only"],
        env=env, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    assert "FACTORY=precision_insertion.zerodex_franka_integration:factory" in lines
    assert str(demo / "run_zerodex_recorded.py") in lines
    assert "--plan-only" in lines
