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
               PRECISION_XY_UPDATE_WEIGHT="0.4")
    completed = subprocess.run(
        ["bash", str(demo / "run_robot_pc.sh"), "--run"],
        env=env, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    assert lines[lines.index("--xy-update-weight") + 1] == "0.4"


def test_zerodex_launcher_provides_demo_collision_compat(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("launch.json", "recipient_gap20_endpoint_catalog_straight.json",
                 "franka_runtime.json"):
        (stage / name).write_text("{}\n", encoding="utf-8")
    fake_check = tmp_path / "fake_check"
    fake_check.write_text(
        "#!/bin/sh\nprintf 'PYTHONPATH=%s\\nPREWARM=%s\\n' "
        "\"$PYTHONPATH\" \"$PRECISION_PREWARM_FIXED_WORLD\"\n",
        encoding="utf-8")
    fake_check.chmod(0o755)
    env = dict(os.environ, PRECISION_STAGE_ROOT=str(stage),
               PRECISION_CHECK_PY=str(fake_check),
               PRECISION_EXTRA_PYTHONPATH=str(demo / "compat"))
    env.pop("PRECISION_PREWARM_FIXED_WORLD", None)
    completed = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    expected = f"PYTHONPATH={demo / 'compat'}:{root}:"
    assert expected in completed.stdout
    assert "PREWARM=1" in completed.stdout

    env["PRECISION_KEY_ALREADY_PLACED"] = "1"
    already_placed = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "PREWARM=0" in already_placed.stdout

    env["PRECISION_PREWARM_FIXED_WORLD"] = "1"
    overridden = subprocess.run(
        ["bash", str(demo / "run_zerodex_capture4.sh"), "--check"],
        env=env, capture_output=True, text=True, check=True)
    assert "PREWARM=1" in overridden.stdout
