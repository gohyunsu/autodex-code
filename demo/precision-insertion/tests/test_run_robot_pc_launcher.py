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
