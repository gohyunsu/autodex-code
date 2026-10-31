"""The launcher forwards the configured XY gain to the live CLI."""

import json
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


def test_run_launcher_applies_current_camera_profile_to_effective_launch(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "launch.json").write_text(json.dumps({
        "camera_ids": ["cam0", "cam1", "cam2", "cam3"],
        "board_fit_camera_ids": ["cam0", "cam1", "cam3"],
    }), encoding="utf-8")
    (stage / "recipient_gap20_endpoint_catalog.json").write_text(
        "{}\n", encoding="utf-8")
    (stage / "franka_runtime.json").write_text("{}\n", encoding="utf-8")
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_planner.chmod(0o755)
    run_root = tmp_path / "runs"
    env = dict(
        os.environ, PRECISION_REPO_ROOT=str(root),
        PRECISION_STAGE_ROOT=str(stage), PRECISION_CHECK_PY=sys.executable,
        PRECISION_PLANNER_PY=str(fake_planner),
        PRECISION_RUN_OUTPUT_ROOT=str(run_root), PRECISION_RUN_TAG="profile",
        PRECISION_CAPTURE4_CAMERA_IDS="cam0,cam1,cam2,cam3",
        PRECISION_BOARD_FIT_CAMERA_IDS="cam0,cam1,cam2,cam3")
    subprocess.run(["bash", str(demo / "run_robot_pc.sh"), "--run"],
                   env=env, capture_output=True, text=True, check=True)
    effective = json.loads((run_root / "_control/profile/launch_profile.json").read_text(
        encoding="utf-8"))
    assert effective["camera_ids"] == ["cam0", "cam1", "cam2", "cam3"]
    assert effective["board_fit_camera_ids"] == ["cam0", "cam1", "cam2", "cam3"]


def test_zerodex_run_keeps_control_and_episode_outputs_out_of_assets(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    stage = (tmp_path / "runtime/active_shared_data/AutoDex" /
             "precision_insertion/zerodex_capture4")
    stage.mkdir(parents=True)
    for name in ("launch.json", "recipient_gap5_endpoint_catalog.json",
                 "franka_runtime.json"):
        (stage / name).write_text("{}\n", encoding="utf-8")
    fake_planner = tmp_path / "fake_planner"
    fake_planner.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_planner.chmod(0o755)
    run_root = tmp_path / "runs/zerodex_capture4"
    env = dict(
        os.environ, PRECISION_REPO_ROOT=str(root),
        PRECISION_STAGE_ROOT=str(stage), PRECISION_GAP_MM="5",
        PRECISION_RUN_OUTPUT_ROOT=str(run_root),
        PRECISION_RUN_TAG="20261030_test_run",
        PRECISION_CHECK_PY=sys.executable,
        PRECISION_PLANNER_PY=str(fake_planner))
    completed = subprocess.run(
        ["bash", str(demo / "run_robot_pc.sh"), "--run"],
        env=env, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    output = Path(lines[lines.index("--output-dir") + 1])
    assert output == (run_root / "precision-insertion/inspire" /
                      "precision_key_cylinder_r15_h80/_pipeline_runs" /
                      "20261030_test_run")
    launch_path = (run_root / "_control/20261030_test_run" /
                   "launch_20261030_test_run.json")
    assert launch_path.is_file()
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    assert Path(launch["capture_root"]) == (
        run_root / "_control/20261030_test_run/raw_captures")
    assert Path(launch["capture_root"]).is_dir()
    assert not (stage / "runs").exists()


def test_zerodex_profile_rejects_only_nas_paths_outside_hyunsu(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    root = demo.parents[1]
    common = dict(
        os.environ, PRECISION_REPO_ROOT=str(root),
        PRECISION_ALLOWED_NAS_ROOT="/mnt/paradex2/hyunsu",
        PRECISION_NAS_MOUNT_ROOT="/mnt/paradex2",
        PRECISION_CHECK_PY=sys.executable,
        PRECISION_PLANNER_PY=sys.executable)

    outside = subprocess.run(
        ["bash", str(demo / "run_robot_pc.sh"), "--check"],
        env={**common,
             "PRECISION_STAGE_ROOT": "/mnt/paradex2/another_user/stage"},
        capture_output=True, text=True)
    assert outside.returncode == 2
    assert "outside the allowed NAS namespace" in outside.stderr

    allowed = subprocess.run(
        ["bash", str(demo / "run_robot_pc.sh"), "--check"],
        env={**common,
             "PRECISION_STAGE_ROOT": "/mnt/paradex2/hyunsu/missing-stage"},
        capture_output=True, text=True)
    assert allowed.returncode == 2
    assert "outside the allowed NAS namespace" not in allowed.stderr
    assert "active cylindrical recipient stage is missing" in allowed.stderr


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
