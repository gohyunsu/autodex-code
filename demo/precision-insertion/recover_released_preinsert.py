#!/usr/bin/env python3
"""Park the ZeroDex FR3 after a release-plan abort with the key released.

This is a one-shot recovery from the measured current arm pose. The caller
supplies the frozen socket session and the failed attempt's saved geometry.
The key is modeled at the nominal inserted pose after it has been released.
Without --execute, this plans and checks the retract without moving the arm.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from autodex.utils.conversion import se32cart
from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT
from precision_insertion.franka_core import (
    FrankaFeedbackSettings, open_franka_core,
)
from precision_insertion.franka_integration import install_curobo_planner_compat
from precision_insertion.live_robot_state import FeedbackTimeAlignmentUnavailable
from precision_insertion.socket_release import _check_path
from precision_insertion.zerodex_franka_integration import (
    inspect_zerodex_daemon, install_zerodex_modbus_compat,
    load_zerodex_clear_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--session-calibration", type=Path, required=True)
    parser.add_argument("--trial-scene", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="follow the planned retract after inspection")
    args = parser.parse_args()

    attempt = args.attempt_dir.resolve()
    calibration = json.loads(args.session_calibration.read_text())
    trial_scene = json.loads(args.trial_scene.read_text())
    postlift = json.loads((attempt / "postlift_preflights/000/report.json").read_text())
    key_pose = np.asarray(postlift["targets"]["T_robot_key_verification"], dtype=float)
    scene = copy.deepcopy(calibration["collision_scene"])
    scene["mesh"]["target"] = {
        "file_path": trial_scene["mesh"]["target"]["file_path"],
        "pose": se32cart(key_pose).tolist(),
    }

    inspect_zerodex_daemon()
    install_zerodex_modbus_compat()
    install_curobo_planner_compat()
    from paradex.utils.system import network_info

    previous_host = network_info["franka"]
    previous_clear_view = FR3_CLEAR_VIEW.copy()
    network_info["franka"] = "127.0.0.1"
    FR3_CLEAR_VIEW[:] = load_zerodex_clear_view(args.robot_config)
    core = None
    try:
        core = open_franka_core(FrankaFeedbackSettings(
            max_arm_hand_skew_s=0.1, max_sample_age_s=1.0,
            max_hand_command_error_raw=30.0,
            max_arm_velocity_rad_s=0.05))
        state = core.read_state()
        print(json.dumps({"current_arm_q": state.full_q[:7].tolist(),
                          "current_hand_q": state.full_q[7:].tolist(),
                          "open_hand_q": INSPIRE_INIT.tolist()}), flush=True)
        # InspireControllerIP starts its worker at raw 800 even though its
        # constructor briefly writes raw 1000. Set the persistent worker
        # target to open before planning an open-hand arm trajectory.
        core.executor.hand.move(np.full(6, 1000, dtype=np.int32))
        state = core.clear_view.wait_for_open_hand(
            np.r_[state.full_q[:7], INSPIRE_INIT],
            phase="recovery_hand_open", timeout_s=8.0)
        raw = core.planner.plan_js_to_init(
            scene, state.full_q[:7], start_hand_qpos=INSPIRE_INIT,
            goal_arm_qpos=FR3_CLEAR_VIEW)
        if raw is None:
            raise RuntimeError("no clear-view path from the released preinsert pose")
        path = np.asarray(raw, dtype=float).copy()
        path[:, 7:] = INSPIRE_INIT
        checked = _check_path(core.planner, scene, path)
        if not checked["all_feasible"]:
            raise RuntimeError(f"clear-view path intersects socket/key: {checked}")
        print(json.dumps({"planned_samples": len(path), "check": checked,
                          "start_arm_q": state.full_q[:7].tolist(),
                          "end_arm_q": path[-1, :7].tolist()}), flush=True)
        if args.execute:
            core.clear_view._follow_autodex(path)
            for read_attempt in range(3):
                try:
                    final = core.clear_view.verify_state(
                        core.read_state(), phase="abort_recovery")
                    break
                except FeedbackTimeAlignmentUnavailable:
                    if read_attempt == 2:
                        raise
                    time.sleep(0.25)
            print(json.dumps({"result": "clear_view", "arm_q": final.full_q[:7].tolist()}),
                  flush=True)
    finally:
        if core is not None:
            core.close()
        network_info["franka"] = previous_host
        FR3_CLEAR_VIEW[:] = previous_clear_view


if __name__ == "__main__":
    main()
