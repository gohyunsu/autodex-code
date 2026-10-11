# Failed-trial held-key return

An insertion failure does not make the pre-insertion hold a drop site. That
pose is directly above the socket on the same XY axis. Opening Inspire there
can leave the key in or on the socket instead of restoring a tabletop trial.

For a **still-held key after insertion failure**, the intended common tail is:

1. A commissioned guarded controller withdraws axially to a clear hold and
   records its result. Do not shift laterally while the key is engaged.
2. From fresh measured Franka/Inspire joints and the established post-lift
   `T_key_hand`, plan the held key to an explicitly chosen tabletop release
   site. The trial-start site is a useful target, but its full key footprint
   must fit the measured ChArUco area and clear the frozen socket.
3. Preflight hand opening, upward exit and retract using the existing reset
   release planner. Then call `ClearViewForPerception.finish_planned_release_exit()`
   with blocking commissioned motion callbacks. It opens Inspire fully,
   checks measured opening, follows the vertical exit and socket-aware
   retract, then confirms measured `FR3_CLEAR_VIEW` / `INSPIRE_INIT` joints.
   Do not use free-space `home(clear_view=True)` after a trial.
4. Capture the landed key afresh only after clear-view feedback passes. A
   commanded drop does not prove its pose.
   `SessionRunner.observe_reset_landing()` currently admits only a supervised
   manual return; a bound automatic-return execution/landing adapter is still
   needed before the action loop can repeat trials this way.

For a missed grasp, there is no held key to withdraw or drop; recapture the
tabletop key. A lost grasp or abort cannot use the held-key return path.

The new planning primitive starts **after** the withdrawal. It shares the
socket-aware held transfer/descent and release checks used by directed v8
repose; it neither runs AutoDex's pickup planner again nor invents a second
lift. Supply the actual measured 13-DoF withdrawal endpoint, a previously
established held relation, the original trial scene, and an explicit safe
tabletop target:

```python
from precision_insertion.repose_preflight import plan_repose_held_from_state
from precision_insertion.repose_release import plan_repose_release_exit

held = plan_repose_held_from_state(
    planner=planner, trial_scene=trial_scene, shared_root=shared_root,
    calibration=calibration, mode=mode,
    T_key_hand=postlift_relation.T_key_hand,
    measured_start_q=withdrawn_state.full_q,
    T_robot_key_rest=trial_start_key_pose_robot,
    release_height_m=release_height_m,
    minimum_rest_socket_clearance_m=socket_clearance_m,
    minimum_board_edge_clearance_m=board_edge_margin_m,
    limits=path_audit_limits,
)
if held.status != "sampled_held_path_pass_release_unplanned":
    raise RuntimeError(held.status)

release = plan_repose_release_exit(
    planner=planner, trial_scene=trial_scene, shared_root=shared_root,
    calibration=calibration, mode=mode, held_plan=held,
    release_hand_q=open_hand_q, retreat_goal_arm_q=retreat_arm_q,
    minimum_release_key_clearance_m=release_key_clearance_m,
    limits=path_audit_limits,
)
if release.status != "nominal_release_exit_path_pass_drop_unobserved":
    raise RuntimeError(release.status)

# First execute held.transfer_trajectory and held.descent_trajectory through
# the commissioned controller. The callbacks below must block and report
# physical execution failures; the gate confirms release-site feedback.
clear_view.finish_planned_release_exit(
    release_plan=release,
    open_hand=commissioned_open_inspire,
    follow_joint_trajectory=commissioned_follow_franka_inspire,
)
```

The first two identical `held.lift_trajectory` rows are a stationary audit
hold, **not** an executable lift. Only `held.transfer_trajectory` and
`held.descent_trajectory` are new held-motion plans. A caller must still
verify the withdrawal log, measured-state timestamp and key/hand relation;
the new primitive does not bind these sources itself. Collision checks are
sampled, and release/drop dynamics and the actual landing pose are unknown.
An XY correction learned during the failed attempt is not yet carried into
the next `SessionRunner` attempt; its initial offset remains zero.
