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
   release planner. `ClearViewForPerception.complete_failed_insertion_return_with_autodex()`
   follows the planned held transfer/descent through the stock
   `FrankaExecutor.follow_joint_trajectory()` with converted Inspire actions,
   then calls its ramped `release()` and follows the vertical exit/retract.
   It checks measured endpoints and final `FR3_CLEAR_VIEW` / `INSPIRE_INIT`.
   Do not use free-space `home(clear_view=True)` after a trial.
4. Capture the landed key afresh only after clear-view feedback passes. A
   commanded drop does not prove its pose.
   `SessionRunner.observe_reset_landing()` accepts either a supervised manual
   return or an ordered, source-bound automatic-return execution record. A
   commissioned controller still has to produce that independent record;
   neither a commanded drop nor a planned trajectory proves the landing.

For a **judged missed grasp**, there is no held key to withdraw or drop.
`LiveKeyPlanningHandlers` calls stock AutoDex
`FrankaExecutor.release(..., open_to_init=True)` and `reset()` before
recapture. The reset scene preserves the original tabletop key and frozen
socket as obstacles, then checks measured open-hand/clear-view feedback.
If reset planning or execution fails, key perception does not restart. A
lost grasp or ambiguous abort cannot use this missed-grasp path or the
held-key return path automatically.

The new planning primitive starts **after** the withdrawal. It shares the
socket-aware held transfer/descent and release checks used by directed v8
repose; it neither runs AutoDex's pickup planner again nor invents a second
lift. Supply the actual measured 13-DoF withdrawal endpoint, a previously
established held relation, the original trial scene, and an explicit safe
tabletop target:

In a live session, prefer `runner.prepare_failed_insertion_return(...)`.
It accepts the externally logged guarded withdrawal, checks the measured
Franka/Inspire start, reuses that attempt's saved post-lift key/hand relation
and original tabletop pose, and writes an immutable return plan with source
hashes under the attempt. The explicit lower-level equivalent is shown here
to clarify the geometry:

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

# After the commissioned guarded withdrawal and a fresh measured start,
# this reuses the existing FrankaExecutor for all return/exit motion.
clear_view.complete_failed_insertion_return_with_autodex(
    held_plan=held, release_plan=release)
```

The live call supplies the same physical quantities but also binds sources:

```python
held, release, plan_report = runner.prepare_failed_insertion_return(
    planner=planner, withdrawn_state=withdrawn_state,
    withdrawal_evidence_path=withdrawal_log_path,
    limits=path_audit_limits, release_height_m=release_height_m,
    minimum_rest_socket_clearance_m=socket_clearance_m,
    minimum_board_edge_clearance_m=board_edge_margin_m,
    minimum_release_key_clearance_m=release_key_clearance_m,
    max_withdrawal_state_gap_s=commissioned_state_gap_s,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_hand_command_error_raw=commissioned_hand_error_raw,
    max_arm_velocity_rad_s=commissioned_arm_velocity_rad_s,
)
clear_view.complete_failed_insertion_return_with_autodex(
    held_plan=held, release_plan=release)
```

The commissioned execution recorder must then save an ordered
`precision_insertion_failed_return_execution_v1` record (phases:
`axial_withdrawal`, `held_transfer`, `held_descent`, `release_open`,
`post_release_lift`, `post_release_retract`) with the exact `plan_report`
path/hash, session/catalogue hashes and independent hashed controller
feedback files. See
[`failed_return_evidence.py`](precision_insertion/failed_return_evidence.py).
After the arm is clear, capture the key anew and call
`runner.observe_reset_landing(..., recovery_log_path=execution_log_path)`.
Only an observed supported tabletop pose in the original pose class and
within the configured return-position tolerance sets `reset_success=True`.
The demo's [`LiveHeldReturnHandlers`](precision_insertion/live_recovery_handlers.py)
connects these steps to the existing action loop and reuses the initial
AutoDex key-capture path for the landing observation. The robot PC must
provide the real guarded-withdrawal callback, independent controller-log
callback and exposure-time robot-state sampler. Neither the handler nor the
motion wrapper fabricates those measurements.

The first two identical `held.lift_trajectory` rows are a stationary audit
hold, **not** an executable lift. Only `held.transfer_trajectory` and
`held.descent_trajectory` are new held-motion plans. A caller must still
verify the withdrawal log, measured-state timestamp and key/hand relation;
the new primitive does not bind these sources itself. Collision checks are
sampled, and release/drop dynamics and the actual landing pose are unknown.
An XY correction learned during the failed attempt is not yet carried into
the next `SessionRunner` attempt; its initial offset remains zero.
