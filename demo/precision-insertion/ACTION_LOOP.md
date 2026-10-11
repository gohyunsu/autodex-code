# One-session action loop

`run_live_precision_session()` joins the already implemented AutoDex-camera
ChArUco→socket startup to `SessionRunner`'s existing trial policy. It keeps
the same initialized cameras available across key captures, attempts and
reobservations. It does **not** duplicate v8 grasp ranking, FoundPose,
cuRobo planning, label policy or the guarded execution boundaries.

Its handler factory receives the live session after the socket/collision
world is frozen. It returns an action-name → callable mapping. Each callable
receives `(runner, decision)` and completes the entire indicated boundary
using the existing demo functions before returning. For example:

| Runner action | Handler's existing boundary |
| --- | --- |
| `capture_fresh_key`, `reobserve_key_and_preflight` | Move to measured clear view, then `prepare_next_live_key()` with the same cameras and frozen runner. For a judged grasp miss, reuse AutoDex `release(..., open_to_init=True)` and `reset()` before recapture. The attempted grasp is excluded. |
| `continue_candidate_preflight` | `runner.preflight_next_key()` on the *same* saved key capture and measured state, continuing a budget-limited prefix rather than falsely declaring pose exhaustion. |
| `execution_gate_required` | [`LivePickupHandlers`](precision_insertion/live_pickup_handlers.py) starts the selected v8 attempt, calls the existing `execute_bound_pickup()` through a commissioned executor, and preflights lift from *measured* post-squeeze joints. It does not execute lift or write a grasp-success label. |
| `await_lift_observation` | [`LiveLiftHandlers`](precision_insertion/live_lift_handlers.py) runs the saved measured-start lift through a commissioned adapter, captures fresh raw frames from the *same* AutoDex cameras, and calls `runner.prepare_raw_lift_label()` with a Qwen-compatible backend. The runner records a grasp label only for a decisive visual checkpoint. |
| `postlift_observed_preflight_required`, `held_relation_evidence_required` | Obtain the post-squeeze key/hand relation or its explicit uncertainty bound, then call the matching `runner.prepare_*postlift_transfer()` method. |
| `transfer_execution_gate_required` | Execute the bound held transfer, capture the pre-insertion hold, and record the separate `preinsert_reached` label. |
| `await_guarded_insertion_and_observation` | Prepare the bound axial handoff, run a commissioned guarded stroke, then use paired Qwen views plus measured depth/alignment/force in `runner.prepare_observed_insertion_label()`; only its checkpoint may set `insertion_success`. |
| `guarded_withdrawal_then_xy_assessment` | Withdraw the still-held key and use the existing native-pixel multi-view cylinder landmark/XY diagnostic; replan and record a bounded retry. Do not turn a VLM suggestion directly into a motor command. |
| `await_retry_execution_and_observation` | Execute the bound retry path and record a *new* insertion checkpoint. |
| `recover_held_key_then_reobserve` | [`LiveHeldReturnHandlers`](precision_insertion/live_recovery_handlers.py) joins logged guarded withdrawal, `runner.prepare_failed_insertion_return()`, stock Franka held return/release/clear-view motion, independently recorded controller phases, a fresh key capture on the same cameras, and `runner.observe_reset_landing()`. Guarded withdrawal and independent logging remain commissioned robot-PC callbacks. |
| `recover_key_then_reobserve` | Transfer/hold failure may leave the key in an unknown state. Do not use the held-key or grasp-miss motion blindly; require observed recovery or operator intervention before a new capture. |
| `preflight_repose`, `continue_repose_seed_preflight`, `preflight_repose_release`, `preflight_alternative_repose_cell` | Use the same captured key and `runner.preflight_repose()` with directed v8 reset seeds; a staged seed alone is not a plan. |
| `repose_execution_gate_required`, `await_repose_observation` | Execute a commissioned reset plan, observe the landed tabletop pose, and record `reorient_success` before a fresh key capture. |

The first two rows are implemented as
[`LiveKeyPlanningHandlers`](precision_insertion/live_planning_handlers.py).
It obtains capture IDs and uses the *same* `live.cameras` context and frozen
runner for every new key. A pilot prefix continues from its saved
`PreparedLiveKeyTrial` without another camera exposure; the runner retains
its exclusions and checks that the key/state record is the same. The caller
must supply a running exposure-time feedback sampler, AutoDex planner,
`PathAuditLimits`, a specific key prompt and the numeric keyword arguments
of `prepare_next_live_key()`. No default measurement thresholds are invented.

[`LivePickupHandlers`](precision_insertion/live_pickup_handlers.py) is the
next action boundary. It calls `SessionRunner.begin_selected_attempt()`,
`execute_bound_pickup()`, then
`SessionRunner.prepare_measured_lift_chain()` in that order. The selected
candidate and saved v8 trajectory are checked by the existing pickup
boundary. The measured post-squeeze state—not the nominal BODex hand pose—is
fed to the lift planner. Pickup needs an explicitly enabled, separately
commissioned executor and motion interlock. The stock `FrankaExecutor`
cannot be supplied unchanged to `execute_bound_pickup()` because that
function rejects its non-fail-closed trajectory follower. A passing chain
only unlocks the next `await_lift_observation` boundary; it is not evidence
that lift or grasp succeeded.

[`LiveLiftHandlers`](precision_insertion/live_lift_handlers.py) uses
`execute_bound_measured_lift()` and the existing raw lift checkpoint. Its
camera callback must produce a `RawCameraCapture` with the open session's
undistorted, acquisition-time-stamped images; its state callback must return
joint feedback buffered for that exposure. The handler writes the existing
raw-camera evidence format and passes it to
`SessionRunner.prepare_raw_lift_label()`. The VLM backend can be loaded once
with `load_vlm_backend(mode="local", max_input_size=(W, H))`, whose default
model is Qwen3-VL-2B-Instruct. Unknown visual outcomes stay unlabelled; a
positive visual outcome advances to held-relation calibration, not directly
to nominal BODex transfer. The external lift adapter and raw-camera producer
remain robot-PC integration requirements.

The required [`ClearViewForPerception`](precision_insertion/clear_view.py)
calls stock `FrankaExecutor.home(clear_view=True)` **before** the initial
ChArUco/socket image: Inspire opens and FR3 parks at `FR3_CLEAR_VIEW`.
Measured stationary joints must agree. Every fresh key capture, including
`reobserve_key_and_preflight`, independently checks the arm and open hand
before capture and at camera exposure time. Same-capture candidate
continuation needs no new image. For a judged grasp miss,
`LiveKeyPlanningHandlers` calls the stock AutoDex release/reset with the
selected v8 pickup plan. Its copied reset world preserves the fixed socket
and original tabletop key as obstacles. For a still-held insertion failure,
the checked held return and release/exit are required instead. Neither path
uses free-space `home` after a trial. If measured clear view is not reached,
key recapture stops. The held-insertion-failure handler invokes the same
`ClearViewForPerception` return/exit motion, then uses the **same**
`collect_key_capture()` → SAM/FoundPose → `admit_key_capture()` path used by a
new trial. It captures the returned tabletop key *before* planning the next
grasp, and the runner labels the reset only from that fresh observation.

`stop_for_review` is a non-success terminal result. A verified positive
`insertion_success` produces `session_success`, which immediately returns
`status="success"` without release, reset, repose, or another grasp attempt.
The live wrapper then closes its camera context. A visual guess, nominal
trajectory, or an unknown insertion verdict cannot end the session as
success. Recovery/reset actions not listed above still need explicit handlers;
otherwise the result is `needs_handler`. There is no default robot-motion
handler. A `max_steps` limit returns `step_budget_exhausted` instead of
silently continuing forever. If a handler returns without changing the
decision, the loop stops with `decision_unchanged` rather than repeating a
physical action. For a budget-limited candidate prefix, the handler should
continue that same saved capture until the runner's decision changes, or let
an operator inspect the unchanged decision. Handler exceptions propagate while the camera
context closes, leaving the runner's already written evidence intact.

Programmatic use from the repository root (after providing commissioned
camera, planner, feedback, Qwen and motion adapters):

```python
from pathlib import Path
from precision_insertion.live_orchestration import run_live_precision_session
from precision_insertion.live_planning_handlers import LiveKeyPlanningHandlers
from precision_insertion.live_pickup_handlers import LivePickupHandlers
from precision_insertion.live_lift_handlers import LiveLiftHandlers
from precision_insertion.observer import load_vlm_backend
from precision_insertion.clear_view import ClearViewForPerception
from precision_insertion.live_robot_state import read_live_franka_inspire_state

def read_current_state():
    return read_live_franka_inspire_state(
        arm=franka_executor.arm, hand=franka_executor.hand,
        max_arm_hand_skew_s=commissioned_key_parameters["max_arm_hand_skew_s"],
        max_sample_age_s=commissioned_feedback_age_s,
        max_hand_command_error_raw=commissioned_key_parameters["max_hand_command_error_raw"],
        max_arm_update_wait_s=commissioned_arm_update_wait_s,
        max_arm_velocity_rad_s=commissioned_key_parameters["max_arm_velocity_rad_s"],
    )

clear_view = ClearViewForPerception(
    executor=franka_executor,
    read_state=read_current_state,
    max_arm_hand_skew_s=commissioned_key_parameters["max_arm_hand_skew_s"],
    max_hand_command_error_raw=commissioned_key_parameters["max_hand_command_error_raw"],
    max_arm_velocity_rad_s=commissioned_key_parameters["max_arm_velocity_rad_s"],
    max_sample_age_s=commissioned_feedback_age_s,
)
qwen_backend = load_vlm_backend(
    mode="local", max_input_size=commissioned_native_image_size)

def build_handlers(live):
    planning = LiveKeyPlanningHandlers(
        live=live, planner=planner,
        state_for_observation=feedback_sampler.state_for_observation,
        limits=path_audit_limits,
        key_prompt="blue precision key on the board, excluding the fixed red socket",
        key_parameters=commissioned_key_parameters,
    )
    pickup = LivePickupHandlers(
        live=live, planner=planner, executor=commissioned_pickup_executor,
        read_state=read_current_state, motion_interlock=motion_interlock,
        pickup_limits=pickup_execution_limits,
        lift_bounds=measured_surface_deviation_bounds,
        path_limits=path_audit_limits,
        measured_lift_parameters=commissioned_measured_lift_parameters,
        enable_robot_motion=True,
    )
    lift = LiveLiftHandlers(
        live=live, adapter=commissioned_lift_adapter,
        read_state=read_current_state,
        state_for_capture=feedback_sampler.state_for_raw_capture,
        capture_after_lift=autodex_raw_after_lift_capture,
        backend=qwen_backend, motion_interlock=motion_interlock,
        limits=lift_execution_limits,
        commissioning_record_path=commissioning_record_path,
        assessment_parameters=commissioned_lift_assessment_parameters,
        enable_robot_motion=True,
    )
    return {**planning.handlers(), **pickup.handlers(), **lift.handlers()}

result = run_live_precision_session(
    launch_config_path=Path("/absolute/commissioned_session_launch.json"),
    endpoint_catalog_path=Path("/absolute/current_endpoint_catalog.json"),
    run_output_dir=Path("/absolute/new_trial_run"),
    max_xy_retries=2, clear_view=clear_view,
    build_handlers=build_handlers,
)
print(result.to_record())
```

`planner`, `feedback_sampler`, `path_audit_limits`, execution limits and
`commissioned_key_parameters` are **operator-supplied values**. The feedback
sampler must already run before the key exposure; configure it as shown in
the main [README](README.md). `commissioned_key_parameters` contains the
threshold arguments of `prepare_next_live_key()` other than the session,
camera, planner, prompt and limit objects supplied above, for example
`max_key_state_skew_s`, `max_pose_error_deg` and
`axial_waypoint_step_m`. The example deliberately requires
`commissioned_pickup_executor`, `motion_interlock`,
`pickup_execution_limits`, `measured_surface_deviation_bounds` and
`commissioned_measured_lift_parameters` from the robot PC. Without them,
omit `pickup.handlers()` and the loop stops at `execution_gate_required`.
With them, and commissioned lift/camera callbacks, the next unimplemented
boundary after a positive grasp label is post-lift relation/transfer.
Transfer, insertion, guarded
withdrawal, independent controller logging, VLM observation and repose
execution remain AutoDex-PC integration work. The judged grasp-miss clear-view
reset is connected; the held-key return handler can be composed as shown
below once its commissioned callbacks exist. Returning from
`run_live_precision_session()` closes the cameras and loses the in-memory
runner, so `needs_handler` is a diagnostic stop, **not** a resumable paused
trial. Supply the remaining handlers before starting a physical session.
A passing offline endpoint
catalogue and synthetic plan are not physical-success evidence.

For a failed still-held insertion, the held-key return preflight targets the
original tabletop area, not the socket. After a commissioned guarded axial
withdrawal, `clear_view.complete_failed_insertion_return_with_autodex()`
reuses the stock Franka executor for held transfer/descent, full hand
release, vertical exit and retract. It checks measured joints at each
boundary. The [`failed-return evidence`](precision_insertion/failed_return_evidence.py)
binds the plan and ordered external controller log; `observe_reset_landing()`
then accepts a new camera-observed tabletop pose to label the reset. The
[`LiveHeldReturnHandlers`](precision_insertion/live_recovery_handlers.py)
joins these existing steps without a second camera or robot executor. It
requires three operator-supplied numeric mappings (the key-capture view
limits, `SessionRunner.prepare_failed_insertion_return()` return limits, and
`SessionRunner.observe_reset_landing()` landing limits), plus:

```python
from precision_insertion.live_recovery_handlers import LiveHeldReturnHandlers

def build_handlers(live):
    planning = LiveKeyPlanningHandlers(...)  # as above
    recovery = LiveHeldReturnHandlers(
        live=live, planner=planner, limits=path_audit_limits,
        state_for_observation=feedback_sampler.state_for_observation,
        withdraw_guarded=commissioned_guarded_withdrawal,
        record_execution=commissioned_controller_recorder,
        key_prompt="blue precision key on the board, excluding the fixed red socket",
        capture_parameters=commissioned_capture_parameters,
        return_parameters=commissioned_return_parameters,
        landing_parameters=commissioned_landing_parameters,
    )
    return {**planning.handlers(), **recovery.handlers(),
            **other_commissioned_handlers}
```

`withdraw_guarded(runner, decision)` must execute the axial withdrawal and
return `(measured_withdrawn_state, guarded_withdrawal_log_path)`.
`record_execution(run_planned_return, report_path, runner, decision)` must
invoke the one-shot blocking `run_planned_return()` exactly once while it
independently records the six ordered controller phases, then return the
`precision_insertion_failed_return_execution_v1` log path. A skipped or
invalid execution cannot trigger key capture. Those two callbacks, plus
physical robot/camera commissioning, are **not** implemented by this repo;
the handler is not proof of an executed reset.
