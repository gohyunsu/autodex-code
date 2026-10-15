# One-session action loop

The current insertion-failure route stores XY advice for a **new physical
attempt**, never a same-held-key reinsert. See
[`NEXT_TRIAL_XY.md`](NEXT_TRIAL_XY.md).
The active handler contract does not require the historical
`await_retry_execution_and_observation` same-held-key action. The cylinder
runner opens the hand after the 20 mm stroke, observes the released key, and
starts a new pickup only after restoring the key to the trial-start pose.

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
| `capture_fresh_key`, `reobserve_key_and_preflight` | From measured clear view, run `prepare_next_live_key()` with the same cameras and frozen runner. This action begins only after a failed attempt's original-pose return has been observed; the attempted grasp is excluded. |
| `continue_candidate_preflight` | `runner.preflight_next_key()` on the *same* saved key capture and measured state, continuing a budget-limited prefix rather than falsely declaring pose exhaustion. |
| `execution_gate_required` | [`LivePickupHandlers`](precision_insertion/live_pickup_handlers.py) starts the selected v8 attempt, calls the existing `execute_bound_pickup()` through a commissioned executor, and preflights lift from *measured* post-squeeze joints. It does not execute lift or infer grasp success. A stopped pickup or rejected post-squeeze lift plan records `grasp_success=unknown` and routes to supervised return. An unacknowledged motor stop remains an exception. |
| `await_lift_observation` | [`LiveLiftHandlers`](precision_insertion/live_lift_handlers.py) runs the saved measured-start lift through a commissioned adapter, captures fresh raw frames from the *same* AutoDex cameras, and calls `runner.prepare_raw_lift_label()` with a Qwen-compatible backend. A decisive visual checkpoint records true/false. A stopped lift or Qwen abstention records `unknown` and routes to supervised return; neither is mislabeled a grasp miss. |
| `held_relation_evidence_required` | In the active cylinder live route, [`LiveNominalRelationHandlers`](precision_insertion/live_nominal_relation_handlers.py) uses the selected v8 key–hand hypothesis plus fresh robot joints. There is no post-grasp key–hand measurement. Historical physical/observed routes are not allowed for cylinder live execution. |
| `transfer_execution_gate_required` | [`LiveTaskHandlers`](precision_insertion/live_task_handlers.py) runs the bound held transfer and saves the synchronized raw multi-view pre-insertion snapshot. The Qwen arrival check receives each raw image only; separately saved CAD renders remain audit/presentation artifacts, never VLM inputs. |
| `await_guarded_insertion_and_observation` | The same handler runs the commissioned 20 mm stroke, requires the robot-PC `release_and_clear` adapter to fully open Inspire and exit to clear view, then captures final raw views. Qwen receives one **initial tabletop raw + final clear-view raw** pair per camera and votes `inside_socket`, `outside_socket`, or `unobservable`. Success ends the session. |
| `assess_saved_preinsert_xy` | Cylinder only: [`LiveGroundedRetryHandlers`](precision_insertion/live_retry_handlers.py) reopens the immutable raw snapshot captured before the failed stroke. Qwen grounds the visible tip/axis and calibrated geometry converts it to at most 10 mm socket-XY advice for the **next** attempt. This action performs no robot motion and takes no post-failure picture. |
| `recover_released_key_then_reobserve` | [`LiveHeldReturnHandlers`](precision_insertion/live_recovery_handlers.py) perceives the already released key from measured clear view. If it differs from the trial-start tabletop pose/CAD-center XY, [`plan_original_pose_restoration()`](precision_insertion/original_pose_restoration.py) reuses v8 pickup and socket-aware placement; a commissioned `execute_restoration` callback runs it and the key is recaptured. The attempt ends only after `runner.observe_reset_landing()` verifies the restored state. |
| `recover_key_then_reobserve` | [`LiveTransferRecoveryHandlers`](precision_insertion/live_transfer_recovery.py) uses stock AutoDex release/reset for a judged `grasp_miss`, then recaptures the key and restores its original tabletop pose/position if disturbed **within the same attempt**. A completed, still-held transfer with an off-target measured hand **or visible gross XY misalignment** may instead use a planned held return. Stopped pickup/lift/transfer, visible slip/miss and unknown VLM verdicts require supervised tabletop recovery. Every route recaptures from clear view and verifies original-pose landing before a new trial. |
| `preflight_repose`, `continue_repose_seed_preflight`, `preflight_repose_release`, `preflight_alternative_repose_cell` | [`LiveReposePlanningHandlers`](precision_insertion/live_repose_handlers.py) reuses the exhausted pose's saved key capture and measured joints, selects a target pose/release cell, and invokes `runner.preflight_repose()` with directed v8 reset seeds. |
| `repose_execution_gate_required`, `await_repose_observation` | [`LiveReposeExecutionHandlers`](precision_insertion/live_repose_execution.py) begins a separate reset attempt, invokes a commissioned executor for the planned pickup/place/clear-view exit, checks its log, recaptures the landed key on the same AutoDex cameras and records `reorient_success`. |

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

[`LiveHeldRelationHandlers`](precision_insertion/live_held_relation_handlers.py)
connects the likely occluded-key branch. Its
`calibration_for_candidate(candidate_id)` callback must return the reviewed
`precision_insertion_physical_grasp_calibration_v1` for **that** selected v8
grasp and mode. The runner verifies its source samples, candidate and saved
lift checkpoint before planning a bounded transfer. No such calibration
has been found in the local shared assets, so this branch cannot yet advance
on real data; a simulated squeeze transform is not an acceptable substitute.
For a visible key, select `held_relation_route="observed"` in
`assemble_live_components()`. Its separate handler captures the key after
lift, aligns the measured wrist to the image exposure, and replans transfer
from the observed relation. This fresh observed relation supersedes a saved
previous-trial wrist-bias hypothesis; the default physical-calibration route
applies the saved bias to the next attempt's wrist target. Failed pose
admission never falls back to the nominal BODex relation.

[`LiveTaskHandlers`](precision_insertion/live_task_handlers.py) connects the
first non-contact transfer and 20 mm guarded insertion to their existing
execution/checkpoint primitives. It reuses the opened AutoDex camera IDs,
records a separate pre-insertion arrival label, and then passes the guarded
controller's depth/alignment/force record plus final images to the insertion
checkpoint. Its robot adapters, raw-frame producers and numeric limits are
injected on the AutoDex PC. The separate
[`LiveGroundedRetryHandlers`](precision_insertion/live_retry_handlers.py)
only composes guarded withdrawal, a fresh raw multi-view diagnostic, and a
candidate/pose-specific next-attempt advice record. The held-return handler
then releases the key; a fresh SAM/FoundPose key capture and v8 pickup plan
are mandatory before that candidate is tried again. The next physical
attempt adjusts its calibrated wrist command, while the CAD key insertion
goal remains socket-centered and endpoint/path checks are rerun. Square
tip/shoulder grounding supplies XY and tilt but **not yaw**; a yaw error
needs separate evidence or a different grasp. No same-attempt lateral shift
or second guarded stroke is registered in the live action mapping.

[`LiveReposePlanningHandlers`](precision_insertion/live_repose_handlers.py)
handles pose exhaustion without another key exposure. Its target selector
chooses an eligible tabletop pose, release height and robot-frame XY cell;
the existing v8 reset planner decides whether that target is feasible.
[`LiveReposeExecutionHandlers`](precision_insertion/live_repose_execution.py)
connects the resulting plan to an injected robot-PC execution callback and
then reuses the *same* key SAM/FoundPose capture for the observed landing.
The callback must execute the selected reset plan and supply an independent
`precision_insertion_repose_execution_evidence_v1` log; no local physical
reset controller or measured landing can be fabricated here.

The required [`ClearViewForPerception`](precision_insertion/clear_view.py)
calls stock `FrankaExecutor.home(clear_view=True)` **before** the initial
ChArUco/socket image: Inspire opens and FR3 parks at `FR3_CLEAR_VIEW`.
Measured stationary joints must agree. Every fresh key capture, including
`reobserve_key_and_preflight`, independently checks the arm and open hand
before capture and at camera exposure time. Same-capture candidate
continuation needs no new image. For a judged grasp miss,
`LiveTransferRecoveryHandlers` calls the stock AutoDex release/reset with the
selected v8 pickup plan, then recaptures and, if needed, repositions the key
before closing that attempt. Its copied reset world preserves the fixed socket
and original tabletop key as obstacles. For a still-held insertion failure,
the checked held return and release/exit are required instead. Neither path
uses free-space `home` after a trial. If measured clear view is not reached,
key recapture stops. The held-insertion-failure handler invokes the same
`ClearViewForPerception` return/exit motion, then uses the **same**
`collect_key_capture()` → SAM/FoundPose → `admit_key_capture()` path used by a
new trial. It captures the returned tabletop key *before* planning the next
grasp, and the runner labels the reset only from that fresh observation.

`compose_live_handlers(live, planning, pickup, lift, held, task, retry,
held_return, transfer_recovery, repose_planning, repose_execution)` in
[`live_orchestration.py`](precision_insertion/live_orchestration.py) checks
that all nonterminal action names are provided exactly once by components
bound to the same camera session. Use it as the `build_handlers` factory's
return value after constructing those ten components. The injected robot-PC
adapters, motion interlock, exposure-aligned feedback, Qwen backend and
per-stage limits are still required; composition does not supply them.

`stop_for_review` is a non-success terminal result. A verified positive
`insertion_success` produces `session_success`, which immediately returns
`status="success"` without release, reset, repose, or another grasp attempt.
For a positive **20 mm** label, configure the optional reviewed
[exposed-depth admission](DEPTH_ADMISSION.md) on the task handler;
an external controller's wrist-stroke/depth claim is not admitted by itself.
The live wrapper then closes its camera context. In `vlm_only` mode a
positive visual label can end the session as **experimental visual success**,
but it does not verify 20 mm penetration. A nominal trajectory or unknown
verdict does not establish task success. Recovery/reset actions not listed
above still need explicit handlers;
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

`franka_executor` below must be `src.execution.franka_executor.FrankaExecutor`
with the Inspire hand. `autodex.executor.real.RealExecutor` is the xArm
executor even though stock `run_auto.py` can select Franka through its
`--arm franka` branch. The demo clear-view gate now rejects the xArm
executor before its first `home()` command.

```python
from pathlib import Path
from precision_insertion.live_orchestration import run_live_precision_session
from precision_insertion.live_planning_handlers import LiveKeyPlanningHandlers
from precision_insertion.live_pickup_handlers import LivePickupHandlers
from precision_insertion.live_lift_handlers import LiveLiftHandlers
from precision_insertion.live_held_relation_handlers import LiveHeldRelationHandlers
from precision_insertion.observer import load_vlm_backend
from precision_insertion.franka_core import FrankaFeedbackSettings, open_franka_core

core = open_franka_core(FrankaFeedbackSettings(
    max_arm_hand_skew_s=commissioned_key_parameters["max_arm_hand_skew_s"],
    max_sample_age_s=commissioned_feedback_age_s,
    max_hand_command_error_raw=commissioned_key_parameters["max_hand_command_error_raw"],
    max_arm_velocity_rad_s=commissioned_key_parameters["max_arm_velocity_rad_s"],
))
franka_executor, planner = core.executor, core.planner
read_current_state, clear_view = core.read_state, core.clear_view
qwen_backend = load_vlm_backend(
    mode="local", max_input_size=commissioned_native_image_size)

def build_handlers(live):
    planning = LiveKeyPlanningHandlers(
        live=live, planner=planner,
        state_for_observation=feedback_sampler.state_for_observation,
        limits=path_audit_limits,
        key_prompt="white plastic cylindrical peg",
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
    held = LiveHeldRelationHandlers(
        live=live, planner=planner, read_state=read_current_state,
        calibration_for_candidate=commissioned_grasp_calibration_for_candidate,
        bounds=measured_surface_deviation_bounds,
        path_limits=path_audit_limits,
        parameters=commissioned_held_relation_parameters,
    )
    return {**planning.handlers(), **pickup.handlers(), **lift.handlers(),
            **held.handlers()}

try:
    result = run_live_precision_session(
        launch_config_path=Path("/absolute/commissioned_session_launch.json"),
        endpoint_catalog_path=Path("/absolute/current_endpoint_catalog.json"),
        run_output_dir=Path("/absolute/new_trial_run"),
        max_xy_retries=1, clear_view=clear_view,
        build_handlers=build_handlers,
    )
    print(result.to_record())
finally:
    core.close()
```

`feedback_sampler`, `path_audit_limits`, execution limits and
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
The subsequent transfer, guarded insertion, withdrawal, retry, reset and
repose boundaries are now connected as handlers in the demo; their real
controllers, independent execution logs and camera callbacks remain
AutoDex-PC integration work. The judged grasp-miss clear-view
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
        key_prompt="white plastic cylindrical peg",
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
