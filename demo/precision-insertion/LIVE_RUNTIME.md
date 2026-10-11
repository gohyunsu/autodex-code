# Run one live AutoDex precision-insertion session

`run_live.py` is the demo-local replacement for a precision-specific
`run_pipeline.py`/`run_auto.py` robot loop. It does not modify either stock
AutoDex file. One invocation parks FR3/opens Inspire, measures the ChArUco
board and fixed socket, freezes their collision world, then repeatedly
captures the key, selects a source-verified v8 grasp, executes/labels pickup,
lift, transfer and guarded 20 mm insertion, optionally obtains a VLM
multi-view XY update after each eligible failure and accumulates it for the
next physical attempt, then follows the existing recovery/repose decisions.
It ends immediately on an admitted insertion-success label.

The script does **not** include a robot/force controller or invent camera
exposure timestamps. The optional `bounded_request_interval` camera mode
uses the complete new-frame request/response interval as timing uncertainty
for stationary bring-up; slow FoundPose calls may still fail the unchanged
camera/feedback timing gates. On the AutoDex PC, an importable `module:factory` must return a
`LiveRuntimeIntegration(clear_view, build_components, close)`; the factory
receives `RunLiveOptions`. `build_components(live)` is called after board and
socket capture and must return these ten components, all bound to that same
`live` object:

1. `LiveKeyPlanningHandlers` — key SAM/FoundPose and v8 trial preflight.
2. `LivePickupHandlers` — selected grasp, measured squeeze and lift plan.
3. `LiveLiftHandlers` — executed lift and fresh Qwen grasp label.
4. `LiveHeldRelationHandlers` — physical grasp-specific key/hand relation;
   `LiveObservedHeldRelationHandlers` for a newly visible post-lift key; or
   explicitly selected `LiveNominalRelationHandlers` for an **unmeasured v8
   key–hand hypothesis** in the repeatable-bias experiment.
5. `LiveTaskHandlers` — transfer, arrival label, first guarded insertion.
6. `LiveGroundedRetryHandlers` — guarded withdrawal and continuous multi-view
   cylinder XY diagnosis; an at-most-1 mm update is saved for a **new pickup**,
   not a same-held-attempt movement. Square live execution is disabled.
7. `LiveHeldReturnHandlers` — held-key return after failed insertion.
8. `LiveTransferRecoveryHandlers` — optional planned held return when a
   completed transfer has measured hand-pose error but the pre-insertion VLM
   still sees the key matching the held prediction; supervised recovery for
   stopped pickup/lift/transfer, Qwen abstention, suspected slip or other
   uncertain states.
9. `LiveReposePlanningHandlers` — directed v8 reorient planning.
10. `LiveReposeExecutionHandlers` — selected reset execution and new key pose.

The demo provides `open_franka_core(FrankaFeedbackSettings(...))` in
`precision_insertion.franka_core` for the common robot resources. It creates
a [demo-local completion-checked subclass](FRANKA_COMPLETION_CHECK.md) of the
unchanged `src.execution.franka_executor.FrankaExecutor(hand_name="inspire")`,
`GraspPlanner(hand="fr3_inspire")`, measured FR3/Inspire state reader and
clear-view gate. The integration factory can pass `core.clear_view` and
`core.close` to `LiveRuntimeIntegration`, and share `core.planner` and
`core.read_state` across stages. Construction connects to the robot but does
not command `home()`; the live session does that after explicit motion opt-in.
Do not use `autodex.executor.real.RealExecutor`: it controls xArm, and the
clear-view gate rejects it. This core does not supply a robot-side watchdog,
guarded contact, camera timestamps or the remaining hardware callbacks.

For the first-party FR3 resource path, create a robot-PC JSON file such as:

```json
{
  "schema": "precision_insertion_franka_integration_v1",
  "feedback": {
    "max_arm_hand_skew_s": 0.1,
    "max_sample_age_s": 1.0,
    "max_hand_command_error_raw": 20.0,
    "max_arm_update_wait_s": 0.2,
    "max_arm_velocity_rad_s": 0.05
  },
  "sampling": {
    "max_samples": 100,
    "max_bracket_span_s": 0.1,
    "max_key_state_skew_s": 0.05,
    "max_arm_hold_drift_rad": 0.002,
    "max_hand_hold_drift_raw": 2.0,
    "max_sampling_hand_command_error_raw": 200.0,
    "max_sampling_arm_velocity_rad_s": 2.0,
    "period_s": 0.01,
    "warmup_timeout_s": 2.0
  },
  "vlm": {
    "mode": "local",
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "max_new_tokens": 512,
    "require_native_pixels": true,
    "capture_timeout_s": 10.0
  },
  "stages_factory": "my_robot_pc_binding:configure_stages"
}
```

These numbers illustrate the file format, **not** rig-approved limits.
The `vlm` block is optional. If the stage factory does not supply `backend`,
the integration loads local Qwen by default using the active camera image
size; Gemini requires an explicit `mode: "gemini"` and `model_id`. The
integration also fills missing lift, pre-insertion/final, and retry checkpoint
callbacks with fresh snapshots from the already-open AutoDex cameras. A
stage factory may supply a custom callback for any field. These defaults do
not invent exposure timestamps or measured robot state. The native-pixel
setting pads only the right/bottom edge to Qwen's patch unit without moving
source pixels. Images outside the model's pixel-area budget still need the
existing pixel-preserving ROI route for metric grounding.
For temporal lift and insertion labels, local Qwen now receives one
same-camera before/after pair per call, not all 20 camera pairs in one
request. A positive visual class needs at least two independently agreeing
camera replies and no conflicting decisive reply. Each raw per-camera answer,
prompt and image order is retained inside the aggregate checkpoint record.
This bounds the image count per model call; it does not establish VLM accuracy
or guarantee that a full-resolution two-image call fits the robot PC GPU.
`configure_stages(core, live, options)` returns `{"shared": {...},
"stages": {...}, "held_relation_route": "physical"}` for the existing
`assemble_live_components`. The core injects its own `planner` and
`read_state`. If the optional `sampling` block is present, it starts the
long-lived feedback sampler before camera capture and also injects
`state_for_observation` and `state_for_capture`. It records moving samples
without killing the thread, but image lookups still require a stationary
Franka/Inspire hold throughout exposure. The stage factory may omit
`pickup.executor`: the integration then passes the same demo-local
completion-checked FR3 executor used for clear-view and feedback. It must
still supply pickup limits and a real motion interlock; the subclass does
not replace a robot-side watchdog or physical review. The stage factory
supplies the remaining commissioned contact, recovery and repose callbacks.
An optional [`held_motion_monitor` block](MEASURED_LIFT_ADAPTER.md) builds
one measured non-contact adapter for lift and first transfer. Retry requires
only the independently commissioned guarded axial withdrawal; there is no
same-attempt lateral or retry-transfer adapter. The older
`lift_monitor` block still fills only `lift.adapter`. The factory may override
each field. Neither setting supplies a reviewed daemon watchdog record,
stage limits, interlock, contact controller or real-robot validation.

For the default physical held-relation route, the stage factory may omit
`held_relation.calibration_for_candidate`: assembly then looks for the exact
candidate under the session's canonical
`shared_root/AutoDex/precision_insertion/physical_grasp_calibrations` layout.
Generate those grasp-specific records from independent physical pickups using
[`INSERTION_GRASP_CALIBRATION.md`](INSERTION_GRASP_CALIBRATION.md). A missing
record stops at the held-relation stage; no simulated key pose is substituted.
For a deliberate no-physical-calibration **experimental cylinder** run,
return `"held_relation_route": "nominal"` from the robot-PC stage factory.
Its `held_relation` stage supplies `limits` (`PathAuditLimits`) and a
`parameters` mapping containing `max_state_age_s`, `max_arm_hand_skew_s`,
`max_hand_command_error_raw`, `max_arm_velocity_rad_s`,
`max_postlift_arm_drift_rad`, `max_postlift_hand_drift_raw`, and
`axial_waypoint_step_m`. These values come from the actual robot setup.
The selected v8 `wrist_se3.npy` is an *assumed* key–hand relation, never a
post-squeeze measurement. A fresh measured joint state, the exact 20 mm
endpoint screen and sampled transfer/axial preflight still run. The saved
report uses a separate nominal schema, so downstream transfer, visual
checkpoint, guarded insertion and failed-trial diagnostic retain that
provenance. This mode tests whether next-trial XY learning absorbs a
repeatable translation bias; it does not establish physical held-key pose,
axis tilt or grasp repeatability.

Run:

```bash
python demo/precision-insertion/run_live.py \
  --launch-config /absolute/session_launch.json \
  --endpoint-catalog /absolute/endpoint_catalog.json \
  --output-dir /absolute/new_run \
  --integration precision_insertion.franka_integration:factory \
  --robot-config /absolute/franka_runtime.json \
  --enable-robot-motion
```

The JSON/stage module is read before the FR3 connection; the planner is
initialized before the executor. On process exit, the built-in integration
shuts down the Franka executor. This command still cannot run with only the
illustrative file: robot-PC stage callbacks and commissioned contact control
remain required.

The handler composer rejects missing/duplicate actions before any grasp. The
module should supply commissioned robot-side adapters and may override the
first-party Qwen backend or checkpoint callbacks when needed.
The pickup executor should implement `stop_and_acknowledge()` in addition to
`execute(...)`; the lift and transfer adapters already need that stop method.
Only an acknowledged stop enters the supervised-recovery action. Without one,
the loop raises and does not move to another grasp.
See [ACTION_LOOP.md](ACTION_LOOP.md) for each constructor's existing
parameters and the exact decision-to-handler mapping. The `close` callback
must release robot-side resources even if a handler raises. The demo closes
its own camera context first.

For automatic recovery after a completed but misaligned transfer, configure
`transfer_recovery` with the same `planner` and `limits` as held return,
`read_state`, `record_execution`, and `return_parameters`. The latter has the
held-return release/clearance limits plus `max_transfer_state_gap_s` instead
of `max_withdrawal_state_gap_s`. The execution recorder writes the existing
`precision_insertion_failed_return_execution_v1` schema with
`return_origin: "transfer"` and five ordered phases: `held_transfer`,
`held_descent`, `release_open`, `post_release_lift`,
`post_release_retract`. There is no fictitious guarded withdrawal. If the
visual checkpoint says `slip_or_miss` or `gross_misalignment`, the transfer
was incomplete/aborted, or these automatic-return adapters are absent, the
handler uses `recover_to_tabletop` and its supervised reset record. A real
key image after clear view decides whether the return actually restored the
original tabletop pose; the plan alone never counts as reset success.

To avoid writing ten separate constructor calls on the robot PC, use
`assemble_live_components` from `precision_insertion.runtime_assembly` in
`build_components(live)`. Supply a `shared` mapping for the single planner,
feedback sampler, Qwen backend and other identical dependencies, plus a
`stages` mapping with exactly these keys: `key_planning`, `pickup`, `lift`,
`held_relation`, `task`, `retry`, `held_return`, `transfer_recovery`,
`repose_planning`, `repose_execution`. Each stage mapping contains only that
handler's remaining constructor fields. The assembler injects `live` and the
CLI motion opt-in into every applicable handler, and connects
`repose_planning.key_planning` to the same instance used for key capture.
It rejects missing stages, unknown fields and an accidental override of a
shared dependency. For example, the factory's `build_components` can end with:

```python
from precision_insertion.runtime_assembly import assemble_live_components

def build_components(live):
    return assemble_live_components(
        live, shared=shared_robot_objects,
        stages=stage_specific_callbacks_and_limits,
        enable_robot_motion=options.enable_robot_motion,
        held_relation_route="physical")
```

This removes integration boilerplate; it does not synthesize camera frames,
robot feedback, physical grasp calibration or an insertion controller.
Set `held_relation_route="observed"` only when the held key is visible enough
for a **new** multi-view FoundPose measurement after Qwen's positive lift
label. This route uses the measured wrist plus selected v8 grasp solely as a
loose search prior; the wrist sample is selected from the buffered feedback
at the new images' exposure time, not before the camera request. It then
replans from the observed key–hand relation. Failed pose admission stops that
attempt; it does not silently substitute the nominal BODex transform. The
optional `held_relation.calibration_sample_bounds` exports that accepted
physical key/wrist pair for later grasp-specific commissioning; see
[`INSERTION_GRASP_CALIBRATION.md`](INSERTION_GRASP_CALIBRATION.md). The bounds
must be supplied from rig measurements, not chosen to make a sample pass.
The observed route uses a fresh measured key/hand relation in its next
attempt; that direct evidence supersedes a learned wrist-bias hypothesis.
The physical-calibration route remains the default and adds any saved
previous-trial socket-XY compensation to the grasp-specific medoid command.
The nominal route applies the same correction to the selected v8 hypothesis.
The key CAD target stays centered; the corrected hand/key endpoint and held
path are re-screened before transfer.

The cylinder live path uses one **future-trial advice** handler (see
[`NEXT_TRIAL_XY.md`](NEXT_TRIAL_XY.md) for the attempt boundary and transform):
guarded axial withdrawal, fresh raw AutoDex multi-view images, metric
tip/axis reconstruction, save an at-most-1-mm socket-XY increment, return
the key to the tabletop, clear view, and start a new perception/pickup
attempt. There is no lateral hold shift or second insertion in the failed
attempt. The cylinder VLM marks a tip and visible shaft-axis line.
Square remains an offline asset/planning mode with **no XY correction**;
its effective update weight and retry budget are zero. Historical square
diagnostic helpers are not assembled into the active pipeline.
If the marks are hidden or disagree across views, the diagnostic abstains
and follows held-key return without saving compensation. The older
`LiveObservedXYRetryHandlers` is not selected by runtime assembly and its
same-attempt action is unsupported by the session policy. The robot-PC
factory still supplies actual withdrawal/contact adapters and limits.
A malformed request ID, changed frame provenance or controller failure is
still an error requiring review, not a visual abstention.

`LiveReposePlanningHandlers` now uses `DirectedReposeTargetSelector` by
default. When a tabletop pose has no remaining usable grasp, it checks each
directed cell with the same full-key MuJoCo/fidelity reset-seed loader used
by the later preflight. It skips target poses with no eligible seed, then
ranks the remaining poses by eligible insertion-grasp count. It first tries the key's
current robot-frame XY as the release site, then four nearby 30 mm cells at
the v8 12 cm reset height. Each choice still goes through the existing
socket-aware full pickup/held/release/clear-view preflight. Exhausting the
finite choices writes `repose_targets_exhausted.json` and ends in
`stop_for_review`, rather than planning the same failed reset forever.
The handler defaults to AutoDex `FR3_CLEAR_VIEW` for the exit and 5 mm
minimum release-key clearance; both can be overridden in its `parameters`.
Set `reset_candidate_root` there to the **exact** installed `reset_12`
directory if the curated v8 reset seeds are not at the canonical path.
For the current cylinder staging root
`shared_data/AutoDex/precision_insertion/cylindrical/reorient_curated_handoff_20261011/reset_12`,
the loader with 5 mm center-in-hand and 10° axis-tilt limits accepts one
`0→1` seed and no `1→0` seed. These are offline proposal results, not a
socket-aware Franka path or physical reset success.
For square 1.5 mm, the corresponding
`shared_data/AutoDex/precision_insertion/square/reorient_curated_handoff_20261011/reset_12`
has one eligible `0→4` seed under the same 5 mm drift / 10° full-rotation
limits; `1→4`, `2→4`, and `3→4` have none. Set the root for the chosen task
mode explicitly—these incomplete directed graphs are not universal reset
coverage.
Square 0.5 mm pose `000` is the one remaining tabletop pose without an
eligible insertion grasp in the 1,000-per-scene batch. Its staged
`0→4` reset root is
`shared_data/AutoDex/precision_insertion/square/reorient_0p5_curated_5000_20261011/reset_12`.
Of 33 source-verified MuJoCo-stable seeds, two pass the illustrative 5 mm
drift / 10° full-rotation gate. The selector may therefore offer `0→4` to
the socket-aware Franka preflight; a passing seed alone is not permission
to move, and a failed preflight still stops for review. Three hypothetical
release-site probes of those two seeds found no complete Franka path; one
same-state collision isolation implicated the hypothetical table obstacle.
This is a planner diagnostic, not a claim about every measured session.

After the XY diagnostic (whether advice or abstention), the key is already
at the withdrawn hold. The assembled held-return handler
reuses that withdrawal record and fresh measured joint state, instead of
commanding a second axial withdrawal. It accepts reuse only if the joints
remain at the withdrawn hold.
`max_withdrawal_state_gap_s` in the held-return settings must cover the
elapsed VLM/planning time; otherwise the return raises before motion.
The robot-PC integration must supply a real `read_state` and the
original guarded withdrawal controller; this wiring does not implement it.

For VLM checkpoints, `make_checkpoint_capture` in
`precision_insertion.checkpoint_camera` supplies the existing handlers'
`(live, runner, decision)` callback. Its default `snapshot` transport uses
the **same open AutoDex snapshot cameras** without asking SAM/FoundPose to
recognize a partly occluded held key. It undistorts the original snapshot
using the session camera calibration, and saves both image versions so that
the VLM bundle can replay that exact derivation. Instantiate a callback per
phase (`after_lift`, `preinsert`, `final_or_abort`);
repeated calls create fresh capture IDs. `transport="init_key"` retains the
older key-init capture path when FoundPose outputs are explicitly wanted.
The default still requires complete frame-ID/exposure metadata from the
AutoDex snapshot adapter, but **does not require a key mask or pose**.
`state_for_capture` remains a separate exposure-aligned robot feedback
sampler; reading joints after Qwen inference is not equivalent.

The existing `ExposureStateBuffer`/`RobotFeedbackSampler` in
`precision_insertion.feedback_buffer` now also supply the two callbacks used
at VLM checkpoints: `read_state(max_age_s=...)` for a recent measured hold
and `state_for_capture(raw_capture)` for feedback bracketing **all** camera
exposure times. Start the sampler before the first camera request, using a
reader based on `read_live_franka_inspire_state` and the same
`FrankaExecutor.arm`/`.hand`; close it in the integration's `close` callback.
For a session-long sampler, use the wider sampling velocity/hand-error limits
for both its reader and `ExposureStateBuffer` admission, while retaining
the tighter stationary limits for image lookup. The first-party `sampling`
block above wires this distinction and closes the sampler automatically.
The pre-existing `state_for_observation(key_observation)` remains the input
to key FoundPose planning. No joint state is inferred from a planned path or
read only after Qwen inference. These are robot-PC receipt timestamps, not
hardware-latched encoder times; the camera/robot clock assumptions still
need validation on the recipient PC.

From the repository root, after the integration module and assets are
installed on the AutoDex PC:

```bash
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src:/path/to/robot_pc_integration" \
  python demo/precision-insertion/run_live.py \
  --launch-config /absolute/path/to/live_session.json \
  --endpoint-catalog /absolute/path/to/current_endpoint_catalog.json \
  --output-dir /absolute/path/to/new_run_directory \
  --integration my_robot_integration:create_integration \
  --xy-update-weight 1.0 --enable-robot-motion
```

Use a **new** output directory; the command does not overwrite an earlier
run. `--enable-robot-motion` is mandatory because the first clear-view move
occurs before camera capture. `--max-xy-retries 0` disables next-trial XY
advice; `1` permits one candidate-specific corrected **new physical attempt**
per session. By default there is no retry-count cap: eligible failures can
continue to contribute cumulative updates
`p_(n+1)=p_n+w*d_n`; the weight defaults to `1.0` and must lie in `(0,1]`.
The robot integration's `RetrySessionLimits.max_total_offset_m` must allow
the desired cumulative travel. The live CLI currently accepts only
`cylinder` launch mode; `square` remains an offline/planning mode.
`action_loop_result.json` is saved in the run directory. Exit code 0 means
the session reached its task-success label; 2 means a normal non-success
terminal/step-budget result. An exception is not converted into a success or
automatic retry; existing attempt/capture evidence remains on disk.

Required recipient inputs remain distinct from code completeness:

- A current, complete v8 endpoint catalog rebuilt for the recipient's
  shared-data paths and selected key/socket mode, plus matching object meshes
  and FoundPose representations.
- Acquisition-timestamped multi-view AutoDex cameras and exposure-aligned
  measured Franka/Inspire feedback.
- Either a per-grasp physical held-relation calibration, a visible
  post-lift multi-view key observation, or an explicitly opted-in unmeasured
  nominal v8 hypothesis; and commissioned pickup, lift, transfer,
  guarded-contact, withdrawal and reset controllers with their independent
  execution records.
- Under `evidence_fused`, for a **positive verified 20 mm label**, the optional reviewed visual-depth error
  audit described in [DEPTH_ADMISSION.md](DEPTH_ADMISSION.md). Without it,
  unverified controller stroke depth remains diagnostic and a visually
  normal insertion is not marked successful by that mode.

The current workstation has neither a real AutoDex-PC integration module nor
the physical depth commissioning audit. Running this command here cannot
complete a real insertion. The CLI and synthetic tests establish the wiring
contract, not robot readiness.

## Experimental task-label modes

`run_live.py --label-mode vlm_only` is the default development policy. The
post-lift, pre-insertion and final/abort images go to the configured VLM;
the corresponding multi-view visual classes alone determine the three task
labels. `held`, `coarse_match` and `normal_appearance` are positive;
visible failure classes are negative; occlusion or insufficient consensus is
unknown. A visual insertion label is **not** a verified 20 mm penetration.
The controller's motion/abort mechanism and geometric preflight remain
separate from this label policy.

Use `--label-mode manual --annotator YOUR_NAME` for directly entered labels.
After each capture the terminal shows the saved PNG directory and asks for
`y`, `n` or `u`, plus an optional note. The VLM answer is still saved for
comparison but the named human label drives the task state. This terminal
workflow blocks for input; a robot-PC integration needs an operator interface
and watchdog before unattended physical use. The next physical attempt uses
the same label mode. `--label-mode evidence_fused` restores the prior VLM-plus-metric
criterion.

The checkpoint report records `label_mode`, `manual_annotation` and source
pixels; verification recomputes the label and rejects a changed Boolean.
Third-party factories can pass `options.label_mode` and a callable
`manual_label_provider(stage, evidence_dirs, attempt_id, candidate_id)` to
`assemble_live_components`.
