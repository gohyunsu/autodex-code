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

The default camera startup follows stock AutoDex order: measure the empty
board, wait for the operator to place the fixed white socket, then capture
the socket repeatedly. The explicit `socket_present_partial` launch mode
supports the currently installed board 9 with the socket already present:
its visible corners must span the board and pass the same reprojection,
plane, and rigid-fit limits. The stock board-11 default still requires all
54 corners. Neither mode sends a robot motion command.

Socket initialization uses current AutoDex cross-view IoU selection and
silhouette refinement. Per-view IoU scores and threshold supporters remain
evidence; a low-scoring view alone no longer vetoes a passing stock result.
The session still needs acquisition-bound frames and agreement between
separate socket captures before its fixture pose can be frozen.

**Current cylinder contract (2026-10-11):** the assembled held-relation
handler defaults to the selected v8 nominal hypothesis; cylinder live mode
rejects physical/observed post-grasp key–hand routes. Failed-trial advice is
at most 10 mm per update, weighted by `--xy-update-weight` (default 1.0),
with no separate cumulative offset cap. Its VLM, human and metric records
remain distinct. If the first failed-return placement lands outside the
original pose/XY tolerance, the same attempt plans and executes another
pickup/place and recaptures from clear view before recording reset success.
See [NEXT_TRIAL_XY.md](NEXT_TRIAL_XY.md). Older discussions later on this
page describe optional/historical branches, not the active cylinder loop.

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
4. `LiveNominalRelationHandlers` — selected v8 **unmeasured key–hand
   hypothesis** for cylinder live execution; no post-grasp key pose estimate.
5. `LiveTaskHandlers` — transfer, arrival label, first guarded insertion.
6. `LiveGroundedRetryHandlers` — read the **saved pre-insertion** raw views
   after a released-key failure and calculate continuous multi-view cylinder
   XY advice; an at-most-10 mm update is saved for a **new pickup**, with no
   same-attempt movement or new failure-time capture. Square live execution
   is disabled.
7. `LiveHeldReturnHandlers` — after a released-key failure, clear-view
   recapture and, if misplaced, original-pose pick/place restoration in the
   same attempt; separate held-key recovery paths remain for earlier failures.
8. `LiveTransferRecoveryHandlers` — judged grasp-miss AutoDex clear-view
   reset, fresh key capture and same-attempt original-pose restoration;
   optional planned held return when a
   completed held transfer ends with visible `coarse_match` or
   `gross_misalignment`; supervised recovery for stopped pickup/lift/transfer,
   Qwen abstention, suspected slip or other uncertain states.
9. `LiveReposePlanningHandlers` — directed v8 reorient planning.
10. `LiveReposeExecutionHandlers` — selected reset execution and new key pose.

The demo provides `open_franka_core(FrankaFeedbackSettings(...))` in
`precision_insertion.franka_core` for the common robot resources. It creates
a [demo-local completion-checked subclass](FRANKA_COMPLETION_CHECK.md) of the
unchanged `src.execution.franka_executor.FrankaExecutor(hand_name="inspire")`,
`GraspPlanner(hand="fr3_inspire")`, measured FR3/Inspire state reader and
clear-view gate. The integration factory can pass `core.clear_view` and
`core.close` to `LiveRuntimeIntegration`, and share `core.planner` and
`core.read_state` across stages. Construction is **not feedback-only**: the
stock executor performs FR3 stop/recovery/collision setup and the Inspire
constructor writes angle, speed and force registers. Call it only after
explicit motion opt-in. For nonmoving connection checks use
[`probe_robot_feedback_only.py`](probe_robot_feedback_only.py).
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
    "model_id": "/absolute/recipient/qwen3-vl-2b/payload/model",
    "max_new_tokens": 512,
    "require_native_pixels": true,
    "capture_timeout_s": 10.0
  },
  "stages_factory": "my_robot_pc_binding:configure_stages"
}
```

These numbers illustrate the file format, **not** rig-approved limits.
All keys and sockets are white. Use a socket SAM prompt such as
`fixed white cylindrical socket on the ChArUco board` and a key prompt such
as `loose white cylindrical peg on the tabletop`. The launch and capture
adapters reject stale nonwhite color words before sending a SAM request.
Set `vlm.model_id` to the recipient's verified NAS Qwen bundle
`payload/model` directory. If omitted, the built-in FR3 integration selects
`shared_root/AutoDex/models/Qwen3-VL-2B-Instruct`; a custom stage factory
may instead supply `backend`. The first-party Transformers loader reads only
local model files and does not import ZeroDex's `main.vlm_base`. The
integration uses the active camera image size;
Gemini requires an explicit `mode: "gemini"` and `model_id`. The
integration also fills missing lift, pre-insertion/final, and retry checkpoint
callbacks with fresh snapshots from the already-open AutoDex cameras. A
stage factory may supply a custom callback for any field. These defaults do
not invent exposure timestamps or measured robot state. The native-pixel
setting pads only the right/bottom edge to Qwen's patch unit without moving
source pixels. Images outside the model's pixel-area budget still need the
existing pixel-preserving ROI route for metric grounding.
For temporal lift, preinsert and released insertion labels, local Qwen receives
one same-camera raw before/after pair per call, not all 20 camera pairs in one
request. A positive visual class needs at least two independently agreeing
camera replies and no conflicting decisive reply. Each raw per-camera answer,
prompt and image order is retained inside the aggregate checkpoint record.
The preinsert pair is `after_lift` followed by `preinsert`. Its checkpoint
also binds both capture manifests, camera/frame IDs, exposure times and PNG
hashes, and replay verifies the original model text. The visual class only
asks whether the same key remains visibly held and roughly above the socket.
This bounds the image count per model call; it does not establish VLM accuracy
or guarantee that a full-resolution two-image call fits the robot PC GPU.
`configure_stages(core, live, options)` returns `{"shared": {...},
"stages": {...}}` for the existing
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
one measured non-contact adapter for lift and first transfer. The task stage
must additionally supply `release_and_clear`; it opens Inspire after the
single 20 mm stroke only when the bound controller reports no safety abort
and a verified held key. An abort, lost key, or unknown grasp state keeps the
terminal hold for supervised recovery without release or final task capture.
On a permitted release, the callback follows a commissioned collision-checked
exit to FR3 clear view. The callback returns a
`precision_insertion_poststroke_release_clear_view_v1` JSON record bound to
the guarded metric record. Retry itself has no motion or capture adapter: it
reuses the saved pre-insertion snapshot. The active released-failure branch
does not command a guarded withdrawal or take another held-key photograph.
The older
`lift_monitor` block still fills only `lift.adapter`. The factory may override
each field. Neither setting supplies a reviewed daemon watchdog record,
stage limits, interlock, contact controller or real-robot validation.

For the **historical square/offline** physical held-relation route, the stage factory may omit
`held_relation.calibration_for_candidate`: assembly then looks for the exact
candidate under the session's canonical
`shared_root/AutoDex/precision_insertion/physical_grasp_calibrations` layout.
Generate those grasp-specific records from independent physical pickups using
[`INSERTION_GRASP_CALIBRATION.md`](INSERTION_GRASP_CALIBRATION.md). A missing
record stops at the held-relation stage; no simulated key pose is substituted.
For the active cylinder run, omit `held_relation_route` or explicitly return
`"held_relation_route": "nominal"` from the robot-PC stage factory.
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

For released-key original-pose restoration, the robot-PC stage factory must
provide `stages["held_return"]["execute_restoration"]`. Its callable receives
`(runner, decision, plan, preflight_report_path)` and returns an absolute
execution-log path. `plan` is a `ReposeTransitionPreflight` assembled from
the freshly recaptured key, measured FR3/Inspire joints, the original
tabletop class/CAD-center XY, eligible v8 pickup candidates or directed reset
seeds, socket-aware held transfer, and release/exit trajectories. The
callback must execute that exact plan and write a
`precision_insertion_repose_execution_evidence_v1` log for the **current
failed attempt ID**; the handler verifies it before another clear-view key
capture. The same helper is wired into completed-transfer recovery. If no
placement is needed, the callback is not called. No new key/socket mesh or
same-pose reset seed is required: an eligible full-key v8 grasp from the
source tabletop pose is used for same-pose relocation, with the previous
attempt's candidate tried first when it is still eligible. Cylinder live
assembly rejects a missing callback before pickup, instead of discovering
that gap only after a failed insertion. If neither a directed
reset seed nor an eligible source-pose grasp has a complete planned path,
restoration remains unavailable for that observed scene; it is not reported
as a successful reset. `restoration_height_cm` defaults to 12 and
`max_restoration_attempts` to 3 in the held-return stage; both can be set by
the robot-PC factory.

Run:

```bash
python demo/precision-insertion/run_live.py \
  --launch-config /absolute/session_launch.json \
  --endpoint-catalog /absolute/endpoint_catalog.json \
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
visual checkpoint says `slip_or_miss` or is inconclusive, the transfer
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
  held_relation_route="nominal")
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
The physical-calibration route is historical/offline for this cylinder demo.
The active nominal route applies compensation to the selected v8 hypothesis.
The key CAD target stays centered; the corrected hand/key endpoint and held
path are re-screened before transfer.

The cylinder live path uses one **future-trial advice** handler (see
[`NEXT_TRIAL_XY.md`](NEXT_TRIAL_XY.md) for the attempt boundary and transform):
reopen the raw AutoDex multi-view images captured at the pre-insertion hold,
run metric tip/axis reconstruction, save an at-most-10-mm socket-XY increment,
perceive the key already released after the stroke, restore its original
tabletop pose/position, and start a new perception/pickup attempt. There is no
post-failure lift/capture, lateral hold shift, or second insertion in the
failed attempt. The cylinder VLM marks a tip and visible shaft-axis line.
Square remains an offline asset/planning mode with **no XY correction**;
its effective update weight and retry budget are zero. Historical square
diagnostic helpers are not assembled into the active pipeline.
If full tip/axis marks are unavailable, use a calibrated multi-view tip,
single-view nominal-height intersection or explicitly exploratory step,
according to the available evidence. These have distinct diagnostic statuses
and the saved next-trial advice records the source and supporting cameras;
an exploratory step is not a measured XY error. The older
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

The active released-failure branch ends its 20 mm stroke by opening the hand
and moving to clear view **before** taking the final snapshot. If the raw
initial/final camera pairs show the key visibly outside the socket, the
retry handler reuses the earlier pre-insertion snapshot for next-attempt XY
advice. The held-return handler then observes the released key and restores
its original tabletop pose/position before a new attempt. The older guarded
withdrawal and held-key return utilities remain available to other opt-in
paths, but they are **not** the active cylinder insertion-failure route.

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

From the repository root, after the launch files and assets are installed
on the AutoDex PC:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_live.py --preflight \
  --launch-config /absolute/path/to/live_session.json \
  --endpoint-catalog /absolute/path/to/current_endpoint_catalog.json \
  --robot-config /absolute/path/to/franka_runtime.json
```

`--preflight` validates the cylinder launch, camera calibration files, exact
recipient-bound v8 catalog, and robot config syntax. It does not import the
external stage factory, construct CUDA/FR3, connect cameras, create a run
directory or command motion. It always reports `robot_ready: false`; separate
camera capture, stage assembly and controller commissioning are still needed.
The `--robot-config` argument may be omitted when only camera/catalog inputs
are installed. A launch created only to inspect current recipient files may
set `"execution_scope": "static_preflight_only"`; the camera capture and
motion entry points reject that launch, while `--preflight` can read it.

For the full motion-enabled loop, after the stage adapters and guarded
controller are commissioned:

```bash
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src:/path/to/robot_pc_integration" \
  python demo/precision-insertion/run_live.py \
  --launch-config /absolute/path/to/live_session.json \
  --endpoint-catalog /absolute/path/to/current_endpoint_catalog.json \
  --integration my_robot_integration:create_integration \
  --xy-update-weight 1.0 --enable-robot-motion
```

The default exp name is `precision-insertion`. The default run directory is
`~/shared_data/AutoDex/experiment/precision-insertion/inspire/<key_object>/_pipeline_runs/<timestamp>`;
each physical episode is saved at the sibling `<key_object>/<episode_id>/`
with `result.json`, append-only `state_*.json`, and its raw/checkpoint evidence.
The run's `attempts/<episode_id>` is a link to that same episode directory.
The object-level `summary.json` indexes episode results, and the run keeps
`action_loop_result.json` plus frozen source bindings. `--exp-name` selects
another stock experiment directory; `--output-dir` explicitly overrides the
default run path. New run IDs do not overwrite earlier runs.
`--enable-robot-motion` is mandatory because the first clear-view move
occurs before camera capture. `--max-xy-retries 0` disables next-trial XY
advice; `1` permits one candidate-specific corrected **new physical attempt**
per session. By default there is no retry-count cap: eligible failures can
continue to contribute cumulative updates
`p_(n+1)=p_n+w*d_n`; the weight defaults to `1.0` and must lie in `(0,1]`.
There is no separate cumulative XY bound. The live CLI currently accepts only
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
post-lift and raw pre-insertion images go to the configured VLM. Final task
success is evaluated only after the 20 mm stroke, full hand opening and
clear-view exit: each camera contributes an initial-tabletop/final-clear-view
raw image pair. No CAD overlay is supplied to either VLM checkpoint. The
corresponding multi-view visual classes alone determine the three task
labels. `held`, `coarse_match` and `inside_socket` are positive;
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
