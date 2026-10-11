# Run one live AutoDex precision-insertion session

`run_live.py` is the demo-local replacement for a precision-specific
`run_pipeline.py`/`run_auto.py` robot loop. It does not modify either stock
AutoDex file. One invocation parks FR3/opens Inspire, measures the ChArUco
board and fixed socket, freezes their collision world, then repeatedly
captures the key, selects a source-verified v8 grasp, executes/labels pickup,
lift, transfer and guarded 20 mm insertion, optionally performs one VLM
multi-view XY correction, and follows the existing recovery/repose decisions.
It ends immediately on an admitted insertion-success label.

The script does **not** include a robot/force controller or invent camera
timestamps. On the AutoDex PC, an importable `module:factory` must return a
`LiveRuntimeIntegration(clear_view, build_components, close)`; the factory
receives `RunLiveOptions`. `build_components(live)` is called after board and
socket capture and must return these ten components, all bound to that same
`live` object:

1. `LiveKeyPlanningHandlers` — key SAM/FoundPose and v8 trial preflight.
2. `LivePickupHandlers` — selected grasp, measured squeeze and lift plan.
3. `LiveLiftHandlers` — executed lift and fresh Qwen grasp label.
4. `LiveHeldRelationHandlers` — physical grasp-specific key/hand relation,
   or `LiveObservedHeldRelationHandlers` for a newly visible post-lift key.
5. `LiveTaskHandlers` — transfer, arrival label, first guarded insertion.
6. `LiveGroundedRetryHandlers` — guarded withdrawal, VLM ≤1 mm XY correction,
   renewed planning and one guarded reattempt.
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
the existing `src.execution.franka_executor.FrankaExecutor(hand_name="inspire")`,
`GraspPlanner(hand="fr3_inspire")`, measured FR3/Inspire state reader and
clear-view gate. The integration factory can pass `core.clear_view` and
`core.close` to `LiveRuntimeIntegration`, and share `core.planner` and
`core.read_state` across stages. Construction connects to the robot but does
not command `home()`; the live session does that after explicit motion opt-in.
Do not use `autodex.executor.real.RealExecutor`: it controls xArm, and the
clear-view gate rejects it. This core does not supply guarded contact,
camera timestamps or the remaining hardware callbacks.

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
  "stages_factory": "my_robot_pc_binding:configure_stages"
}
```

These numbers illustrate the file format, **not** rig-approved limits.
`configure_stages(core, live, options)` returns `{"shared": {...},
"stages": {...}, "held_relation_route": "physical"}` for the existing
`assemble_live_components`. The core injects its own `planner` and
`read_state`; the stage factory supplies the remaining commissioned camera,
feedback-buffer, Qwen, pickup, contact, recovery and repose callbacks. Run:

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
module should create one long-lived feedback sampler, one Qwen backend and
commissioned robot-side adapters, then share them across the components.
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
grounded 1 mm XY retry now accepts this same observed relation through the
existing lateral, re-observation and 20 mm endpoint/path stages. The observed
route still requires an **independently commissioned** `SurfaceDeviationBounds`
from the retry stage's `deviation_bounds`, covering the entire visible held
motion **including** the VLM tip/axis estimation error at subsequent holds;
a visual point estimate is not a future-trial error bound. The post-shift
endpoint checks that the supplied bound covers the stated visual error.
An unavailable or under-bounded input stops the
retry. The physical-calibration route remains the default and retains its
separate medoid evidence.

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

For VLM checkpoints, `make_checkpoint_capture` in
`precision_insertion.checkpoint_camera` supplies the existing handlers'
`(live, runner, decision)` callback. Its default `snapshot` transport uses
the **same open AutoDex snapshot cameras** without asking SAM/FoundPose to
recognize a partly occluded held key. It undistorts the original snapshot
using the session camera calibration, and saves both image versions so that
the VLM bundle can replay that exact derivation. Instantiate a callback per
phase (`after_lift`, `preinsert`, `post_lateral_hold`, `final_or_abort`);
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
  --max-xy-retries 1 --enable-robot-motion
```

Use a **new** output directory; the command does not overwrite an earlier
run. `--enable-robot-motion` is mandatory because the first clear-view move
occurs before camera capture. `--max-xy-retries 0` disables XY correction;
the current grounded implementation supports one correction, not two.
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
- Either a per-grasp physical held-relation calibration, or a visible
  post-lift multi-view key observation plus independently commissioned total
  surface bounds; and commissioned pickup, lift, transfer, lateral-shift,
  guarded-contact, withdrawal and reset controllers with their independent
  execution records.
- For a **positive 20 mm label**, the optional reviewed visual-depth error
  audit described in [DEPTH_ADMISSION.md](DEPTH_ADMISSION.md). Without it,
  unverified controller stroke depth remains diagnostic and a visually
  normal insertion is not marked successful.

The current workstation has neither a real AutoDex-PC integration module nor
the physical depth commissioning audit. Running this command here cannot
complete a real insertion. The CLI and synthetic tests establish the wiring
contract, not robot readiness.
