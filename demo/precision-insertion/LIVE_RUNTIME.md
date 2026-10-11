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
4. `LiveHeldRelationHandlers` — physical grasp-specific key/hand relation.
5. `LiveTaskHandlers` — transfer, arrival label, first guarded insertion.
6. `LiveGroundedRetryHandlers` — guarded withdrawal, VLM ≤1 mm XY correction,
   renewed planning and one guarded reattempt.
7. `LiveHeldReturnHandlers` — held-key return after failed insertion.
8. `LiveTransferRecoveryHandlers` — supervised recovery when transfer leaves
   the key state uncertain.
9. `LiveReposePlanningHandlers` — directed v8 reorient planning.
10. `LiveReposeExecutionHandlers` — selected reset execution and new key pose.

The handler composer rejects missing/duplicate actions before any grasp. The
module should create one long-lived feedback sampler, one Qwen backend and
commissioned robot-side adapters, then share them across the components.
See [ACTION_LOOP.md](ACTION_LOOP.md) for each constructor's existing
parameters and the exact decision-to-handler mapping. The `close` callback
must release robot-side resources even if a handler raises. The demo closes
its own camera context first.

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
- Per-grasp physical held-relation calibration and commissioned pickup,
  lift, transfer, lateral-shift, guarded-contact, withdrawal and reset
  controllers with their independent execution records.
- For a **positive 20 mm label**, the optional reviewed visual-depth error
  audit described in [DEPTH_ADMISSION.md](DEPTH_ADMISSION.md). Without it,
  unverified controller stroke depth remains diagnostic and a visually
  normal insertion is not marked successful.

The current workstation has neither a real AutoDex-PC integration module nor
the physical depth commissioning audit. Running this command here cannot
complete a real insertion. The CLI and synthetic tests establish the wiring
contract, not robot readiness.
