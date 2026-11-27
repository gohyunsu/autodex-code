# ZeroDex capture4 precision-insertion operations

This is the short operational source of truth for the current
`feat/zerodex-precision-insertion` profile. Historical AutoDex-PC commands in
the other dated reports are evidence, not commands for this rig.

The implementation/verification ledger is
[`CURRENT_STATUS.md`](CURRENT_STATUS.md). It distinguishes implemented code,
physical evidence and external prerequisites so that an old TODO or a passing
offline catalogue is not mistaken for current robot validation.

## Machine ownership

| Resource | Owner in the live run |
| --- | --- |
| Four camera acquisition streams | capture4's existing ParaDex installation |
| Namespaced exposure, SAM3 and FoundPose services | the separate capture4 adapter |
| MV-GoTrack and distributed Qwen | capture4 RTX 5060 Ti |
| cuRobo, insertion state machine and Franka control | this kds-Z790 RTX 3090 PC |
| Franka FCI streaming and selected Inspire/gripper control | this kds-Z790 PC |
| Static assets and authoritative run records | `/mnt/paradex2/hyunsu/precision_insertion` |
| Repeated-use model caches | `~/precision_insertion_project/cache/models` |
| Full-resolution selected-view recording during a run | capture4 local disk |

The capture4 ParaDex camera installation is the only external code/data
exception. Live assets and outputs never use another NAS user's directory.

## Active files

- machine profile: `config/zerodex_capture4.env.sh`
- fixture candidate policy: `config/candidate_policies/cylinder_gap5.env.sh`
- launcher: `run_zerodex_capture4.sh`
- main state machine: `run_live.py` and `precision_insertion/`
- ZeroDex FR3 binding: `precision_insertion/zerodex_franka_integration.py`
- stage parameters: `robot_pc_stages.py`
- current detailed rig record: `ZERODEX_CAPTURE4.md`
- asset/storage contract: `ZERODEX_NAS_PROJECT.md`
- latency schema and reports: `TIMING_PROFILE.md`
- experiment protocol: `RESEARCH_EVALUATION_PROTOCOL.md`

The AutoDex executable entry points are unchanged. A reviewed set of optional
low-level adapter surfaces, dual-end-effector robot assets and executor helper
modules still lives outside this demo; [PLAN.md](PLAN.md) and
`config/source_boundary.json` are their exact allowlist and compatibility
contract.
Do not describe the branch as byte-identical to stock AutoDex, and do not add
another shared-file dependency without updating that boundary and its tests.
Audit it without opening cameras or robot devices:

```bash
python3 demo/precision-insertion/audit_source_boundary.py
```

## Before a run

The robot may remain off for the first two checks.

The ZeroDex profile defaults to `PRECISION_END_EFFECTOR=robotiq_2f85`.
Set `PRECISION_END_EFFECTOR=inspire` on the same command to retain the original
Inspire/BODex route.  These modes have separate URDFs, joint widths and
candidate namespaces; selecting one never deletes or rewrites the other.

```bash
cd /home/hyunsu/autodex-code

# Read-only: validates active JSON, candidate catalog and project namespace.
PRECISION_GAP_MM=5 \
  bash demo/precision-insertion/run_zerodex_capture4.sh --check

# Read-only: audits the independent NAS bundle inventory and current code
# snapshot. This does not open CUDA, cameras, Franka or Inspire.
source demo/precision-insertion/config/zerodex_capture4.env.sh
PYTHONPATH=demo/precision-insertion \
  "$PRECISION_CHECK_PY" demo/precision-insertion/prepare_zerodex_nas_project.py \
  --verify-project --nas-root "$PRECISION_NAS_PROJECT_ROOT"

# Read-only remote status; does not start or change camera services.
python3 demo/precision-insertion/manage_zerodex_capture4_services.py check
```

For the local-Qwen profile the static report identifies
`~/precision_insertion_project/cache/models/Qwen3-VL-2B-Instruct`.  The
recommended MV-GoTrack profile instead uses the isolated capture4 deployment
described below. Static checks validate paths only; they deliberately do not
load CUDA or open the robot.
The bundle verifier is independent of the live launcher. A nonzero result
means the handoff inventory or archived code snapshot needs refreshing; it is
not an added grasp, perception, planning or insertion failure condition.

For a camera/perception-only session with no robot command:

```bash
PRECISION_GAP_MM=5 \
  bash demo/precision-insertion/run_zerodex_capture4.sh --camera-only
```

When a physical run is intended, activate FCI in Desk and keep this command
in a separate terminal:

```bash
cd /home/hyunsu/autodex-code
sudo bash demo/precision-insertion/run_zerodex_franka_daemon.sh
```

Then launch one run:

```bash
cd /home/hyunsu/autodex-code
PRECISION_END_EFFECTOR=franka_gripper \
PRECISION_EXPERIMENT_PROFILE=mvgotrack_axis_alignment_capture4_remote_qwen.env.sh \
PRECISION_GAP_MM=5 \
  bash demo/precision-insertion/run_zerodex_capture4.sh --run
```

The recommended profile keeps both GPU-heavy visual models beside the camera
streams while leaving deterministic robot planning/control on kds. It creates
one persistent Qwen process for all four camera IDs; one model is not loaded
per view. Prepare or verify that experiment-owned deployment without changing
capture4's shared camera environments or services:

```bash
cd /home/hyunsu/autodex-code
bash demo/precision-insertion/deploy_qwen_capture4.sh --check

PRECISION_END_EFFECTOR=franka_gripper \
PRECISION_EXPERIMENT_PROFILE=mvgotrack_axis_alignment_capture4_remote_qwen.env.sh \
  bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

`--deploy` is required only to create or refresh
`/home/capture4/precision_insertion_qwen`. The worker receives selected
lossless PNG pixels over one SSH request stream; it never opens a camera and
never controls the robot. A 2026-11-27 synthetic-image smoke test measured
4.39 s cold model load, 0.615 s inference round trip and 4.30 GB CUDA peak
allocation on capture4. This is a transport/runtime check, not VLM accuracy or
full-episode latency evidence.

The default experiment labels are manual while the same snapshots and Qwen
answers are retained. The current profile uses the four commissioned capture4
cameras and the current 5 mm straight-mount candidate policy.

## Runtime sequence

1. Move the open hand to the measured ZeroDex clear-view joint pose.
2. Capture the fixed socket once, estimate its pose in the calibrated robot
   frame and freeze the collision world for the session. The default
   `board_free` profile uses the commissioned extrinsics and measured table
   plane; no ChArUco observation is required at runtime.
3. Observe the newly placed key. After SAM3+FoundPose admits its same-session
   6D pose and exposure-time robot state, start MV-GoTrack on capture4. Its
   remote initialization and first tracked frame overlap local cuRobo
   candidate preflight; the pipeline joins that initializer before exposing
   the selected endpoint-screened grasp to pickup.
4. Plan and execute pickup, squeeze and lift; wait two seconds before the lift
   snapshot so a later slip is recorded as transfer failure rather than grasp
   failure.
5. Transfer the held key to the planned pre-insertion wrist pose.
6. Capture pre-insertion multiview images. In MV-GoTrack mode, establish a
   sequence barrier after the stopped hold and consume the first newer native-
   valid 6D pose. Reduce its cylindrical D-infinity symmetry to the physical
   free tip and unoriented axis, then compute socket-frame XY and optional
   axis correction. The image estimator remains available in its separate
   non-tracker mode.
7. If a metric multiview estimate exists, apply a bounded incremental wrist
   correction and recapture in the same held-key attempt. A single-view point
   is only an active-observation probe and is never compounded as metric XY.
8. Replan the socket-axis stroke from the measured corrected hold. Try the
   20 mm endpoint first; use the implemented 10 mm endpoint only when the
   corrected 20 mm path cannot be planned.
9. Execute the axial descent, record the human `tip_entry_success` diagnostic,
   open the hand, withdraw along the planned path and return to clear view.
10. Capture the released final state. For each camera, submit its saved raw
    initial-key crop followed by its final clear-view socket crop to Qwen; no
    CAD overlay is part of this label input. Store the visual/VLM task label;
    success ends the session. Failure performs the existing key-return/repose
    flow and preserves the observed correction for the next trial.

No extra key-in-hand threshold, fixed camera blacklist or cumulative XY cap is
added by this profile. Existing geometry collision checks, exact endpoint
screening and measured executor completion remain the inherited execution
contract.

## Recording and output

The selected full-resolution camera recording begins only after the
pre-insertion label is confirmed and stops after release plus clear-view
recovery. It stays on capture4 during the action loop. Upload it after the run:

```bash
python3 demo/precision-insertion/upload_zerodex_recording.py --help
```

Pipeline records are written under:

```text
/mnt/paradex2/hyunsu/precision_insertion/
  runs/zerodex_capture4/precision-insertion/<end_effector>/<key>/_pipeline_runs/<run>/
  runs/zerodex_capture4/_recording_receipts/<run>/
  development/zerodex_capture4/                 # offline diagnostics only
```

The launcher creates run/control directories before any robot integration is
opened. A read-only NAS mount therefore fails before motion rather than midway
through an episode.

## Timing and post-run inspection

Each run writes the hierarchical timing trace and summary documented in
`TIMING_PROFILE.md`. To summarize one completed run:

```bash
python3 demo/precision-insertion/report_timing.py \
  /absolute/path/to/RUN.timing/timing.jsonl
```

Interpret cumulative worker time separately from occupied wall time: parallel
PNG/VLM workers may sum to more than the episode duration. The main latency
optimizations currently active are local model caches, persistent capture4
perception services, fixed-world prewarm, Qwen preload and post-run-only video
upload.

## Current external prerequisites

- The NAS mount must be writable for a live run. Static `--check` remains
  read-only.
- `/dev/nvidia*` must exist and `nvidia-smi` must work before local cuRobo CUDA
  execution. Model-file presence alone does not prove this. On this PREEMPT_RT
  host, the 535.309.01 modules can be loaded while the device nodes are absent
  after reboot. In that state run `sudo /sbin/ub-device-create --verbose`,
  then verify `nvidia-smi`; this creates device nodes only and does not start
  the robot or precision pipeline. The nodes and PyTorch CUDA access were
  verified on 2026-11-27. An initial 1,024-seed OOM exposed that explicit
  checkout-local planner YAMLs had selected the generic custom-robot
  numerics. The ZeroDex factory now explicitly preserves the FR3 profile's
  32 trajectory-optimization seeds. With that fix, real cuRobo planning ran
  while the unrelated 6.62 GB process remained resident; a hypothetical
  `side_down` key passed collision, IK, approach and held lift. Its synthetic
  transfer target remained unreachable and is not physical insertion proof.
  The device nodes later disappeared again while the modules stayed loaded,
  so repeat the node-creation command if `nvidia-smi` fails.
- Robot, Inspire hand, FCI and the Franka daemon are required only for
  `--run`/`--plan-only`.
- Do not start a second capture4 adapter if another experiment owns its
  cameras. `check` is non-mutating; `start` is an explicit operator action.
