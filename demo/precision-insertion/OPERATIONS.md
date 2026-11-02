# ZeroDex capture4 precision-insertion operations

This is the short operational source of truth for the current
`feat/zerodex-precision-insertion` profile. Historical AutoDex-PC commands in
the other dated reports are evidence, not commands for this rig.

## Machine ownership

| Resource | Owner in the live run |
| --- | --- |
| Four camera acquisition streams | capture4's existing ParaDex installation |
| Namespaced exposure, SAM3 and FoundPose services | the separate capture4 adapter |
| cuRobo, insertion state machine, SAM key-tip ROI and Qwen | this kds-Z790 PC |
| Franka FCI streaming and Inspire control | this kds-Z790 PC |
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

## Before a run

The robot may remain off for the first two checks.

```bash
cd /home/hyunsu/autodex-code

# Read-only: validates active JSON, candidate catalog and project namespace.
PRECISION_GAP_MM=5 \
  bash demo/precision-insertion/run_zerodex_capture4.sh --check

# Read-only remote status; does not start or change camera services.
python3 demo/precision-insertion/manage_zerodex_capture4_services.py check
```

The static report must identify the effective local Qwen directory as
`~/precision_insertion_project/cache/models/Qwen3-VL-2B-Instruct`. It checks
path presence only; it deliberately does not load CUDA or open the robot.

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
PRECISION_GAP_MM=5 \
  bash demo/precision-insertion/run_zerodex_capture4.sh --run
```

The default experiment labels are manual while the same snapshots and Qwen
answers are retained. The current profile uses the four commissioned capture4
cameras and the current 5 mm straight-mount candidate policy.

## Runtime sequence

1. Move the open hand to the measured ZeroDex clear-view joint pose.
2. Capture the fixed socket once, estimate its board/robot-frame pose and
   freeze the collision world for the session.
3. Observe the newly placed key and select an endpoint-screened grasp.
4. Plan and execute pickup, squeeze and lift; wait two seconds before the lift
   snapshot so a later slip is recorded as transfer failure rather than grasp
   failure.
5. Transfer the held key to the planned pre-insertion wrist pose.
6. Capture pre-insertion multiview images. Estimate the visible key end/axis
   with calibrated, socket-pose-inlier views.
7. If a metric multiview estimate exists, apply a bounded incremental wrist
   correction and recapture in the same held-key attempt. A single-view point
   is only an active-observation probe and is never compounded as metric XY.
8. Replan the socket-axis stroke from the measured corrected hold. Try the
   20 mm endpoint first; use the implemented 10 mm endpoint only when the
   corrected 20 mm path cannot be planned.
9. Execute the axial descent, record the human `tip_entry_success` diagnostic,
   open the hand, withdraw along the planned path and return to clear view.
10. Capture the released final state. Store the visual/VLM task label; success
    ends the session. Failure performs the existing key-return/repose flow and
    preserves the observed correction for the next trial.

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
  runs/zerodex_capture4/precision-insertion/inspire/<key>/_pipeline_runs/<run>/
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
- `/dev/nvidia*` must exist and `nvidia-smi` must work before local Qwen/SAM or
  cuRobo CUDA execution. Model-file presence alone does not prove this.
- Robot, Inspire hand, FCI and the Franka daemon are required only for
  `--run`/`--plan-only`.
- Do not start a second capture4 adapter if another experiment owns its
  cameras. `check` is non-mutating; `start` is an explicit operator action.

