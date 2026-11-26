# MV-GoTrack mode on ZeroDex capture4

## What this mode changes

The existing precision-insertion state machine remains the owner of grasp
selection, cuRobo motion, 20 mm insertion, release/clear-view and the final VLM
label. `--xy-estimator mvgotrack` replaces only the held-key pose observation
used at pre-insertion.

1. The normal SAM3 + FoundPose tabletop observation produces `T_world_key`.
2. After the same observation passes v8 grasp preflight, it is sent to
   MV-GoTrack as the initial pose. This happens before pickup.
3. capture4 tracks the key through pickup, lift, transfer and pre-insertion.
4. At a stopped pre-insertion hold, the main PC consumes a tracker message
   newer than the previous consumed sequence number.
5. The raw pose is reduced with the key asset's `Dinf` symmetry: axial yaw is
   discarded and the equivalent end is signed toward the planned insertion
   end. The physical CAD center is preserved.
6. The reduced free-tip center and key axis are expressed in the frozen socket
   frame. `-tip_xy` drives the existing held XY planner; the minimal rotation
   from the measured key axis to socket `-Z` drives the existing optional tilt
   planner.
7. After each completed XY or tilt move, a newer tracker pose is consumed.
   No extra still image or Qwen call is made for this reobservation.
8. The existing Jacobian socket-axis handoff, guarded 20 mm insertion,
   release/clear-view and VLM success label are unchanged.

The integration does **not** add score, age, workspace, inter-frame jump or
task-success thresholds. The capture4 launcher explicitly sets the optional
MV-GoTrack score/workspace/jump gates to zero. A message must only satisfy the
tracker's own `valid` flag and contain a finite rigid transform. Existing
planner motion limits still apply because the correction uses the same motion
implementation as the image-based mode.

## Process boundary

- kds-Z790 main PC: FoundPose, session state machine, D∞ reduction, cuRobo,
  FR3/Inspire control, final VLM label.
- capture4: four-camera SHM read-only HTTP streamer and MV-GoTrack GPU process.
- Pose transport: ZMQ `forgehand.tracked_pose.v1` on
  `tcp://147.46.219.136:5591`.

The deployment lives only in
`/home/capture4/precision_insertion_mvgotrack`. It reads the existing ParaDex
camera SHM and does not edit or restart shared camera services.

## Installed capture4 environment

Validated on 2026-11-26:

- cameras: `26053248`, `25322639`, `25305462`, `25322642`;
- Python: `/home/capture4/anaconda3/envs/gotrack_cu128/bin/python`;
- PyTorch `2.11.0+cu128`, CUDA available, RTX 5060 Ti;
- OpenCV `4.13.0`, pyzmq `27.1.0`, nvdiffrast importable;
- checkpoint SHA256:
  `f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086`;
- read-only streamer `/cameras` smoke test returned all four serials.

The live tracker was not initialized against a physical key during this setup,
because an accepted same-session FoundPose initializer was not available. That
is the remaining hardware validation, not a missing code path.

## Deploy or refresh capture4

Use an initialized checkout of the shared MV-GoTrack repository, the official
checkpoint, and the valid ZeroDex calibration:

```bash
cd /home/hyunsu/autodex-code
bash demo/precision-insertion/deploy_mvgotrack_capture4.sh \
  --source /path/to/MV-GoTrack \
  --checkpoint /path/to/gotrack_checkpoint.pt \
  --cam-param /mnt/paradex2/hyunsu/precision_insertion/runtime/active_shared_data/AutoDex/precision_insertion/zerodex_capture4/calibration
```

The script preserves an already matching checkpoint, copies the four-camera
calibration, installs the experiment-owned streamer/manager, and applies only
the RTX 5060 Ti compatibility change to the isolated source copy.

Status and logs:

```bash
ssh -p 77 capture4@147.46.219.136 \
  /home/capture4/precision_insertion_mvgotrack/manage_mvgotrack.py status

ssh -p 77 capture4@147.46.219.136 \
  'tail -f /home/capture4/precision_insertion_mvgotrack/state/tracker.log'
```

## Static check and live command

Use the normal ZeroDex environment plus the MV profile:

```bash
cd /home/hyunsu/autodex-code
source demo/precision-insertion/config/zerodex_capture4.env.sh
source demo/precision-insertion/config/experiments/mvgotrack_axis_alignment_capture4.env.sh

bash demo/precision-insertion/run_robot_pc.sh --check
```

With the commissioned robot services running and explicit motion permission:

```bash
source demo/precision-insertion/config/zerodex_capture4.env.sh
source demo/precision-insertion/config/experiments/mvgotrack_axis_alignment_capture4.env.sh
bash demo/precision-insertion/run_robot_pc.sh --run
```

The profile enables XY and axis correction and keeps the ordinary 10 mm
per-motion XY discretization. It does not change the VLM-only final label or
the 20 mm insertion target.

## Evidence written per run

- `mvgotrack/init_<capture_id>/initialization.json`: accepted FoundPose source,
  pose/frame contract and endpoint;
- `mvgotrack/init_<capture_id>/first_pose.json`: first native-valid tracked
  pose after start/reinit;
- `attempts/<id>/same_trial_xy/observations/*.json`: each symmetry-reduced tip,
  axis, raw tracker message and chosen XY/tilt move;
- capture4 session `tracker_output/`: upstream inputs, messages and latency
  evidence;
- normal transfer, insertion, final clear-view and VLM records remain in their
  existing paths.
