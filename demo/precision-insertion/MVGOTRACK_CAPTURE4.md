# MV-GoTrack mode on ZeroDex capture4

## What this mode changes

The existing precision-insertion state machine remains the owner of grasp
selection, cuRobo motion, 20 mm insertion, release/clear-view and the final VLM
label. `--xy-estimator mvgotrack` replaces only the held-key pose observation
used at pre-insertion.

1. The normal SAM3 + FoundPose tabletop observation identifies the key and
   supplies its raw pose/tabletop-axis evidence. For the metric cylinder, the
   runtime then resolves the otherwise ambiguous camera-ray depth by placing
   the complete key mesh on the frozen support plane and fitting its two
   tangential coordinates to all admitted SAM masks. The result is the
   support-constrained `T_world_key` used by the rest of the run.
2. After the same support-constrained observation passes v8 grasp preflight,
   it is sent to
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
8. The existing Jacobian socket-axis handoff and guarded 20 mm insertion run.
   The gripper then opens and the arm reaches clear view before the final VLM
   label.  Each VLM request receives two raw, overlay-free crops from one
   camera: the initial tabletop key and the final socket.  The per-camera A/B
   answers are combined by the configured-camera majority.

The tabletop support fit is a pose recovery step, not a new pass/fail gate.
It handles both the `000` end-down and `001` side-down cylinder classes. The
known metric mesh supplies support contact; axial yaw and equivalent ends are
reduced by D∞ symmetry. Raw FoundPose pose, raw class, per-view fitted XY,
mask-hull IoU and the resulting robot-frame centre correction are retained in
the observation record for diagnosis.

The integration does **not** add score, age, workspace, inter-frame jump or
task-success thresholds. The capture4 launcher explicitly sets the optional
MV-GoTrack score/workspace/jump gates to zero. A message must only satisfy the
tracker's own `valid` flag and contain a finite rigid transform. Existing
planner motion limits still apply because the correction uses the same motion
implementation as the image-based mode.

## Process boundary

- kds-Z790 main PC: session state machine, D∞ reduction, cuRobo and
  FR3/Inspire-or-gripper control.
- capture4: four-camera SHM read-only HTTP streamer, namespaced SAM3/FoundPose,
  MV-GoTrack and (in the remote-Qwen profile) one shared Qwen GPU process.
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

The live tracker has since been initialized from a physical-key observation on
capture4 and produced its first native-valid pose before pickup. A 2026-11-27
run also exposed why the initializer must be support-constrained: a valid SAM
mask and single-view raw FoundPose silhouette required a
`[-58.0, -10.7, -63.5]` mm robot-frame centre correction to coincide with the
visible tabletop key. Replaying the exact mask with the support fit produced
mask-hull IoU `0.953`; that corrected pose passed the stock Robotiq pickup, lift,
transfer and sampled 20 mm insertion preflight. Physical execution of the
corrected run remains separate evidence from this planning result.

The companion Qwen deployment is independently namespaced at
`/home/capture4/precision_insertion_qwen`. All four camera IDs share one model
process and same-host batch request path; Qwen does not open the cameras. This
avoids duplicating model weights four times and removes local-3090 contention
between visual labeling and cuRobo.

## Deploy or refresh capture4

Use an initialized checkout of the shared MV-GoTrack repository, the official
checkpoint, and the valid ZeroDex calibration:

```bash
cd /home/hyunsu/autodex-code
source demo/precision-insertion/config/zerodex_capture4.env.sh
bash demo/precision-insertion/deploy_mvgotrack_capture4.sh \
  --source /path/to/MV-GoTrack \
  --checkpoint /path/to/gotrack_checkpoint.pt \
  --cam-param "$PRECISION_CALIB_DIR"
```

The script preserves an already matching checkpoint, copies the four-camera
calibration selected by the active kds-Z790 machine profile, installs the
experiment-owned streamer/manager, and applies only
the RTX 5060 Ti compatibility change to the isolated source copy.

Do not substitute the older calibration snapshot under
`runtime/active_shared_data/.../zerodex_capture4/calibration`: the live launch
overrides that portable asset snapshot with `PRECISION_CALIB_DIR`, and the
tracker must receive the same files.  Verify the deployment after a calibration
change with:

```bash
sha256sum "$PRECISION_CALIB_DIR"/{intrinsics,extrinsics}.json
ssh -p 77 capture4@147.46.219.136 \
  'sha256sum /home/capture4/precision_insertion_mvgotrack/cam_param/{intrinsics,extrinsics}.json'
```

For a calibration-only refresh while the experiment-owned tracker is stopped:

```bash
source demo/precision-insertion/config/zerodex_capture4.env.sh
bash demo/precision-insertion/sync_mvgotrack_calibration_capture4.sh
```

The capture4 manager publishes a canonical calibration identity computed from
the four selected camera entries. The main runtime computes the same identity
from `PRECISION_CALIB_DIR` and compares it with the first tracked pose before
pickup. JSON formatting and unused camera entries are ignored; an actual
intrinsic/extrinsic mismatch in one of the four tracker views is rejected as a
coordinate-frame configuration error. This is not a tracking-confidence or
task-outcome threshold.

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
PRECISION_EXPERIMENT_PROFILE=mvgotrack_axis_alignment_capture4_remote_qwen.env.sh \
  bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

With the commissioned robot services running and explicit motion permission:

```bash
PRECISION_EXPERIMENT_PROFILE=mvgotrack_axis_alignment_capture4_remote_qwen.env.sh \
  bash demo/precision-insertion/run_zerodex_capture4.sh --run
```

The ZeroDex integration opts the initial empty-hand clear-view return into the
same continuous velocity-stream follower used by planned trajectories, at
`0.2x`. The local daemon's blocking joint-position generator was observed to
trip Franka's velocity/acceleration-discontinuity reflex at its first sample,
even at a reduced `speed_scale`; the streamed route reached the measured
clear-view target with `0.003 rad` maximum residual. This switch is scoped to
the ZeroDex integration. AutoDex/Inspire outside this profile retain their
existing home behavior.

The profile enables XY and axis correction and keeps the ordinary 10 mm
per-motion XY discretization. It does not change the VLM-only final label or
the 20 mm insertion target.

The current released-outcome prompt is intentionally short:

```text
Image 1 shows the narrow solid white key before the trial. Image 2 shows the
large white socket after release and clear view. B = a key of that shape
protrudes from inside the socket rim. A = the socket opening is empty or the
key is outside it. Reply A or B.
```

The two images always come from the same calibrated camera. No CAD overlay is
shown to this classifier. The checkpoint records both crop hashes, the raw
token from every camera and the deterministic majority result.

## Evidence written per run

- `mvgotrack/init_<capture_id>/initialization.json`: accepted FoundPose source,
  pose/frame contract and endpoint;
- `key_captures/<capture_id>/key_observation.json`:
  `selection.support_body_fit` records the support-constrained pose derivation,
  raw pose, per-view XY/IoU and robot-frame centre correction;
- `mvgotrack/init_<capture_id>/first_pose.json`: first native-valid tracked
  pose after start/reinit;
- `attempts/<id>/same_trial_xy/observations/*.json`: each symmetry-reduced tip,
  axis, raw tracker message and chosen XY/tilt move;
- capture4 session `tracker_output/`: upstream inputs, messages and latency
  evidence;
- normal transfer, insertion, final clear-view and VLM records remain in their
  existing paths.

## 2026-11-27 trial reinitialization and automatic drop recovery

Each trial copies the same key mesh into a session-owned evidence directory.
The capture4 manager now hashes that mesh and keeps the active tracker mesh
path in `state/tracker.active.json`. A subsequent trial with the same object
and identical geometry reinitializes the running tracker with its new pose but
passes the original active mesh path expected by MV-GoTrack. A genuinely
different object or mesh restarts only the experiment-owned tracker process;
the shared capture4 camera daemon is not modified or restarted.

The read-only SHM-to-JPEG bridge treats an empty ParaDex double-buffer slot as
a transient acquisition slot and waits for the next frame. It no longer lets
one empty slot terminate a camera thread while the HTTP process remains alive.
Remote manager failures include capture4 stderr in the local exception and are
owned by session cleanup even when startup fails partway through.

For an unknown key state after pickup/squeeze/lift preflight failure, the
ZeroDex Robotiq route no longer pauses for a manual `yes`. It opens the gripper
at the stopped pose, uses the commissioned streamed clear-view motion, records
the measured open/clear state, then runs the ordinary four-view tabletop key
capture. The same classifier accepts an upright or side-down landing. This
changes recovery execution only; it does not add a pose-quality or task-failure
threshold, and the separate AutoDex/Inspire route is unchanged.
