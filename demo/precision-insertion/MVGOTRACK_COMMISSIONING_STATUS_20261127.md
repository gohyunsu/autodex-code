# MV-GoTrack commissioning status

Date: 2026-11-27  
Scope: static checks plus an isolated capture4 Qwen inference probe; no camera
acquisition and no robot motion

This record separates a configured software path from a physically validated
tracking episode.  It is evidence for undergraduate-thesis completion gate 2,
not a claim that the gate has passed.

## Commissioned inputs

- Machine profile: `config/zerodex_capture4.env.sh`
- Experiment profile:
  `config/experiments/mvgotrack_axis_alignment_capture4_remote_qwen.env.sh`
- Active extrinsics:
  `~/calibration/zerodex_capture4/extrinsic/20261118_232125/solution_filtered`
- Active hand-eye:
  `~/calibration/zerodex_capture4/handeye/20261118_233621/solution/C2R.npy`
- Active support-plane link:
  `~/precision_insertion_project/calibration/table_plane/current/table_plane.json`
- Resolved support-plane measurement:
  `20261126_212743_final_multiview/table_plane.json`
- Support height in the Franka base frame: `0.06336083895817991 m`
- Plane fit: 54/54 triangulated corners, four cameras, RMS `0.424 mm`
- Runtime end effector: Franka gripper
- Runtime object family: cylindrical key, gap-5 fixture

The support-plane record was produced with the active 2026-11-18 extrinsic
and hand-eye pair.  Board-free execution consumes this saved measurement; it
does not require a board to remain in the live scene.

## Static end-to-end preflight

The following command completed on 2026-11-27:

```bash
source demo/precision-insertion/config/zerodex_capture4.env.sh
source demo/precision-insertion/config/experiments/\
mvgotrack_axis_alignment_capture4_remote_qwen.env.sh
bash demo/precision-insertion/run_robot_pc.sh --check
```

Observed result:

- board-free launch config resolved;
- four capture4 cameras resolved;
- the gap-5 Franka-gripper endpoint catalog loaded;
- both cylinder tabletop pose stems (`000`, `001`) had one eligible candidate;
- MV-GoTrack endpoint resolved to `tcp://147.46.219.136:5591`;
- capture4 manager and session directories passed configuration preflight;
- all runtime paths were independent of the legacy AutoDex asset roots;
- no camera, CUDA model or robot connection was opened by this command.

The reported `robot_ready=false` is expected: `--check` is a static-files-only
preflight and deliberately does not infer hardware readiness.

## Capture4 tracker host

A read-only remote check reported:

- GPU: NVIDIA GeForce RTX 5060 Ti;
- PyTorch: `2.11.0+cu128`;
- CUDA available: true;
- pyzmq: `27.1.0`;
- GoTrack checkpoint SHA256:
  `f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086`;
- experiment-owned stream and tracker processes: stopped at check time.

The stopped state is not a failure.  The session runtime starts or reinitializes
the experiment-owned tracker only after it accepts the same-session SAM3 +
FoundPose key observation.  It stops the process on session close.

The capture4 manager invokes the native tracker with the four commissioned
cameras and with `min-score`, workspace and pose-jump rejection disabled.  The
precision runner likewise adds no confidence, age, workspace or task-success
gate to tracker output.

## Capture4 Qwen execution check

The remote-Qwen experiment profile passed the same static preflight with all
four camera IDs mapped to one persistent capture4 worker.  A fresh end-to-end
probe then started that worker over its normal SSH request stream, loaded the
deployed Qwen3-VL-2B-Instruct model on the RTX 5060 Ti, submitted one lossless
RGB PNG, received the requested `WHITE_CIRCLE` label and shut the worker down
cleanly.  The measured cold ready time was `3.592 s` and the inference
round-trip was `0.589 s`.  This proves the deployed model, CUDA runtime, SSH
transport and request protocol; it is not a task-label accuracy result.

At the end of this check, `manage_mvgotrack.py status` still reported
`stream_pid=null` and `tracker_pid=null`.  That is the intended idle state:
the normal live state machine starts tracking only from the accepted
same-session key pose.

## Franka connection check

The host is running `6.8.1-1061-realtime`, `enp4s0` is up with
`172.16.1.10/24`, and the selected route to `172.16.1.11` uses that interface.
The robot address nevertheless remains `INCOMPLETE` in the ARP table and two
ICMP probes returned `Destination Host Unreachable`.  Local daemon ports 5555
and 5556 are listening.  Therefore no current robot state or motion can be
claimed until the robot-side power/cable/switch/IP/Desk path responds; no
robot command was issued during this check.

## Evidence still required

Gate 2 passes only after one physical episode preserves all of the following:

1. same-session raw multi-view key capture;
2. accepted SAM3 mask and FoundPose `T_world_key` initializer;
3. the exact initializer copied to capture4;
4. first native-valid MV-GoTrack message;
5. continuous timestamped tracker poses through pickup, lift and transfer;
6. a fresh stopped pose at the pre-contact hold;
7. symmetry-reduced tip/axis residual and the commanded correction;
8. post-correction tracker pose;
9. robot state and synchronized video covering the same interval.

No archived run under the current ZeroDex run root contains
`initialization.json`, `first_pose.json` or an MV-GoTrack pose trace.  Therefore
the software path is preflight-ready, but physical continuous tracking is not
yet established.

## Next physical command boundary

Do not start the tracker from an old pose file.  Place the cylindrical key in
the current calibrated scene and run the normal live pipeline so the current
FoundPose observation becomes the initializer.  The tracker initialization
then overlaps candidate planning and is joined before pickup.  This preserves
the intended sequence:

```text
SAM3 + FoundPose -> begin tracker -> plan -> first tracker pose -> pickup
-> lift -> transfer -> stopped pre-contact pose -> XY/tilt correction
```

This commissioning run should stop after evidence collection if robot motion
is not explicitly authorized; no additional conservative rejection threshold
is required.
