# Recorded runtime, stock comparison, and recovery latency review (2026-10-22)

This review uses read-only traces from the separate precision-insertion
checkout's October 21 physical sessions. It did not connect a camera or robot.
The stock comparison is `/home/robot/AutoDex` at `448732d`. The checkout's
`src/execution/run_auto.py` has the same SHA-256 as stock; the planner adds a
selectable right-angle-adapter URDF, and the FR3 executor adds current
completion and speed handling. The precision demo orchestrates the stock v8
pickup, calibrated cameras, FR3 follower, force stop, and reset primitives.

## Measured wall time

The freshest socket-acquisition run and the furthest downstream run are
**different sessions**. Their durations must not be summed as one physical
trial. Run A (`20261021_090945_053190`) captured the socket anew but stopped
after transfer. Run B (`20261021_111900_186891`) reused the frozen socket
session and reached the post-release final visual label, then stopped in an
optional XY diagnostic. These are measurements of the 2026-10-21 binaries,
not predictions for today's changed recovery path.

| Sequential stage, measured by parent span | Run A: new socket | Run B: saved socket | Work performed |
| --- | ---: | ---: | --- |
| Startup and board/socket session or saved-session opening | 86.43 s | 11.73 s | FR3 planner/camera startup; A measured board/socket in 76.14 s; B verified the saved session in 1.86 s. |
| Operator key placement wait | 23.03 s | Outside this run | A waited for Enter; B began with the key already placed, so its human setup time was not recorded. |
| Key capture and candidate preflight | 103.48 s | 38.32 s | FoundPose/key admission, initial visual snapshot, v8 pickup and sampled insertion plan. B's candidate preflight alone was 31.36 s. |
| Pickup and squeeze action | 7.23 s | 6.30 s | Includes actual commanded pickup/squeeze: 4.79 s / 4.29 s. |
| Lift and visual verdict action | 11.96 s | 12.38 s | Includes commanded lift: 6.41 s / 7.42 s, then capture/VLM. |
| Held relation and post-lift preflight | 11.37 s | 25.04 s | Measured-state held path and insertion preflight. |
| Transfer, preinsert image and verdict | 12.79 s, then error | 10.51 s | Includes commanded transfer: 5.06 s / 3.06 s. |
| Axial insertion, release, final image and verdict | Not reached | 21.16 s | B: axial command 4.75 s, release/clear 5.85 s, final label 3.81 s; handoff and captures fill the remainder. |
| Saved preinsert XY diagnostic | Not reached | 2.60 s, then error | Optional next-attempt advice; no new motor command. |
| Unattributed residual | 1.15 s | 0.83 s | Root session duration minus the preceding disjoint rows; it is not assigned to a specific operation. |
| **Session wall time** | **257.44 s** | **128.88 s** | Both recorded a failed process outcome; neither is an uninterrupted successful full session. |

For A, the `socket.capture` span was 66.67 s and board/socket orchestration
76.14 s. The key capture span was 68.95 s, reflecting the older capture path;
B's was 3.29 s after the pose-only key/independent visual split. A failed
with `preinsert stage contradicts verified checkpoint`; the episode symlink
versus canonical transfer-report path was corrected later on October 21 in
commit `1b539bc5`. B's original final VLM label was `outside_socket` despite
the post-hoc corrected visual review recording `inside_socket`; a camera-PC
Qwen CUDA out-of-memory exception then stopped its XY diagnostic. Commit
`bbbd5df3` made that optional backend outage nonfatal and corrected the
released-key visual vote. The post-hoc label is a visual reanalysis, not a
metric insertion-depth measurement.

The timing summary includes parallel work. In B, five-camera VLM inference
totalled 15.09 worker-seconds but occupied only 3.79 wall-seconds, with
11.30 seconds of overlap. PNG write/hash work totalled 26.85 worker-seconds;
the four enclosing batch spans summed to 6.93 seconds. Those worker-seconds
cannot be added to the sequential wall-time table. A's operator wait was
23.03 seconds. `report_timing.py` and each `timing_summary.json` retain
the detailed span counts, failures and phase attribution.

Two sampled held-path geometry audits in B occupied 37.06 s in aggregate
(33.92 s of self time), making collision geometry the clearest remaining
planning bottleneck. These are nested inside the preflight/held-relation
rows above and must not be added to the session total. The checks include
the held key and hand against the fixed board/socket world; no equivalent
stock fixed-socket path exists, and the available evidence does not justify
skipping samples or loosening collision clearance to reduce that time.

The later run `20261021_140856_019116` lasted 466.29 s and stopped by
operator `KeyboardInterrupt` while waiting for recovery frames. Its first
missed-grasp recovery captured the returned key in 65.2 s, then unnecessarily
planned and executed an original-XY restoration (191.54 s planning and
34.29 s motion) even though the lift VLM label was `miss`. A second recovery
key capture was interrupted after 60.59 s. The current branch already skips
restoration for a definite miss (commit `35ad1db3`), so the 191.54 s plan is
not a valid forecast for the same current-route case.

## Current-route change and gate comparison

The active recovery observation previously called `collect_key_capture`
with `pose_only=False`, waiting for and rereading every remotely written
undistorted PNG before admitting the tabletop pose. That made the last run's
`camera.saved_frame_wait` spans occupy 114.85 s across two captures; 40
individual saved-frame reads accounted for 459.23 worker-seconds and were
parallel, not 459.23 s of wall time. The recovered pose is used to decide
whether to restore and where to pick up next. The next episode makes its own
new key and VLM captures. The current recovery path therefore uses the same
pose-only SAM/FoundPose and frame-metadata admission already used by the
fresh key: no remote PNG completion gate, while retaining request ID,
camera/frame ID, exposure timestamp, mask/pose binding and observed robot
state. Optional registered-image assistance still takes the full-image path.
There is no post-change hardware timing yet; 114.85 s is observed *old-path*
wait, not a promised saving.

Stock AutoDex does not have a fixed socket, 20 mm insertion, preinsert VLM,
released-key XY feedback, or exact fixed-socket collision world. Those task
dependencies cannot be removed merely for lacking a stock counterpart.
Current active-route comparison found no new evidence that the remaining
measured start/arrival, collision, force-stop, and supported-return checks
caused an unjustified halt. Prior changes removed image-age vetoes,
redundant held-hand thresholds, the missed-grasp original-XY restoration,
and an optional VLM outage as a terminal exception. The historical
`xy_retry.py` and observed held-relation routes remain separate from the
active nominal cylinder route.

Neither saved timing can measure the latency of the changed recovery code,
another radial-gap fixture, future network contention, or independently
surveyed key penetration. Those remain unmeasured rather than inferred.
Three older traces have `live.session=success` but end after alternative
repose preflight, before a full insertion; they do not establish a measured
successful end-to-end insertion time.

Validation after the recovery-path edit: 1,091 tests passed and four were
skipped with
`/home/robot/anaconda3/envs/planner/bin/python -m pytest -q demo/precision-insertion/tests --ignore=demo/precision-insertion/tests/test_franka_completion_checked.py`.
The separate Franka-completion test module requires CUDA hardware unavailable
in this offline environment. The current pose-only verifier also accepted
the real 20-camera key evidence bundle from run B. `git diff --check` and
Python compilation passed. The host's default Python 3.13 does not have this
project's `cv2`, `trimesh`, and `coal` dependencies; testing uses the
configured `planner` environment above.
