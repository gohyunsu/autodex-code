# Cylinder recovery and retry review (2026-10-21)

This review uses saved robot-PC episodes and code only. Its path result is an
offline plan, not an executed placement or a newly measured socket pose.

## Failure to next episode

| Outcome | Existing AutoDex behavior | Precision insertion behavior after this change |
| --- | --- | --- |
| Judged grasp miss | Mark attempted v8 grasp, release/reset, reacquire the object. | Open/clear view, observe the supported key at its current pose, exclude the missed grasp. No extra pickup solely to restore its old XY. |
| Held insertion/transfer failure | Stock place uses a force-aware descent to support, opens, lifts, then resets. | Preflight held return plus support placement (`release_height_m=0`), force-aware descent, post-release lift/exit, fresh key capture, restore if pose class differs or CAD-center shift exceeds 30 mm. |
| Released insertion failure | Stock has no insertion-specific XY diagnostic. | Reopen the pre-stroke `preinsert` snapshot, ground tip/axis, save the source frames and bounded XY increment, recapture the released key, restore if needed, then begin a fresh episode. |
| Corrected next trial | Stock excludes attempted grasps within a session. | The same v8 grasp and tabletop class are required for the saved XY correction. A failed preflight reports `repeat_grasp_unavailable`; another grasp cannot silently receive the same correction. |

The next wrist correction is `previous + weight × measured_increment`. The
configured weight is applied once, the increment is capped at 10 mm per
diagnosis, and the record retains source attempt, candidate, frame provenance,
camera IDs, prior and next corrections. The command relation shifts the wrist
in the frozen socket XY frame; the modeled key target remains centered. A new
tabletop class at the next insertion attempt invalidates that previous
correction. An intermediate `001→000` reorientation retains the correction
until the recovered `000` insertion attempt consumes it.

The 30 mm rest-position tolerance is a reset-label rule, not an XY correction
cap. If a supported key lies farther away, restoration plans a new v8 pickup
and placement using fresh key/robot state. Lying cylinder keys are directed
to upright `000` before insertion. Reorientation preserves the key's CAD
center for its first board site; this avoids shifting the physical cylinder
when its tabletop frame changes. All rest sites retain the frozen socket and
board in the collision scene.

## Saved offline path result

The actual captured `001` key pose and measured joints in
`attempt_d729c8e4814a401abc219bb3371c2747` were replayed through the
`reset_12` directed seed with a physical release height of zero. The original
10 cm elevated transfer intersected the table while reorienting the key.
Using 15 cm for the reorientation lift and transfer removed that intersection.
The sampled path now allows only the final 1 mm of intended key/table support
contact during descent. The full pickup, held lift, transfer, descent,
release and exit preflight returned
`nominal_reset_preflight_pass_drop_unobserved`, with no sampled held-path
failure. This is a planner result for one saved state, not a motor test.

## Socket pose evidence

Existing preinsert overlays for real episodes are saved under each episode's
`preinsert_assessments/000/overlays/` directory. For this review the episodes
were `attempt_a5325a0dd03f4835b3bd575356cef413`,
`attempt_b8272606879045318418341b93334777`, and
`attempt_f39d3f2468944e199496c2469bcd286f`. Purple is the frozen socket
model projection on an actual undistorted camera frame. The socket capture
is not repeated at each episode.

| Session | Socket axis versus board normal | Mean refined mask IoU | Views reaching 0.8 IoU |
| --- | ---: | ---: | ---: |
| `20261021_090945_053190` | 25.75° | 0.697 | 1/18 |
| `20261021_104732_950124` (also used by `20261021_111900_186891`) | 27.25° | 0.697 | 1/19 |

Both sessions used the stock `ignore_sil_loss` socket refinement and one
capture. The axis tilt and weak cross-view fit are evidence that the frozen
socket pose may be inaccurate. An XY wrist retry does not correct a wrong
socket axis or entry plane. The follow-up below constrains the fixed upright
socket axis to the measured board normal and fits board-plane translation
against the original masks. Axial yaw is unobservable for this cylinder.
Existing frozen calibrations remain unchanged; a new session capture is
required to use the corrected method.

### Follow-up: board-supported socket fit

The later 13:55 capture showed a 35.05° frozen-axis deviation from the
measured board normal and only 2/19 views
above the old 0.8 whole-mesh IoU. Inspection of the saved undistorted socket
frames showed that SAM usually masks the 40 mm-radius cylindrical body but
omits the 60 mm-radius flange; several overhead views mask the flange instead.
The stock whole-mesh silhouette objective therefore compares different parts
of the same object. `ignore_sil_loss` permits the resulting bad pose to freeze.

The active cylinder launch now uses `board_constrained_body`, recorded as
`per_capture_board_constrained_body` in session evidence. It retains the
existing AutoDex camera, SAM, FoundPose, acquisition-time and evidence path,
then measures the board once, intersects mask-centroid rays at the CAD body's
mid-height, takes the median board XY, and fixes the open socket axis to the
measured board normal. A CPU projection of the CAD cylindrical wall supplies
the mean IoU using the same undistorted camera calibration. The configured
0.8 threshold applies to that mean, allowing individual overhead flange
masks to disagree. Axial yaw remains unobservable. The full raw and collision
meshes remain unchanged for planning; only the localization objective uses
the visible wall. The renderer warmup and 100-iteration stock silhouette fit
are skipped for this fixed-socket route; stock modes remain available for
comparison and are unchanged.

A local diagnostic image at `/tmp/precision_socket_body_fit_20261021.png`
projects the old and proposed fits onto three saved undistorted 13:55 frames.
It is a read-only overlay, not a new camera observation or frozen calibration.

Read-only replay of three *different* saved socket captures, with no newly
acquired frames or robot motion, gave mean wall IoUs of 0.943 (09:09), 0.925
(10:47), and 0.921 (13:55), versus approximately 0.697 with the old frozen
whole-mesh poses. For the 13:55 masks already loaded in memory, five CPU runs
of the new fit averaged 0.135 s. The saved stock selection and silhouette
timings total 2.836 s for that capture. These are localization compute times,
not full session times. The new result has not been used in a physical trial;
new board/socket movement still requires a fresh session capture. The stock
FoundPose pose admission still runs because the capture/evidence interface
stores its same-frame per-camera pose. This review did not measure that
inference cost separately or prove that it can be skipped without changing
the interface.

## Tests

`run_robot_pc.sh --check` passed without camera or robot connection and
reported the new `board_constrained_body` launch mode. The `flir_env` demo
suite passed 1,085 tests with 4 skipped, excluding the separate
`test_franka_completion_checked.py` module whose cuRobo dependency is not
installed in that environment. The saved-pose GPU preflight above passed
before this socket-fit change; no new live motion test was run.

## 2026-10-29: corrected axial plan unavailable while the key is held

The saved four-view attempt `9542c1bd` completed three same-trial XY shifts,
then rejected the corrected 20 mm endpoint because the measured hand
intersected the socket. No insertion stroke was commanded. At the time of
this saved attempt, `preinsert_path_unavailable` stopped with the key held.
The implementation added below now routes this condition to the measured
held-return procedure.

Replanning the existing held-key return from the original pre-insertion hold
also returned `held_transfer_unreachable` at endpoint IK. Thus simply
reversing the three shifts and calling the old return planner does not solve
this saved attempt. The direct return failure is reproducible with
`~/precision_insertion_project/development/zerodex_capture4/reviews/attempt_9542c1bd6c174baa9cc8a7f5b3a03211/probe_preinsert_return.py`.

A second offline route reversed the archived three XY paths, transfer and
lift while keeping the measured held Inspire joints fixed. Four short linear
bridges, sampled at 11 configurations each, joined the adjacent archives.
All 554 sampled configurations passed the existing key/hand versus
fixed-world collision screen **and** cuRobo's whole-arm world/self constraint
query. This is a sampled check, not a time-parameterized swept-motion proof.
The reversed key center lies approximately 22 mm from the original tabletop
center and 4.7 mm above it, using the cylinder's end-for-end symmetry. From
that inferred release pose, the stock `plan_js_to_init` found an open-hand
clear-view retract with 103 samples in the saved run. Its scene included the
key at the inferred landing pose and the frozen socket/table.
The replay script, result JSON, and planned retract array are grouped in the
same development review folder as the direct-return probe.

At this point in the investigation the route was **not installed for live
execution**. It still needed a start bridge from fresh measured joints,
time parameterization for the interpolated joins, and a verified hand opening
at the tabletop site. The whole-arm result
above covers only archived/configured joint samples. After release, the
existing clear-view capture and original-pose restoration can check the
actual landing and correct it. No robot or hand command was sent during this
review.

## 2026-10-29: corrected held-return implementation

`preinsert_path_unavailable` now routes to
`recover_corrected_hold_then_reobserve`. The new
`precision_insertion/held_reverse_route.py` assembles the executed XY shifts
in reverse order, then the executed transfer and lift. It bridges their
measured joint gaps at the stock follower's 0.75 velocity scale, holds the
measured Inspire joints, and runs the existing held-key geometry audit and
cuRobo world/self collision query over every sampled state. It then plans
the stock open-hand clear-view retract with the key at the inferred release
pose. The separate `corrected_hold_recovery.py` reads and hashes the exact
attempt archives, writes the plan, follows it through the existing Franka
executor, opens the hand, moves to clear view and records measured feedback.
The existing reset/repose handler then captures the actual landed key and
returns it to the original pose and XY before the next attempt.

The saved `9542c1bd` replay produced a 513-sample held route with all 513
cuRobo states feasible, followed by a stock clear-view plan of 103–105
samples across two planner runs. The source loader independently matched the
three saved XY paths (67, 64, 68 samples), transfer (89) and lift (222).
These are sampled offline results. The hand opening, actual key landing and
live execution of this route are untested while the robot/hand are off. If a
future measured state cannot pass the same preflight, the route is not
executed and its held-key state must be handled with the robot present.

## 2026-10-29: held XY planning failure

The same recovery route now also receives a held XY correction that fails
**before motion**: either cuRobo cannot produce the lateral path or the
existing held-key collision audit rejects it. The failure report records the
measured joints and the current key-pose hypothesis in
`same_trial_xy/plan_failures/NNN.json`. The session records
`preinsert_path_unavailable`, then reverses only XY shifts that actually
completed, followed by transfer and lift. A controller/execution exception
after the XY command is kept distinct; this pre-motion return is not applied
to a path whose completion is unknown. Focused policy, handler and recovery
tests passed offline; no live return has been attempted.
