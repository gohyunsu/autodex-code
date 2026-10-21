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
socket axis or entry plane. The next calibration improvement should constrain
the fixed upright socket axis to the measured board normal, fit translation
and axial yaw against the original synchronized masks, and replay the
projection on all selected preinsert cameras before freezing a new session.
This review does not overwrite the existing calibration without that replay.

## Tests

`run_robot_pc.sh --check` passed without camera or robot connection. The
`flir_env` demo suite passed 1,080 tests with 4 skipped after the recovery,
same-grasp and decode changes. The saved-pose GPU preflight above passed.
