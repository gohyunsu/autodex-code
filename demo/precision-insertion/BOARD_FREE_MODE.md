# Board-free cylinder session

`PRECISION_SCENE_MODE=board_free` is the default ZeroDex cylinder setup. The
ChArUco board is physically absent during an episode. The socket rests directly
on the working surface, and the key stays out of view until the socket capture
has completed. The session does **not** capture or fit a board, does not save
board images, and does not use a board footprint for repose. It retains the
commissioned camera intrinsics/extrinsics and hand-eye `C2R`, loads the frozen
support plane from `PRECISION_TABLE_PLANE_CONFIG`, and measures a **new** socket
pose from live SAM masks with the table-constrained metric socket body fit. It
never reuses an old socket position and does not require socket FoundPose.

The kds-Z790 machine profile points `PRECISION_TABLE_PLANE_CONFIG` at
`~/precision_insertion_project/calibration/table_plane/current/table_plane.json`.
That plane was measured once while a temporary ChArUco board made the physical
surface observable; the board is not a runtime dependency. An explicit
`PRECISION_TABLE_SURFACE_Z_M` remains a fallback for installations without a
saved plane. Static `--check` validates the file and coordinate contract only;
it does not remeasure the surface. For a camera-only/static example:

The command also needs an active recipient stage at `PRECISION_STAGE_ROOT`.
The former stage was moved under `hyunsu/tmp/20261023_prev` on this PC; its
embedded absolute paths are stale, so restore and rehydrate the stage before
using it. The code tests below do not establish live readiness.

```bash
cd /home/robot/autodex-precision-insertion
PRECISION_STAGE_ROOT=/absolute/path/to/active/AutoDex/precision_insertion/cylindrical \
PRECISION_SCENE_MODE=board_free \
PRECISION_GAP_MM=10 \
PRECISION_TABLE_PLANE_CONFIG=/absolute/path/to/table_plane.json \
bash demo/precision-insertion/run_robot_pc.sh --check
```

The camera calibration, hand-eye transform and saved plane require the physical
rig/table to be unchanged. A key observation uses the same plane: raw
FoundPose selects the cylinder tabletop geometry, while the metric key mesh
and SAM masks recover support-normal depth and support-plane XY. Thus the
planner and MV-GoTrack receive a board-free, support-constrained 6D key pose
rather than the raw single-view silhouette translation.

The last saved 10 mm board-9 session, run `20261022_105828_126084`, recorded:

| Quantity | Saved value | Reuse in board-free session |
| --- | --- | --- |
| Board top in FR3 base | `0.04154719460865922 m` | No; the board is removed. |
| Board origin in FR3 base | `[0.4093130966, -0.0923178451, 0.0403741529] m` | No. |
| Board center in FR3 base | `[0.5862410633, 0.1563184245, 0.0415989116] m` | No. |
| Socket origin in FR3 base | `[0.6262827268, 0.2682891255, 0.0417851765] m` | No; socket pose is remeasured. |
| `C2R` translation | `[0.5168608282, -0.1299272432, 1.1501426218] m` | Yes, via the saved hand-eye file if the camera and robot mounts are unchanged. |

The saved `C2R` rotation is
`[[-0.8189964721,-0.4122175736,0.3991509122],
[-0.4613221669,0.8866965829,-0.0308387452],
[-0.3412134772,-0.2093939873,-0.9163664775]]`.
These numbers are from `frozen_session_calibration.json`; they are historical
measurements, not measurements of the new board-free scene.

The released final VLM comparison for that run voted `outside_socket` in
three views and `inside_socket` in one, while the operator observed the key
inside the socket. Removing the checkerboard may improve visual attention,
but that causal explanation is untested. The final VLM remains a visual task
label, not physical insertion-depth proof. Compare its saved per-view labels
with the operator observation during the board-free trial.
