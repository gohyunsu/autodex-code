# Board-free cylinder session

`PRECISION_SCENE_MODE=board_free` is a separate scene setup for the cylinder
pipeline. The ChArUco board is physically absent. The socket rests directly
on the working surface, and the key stays out of view until the socket capture
has completed. The session does **not** capture or fit a board, does not save
board images, and does not use a board footprint for repose. It retains the
same camera intrinsics/extrinsics and saved hand-eye `C2R` as the normal
pipeline, then measures a **new** socket pose from live SAM/FoundPose views
using stock multi-view silhouette refinement (`ignore_sil_loss`). It never
reuses the old board-supported socket position.

The operator must supply `PRECISION_TABLE_SURFACE_Z_M` in metres in the FR3
base frame. This is the physical surface supporting the socket and key after
the board has been removed. Static `--check` verifies the file and numeric
contract only; it cannot establish that physical height. Do not substitute
the old board's top height automatically. For a camera-only/static example:

The command also needs an active recipient stage at `PRECISION_STAGE_ROOT`.
The former stage was moved under `hyunsu/tmp/20261023_prev` on this PC; its
embedded absolute paths are stale, so restore and rehydrate the stage before
using it. The code tests below do not establish live readiness.

```bash
cd /home/robot/autodex-precision-insertion
PRECISION_STAGE_ROOT=/absolute/path/to/active/AutoDex/precision_insertion/cylindrical \
PRECISION_SCENE_MODE=board_free \
PRECISION_GAP_MM=10 \
PRECISION_TABLE_SURFACE_Z_M=0.040 \
bash demo/precision-insertion/run_robot_pc.sh --check
```

Here `0.040` m is AutoDex's nominal tabletop default, **not a measurement of
the board-free setup**. Before any robot run, measure or otherwise verify the
actual support height and set that value instead. The camera calibration and
hand-eye transform also require the physical rig to be unchanged.

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
