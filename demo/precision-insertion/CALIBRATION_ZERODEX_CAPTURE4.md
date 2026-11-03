# ZeroDex capture4 calibration

This calibration path belongs to the precision-insertion ZeroDex profile. It
uses only cameras `26053248`, `25322639`, `25305462`, and `25322642`. It does
not edit ParaDex `system/current`, the shared capture4 checkout, or an active
runtime calibration.

## Board

The board observed on 2026-11-03 is the board-11 pattern:

- 10 by 7 checker squares
- 50 mm checker edge
- 30 mm marker edge
- OpenCV `DICT_6X6_250`
- marker IDs 70 through 104
- 54 internal ChArUco corners

The machine-readable project copy is
[`config/zerodex_charuco_board11_50mm.json`](config/zerodex_charuco_board11_50mm.json).

## Capture extrinsic observations

Keep all cameras fixed. Move and tilt only the board. Source the machine
profile and start from the repository root:

```bash
cd /home/hyunsu/autodex-code
source demo/precision-insertion/config/zerodex_capture4.env.sh
export PYTHONPATH="$PRECISION_REPO_ROOT/demo/precision-insertion:$PRECISION_REPO_ROOT:$PRECISION_PARADEX_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SESSION=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/extrinsic_sessions/20261103_board11_4cam

"$PRECISION_PLANNER_PY" demo/precision-insertion/capture_zerodex_extrinsic.py \
  --launch-config "$PRECISION_STAGE_ROOT/launch.json" \
  --board-config demo/precision-insertion/config/zerodex_charuco_board11_50mm.json \
  --session-dir "$SESSION" \
  --monitor
```

The Tk/Pillow monitor displays a refreshing 2-by-2 grid with full-resolution ChArUco
detections overlaid. Press `c` once for each new board placement and `q` when
finished. Only the preview is resized; saved calibration images remain
2048 by 1536 pixels. It does not require OpenCV HighGUI, which is absent from
the planning environment's headless OpenCV build. Without `--monitor`, the
same tool uses Enter to capture.
Capture
roughly 15--25 placements spanning the shared workspace. Include front/back,
left/right, high/low in the image, plus moderate pitch, roll, and yaw. Keep a
set of central placements visible in all four cameras; other placements may
be visible to only a subset as long as the camera-overlap graph remains
connected. The tool stores every observation and reports the detected corner
count; it does not silently discard a placement.

The first placement from 2026-11-03 is already stored as `0000` in the session
above and detected all 54 corners in all four views. Re-running the command
continues at `0001` rather than overwriting it.

## Solve

The solver keeps the commissioned intrinsics fixed. Per-view metric PnP
initializes each camera and board pose; one joint robust least-squares problem
then minimizes distorted-pixel reprojection error. Camera `26053248` is the
default arbitrary world frame. A later hand-eye solve determines the robot
pose in that world frame.

```bash
SOLUTION=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/extrinsic_solutions/20261103_board11_4cam_v1

"$PRECISION_PLANNER_PY" demo/precision-insertion/solve_zerodex_extrinsic.py \
  --session-dir "$SESSION" \
  --intrinsics "$PRECISION_STAGE_ROOT/calibration/intrinsics.json" \
  --board-config demo/precision-insertion/config/zerodex_charuco_board11_50mm.json \
  --camera-ids 26053248,25322639,25305462,25322642 \
  --output-dir "$SOLUTION"

python3 -m json.tool "$SOLUTION/report.json"
```

The solution directory is new and separate from the active runtime. Review
its per-camera reprojection statistics and overlays before using it for
hand-eye calibration. Activate the new extrinsics and their matching hand-eye
result together; do not combine new extrinsics with the old `C2R.npy`.

## FR3 hand-eye with the existing AutoDex board 2

The rigid board currently attached to the FR3 is not board 11. A live
four-view probe on 2026-11-03 decoded marker IDs 15 through 29, identifying it
as the existing AutoDex hand-eye board 2:

- 5 by 6 checker squares
- 60 mm checker edge
- 50 mm marker edge
- OpenCV `DICT_6X6_250`
- legacy ChArUco layout

The project-owned metric definition is
[`config/zerodex_charuco_handeye_board2_60mm.json`](config/zerodex_charuco_handeye_board2_60mm.json).
The shared ParaDex board configuration is read-only and is not changed.

Start the Franka daemon in its own terminal after selecting the Desk mode that
will be used for hand-guiding:

```bash
cd /home/hyunsu/autodex-code
sudo bash demo/precision-insertion/run_zerodex_franka_daemon.sh
```

In a second terminal, start the read-only teaching capture. This process never
commands robot motion:

```bash
cd /home/hyunsu/autodex-code
source demo/precision-insertion/config/zerodex_capture4.env.sh
export PYTHONPATH="$PRECISION_REPO_ROOT/demo/precision-insertion:$PRECISION_REPO_ROOT:$PRECISION_PARADEX_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CAM_CAL=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/extrinsic_solutions/20261103_board11_4cam_v1
HE_SESSION=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/handeye_sessions/20261103_board2_ee_fr3_manual_v1

"$PRECISION_PLANNER_PY" demo/precision-insertion/capture_zerodex_handeye.py \
  --launch-config "$PRECISION_STAGE_ROOT/launch.json" \
  --board-config demo/precision-insertion/config/zerodex_charuco_handeye_board2_60mm.json \
  --camera-calibration-dir "$CAM_CAL" \
  --session-dir "$HE_SESSION"
```

The monitor shows the four full camera views as a 2-by-2 grid. Green marks are
the current detected ChArUco corners. The white cross/magenta ring is the
current board centre. Thin coloured points are all corners from already saved
poses, and their numbered coloured circles mark the corresponding centres.
Move the arm by hand, release it, wait until it is stationary, and
press `c`. Press `q` to finish. Capture roughly 15--25 poses with translation
and rotations about multiple axes. The software records every requested pose;
it does not add a hidden visual-quality acceptance threshold.

Each capture stores the measured seven-joint vector and `T_robot_ee` alongside
the synchronized camera data. It also refreshes `trajectory_qpos.npy`,
`trajectory_T_robot_ee.npy`, and the human-readable `trajectory.json`. Thus the
exact taught pose sequence can be repeated later.

Solve the two rigid transforms jointly using all saved four-view pixels:

```bash
HE_SOLUTION=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/handeye_solutions/20261103_board2_ee_fr3_v1

"$PRECISION_PLANNER_PY" demo/precision-insertion/solve_zerodex_handeye.py \
  --session-dir "$HE_SESSION" \
  --camera-calibration-dir "$CAM_CAL" \
  --board-config demo/precision-insertion/config/zerodex_charuco_handeye_board2_60mm.json \
  --output-dir "$HE_SOLUTION"
```

`C2R.npy` is `T_world_robot`, matching the precision-insertion runtime. The
solver also writes the estimated rigid board mounting transform to
`T_ee_board.npy` and pixel residuals to `report.json`.

To repeat the same learned waypoint sequence and automatically recapture it,
start the daemon and run:

```bash
REPLAY_SESSION=/mnt/paradex2/hyunsu/precision_insertion/calibration/zerodex_capture4/handeye_sessions/20261103_board2_ee_fr3_replay_v1

"$PRECISION_PLANNER_PY" demo/precision-insertion/replay_zerodex_handeye.py \
  --launch-config "$PRECISION_STAGE_ROOT/launch.json" \
  --board-config demo/precision-insertion/config/zerodex_charuco_handeye_board2_60mm.json \
  --camera-calibration-dir "$CAM_CAL" \
  --trajectory-session "$HE_SESSION" \
  --session-dir "$REPLAY_SESSION" \
  --speed-scale 0.1 \
  --settle-s 0.5 \
  --monitor
```

The replay command does move the FR3. Its joint speed and stationary dwell are
explicit command-line parameters rather than values embedded in the code.
If a run is interrupted after saving earlier poses, append
`--start-waypoint N` to resume at source waypoint `N` in the same output
session. Each capture records that source index explicitly.
The default `velocity_stream` mode follows each straight joint-space segment
with minimum-jerk time scaling and the same daemon velocity stream used by the
live pipeline. It records the measured endpoint without adding a visual or
joint-error admission gate. `--motion-mode blocking` retains the daemon's
single blocking joint command as an explicit alternative.
