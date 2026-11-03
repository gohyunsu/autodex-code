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
  --session-dir "$SESSION"
```

Press Enter once for each new board placement and `q` when finished. Capture
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
