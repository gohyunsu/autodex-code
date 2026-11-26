# Board-free ZeroDex runtime

The ZeroDex capture4 launcher now defaults to `board_free`. The ChArUco board
is used once to measure the support surface, but is not captured or fitted
during an insertion session.

The active commissioned inputs are:

- Extrinsics: `~/calibration/zerodex_capture4/extrinsic/20261118_232125/solution_filtered`
- Robot/world transform: `~/calibration/zerodex_capture4/handeye/20261118_233621/solution/C2R.npy`
- Table plane: `~/precision_insertion_project/calibration/table_plane/current/table_plane.json`
- Active capture4 cameras: `26053248,25322639,25305462,25322642`

The hand-eye manifest explicitly depends on the extrinsic calibration above.
The table record stores both paths in `calibration_inputs`.

Run the static pipeline preflight with:

```bash
cd ~/autodex-code
bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

To intentionally run the historical board-constrained mode, override the
default explicitly:

```bash
PRECISION_SCENE_MODE=charuco_board \
  bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

To remeasure the table after the board or camera rig changes, use
`measure_zerodex_table_plane.py` with the same extrinsic and C2R paths. Move
the resulting directory behind the `current` symlink only after reviewing its
four-view overlay and fit metrics.
