# Fallen cylinder key: ZeroDex capture4 trial (2026-10-28)

This is a measured 5 mm radial-gap socket session on the kds-Z790/FR3/Inspire
system. The key was lying at the upper right edge of the ChArUco board. No
reorientation pickup or release was executed in the trials below.

## Findings

- The camera-only run initially chose an upright board-body fit for the lying
  key. The three admitted FoundPose views instead had nearly horizontal key
  axes. The tabletop selector now uses the view-majority axis class to choose
  the upright or side-down model. The side-down result is refined by stock
  AutoDex silhouette fitting and placed on the measured board plane. A fresh
  camera-only run classified it as v8 pose `001` with six gap-5 insertion
  candidates. Its evidence is at
  `/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/camera_only_side_down_fixed_20261028_2300`.
- With `upright_cylinder_only=true`, a separate launch template routed pose
  `001` to directed `001→000` reorientation. The first planning run found no
  release site because the selector searched only the original key center
  and four sites 3 cm away, while the old ChArUco interior placement rule
  excluded every site. The selector now also tries the measured board center.
  The ChArUco interior is a calibration footprint rather than the physical
  tabletop boundary; its placement margin is disabled in the ZeroDex stage
  profile. The socket collision check remains.
- The next preflight tested three existing MuJoCo-stable `001→000` reset seeds
  (`1487`, `1687`, `2569`) from
  `AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12`.
  All three passed the stock candidate geometry filter (`n_valid=1`,
  `n_world_collision=0`, `n_self_collision=0` per seed), but **none had an
  inverse-kinematics solution** (`n_ik_success=0`). Approach, lift, held
  transfer, opening and release were therefore never planned or executed.
  The report is under
  `/home/hyunsu/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261028_223648_713821/repose_preflights/0000/report.json`.
- After moving the lying key closer to the board center, the same three reset
  seeds again passed candidate collision filtering and all returned
  `n_ik_success=0` in
  `_pipeline_runs/20261028_224018_043250/repose_preflights/0000/report.json`.
  The six existing side-down insertion grasps were also tried for pickup in
  `_pipeline_runs/20261028_224227_884266/trial_preflights/0000/report.json`;
  all six passed candidate filtering and had `n_ik_success=0`. Thus the
  current saved side-down grasp pool does not provide a planned pickup at
  either observed key location. Key yaw and the straight Inspire mount remain
  possible causes; the records do not isolate which one matters.
- The earlier missed-grasp recovery aborted because it checked a freshly
  observed fallen key as though it were a *planned placement target*. That
  target-only check is now skipped for a visual miss; the key is still
  reobserved before the next attempt.

## Reproduce the stand-up planning probe

Copy the active ZeroDex `launch.json` into a development folder and set only
`upright_cylinder_only` to `true`. Leave the ordinary launch unchanged. Set
`PRECISION_MAX_STEPS=2` to stop after one reorientation preflight, before
any pickup. For example:

```bash
PRECISION_GAP_MM=5 \
PRECISION_LAUNCH_TEMPLATE=/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/stand_fallen_key/launch_upright_only_gap5.json \
PRECISION_REUSE_SESSION_EVIDENCE=/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/socket_session_gap5_clean_20261028/socket_evidence \
PRECISION_KEY_ALREADY_PLACED=1 \
PRECISION_MAX_STEPS=2 \
PRECISION_RESET_CANDIDATE_ROOT=/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12 \
bash demo/precision-insertion/run_zerodex_capture4.sh --run
```

The current key location does not yield a pickup IK solution for these three
saved seeds. A new capture after moving the key to a reachable empty tabletop
location can distinguish a workspace issue from a grasp-orientation issue.
Keep the socket and board fixed so the saved socket session remains valid.
