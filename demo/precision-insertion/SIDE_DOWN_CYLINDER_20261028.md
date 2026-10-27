# ZeroDex side-down cylinder: offline pipeline review (2026-10-28)

This is an offline expansion for a key lying on the board (`table/1`, tabletop
stem `001`). It does **not** establish a Franka trajectory, a live pickup, or
an insertion success. The separate ZeroDex launch already permits direct
planning from this pose (`upright_cylinder_only=false`); the AutoDex launch
keeps its original upright-only policy.

## Evidence and results

Using the existing straight-mount Franka/Inspire geometry and an independent
seed-456 BODex run, the full 80 mm key was used for the unchanged AutoDex v8
scene/contact/MuJoCo filter. Only the requested nominal, axis-aligned 20 mm
key/socket and whole-hand/socket endpoint screen was added. There is no new
key-in-hand drift acceptance threshold or contact-region restriction.

| Stage | Side-down result |
| --- | ---: |
| BODex raw proposals | 1,000 |
| Stock v8 scene collision clear | 890 |
| Stock v8 squeeze contact | 499 |
| Stock v8 MuJoCo gravity stable | 6 |
| 20 mm endpoint clear, gaps 1/3/5/10 mm | 2 (`table/1/699`, `795`) |
| 20 mm endpoint clear, gaps 15/20 mm | 1 (`table/1/699`) |

The six-gap endpoint report has no screening errors. A diagnostic audit of all
six MuJoCo-stable candidates measured `table/1/699` at 2.4 mm center-in-hand
displacement at the end of squeeze and 4.9 mm at the end of gravity. These
are simulated observations, not a filter or real-world calibration. The
nominal fixed-key hand render penetrates the key; the achieved MuJoCo pose
reduces sampled overlap markedly. Do not present a nominal image as a
physically achieved grasp.

Two endpoint-eligible candidates were appended to the v8 pool under new IDs
`table/1/30699` and `table/1/30795`; historical candidates remain intact.
Only `30699` is eligible for the 20 mm-gap socket. A fresh complete endpoint
catalog is required after this append, because older catalogs predate both
IDs. Even a refreshed catalog only enables pose-matched **planning attempts**;
the current fixed socket/key/joints must still yield a full pickup and held
transfer/insertion plan at runtime.

The 20 mm-gap ZeroDex default catalog was refreshed with the explicit straight
mount: 126 directories, 111 eligible, zero errors, including five eligible
side-down candidates. The launch's `--check` and a full source-verified
`select_pose_candidates(..., tabletop_pose_stem="001")` both pass. The default
profile path is
`~/shared_data/AutoDex/precision_insertion/zerodex_capture4/recipient_gap20_endpoint_catalog_straight.json`;
its previous version was moved into the archive below. The other five gap
catalogs still contain the original pool and have not been rescanned with the
two added IDs. They remain valid records of their prior candidates but are
not current complete scans of the expanded pool.

Run-specific evidence is under `~/shared_data/AutoDex/precision_insertion/`:

- `cylinder_side_down_sim_pass_1000_seed456_20261028/`
- `cylinder_side_down_endpoint_straight_1000_seed456_20261028/summary.json`
- `cylinder_side_down_fidelity_straight_1000_seed456_20261028/summary.json`
- `cylindrical/full_length_side_down_straight_1000_seed456_promotion_20261028.json`

The raw and full-key staged proposals are respectively under
`~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_side_down_1000_seed456_20261028/`
and `.../precision_insertion_cylinder_side_down_fullkey_1000_seed456_20261028/`.

## What remains unverified in the current setup

The most recent four-camera capture is almost black (per-camera grayscale
99th percentiles 6, 8, 9, 19 of 255), so this exact present placement has no
reliable key pose. A saved bright lying-key capture from 2026-10-27 did pass
all four key masks and FoundPose initial poses, and current silhouette
refinement classified stem `001` at 2.38 degrees residual; that is a replay,
not a fresh same-session socket/key/robot-state observation. The robot and
hand are off. No live Franka preflight or physical insertion is claimed.

The unchanged full-chain planner was also rerun in the *same hypothetical*
lying-key layout used for the earlier four-candidate diagnostic (key XY
`(0.4, 0.0)` m, socket XY `(0.6, 0.0)` m, 40 mm table, stock home joints).
All five endpoint-eligible candidates were attempted; the new `30699` was
scene-clear but had no pickup IK at that hypothetical location. Overall
status remained `repose_required_unplanned`. The report is under
`zerodex_capture4/runs/synthetic_latency_20261028_pose001_seed456_all5/`.
This does not rule out a plan at the **measured** key/socket poses and robot
joints; those measurements are presently unavailable.

The offline scripts now accept a selected scene (`--scenes 1` for staging),
and the endpoint screen, fidelity audit, and append-only promotion derive
their scene set from the saved run. Their two-scene default is unchanged.
Run standalone endpoint/audit/promotion/catalog tools with
`PRECISION_FR3_INSPIRE_URDF=/home/hyunsu/autodex-code/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf`.
Without that explicit ZeroDex override, `AssetPaths` selects the AutoDex
right-angle mount. A first screen mistakenly used that default; its endpoint
reports, audit, manifest, and two installed candidate copies were moved to
`.../precision_insertion/archive/20261028_side_down_wrong_mount/`. No old
candidate was removed. The correct straight-mount report binds URDF SHA256
`e5ecd9466883d15c5691895db3e5126dbd76844d02ff7972c8d8996c8492f5f3`.
