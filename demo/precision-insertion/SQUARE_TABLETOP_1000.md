# Square full-key tabletop v8 candidate refresh

This refresh uses the existing `sim_inspire/precision_insertion.yml` and
unchanged AutoDex `BODex/generate.py` and `run_sim_filter()` implementations.
It creates **1,000 full-key proposals for each of five tabletop scenes and
each of four square-key gaps**. No object-side contact-region mask or separate
project BODex numerical quality threshold is used. The precision config
still chooses four Inspire contact fingers; this is not a claim that all five
fingers or all physical key contacts were tested.

Stock v8 filtering checks pregrasp scene/self collision, squeeze contact,
then MuJoCo squeeze and tabletop-gravity stability. The demo's additional
screen checks the centered, axis-aligned 20 mm CAD key/socket fit and the
whole Inspire hand/socket endpoint, including the stock simulated squeeze
pose and nominal AutoDex controller hold pose. It **does not** screen the
continuous Franka path, contact-rich insertion, or physical success.

From the repository root, use the isolated `autodex_bodex` environment and
RTX GPU for BODex and the stock filter:

```bash
env PYTHONPATH=demo/precision-insertion/compat \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file demo/precision-insertion/configs/square_tabletop_full_keys.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_type table --seed_num 1000 --seed 12011 \
  --exp_name NEW_SQUARE_TABLETOP_RAW --output_dir /absolute/NEW_SQUARE_TABLETOP_RAW

env PYTHONPATH=demo/precision-insertion/compat \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_1p0mm", "/absolute/NEW_SQUARE_TABLETOP_RAW", "/absolute/NEW_SQUARE_TABLETOP_STOCK_PASS", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'
```

Repeat the stock filter call for `precision_key_0p3mm`, `0p5mm`, and
`1p5mm`. Use new output roots; do not overwrite a previous batch. The stock
single-process function is intentional: its separate multiprocessing CLI
mode can skip the contact check.

```bash
/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/screen_square_tabletop_batch.py \
  --shared-root /home/hyunsu/shared_data \
  --raw-root /absolute/NEW_SQUARE_TABLETOP_RAW \
  --pass-root /absolute/NEW_SQUARE_TABLETOP_STOCK_PASS \
  --output-root /absolute/NEW_SQUARE_1P0_SCREEN \
  --expected-per-scene 1000 --minimum-hand-clearance-m 0.0002 \
  --gap-mm 1.0

/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/promote_square_stock_batch.py \
  --shared-root /home/hyunsu/shared_data --gap-mm 1.0 \
  --screen-root /absolute/NEW_SQUARE_1P0_SCREEN \
  --candidate-id-offset 12011000 \
  --output-manifest /absolute/NEW_SQUARE_1P0_PROMOTION.json
```

The promotion command is **read-only by default**. Add `--install` only after
reviewing `summary.json`: it appends eligible grasps as v8 numeric IDs
`candidate-id-offset + original-seed-id`, refuses to replace an existing
candidate, and records source, scene, mesh and endpoint hashes. A complete
current-pool endpoint catalog must then be rebuilt with
`run_pipeline.py screen-catalog`; an older catalog becomes stale when the
candidate pool grows. The 0.2 mm hand clearance here is an uncommissioned
numerical screen, not a measured safety margin.

## 2026-10-11 full-key result

All 20 scenes contain exactly 1,000 proposals. The unchanged AutoDex v8
filter and exact 20 mm endpoint screen produced:

| Square gap | MuJoCo-stable / 5,000 | New endpoint-eligible | Eligible by tabletop pose `000`–`004` |
| --- | ---: | ---: | --- |
| 0.3 mm | 115 | 31 | `1 / 9 / 10 / 2 / 9` |
| 0.5 mm | 115 | 24 | `0 / 7 / 10 / 1 / 6` |
| 1.0 mm | 89 | 24 | `1 / 13 / 5 / 2 / 3` |
| 1.5 mm | 114 | 34 | `1 / 5 / 13 / 7 / 8` |

Screen errors were zero for all modes. The 1.5 mm key also retains its seven
older pose-`004` eligible grasps; its refreshed, source-verified current-pool
catalog contains **41** eligible grasps across all five poses. The sole
remaining square no-candidate tabletop condition is **0.5 mm pose `000`** in
this finite search. Neither the 20 mm endpoint count nor five-pose coverage
proves a continuous Franka trajectory or physical insertion.

The four current-pool catalogs are
`shared_data/AutoDex/precision_insertion/endpoint_catalogs/`
`square_{0p3,0p5,1p0,1p5}mm_20261011_fullkey1000_schema3.json`.
The all-mode [offline asset audit](OFFLINE_ASSET_STATUS.md) checks all four
against their installed candidate files.

## Four-gap NAS handoff and path relocation

The [versioned NAS handoff](/mnt/paradex2/hyunsu/autodex_precision_insertion_square_fullkey1000_handoff_20261011_10c455d)
contains all four key object-processing trees, the shared socket/fixture,
their five v8 tabletop scenes each, and the full current candidate pools
(124 candidate directories, of which 120 pass the nominal endpoint screen).
Its source catalogs are **audit-only**. All 1,266 payload/support files were
read back from NAS and SHA256-checked against `MANIFEST.json`.

On another PC, first provide the identical FR3/Inspire URDF and its meshes
at that PC's active `shared_data/AutoDex/content` path. Dry-run
`rehydrate_square_fullkey_handoff.py --bundle-root NAS_HANDOFF
--target-shared-root NEW_SHARED_ROOT`; add `--install` only after confirming
it reports no conflicting files. The installer never overwrites existing
files. It changes only the known scene/fixture absolute paths and the
candidate hashes that bind those scenes, preserving the original source
hash in each relocation record. Then run `run_pipeline.py screen-catalog`
for **each** square gap at the recipient root; never use the NAS copy's
source-root catalog directly. A scratch relocation into a different path
reproduced 31/24/24/41 eligible counts with zero screen errors.
FoundPose PTHs are distributed separately; neither bundle proves camera
pose accuracy or physical insertion.

## Missing 0.5 mm pose `000`: directed reorientation attempt

The stock v8 reorient scene generator prepared both BODex and sim-filter
`reorient_12` scenes for all 20 directed 0.5 mm pose cells. Two separate
NVIDIA scans in cell `0_4` used 1,000 and 5,000 full-key proposals. The
larger scan had 4,245 pregrasp-scene-clear, 1,332 squeeze-contact and 33
MuJoCo gravity-stable seeds. `stage_square_reorient_passes.py` bound those
33 to their proposal arrays, original MuJoCo trajectory, key geometry and
both scene files. The [curated staging manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_0p5_curated_5000_20261011/handoff_manifest.json)
re-verifies the copied seeds. It is **not** a reset trajectory or canonical
runtime pool.

Under the *illustrative, uncommissioned* 5 mm key-center drift / 10° full
relative-rotation gate, **two** of the 33 pass: seeds `274` and `1152`.
Their end-gravity drifts/rotations are respectively 4.204 mm/5.705° and
1.010 mm/5.412°. The source-verified [fidelity audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_0p5_handoff_0_4_5000_seed12013_20261011/fidelity_audit_5mm_10deg.json)
retains measurements for every stable seed. This makes the directed reset
seed loader usable for the *example* gate; socket-aware Franka
pickup–lift–repose–release preflight and observed landing are still absent.
The smaller 1,000-proposal scan remains archived separately and had seven
MuJoCo passes but zero passes at the same illustrative gate.

The [versioned NAS reset handoff](/mnt/paradex2/hyunsu/autodex_precision_insertion_reorient_square_0p5_5000_20261011_10c455d)
includes all 33 seeds, the original key and scene snapshot, a source-hash
manifest, and the 5 mm/10° audit. All files were byte-compared against the
local curated root. `rehydrate_curated_reorient.py --mode square --gap-mm 0.5`
relocated that NAS bundle into a distinct scratch shared-data path; its
direct-v8 loader reproduced 33 stable and two illustrative fidelity passes.
The required `--target-handoff-root` must be a new directory **inside** the
recipient's `--target-shared-root`, outside canonical AutoDex candidates.

Three hypothetical release sites were tested with the two illustrative
fidelity-pass seeds. All failed complete Franka repose preflight: at
`(0.45, 0.08)` m and `(0.40, 0.00)` m both failed held descent; at
`(0.35, -0.12)` m one failed descent by joint limit and the other failed
held transfer. For the first site's seed `274`, a [same-state obstacle
isolation](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/synthetic_repose_0p5_0_4_isolation_20261011/collision_isolation.json)
found the synthetic table obstacle necessary for its cuRobo world-collision
verdict: table-only failed, socket-only and neither passed. This does **not**
identify a contacting robot link or prove failure for every measured layout.
No 0.5 mm reset path is yet accepted for runtime execution.

The normal sandbox hides CUDA and exposes NAS as read-only, whereas an
explicitly authorized host execution saw RTX 3090 and a writable NFS mount.
The 5,000-proposal run used that host execution; later reproductions must
check which environment they actually run in.
