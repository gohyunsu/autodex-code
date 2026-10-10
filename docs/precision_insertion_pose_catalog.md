# Pose-conditional insertion and reorientation inventory

This is a **planning/evidence inventory**, not an insertion-capable robot
pipeline. It answers what is known for each observed key tabletop pose without
equating "no saved passing candidate" with "no feasible insertion exists".

## How it is constructed

For each square key (0.3, 0.5, 1.0, 1.5 mm nominal gap) and each cylindrical
socket (1, 3, 5, 10, 15, 20 mm **radial** gap), the builder reads the key's
`object_processing/<key>/processed_data/info/tabletop/*.npy`. Square keys have
five pose IDs; the symmetric cylinder has two. A catalog row is made for every
mode/gap/pose. The 20 mm insertion **depth** is a different dimension from the
socket gap.

Each row inventories three distinct things:

1. Files in `AutoDex/candidates/inspire/v8/<key>/table/<pose>/<candidate>/`.
   These are candidate files only, not insertion evidence.
2. Matching records from `AutoDex/precision_insertion/scenario_catalog.json`,
   including their exact validation level and evidence. The 1.5 mm square
   pose-004 pilot stops at grasp stability and sampled endpoint geometry.
3. Whether any record explicitly passed **full-task** simulation or hardware
   validation. A missing pass leaves `absence_verdict=unknown_not_proven`.

The corresponding directed reorientation inventory has one `i_j` row for
every ordered pair of different tabletop poses. It locates:

- paired BODex reorientation scenes (`reorient_12/i_j.json`, etc.);
- AutoDex runtime reset grasp seeds under
  `AutoDex/candidates/inspire/reset_<height>/<key>/reorient_<height>/i_j/`;
- separately staged diagnostic grasps; and
- source-tabletop grasps as **hypotheses** for that transition (the same
  object-relative grasp can be tried against several target poses, but this
  does not establish pair-scene or whole-chain feasibility); and
- a cuRobo preview only when its report actually names the same cell.

For the native-v8 precision keys, `i_j` uses the object-processing tabletop
stems directly, matching `_reset_cell_indices()` in
`src/experiment/reset/reorient.py`. The scene pair encodes source support and
target support orientation. It does **not** itself contain a hand grasp. BODex
must optimize a grasp common to the paired scene. Raw BODex output retains
failed optimization seeds; it is inventoried separately and never counted as
a viable grasp. Whole-hand contact and
MuJoCo stability must then be checked. The existing AutoDex reset planner
tests approach, close, lift, reorient, descend, release, and exit. A robot
transition still requires calibration and physical verification.

## Regenerate

Use the AutoDex BODex environment. The paired scenes below were generated for
the previously missing square key variants and the cylinder. Do not rerun the
command on an existing output without first inspecting that directory:

```bash
cd ~/autodex-code
~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/reorient/gen_all.py \
  --objects precision_key_0p3mm precision_key_0p5mm precision_key_1p0mm \
            precision_key_cylinder_r15_h80 \
  --h 0.12 --hand inspire --version v8
```

The cylinder's BODex contact object is its 25 mm grip proxy in the **same key
frame**. The pair-scene pillars and support surfaces still come from the full
80 mm key. Its 64-sided proposal mesh avoids a COAL convex-hull failure on the
256-sided original; its maximum radial surface approximation is 0.0181 mm.
Only the proposal mesh is coarsened. Regenerate the proxy and its scenes via:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/build_cylindrical_assets.py \
  --shared-root ~/shared_data
~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/reorient/gen_all.py \
  --objects precision_key_cylinder_r15_h80 \
  --target-obj precision_key_cylinder_r15_h80_grip_proxy \
  --h 0.12 --hand inspire --version v8
```

The first BODex pilot used the standard AutoDex
`sim_inspire/paradex_reorient_12.yml` configuration, 100 seeds per directed
cell, and the repository object/filter lists:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/paradex_reorient_12.yml -w 1 \
  --obj_list_file assets/precision_insertion/bodex_cylinder_reorient_object.txt \
  --obj_root_dir ~/shared_data/object_processing \
  --scene_filter_file assets/precision_insertion/bodex_cylinder_reorient_12_filter.json \
  --exp_name precision_insertion_cylinder_reorient_proxy_pilot_100 \
  --seed_num 100 --grasp_threshold 0.2 --distance_threshold 0.01 \
  -o ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_reorient_proxy_pilot_100
```

The output is **raw optimization evidence only**. Both cells saved 100 seeds.
Native BODex `success` was 0/100 for both. Rechecking the project's relaxed
componentwise `grasp_error <= 0.2` and mean absolute contact-distance
`<= 0.01 m` gives 1/100 for `0_1` and 2/100 for `1_0`. Those three are not
contact-screened or MuJoCo-tested and must not be copied to the runtime reset
pool. A finite 100-seed pilot cannot establish that either transition is
impossible. Do not rerun the same experiment name over existing raw outputs;
use a new versioned name/seed offset.

Build the read-only inventory from the currently saved evidence:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/build_pose_task_catalog.py \
  --shared-root ~/shared_data
```

The builder refuses to overwrite its two JSONs unless `--force` is given.
Use `--force` after adding new candidate/evaluation evidence. It replaces only
these two generated catalog files, not BODex seeds or scene assets.

Outputs:

- `~/shared_data/AutoDex/precision_insertion/pose_task_catalog/full_task_catalog.json`
- `~/shared_data/AutoDex/precision_insertion/pose_task_catalog/reorient_seed_catalog.json`

The square 0.1 mm variant is intentionally not included: no corresponding
`object_processing/precision_key_0p1mm` runtime object exists in this local
asset set. Add that object and deliberately add 0.1 to the builder's selected
gap list if it becomes an experimental condition.

## Decision rule for a real session

With observed pose `i`, select a scenario only if its exact
`(mode, gap, key, socket, i)` matches and its full-task evidence covers grasp,
continuous attached-key transfer, fixture-aware approach, guarded 20 mm
insertion, contact outcome, and safe retreat. Recheck the plan against the
**session-measured** socket pose. A simulated pass is not hardware approval.

If no validated scenario exists, the default decision is **unknown / do not
insert**. To call the *finite generated pool* exhausted, first freeze that
pool's identity and show a complete negative full-task evaluation for every
member. This is still not a mathematical proof that no grasp exists. An
automatic `i -> j` reorientation additionally requires a valid transition
seed, full-chain planning and physical validation, a validated insertable
target pose `j`, and fresh perception after placement. Do not treat an
animation or nominal CAD fit as any of those gates.

## Current evidence boundary

- 32 pose rows: 4 square gap variants × 5 poses and 6 cylinder gaps × 2 poses.
- 92 directed transitions: 4 × 5 × 4 plus 6 × 2 × 1.
- 62 newly generated pair-scene JSONs; the 1.5 mm square's 20 scenes already
  existed.
- Two additional full-key-derived, proxy-target cylinder pair scenes and a
  two-cell 100-seed BODex pilot are present, but neither cell has a native
  BODex success or a validated reset seed.
- One archived `4_0` square diagnostic seed (candidate 104) has a continuous
  cuRobo **reorientation motion preview**, not a physical transition or
  insertion pass. It remains outside the runtime reset pool.
- Zero full-task insertion passes and zero automatic-reorient-ready transitions.

The next production steps are (1) generate pose-specific BODex reset grasps,
(2) screen complete-hand contact and grasp stability, (3) run the existing
Franka full-chain reorientation planner with the fixed socket in its collision
world, (4) run continuous insertion/contact simulation for each scenario, and
(5) calibrate and verify on the physical AutoDex rig. The present catalogs
must remain advisory until those records exist.
