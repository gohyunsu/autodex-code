# Cylindrical key: 1,000 BODex proposals per tabletop scene

This is a grasp-level offline run, **not** a robot-executable insertion
scenario. It uses the existing `sim_inspire/precision_insertion.yml` default
`seed_num: 1000`, tabletop scenes `0` (end-down) and `1` (side-down), the
Inspire hand, and the full 80 mm physical key for all validation after BODex.
The 25 mm grip proxy is only the BODex proposal geometry because the exact
256-sided cylinder causes the BODex/COAL convex-hull seeder to fail. The
proxy and full key use identical canonical frames and tabletop poses; the
staging script verifies this before copying object-frame candidate poses.

## Filter contract, in order

1. **BODex proposal:** save 1,000 raw seeds for each scene. The BODex native
   `success` flag is recorded but is *not* a gate in stock AutoDex's sim
   filter; its QP threshold is known to be overly strict. A raw seed is not a
   validated grasp.
2. **Original AutoDex scene collision:** `run_sim_filter.py` uses cuRobo to
   reject pregrasp hand/world or self-collision in the *full-key* tabletop
   scene. This is a single pregrasp configuration, not a planned Franka path.
3. **Original AutoDex object contact:** its squeeze-pose hand collision
   spheres must touch the *full physical key*. Noncontact candidates fail.
4. **Original AutoDex MuJoCo stability:** it closes/squeezes the Inspire hand
   on the *full physical key*, applies gravity expressed in that tabletop
   pose's object frame, and fails if key translation exceeds 50 mm or rotation
   exceeds 15 degrees during the 50-step test. This is only simulated grasp
   stability, not pickup/reorientation/insertion.
5. **New insertion endpoint gate:** for each socket gap separately, place the
   key on the socket's CAD centerline with its axis aligned at exactly 20 mm
   insertion. Test exact key/socket mesh fit and every Inspire visual hand
   link against the exact concave socket collision mesh. Test both the
   MuJoCo squeeze command and AutoDex's nominal default controller hold
   command (`squeeze_level=2`); *both* must avoid socket collision. A tiny
   positive numerical clearance (1 µm for this run) is not a calibrated
   physical safety margin. The grasp is kept rigid to the key. No Franka
   arm or continuous transfer/insertion path is screened offline.

No extra hand/key contact-region policy, BODex numerical error threshold, or
Franka arm filter is silently added to this requested experiment. The socket
gaps in the cylindrical assets are **radial 1, 3, 5, 10, 15, and 20 mm**;
the object names `gap_01mm` etc. are literal millimetres, not tenths.

## Reproduce on the CUDA host

Run from the repository root. Use new run-specific output names: these tools
refuse to overwrite an existing staged or screened run. The CUDA sandbox may
need GPU access; no robot connection is made.

```bash
cd /home/hyunsu/autodex-code
export PYTHONPATH=/home/hyunsu/autodex-code/demo/precision-insertion/compat
PY=/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python
SHARED=/home/hyunsu/shared_data
RAW=$SHARED/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_proxy_1000_20261010
STAGED=$SHARED/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_fullkey_eval_1000_20261010
SIMPASS=$SHARED/AutoDex/precision_insertion/cylinder_fullkey_sim_pass_1000_20261010
SCREEN=$SHARED/AutoDex/precision_insertion/cylinder_endpoint_screen_1000_squeeze_20261010

$PY src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 2 \
  --obj_list_file demo/precision-insertion/configs/cylinder_grip_proxy.txt \
  --obj_root_dir "$SHARED/object_processing" \
  --scene_filter_file demo/precision-insertion/configs/cylinder_tabletop_scenes.json \
  --scene_type table --exp_name precision_insertion_cylinder_proxy_1000_20261010 \
  --output_dir "$RAW" --seed 123

$PY demo/precision-insertion/stage_cylinder_tabletop_proposals.py \
  --raw-root "$RAW" --stage-root "$STAGED" --shared-root "$SHARED" \
  --expected-per-scene 1000

$PY -c 'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_cylinder_r15_h80", "/home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_fullkey_eval_1000_20261010", "/home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_fullkey_sim_pass_1000_20261010", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'

$PY demo/precision-insertion/screen_cylinder_tabletop_batch.py \
  --shared-root "$SHARED" --stage-root "$STAGED" \
  --candidate-root "$SIMPASS" --output-root "$SCREEN" \
  --expected-per-scene 1000 --minimum-hand-clearance-m 0.000001
```

The original AutoDex sim filter must run in its normal single-worker mode;
its CLI multi-worker path calls `sim_only=True` and skips the squeeze-contact
check. We call the original `run_sim_filter()` with explicit run-specific
roots instead of modifying the original execution or filtering files.

The authoritative result is `SCREEN/summary.json`; individual endpoint
reports are under `SCREEN/gap_XXmm/table/<scene>/<seed>.json`. `SIMPASS` is
an isolated simulation-stage pool, **not** the live
`AutoDex/candidates/inspire/v8/` tree. Promotion should occur only after
inspection of the final reports and a separate visual hand/key contact audit.
Even a passing endpoint does not prove Franka reachability, grasp replay,
continuous clearance, contact-force safety, or physical insertion success.

## 2026-10-10 result

The exact precision config generated 1,000 seeds in each scene (2,000 total).
BODex's strict native success flag was 0/2,000; stock AutoDex does not use
that flag as its sim-filter gate. The original full-key filter gave:

| Tabletop scene | Raw | Scene clear | Full-key squeeze contact | MuJoCo stable |
| --- | ---: | ---: | ---: | ---: |
| `table/0` end-down | 1,000 | 246 | 176 | 12 |
| `table/1` side-down | 1,000 | 243 | 148 | 7 |
| Total | 2,000 | 489 | 324 | 19 |

With both MuJoCo and default AutoDex controller hold poses checked at the
centered 20 mm endpoint, 13/19 were clear for *each* of the six socket gaps.
All six endpoint failures came from hand/socket collision in scene 1; key
and socket fit geometrically in all 19 tests. The six gap catalogues are
not independent grasp generations: they rescreen the *same* 19 full-key
grasp-stable candidates against six socket meshes. The earlier diagnostic
directory `cylinder_endpoint_screen_1000_20261010` checked only the less-
closed `grasp_pose` and is superseded by the `..._squeeze_...` result.

These 13 are **offline filter passes**, not proven physically usable grasps:
the default controller squeeze is only a nominal commanded pose, there has
been no visual hand/key penetration audit or live Franka/fixture path check,
and the endpoint is not a continuous insertion simulation. Do not publish
them as successful robot insertion demonstrations or copy them into the
active v8 runtime pool without those reviews.
