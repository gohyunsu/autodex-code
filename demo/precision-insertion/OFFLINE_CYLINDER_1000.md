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
SCREEN=$SHARED/AutoDex/precision_insertion/cylinder_endpoint_screen_1000_solid_20261010

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

The latest `..._solid_...` result additionally rejects key/hand vertices
inside the *solid socket*, not just intersecting triangle surfaces. A
surface-only BVH can report a box fully inside another box as clear. The
cylinder uses its validated analytic blind-socket solid; the square socket
uses two oblique ray-parity checks. The new full-family nominal count and
candidate IDs remain 13/19 for each gap, with zero screen errors. The old
`..._squeeze_...` reports and renders remain historical diagnostics rather
than current collision-screen evidence.

These 13 are **offline filter passes**, not proven physically usable grasps:
the default controller squeeze is only a nominal commanded pose, there has
been no live Franka/fixture path check, and the endpoint is not a continuous
insertion simulation. Do not publish them as successful robot insertion
demonstrations or copy them into the active v8 runtime pool without review.
The later 2026-10-11 staged subset and its stricter provenance checks are
documented below; the original 13-count remains a nominal-only result.

## Post-squeeze fidelity audit: do not use nominal renders as success evidence

The stock MuJoCo filter compares key motion **after** the squeeze closure;
it does not require the key to retain the initial BODex hand-relative pose
during closure. The original endpoint screen and 312 still images combine
that initial `T_key_hand` with commanded squeeze joints, not the achieved
MuJoCo object pose and joint angles. The resulting deep-looking finger/key
overlap is therefore not an acceptable visual grasp validation.

Run the independent, non-filtering diagnostic from the repository root:

```bash
PYTHONPATH=demo/precision-insertion \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_cylinder_grasp_fidelity.py \
  --summary /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_endpoint_screen_1000_solid_20261010/summary.json \
  --shared-root /home/hyunsu/shared_data \
  --output-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_grasp_fidelity_solid_20261010
```

The completed audit is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_grasp_fidelity_solid_20261010/summary.json`.
It covers all 19 stock-MuJoCo-stable grasps. All 19 nominal fixed-key images
have sampled hand surface points more than 0.2 mm inside the cylindrical
key; maximum sampled depths are 4–15 mm depending on the grasp and hold.
Using the *achieved* MuJoCo key pose and hand joints greatly reduces this:
14/19 have any sampled overlap above 0.2 mm at either recorded squeeze or
gravity state, and the observed end-squeeze maximum is under 2 mm. These
sampling results are diagnostics, not certified collision depths or proof
of real physical success. The exact 256-sided key mesh is approximated by
its enclosing analytic cylinder for this interior test; the radial bound
differs by at most 1.13 µm, below the 0.2 mm diagnostic threshold.

More importantly, the cylinder center moves **2.4–35.2 mm relative to the
hand** during squeeze across the 19 candidates; the axis tilt, modulo the
key's axial-yaw/end-flip symmetry, ranges approximately **3–26 degrees**.
Thus the initial `T_key_hand` is not a validated rigid relation at lift or
insertion. These data do **not** retroactively change the requested stock
AutoDex plus 20 mm endpoint counts: 13 still pass those narrowly defined
checks per socket. They do prevent treating that count, or the old images,
as the number of robot-ready insertion grasps. The runtime must measure the
grasp relation from physical calibration and/or independently visible
post-lift evidence, reject unacceptable drift using calibrated limits, and
repeat hand/socket endpoint and path checks with that relation. A hand-
occluded key cannot be assumed measured merely because a VLM sees the hand.

## Achieved-MuJoCo-pose endpoint comparison

The independent `screen_cylinder_achieved_endpoints.py` reuses the same
20 mm CAD/whole-hand endpoint checker but pairs each recorded *achieved*
MuJoCo hand joint vector with its corresponding key-relative wrist pose.
It separately tests the end of the first squeeze and the end of AutoDex's
second closure/gravity test. An ID appears in the comparison count only when
both states avoid the socket. No post-squeeze drift acceptance threshold is
assumed, and a simulation-achieved pose is not measured robot feedback.

```bash
PYTHONPATH=demo/precision-insertion \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/screen_cylinder_achieved_endpoints.py \
  --fidelity-audit /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_grasp_fidelity_solid_20261010/summary.json \
  --shared-root /home/hyunsu/shared_data \
  --output-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_achieved_endpoint_solid_verified_20261010 \
  --minimum-hand-clearance-m 0.000001
```

| Radial socket gap | Nominal initial-pose clear | Both achieved MuJoCo states clear |
| --- | ---: | ---: |
| 1, 3, 5, 10 mm (each) | 13/19 | 13/19 |
| 15, 20 mm (each) | 13/19 | 12/19 |

The full report is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_achieved_endpoint_solid_verified_20261010/summary.json`.
At 15 and 20 mm, `table/0/71` loses endpoint clearance: its achieved
index-finger link intersects the larger socket body (the palm also
intersects at the 20 mm end-squeeze state). The per-state reports record
which links intersect and how many vertices lie in solid socket volume.
This is still **endpoint-only offline evidence**: no continuous insertion,
Franka motion, physical grasp, or real 20 mm task success has been tested.

## 2026-10-11 v8 tabletop candidate staging

`promote_cylinder_tabletop_candidates.py` verifies the complete stock
full-key MuJoCo pass set, byte-identical raw/stock grasp arrays, scene and
mesh hashes, the nominal 20 mm reports, and both achieved MuJoCo endpoint
reports for **all six** socket gaps. It intersects those tests across the
socket family before writing to the canonical v8 key candidate tree. The
intersection is **12 candidates**: 11 from `table/0`, one from `table/1`.
The nominal-only `table/0/71` is excluded because the achieved hand collides
with the 15/20 mm socket bodies. This is a conservative offline endpoint
selection, **not** a post-squeeze drift or physical-repeatability pass.

```bash
PY=/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python
SHARED=/home/hyunsu/shared_data
ACHIEVED=$SHARED/AutoDex/precision_insertion/cylinder_achieved_endpoint_solid_verified_20261010/summary.json

# Read-only source preflight (omit --install).
$PY demo/precision-insertion/promote_cylinder_tabletop_candidates.py \
  --shared-root "$SHARED" --achieved-summary "$ACHIEVED"

# First-time, non-overwriting installation only after reviewing that output.
$PY demo/precision-insertion/promote_cylinder_tabletop_candidates.py \
  --shared-root "$SHARED" --achieved-summary "$ACHIEVED" --install \
  --manifest "$SHARED/AutoDex/precision_insertion/cylindrical/tabletop_v8_promotion_1000_20261011.json"
```

The installed files are under
`AutoDex/candidates/inspire/v8/precision_key_cylinder_r15_h80/table/<scene>/<seed>/`.
Each candidate retains the exact stock arrays and original `sim_eval.json`,
`sim_traj.json`, `coll_valid.npy`, plus `simulation_validation.json` with
source-byte hashes, achieved-endpoint report hashes and the squeeze/gravity
drift diagnostic. The demo catalogue now checks those source hashes and its
own implementation hash; editing an input invalidates the saved catalogue.
Do not mistake `simulation_validation.status="passed"` for a real-robot
result: its schema explicitly says `physical_validation=false` and
`robot_ready=false`.

On this host, each of the six socket modes was independently rescreened from
the installed v8 pool at 1 µm **numerical** hand clearance. Each complete
catalogue contains 12/12 offline-eligible nominal endpoints and zero screen
errors. The current catalogues are
`AutoDex/precision_insertion/cylindrical/endpoint_catalog_staged_verified_gap_<N>mm_20261011.json`,
where `N` is `1`, `3`, `5`, `10`, `15` or `20`. The earlier files without
`verified` in their names are superseded because the candidate-evidence
implementation hash was extended after they were written.

The selected 12 still move **4.4–38.5 mm** in hand-relative cylinder center
from the initial BODex relation to the end of gravity replay, with roughly
**4.4–25.9°** symmetry-reduced axis tilt. No acceptable physical drift
threshold has been commissioned. The v8 arrays are therefore proposals for
pose-conditioned live preflight and physical-grasp calibration, not replay-
ready insertion grasps. Canonical FoundPose representations for key and
socket are missing from this `~/shared_data` runtime tree. The mounted
`/mnt/paradex2/hyunsu/autodex_precision_insertion_handoff_20261010_652ac909/pending_foundpose/`
contains synthetic-onboarded **candidates**, explicitly withheld from
canonical installation pending real AutoDex-image pose/mask QA. Physical
transfer paths, guarded contact and insertion labels are also unverified.
The unchanged stock AutoDex execution files remain untouched.
