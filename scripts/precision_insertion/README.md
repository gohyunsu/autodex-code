# Precision-insertion asset workflow

The new **square/cylinder dual-mode** asset inventory, offline scenario
selection, VLM observation, bounded retry policy, execution-sequence
comparison, and exact usage are in
[`docs/precision_insertion_modes.md`](../../docs/precision_insertion_modes.md).
Robot insertion is not yet implemented. The original `run_pipeline.py` remains
the AutoDex grasp/lift/place path and no longer accepts precision-specific
socket preflight options. All future insertion runtime work belongs under
[`demo/precision-insertion/`](../../demo/precision-insertion/README.md).

For the symmetric cylindrical task, build and validate the supplied metric
STLs with the same v8 `object_processing`/`AutoDex/scene` layout:

```bash
~/.venvs/autodex-assets/bin/python \
  scripts/precision_insertion/build_cylindrical_assets.py \
  --shared-root ~/shared_data
~/.venvs/autodex-assets/bin/python \
  scripts/precision_insertion/validate_cylindrical_assets.py \
  --shared-root ~/shared_data
```

The stable object IDs are `precision_key_cylinder_r15_h80` and
`precision_socket_cylinder_gap_01mm` through `..._20mm`. The key is D∞ and
therefore has two symmetry-reduced tabletop classes (end-down and side-down);
the open-rim socket is C∞ about local z but has no end-for-end symmetry.
The independent demo's local geometry helper can ignore unobservable socket
yaw while rejecting excessive center or axis variation; it is not wired to a
robot runner yet. Generated geometry does not supply FoundPose weights, BODex
grasps, full-task trajectories, or physical validation. The future demo must
require an explicitly matching socket and round-opening segmentation prompt
before camera or robot startup.

The **canonical path layout is shared with the square-key assets**. Source
STLs live in this repository under `assets/precision_insertion/`; runtime
objects (key, proposal proxy, sockets) live directly under
`~/shared_data/object_processing/<object_id>/`, with scenes and FoundPose
markers under `~/shared_data/AutoDex/{scene,foundpose_assets,candidates}/`.
Task-specific fixture files live together under
`~/shared_data/AutoDex/precision_insertion/fixtures/`: `unified_socket/`
for the square family and `precision_socket_cylinder_gap_XXmm/` for each
cylinder socket. The six-socket manifest remains under
`precision_insertion/cylindrical/`; that directory is *not* a second runtime
object root. `gap_01mm` means **1 mm radial clearance**, not 0.1 mm or
diametral clearance.

The processed full-key cylinder mesh is split at local `z=25 mm`, so its
`contact_allowed.obj` really includes the lateral grip zone and rear cap.
The original raw mesh remains unchanged for perception. The proposal proxy
keeps the full key's mass/OBB reference but declares no standalone symmetry;
any generated grasp must be rechecked on the full key. `mass` in generated
metadata is AutoDex's unit-density volume proxy, **not measured printed-part
mass**. Measure the real key and bore dimensions before quantitative MuJoCo
or tight-gap conclusions.

This directory builds the assets for the staged unified-socket experiment:

| gap | experiment stage |
|---:|---|
| 1.5 mm | full pipeline bring-up with one fixed socket pose and one grasp |
| 1.0 mm | pose-accuracy measurement |
| 0.5 mm | contact-search introduction |
| 0.3 mm | final precision condition |

All four conditions now have geometry, frames, scenes, contact policies,
gap-specific proposal proxies, an existing pick/lift candidate, FR3 plan
evidence, and fail-closed stage profiles. The existing candidate is **not an
insertion candidate**: a later whole-hand audit found contact-policy and socket
clearance violations. None is physically trusted, and none has a FoundPose
representation yet. The staged experiment must not silently treat simulation,
a controller specification, or an unvalidated transfer as physical success.

The end-to-end intent, transform equations, controller/VLM boundaries,
failure taxonomy, blockers, and implementation order are specified in
`docs/precision_insertion_pipeline_design.md`.

## Current status

Generated under `~/shared_data`:

- four metric key objects under `object_processing/precision_key_*`;
- four proposal-only handle proxies under `object_processing/precision_key*proxy`;
- the unified socket mesh, CAD-relative insertion transforms, and fixture-pose
  template under `AutoDex/precision_insertion/fixtures/unified_socket`;
- the independent `precision_socket_unified` pose-estimation object under
  `object_processing`, with an identity socket-to-mesh frame contract;
- one fail-closed experiment profile per gap under
  `AutoDex/precision_insertion/stages`;
- a canonical AutoDex camera contract under
  `assets/precision_insertion/autodex_camera_profile.json`; the robot host
  writes a runtime audit after matching its active cameras and calibration;
- the same historical pick/lift grasp, candidate `table/0/78`, under each of
  the four `AutoDex/candidates/inspire/v8/precision_key_*` pools. It is retained
  for reproducibility but must be rejected by insertion preflight.

Candidate 78 was proposed with the 0.3 mm key's handle-only proxy. Its four
**declared** contacts are lateral handle contacts and it passed the historical
cuRobo scene-clearance/MuJoCo screen plus a hardware-free FR3 approach and
10 cm vertical-lift plan. That screen did not classify every visual hand-link
surface against the forbidden object region. The new audit proves that the
nominal hand mesh touches forbidden key regions and collides with the socket at
the CAD insertion pose. Do not interpret the old `simulation_validation.json`
scope label as a whole-hand forbidden-region proof.

It is **not physically trusted**. Every runtime pool retains
`PHYSICAL_VALIDATION_REQUIRED.json`; no physical grasp, lift, or insertion has
been claimed. The former 1.5 mm candidate 84 was moved, not deleted, to
`~/shared_data/AutoDex/archive/runtime_before_common_grasp_20261005`.

| gap | geometry | common grasp full-key sim | FR3 plan | physical | controller |
|---:|---|---|---|---|---|
| 1.5 mm | ready | seed 78 insertion rejected | pick/lift passed | required | spec only; open-loop implementation required |
| 1.0 mm | ready | seed 78 insertion rejected | pick/lift passed | required | spec only; accuracy instrumentation required |
| 0.5 mm | ready | seed 78 insertion rejected | pick/lift passed | required | force/contact XY-yaw search required |
| 0.3 mm | ready | seed 78 insertion rejected | pick/lift passed | required | final search controller required |

## Why the geometry is generated this way

The STL inputs are binary millimetre meshes. `build_assets.py` parses them
without relying on an implicit unit convention, scales vertices by `1e-3`,
checks watertightness/orientation, and writes the v8 `object_processing`
contract:

- `raw_mesh/<object>.obj` is the canonical perception mesh;
- `processed_data/mesh/simplified.obj` is the planning/simulation mesh;
- `processed_data/urdf/coacd.urdf` references a conservative convex piece;
- `simplified.json` records CoM, OBB, mass proxy, and scale;
- `tabletop/*.npy` stores five controlled full-key stable poses (tip-down is
  excluded); every handle-only proposal proxy copies those same poses rather
  than recomputing poses from its box-only geometry;
- `AutoDex/scene/inspire/<object>/table/*.json` binds those poses to BODex.

The key frame is fixed across all gaps: `+z` runs from the handle rear toward
the insertion tip, the rear plane is `z=0`, the socket-facing shoulder is
`z=45 mm`, and the tip is `z=85.5 mm`. The socket source frame has its entry
at `z=58.5 mm`. Therefore the seated transform is `Rx(pi)` with translation
`z=103.5 mm`: the shoulder lands at the entry plane and the tip lands at
`z=18 mm`. The true tip-at-entry transform is `z=144.0 mm`; pre-insertion is
30 mm above that at `z=174.0 mm`. Primary task success is 20 mm below entry at
`z=124.0 mm`, leaving a nominal 20.5 mm for the optional separate finish
press. These are CAD-relative transforms only; they do not determine the
physical socket pose in the Franka base frame.

## Contact rule and grasp proposal

The hand may contact only:

- all five axial lateral faces of the pentagonal handle, including its
  diagonal/keyed face; or
- the handle rear face.

Every contact must remain 2 mm away from an edge. The shaft, shaft-tip bevels, tip, and
the entire socket-facing handle shoulder are forbidden. `contact_allowed.obj`
and `contact_forbidden.obj` make the partition inspectable, while
`contact_regions.json` is the machine-readable source of truth.

Direct BODex optimization on the full key repeatedly spent fingertips on the
shaft. Each handle proxy changes only the **proposal surface**: it keeps its
corresponding full key's frame, CoM, OBB, and mass proxy, but exposes only the
handle box.
Four digits (thumb/index/middle/ring) are optimized; the little finger is
omitted because the stock five-finger solution repeatedly occupied the shaft.
No proxy result should enter an insertion runtime pool directly. Each proposal
must be checked again against every full key mesh on which it will run for:

1. numerical BODex quality;
2. every declared object contact belonging to an allowed face;
3. whole-hand forbidden-region contact using the actual Inspire link meshes;
4. environment/table clearance in cuRobo;
5. squeeze contact and gravity stability in MuJoCo;
6. attached-key pickup, lift, transfer, and pre-insertion planning;
7. finally, supervised physical validation.

The historical common pick/lift grasp is deliberately identical across gaps.
This avoids confounding gap difficulty with a changing wrist/finger pose, but
it does not make seed 78 insertion-safe. A new common insertion grasp must pass
the complete seven-stage gate above before promotion.

### Why the runtime path is currently `table/0/78`

The three path components mean tabletop scene type, stable-pose scene 0, and
BODex sample ID 78. Scene 0 is the controlled `handle_rear_down` baseline. It
was intentionally the only runtime scene promoted for the first 1.5 mm
bring-up, so the present pool is **not** a general tabletop grasp library.

Geometry/scenes already exist for five protected stable poses: rear-down and
four handle-side-down poses. Tip-down is excluded to avoid damaging the
insertion shaft. Each side-down scene needs its own proposal and validation
pool because the table makes one handle face inaccessible and changes the
collision-free approach. A candidate transform is object-relative, but that
does not make a scene-0 grasp safe or reachable in another resting pose.

After pickup, the rigidly held key must also be transported to a canonical
socket pre-insertion pose. This is an arm-level object reorientation problem
when a collision-free wrist path exists with the grasp fixed. It becomes an
in-hand reorientation or regrasp problem only when the fixed grasp blocks the
socket, violates wrist/joint limits, or is incompatible with the desired key
orientation. Consequently the next grasp-library expansion must score each
stable-pose candidate jointly for pickup and socket pre-insertion
reachability, not merely add more BODex samples.

## Local setup

On the AutoDex robot PC, keep the lab's existing main checkout untouched and
clone the fork branch into a separate directory:

```bash
git clone --branch feat/precision-insertion --single-branch \
  https://github.com/gohyunsu/autodex-code.git \
  ~/autodex-precision-insertion
cd ~/autodex-precision-insertion
```

The checkout path and runtime data path are independent: this code resolves
objects, candidates, calibration, and experiment output below
`~/shared_data`. Restore the NAS handoff's `payload/shared_data/` into that
writable overlay before validating. Do not point active experiment output at a
read-only NAS mount.

The data overlay keeps the ParaDex2 NAS readable while making new precision
assets and experiment output local and writable:

```bash
python scripts/precision_insertion/setup_overlay.py
bash scripts/precision_insertion/setup_asset_env.sh
bash scripts/precision_insertion/setup_bodex_env.sh
```

The full environment is `~/miniconda3/envs/autodex_bodex` (Python 3.10,
PyTorch 2.4.1 CUDA 12.1, cuRobo native extensions, coal, MuJoCo 3.3.7,
OpenCV ArUco, ParaDex, AutoDex, pytest 9.1.1, and the repository-pinned
Ultralytics 8.4.15). CUDA 12.1 is intentional: it is compatible
with the host's RTX 3090 and NVIDIA driver 535/CUDA 12.2 maximum. Verify it:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/verify_bodex_env.py
```

In a container or restricted shell where the host GPU is intentionally not
passed through, use `--allow-no-gpu` for a CPU/package audit. That mode reports
`src.execution.run_pipeline` as skipped because cuRobo creates CUDA tensors at
import time; it does not claim that GPU planning was tested.

The robot host and remote capture PCs retain the normal AutoDex camera
dependencies. The robot host needs the local timestamp-camera SDK, the UTG900
trigger, and `network.json` entries for both; every capture PC needs working
Spinnaker/PySpin for its FLIR cameras.

The mesh renderer is isolated from the planning environment because Open3D
pulls in a large notebook/web visualization dependency set. Install it in a
venv that can read, but cannot modify, `autodex_bodex` packages:

```bash
bash scripts/precision_insertion/setup_visualization_env.sh
```

The renderer needs headless EGL/OpenGL access. A successful 3D planning run
does not imply that EGL is available inside a container or restricted shell.
The original-mesh presentation renderer additionally requires Blender on
`PATH` (tested with Blender 2.93.18); it uses Blender Workbench headlessly and
does not require the planning Conda environment.

## Build and validate geometry

```bash
source ~/.venvs/autodex-assets/bin/activate
python scripts/precision_insertion/build_assets.py
python scripts/precision_insertion/validate_assets.py
python -m unittest tests.test_precision_insertion_assets -v
```

Normal validation returns success when generated geometry is internally
consistent and prints unresolved runtime blockers. The strict form fails until
physical/runtime requirements are complete:

```bash
python scripts/precision_insertion/validate_assets.py --require-runtime
```

## Reproduce the four-gap common grasp search

Run from the repository root after activating `autodex_bodex`:

```bash
python src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file assets/precision_insertion/bodex_handle_proxy_objects_all.txt \
  --obj_root_dir ~/shared_data/object_processing \
  --scene_filter_file assets/precision_insertion/bodex_baseline_scene_filter.json \
  --exp_name precision_insertion_v4_per_key_proxy --seed_num 1000 \
  --grasp_threshold 0.2 --distance_threshold 0.01 \
  -o ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_v4_per_key_proxy
```

For a controlled common-grasp comparison, screen the most demanding 0.3 mm
proxy proposals against each full key's contact policy. If an output already
exists, pass a new explicit `--replace-backup` path; the tool never silently
replaces it.

```bash
for object in precision_key_1p5mm precision_key_1p0mm \
              precision_key_0p5mm precision_key_0p3mm; do
  python scripts/precision_insertion/filter_contact_safe_grasps.py \
    --raw-scene ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_v4_per_key_proxy/precision_key_0p3mm_handle_contact_proxy/table/0 \
    --output-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_v4_common_grasp/$object/table/0 \
    --contact-policy ~/shared_data/object_processing/$object/processed_data/info/contact_regions.json \
    --scene-json ~/shared_data/AutoDex/scene/inspire/precision_key_0p3mm_handle_contact_proxy/table/0.json
done
```

Run collision and simulation against each real key, never the proxy:

```bash
for object in precision_key_1p5mm precision_key_1p0mm \
              precision_key_0p5mm precision_key_0p3mm; do
  python src/grasp_generation/sim_filter/run_sim_filter.py \
    --hand inspire --version precision_insertion_v4_common_grasp \
    --obj "$object" \
    --bodex-root ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_v4_common_grasp \
    --candidate-root ~/shared_data/AutoDex/sim_filter_pass/inspire \
    --obj_root_dir ~/shared_data/object_processing
done
```

Intersect the passing IDs across all four outputs before promotion. In the
recorded seed-123 run, seed 78 is the common candidate. Promote it separately
for each real mesh with `--candidate-id 78`. The command requires an explicit
backup path for an existing pool, writes mesh-hash provenance and a physical
validation gate, and never writes physical success.

```bash
backup_root=~/shared_data/AutoDex/archive/runtime_before_common_grasp_manual
for object in precision_key_1p5mm precision_key_1p0mm \
              precision_key_0p5mm precision_key_0p3mm; do
  python scripts/precision_insertion/promote_sim_validated_grasps.py \
    --screened-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_v4_common_grasp/$object/table/0 \
    --output-scene ~/shared_data/AutoDex/candidates/inspire/v8/$object/table/0 \
    --replace-backup "$backup_root/$object/table/0" \
    --full-object-mesh ~/shared_data/object_processing/$object/processed_data/mesh/simplified.obj \
    --candidate-id 78
done
```

Choose a new `backup_root` for every rerun; the promoter refuses to overwrite
an earlier backup.

Then validate one shared FR3 object pose and write per-candidate evidence:

```bash
python scripts/precision_insertion/validate_franka_grasp_plans.py \
  --candidate-id 78 --x-grid 0.4 --y-grid 0.0 \
  --yaw-grid 3.141592653589793
```

This planner check covers pickup approach and held-object lift only. It does
not plan or certify the socket insertion trajectory.

## Reproduce the hand/key image and planned lift animation

First export a fresh plan with cuRobo. The exporter re-runs full-key/table
collision checks, approach planning, and the 10 cm held-object lift, then
asserts zero discontinuity at the approach/close/lift boundaries:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/export_planned_grasp_lift.py
```

Then render the real FR3 and Inspire visual meshes. The key is fixed in the
world during approach and hand closure; during lift its pose is recomputed as
a rigid attachment to the hand base link. This video still does **not** show
reorientation or insertion, and the caption deliberately identifies it as a
planning preview rather than physical execution.

```bash
PYOPENGL_PLATFORM=egl ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/render_planned_grasp_lift.py \
  ~/shared_data/AutoDex/precision_insertion/visualizations/common_grasp_78_planned_trajectory.npz \
  --output ~/shared_data/AutoDex/precision_insertion/visualizations/common_grasp_78_fr3_approach_lift.mp4
```

### Lift-to-insertion reachability diagnostic

Pick/lift success is insufficient for candidate promotion. Build an
actual-mesh diagnostic that keeps candidate 78's rigid key-to-hand transform,
moves toward the CAD pre-insertion/seated goals, and checks sampled Inspire
surface points against the exact concave socket mesh:

```bash
PYOPENGL_PLATFORM=egl ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/build_insertion_reachability_preview.py

PYOPENGL_PLATFORM=egl ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/render_insertion_reachability_preview.py
```

The first three phases reuse the saved cuRobo approach/close/lift result. The
post-lift motion is numerical endpoint IK plus joint interpolation and is
**not** collision-planned or robot-executable. A red robot in the video means
that sampled hand points have negative signed distance inside the socket. The
JSON beside the NPZ records the exact fixture assumption, endpoint errors, and
colliding hand links.

Inspect the same artifact interactively with actual FR3, Inspire, key, and
socket visual meshes:

```bash
~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/view_insertion_reachability_preview.py \
  --port 8088
```

Open `http://localhost:8088`, scrub the sample slider, and compare the blue
held key with the transparent green seated goal. This viewer is a diagnostic,
not a replacement for an attached-object cuRobo transfer preflight. Current
candidate `table/0/78` is expected to be rejected for insertion even though
its pick and 10 cm lift plan passed.

### Insertion-safe grasp and geometric success preview

The declared-contact screen is insufficient because it sees only BODex's four
object-side contact points. Audit every visual Inspire link against the actual
key mesh:

```bash
PYTHONPATH=scripts/precision_insertion ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/validate_whole_hand_contact_policy.py \
  --candidate-dir ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/precision_key_handle_contact_proxy/table/0/84 \
  --symmetry rear_x \
  --output ~/shared_data/AutoDex/precision_insertion/visualizations/insertion_safe_rear_grasp_policy.json
```

`rear_x` rotates the handle grasp by 180 degrees about the centre of the
45 mm-long handle. The transformed declared contacts remain at least 2 mm from
an edge on the lateral faces, while the palm moves behind the rear face and
away from the shaft/socket. This is a proposal symmetry, not a claim that the
derived grasp has passed BODex or MuJoCo again.

The old fixture-to-fixture preview is not the task setup: the key starts on the
table, not in a staging socket. The handle proxy now exposes all five stable
poses inherited from the full key. Generate proposals for every scene with:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file assets/precision_insertion/bodex_handle_proxy_objects.txt \
  --obj_root_dir ~/shared_data/object_processing \
  --scene_filter_file assets/precision_insertion/bodex_tabletop_all_scene_filter.json \
  --exp_name precision_insertion_tabletop_v1 --seed_num 1000 \
  --grasp_threshold 0.2 --distance_threshold 0.01 \
  -o ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_tabletop_v1
```

Screen each scene against the real 1.5 mm key policy. Existing output must be
moved with an explicit `--replace-backup`; this example assumes a fresh output
root:

```bash
for scene_id in 0 1 2 3 4; do
  ~/miniconda3/envs/autodex_bodex/bin/python \
    scripts/precision_insertion/filter_contact_safe_grasps.py \
    --raw-scene ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_tabletop_v1/precision_key_handle_contact_proxy/table/$scene_id \
    --output-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_tabletop_v1/precision_key_1p5mm/table/$scene_id \
    --contact-policy ~/shared_data/object_processing/precision_key_1p5mm/processed_data/info/contact_regions.json \
    --scene-json ~/shared_data/AutoDex/scene/inspire/precision_key_handle_contact_proxy/table/$scene_id.json
done
```

`--scene-json` is mandatory for every new non-identity scene. BODex saves
`contact_point` in scene/world coordinates even though `wrist_se3.npy` is
object-relative. The filter converts each contact with `T_object_world` before
applying the machine-readable object-frame policy. Omitting this transform was
the cause of the deprecated pre-frame-fix presentation set.

The full key and keyed socket have no usable task-pose symmetry.  Although the
handle alone looks C2-symmetric, the keyed shaft, exact insertion yaw, and
table accessibility break that apparent symmetry.  Proposal generation,
coverage, and runtime perception therefore retain all five exact poses
`000`--`004` independently.  See `assets/precision_insertion/task_symmetry.json`.

Do not use the old deterministic preview choices `0/346`, `2/511`, or `3/27`
as success examples.  They fail the corrected whole-hand contact gate.  In
particular, older previews changed finger joints after BODex; that is not a
native candidate and is no longer permitted.
Before arm IK, screen a scene's candidates with the exact key/socket and full
Inspire hand:

```bash
PYTHONPATH=scripts/precision_insertion ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/screen_rigid_insertion_grasps.py \
  --scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_tabletop_v1/precision_key_1p5mm/table/0 \
  --tabletop-pose ~/shared_data/object_processing/precision_key_1p5mm/processed_data/info/tabletop/000.npy
```

This sampled screen is not continuous planning or physical validation.  A
valid grasp must have both a thumb and a non-thumb finger on permitted handle
surfaces; two links from the same thumb do not constitute opposing contact.
With the current 5,000 scene-000 seeds, 24 candidates pass the sparse BODex
contact/quality filter, only candidate 1289 passes that corrected contact
gate, and it still collides with the socket.  Scene 000 therefore has no honest
rigid insertion candidate yet.

The animation builder rechecks the complete Inspire visual mesh and samples
key/table, hand/table, key/socket, and hand/socket distances at every frame.
After closure, it derives the key pose exclusively from Franka FK and the
immutable candidate `T_key_hand`, so numerical IK residual cannot appear as
in-hand key motion:

```bash
candidate_id=REPLACE_WITH_A_CURRENT_SCREEN_PASS
candidate=~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_tabletop_v1/precision_key_1p5mm/table/4_six_surface_policy_51000/$candidate_id
output=~/shared_data/AutoDex/precision_insertion/presentation_assets/04_planning/pose_004/grasps/grasp_${candidate_id}

PYTHONPATH=scripts/precision_insertion ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/build_tabletop_pose_animation_set.py \
  --pose-id 004 --candidate-dir "$candidate" --output-dir "$output" \
  --terminal-phase verification \
  --collision-samples 12000 --policy-samples-per-link 12000
```

Render only when the resulting JSON status is
`sampled_geometric_preview_passed_not_physical_validation`.

The honest direct-insertion set is data-dependent, not a requested quota. The
corrected six-surface, pose-004, 20 mm audit is currently:

- 51,000 raw BODex proposals;
- 51 candidates pass the numerical quality plus exact-pentagon declared-contact
  screen;
- 10 pass the sampled whole-Inspire contact policy;
- 5 pass the 20 mm insertion-table environment gate and complete sampled
  numerical-IK preview.

The ten contact-policy passes are `40`, `383`, `6910`, `21601`, `21771`,
`26729`, `26765`, `31763`, `36889`, and `46925`. Candidates `40`, `383`,
`6910`, `31763`, and `46925` pass the sampled 20 mm preview. All ten place part
of the hand below the table if the *same grasp* is held to the fully seated CAD
pose. Full seating therefore uses a separate release/retreat/top-down press
mode rather than invalidating the five primary-task candidates. The earlier
20 videos remain only under
`presentation_assets/audit/deprecated_pre_contact_frame_fix/`.

### Unconstrained-contact 10k ablation

The handle-only six-surface rule is a task prior, not a BODex requirement. To
test whether it is over-constraining, generate against the **full 1.5 mm key**
and treat the old contact regions as labels rather than a rejection gate. The
primary target remains the same 20 mm verification depth. This does not mean
"turn off collision": table and socket clearance remain mandatory.

The reproducible seeds and scope are recorded in
`assets/precision_insertion/unconstrained_contact_ablation_20mm_10k.json`.
After generating its two 5,000-proposal batches, stage numerical-quality
candidates with:

```bash
python scripts/precision_insertion/filter_contact_safe_grasps.py \
  --raw-scene ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_unconstrained_20mm_10k/precision_key_1p5mm/table/4 \
  --output-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_unconstrained_20mm_10k/precision_key_1p5mm/table/4 \
  --contact-policy ~/shared_data/object_processing/precision_key_1p5mm/processed_data/info/contact_regions.json \
  --scene-json ~/shared_data/AutoDex/scene/inspire/precision_key_1p5mm/table/4.json \
  --contact-policy-mode report-only
```

Then run the inexpensive all-candidate geometry screen. `disabled` applies
only to the key contact-region prior; the 20 mm hand/socket and table checks
are still active:

```bash
~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/screen_rigid_insertion_grasps.py \
  --scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_unconstrained_20mm_10k/precision_key_1p5mm/table/4 \
  --tabletop-pose ~/shared_data/object_processing/precision_key_1p5mm/processed_data/info/tabletop/004.npy \
  --task-geometry ~/shared_data/AutoDex/precision_insertion/fixtures/unified_socket/task_geometry.json \
  --contact-policy-mode disabled --samples-per-link 100 --quiet \
  --output ~/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/coarse_task_screen_pose_004.json
```

The 100-sample result is a coarse rejection screen. Re-run survivors at high
sample density and then require attached-object cuRobo planning, MuJoCo, and
physical trials before calling any candidate successful. A fair causal
comparison should eventually also run a constrained 10k batch with the same
full-key target, seeds, pose, and thresholds.

For large pools, pass `--candidate-shard-count 8` and a distinct
`--candidate-shard-index 0` through `7` to eight invocations, with a distinct
output path for each. Merge only a complete shard set; the merger rejects
missing, duplicated, or configuration-mismatched shards:

```bash
python scripts/precision_insertion/merge_rigid_insertion_grasp_screens.py \
  ~/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/shards/screen_*.json \
  --output ~/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/coarse_task_screen_pose_004.json
```

For the high-density second pass, add the merged coarse report as
`--candidate-source-report` and raise `--samples-per-link` to 3000. The tool
screens exactly the earlier `passed_candidates`; it fails if any referenced ID
is absent from the scene.

To run downstream cuRobo/MuJoCo on a bounded pilot without mutating the full
quality-stage pool, copy ranked high-density passes into a separate version.
`--include-candidate` is useful for retaining a scientifically relevant
control outside the top-N cutoff:

```bash
python scripts/precision_insertion/stage_screen_passed_candidates.py \
  --source-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_unconstrained_20mm_10k/precision_key_1p5mm/table/4 \
  --screen-report ~/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/highres_task_screen_pose_004.json \
  --output-scene ~/shared_data/AutoDex/sim_staging/inspire/precision_insertion_unconstrained_20mm_10k_top50/precision_key_1p5mm/table/4 \
  --max-candidates 50 --include-candidate 182
```

Run the full simulation filter (not `--sim-only` on a fresh pool, because that
mode intentionally skips the squeeze-contact prefilter):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/sim_filter/run_sim_filter.py \
  --hand inspire \
  --version precision_insertion_unconstrained_20mm_10k_top50 \
  --obj precision_key_1p5mm \
  --bodex-root ~/shared_data/AutoDex/sim_staging/inspire/precision_insertion_unconstrained_20mm_10k_top50 \
  --candidate-root ~/shared_data/AutoDex/sim_filter_pass/inspire \
  --obj_root_dir ~/shared_data/object_processing
```

The recorded 2026-10-07 funnel and its limitations are in
`assets/precision_insertion/unconstrained_contact_ablation_20mm_10k_results.json`.
The top-50-plus-control pilot produced seven MuJoCo gravity-stable grasps, but
five declare contact above the handle front. They are evidence that the old
gate is selective, not evidence that shaft contact is insertion-safe.

Build a presentation preview only from a candidate that carries successful
`sim_eval.json` evidence. `diagnostic` preserves the original contact-policy
report without using it as the acceptance gate; rigid attachment and the
sampled key/socket/hand/table checks remain mandatory:

```bash
candidate_id=104
candidate=~/shared_data/AutoDex/sim_staging/inspire/precision_insertion_unconstrained_20mm_10k_top50/precision_key_1p5mm/table/4/$candidate_id
output=~/shared_data/AutoDex/precision_insertion/presentation_assets/07_unconstrained_ablation/pose_004/grasp_$candidate_id

PYTHONPATH=scripts/precision_insertion ~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/build_tabletop_pose_animation_set.py \
  --pose-id 004 --candidate-dir "$candidate" --output-dir "$output" \
  --terminal-phase verification --contact-policy-mode diagnostic \
  --require-mujoco-success --collision-samples 12000 \
  --policy-samples-per-link 3000
```

Prepare the original 479,710-face FR3/Inspire visuals, then render the clean
16:9 key/socket view. The renderer places no text in the video:

```bash
~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/prepare_blender_actual_mesh_animation.py \
  "$output/tabletop_004_to_insertion_preview.npz" \
  --output "$output/tabletop_004_to_insertion_preview_blender_bundle.npz" \
  --mesh-cache ~/shared_data/AutoDex/precision_insertion/presentation_assets/07_unconstrained_ablation/mesh_cache/blender_original_robot_visuals

blender --background \
  --python scripts/precision_insertion/render_blender_actual_mesh_animation.py -- \
  "$output/tabletop_004_to_insertion_preview_blender_bundle.npz" \
  --output "$output/grasp_${candidate_id}_pose_004_20mm_key_socket.mp4" \
  --width 1280 --height 720 --fps 20 --view key-socket
```

The machine-readable video inventory is
`~/shared_data/AutoDex/precision_insertion/presentation_assets/07_unconstrained_ablation/manifest.json`.

Generate more proposals in bounded batches with non-overlapping
`--seed_offset`, then rebuild both the contact screen and strict task screen.
Never satisfy the requested count of 20 by changing a hand configuration,
duplicating a camera view, or relabelling a rejected candidate.

The preparation step exports all 41 original URDF visual geometries once and
stores only their per-frame transforms in each bundle. The current FR3/Inspire
model contains 479,710 robot visual-mesh faces; the Blender path does not apply
the 7,000-face decimation used by the fast Matplotlib diagnostic renderer.
It is therefore the presentation renderer, not a segmentation visualization.
The blue key begins at the requested stable tabletop pose and the red socket
stays at the fixed task pose.

`--view task` is the default and deliberately gives the key, socket, and
Inspire contact geometry priority over proximal Franka links. It uses a fixed
camera that contains the full key-to-socket workspace, including the 12 cm
lift, so relative motion remains visually comparable across poses. Use
`--view overview` only when the full arm configuration is more important.
Both views omit captions, progress bars, goal ghosts, axes, and inset panels.
For a quick diagnostic without Blender, the older renderer remains available
with `--clean`; its `--robot-faces` budget is a decimated display mesh.

With the corrected contact-frame and insertion-table gates, pose 004 has five
sampled preview passes at 20 mm. There is deliberately no in-hand transition
or hidden regrasp. These previews are still not continuous cuRobo plans,
MuJoCo grasp-stability results, controller executions, or physical successes.

Build the successful-trial reset in drop mode. It releases and observes the
20 mm insertion, regrasp/extracts the key, returns above a reset region, and
opens to drop it. It does not restore the exact original pose:

```bash
~/.venvs/autodex-viz/bin/python \
  scripts/precision_insertion/build_success_reset_preview.py \
  "$output/tabletop_004_to_insertion_preview.npz" \
  --full-trial --reset-mode drop \
  --output "$output/full_trial_drop_reset_preview.npz"
```

`--full-trial` composes pick, rigid transfer, 20 mm insertion, release,
confirmation, reapproach, regrasp, extraction, reset-zone transfer, release,
and drop. Open-hand, regrasp, and drop segments are marked `-1` in collision
arrays until continuous planning/dynamics validate them. The active reset
video is therefore explicitly a composed presentation preview.

The detailed current-vs-proposed pipeline comparison, including cuRobo and
MuJoCo scope, is in `docs/autodex_vs_precision_insertion.md`.

The pose-by-pose full-task evidence inventory and directed `i_j` reorientation
scene/seed inventory are in
`docs/precision_insertion_pose_catalog.md`. Regenerate their two JSON catalogs
after any BODex, screening, simulation, or hardware-evidence update:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/build_pose_task_catalog.py \
  --shared-root ~/shared_data --force
```

The files appear under `~/shared_data/AutoDex/precision_insertion/pose_task_catalog/`.
They inventory evidence; they do not create or approve runtime candidates.

The full presentation set lives under
`~/shared_data/AutoDex/precision_insertion/presentation_assets`. Rebuild its
pose-local inventory after rendering:

```bash
python scripts/precision_insertion/organize_presentation_assets.py
```

Read `presentation_assets/manifest.json`, not filenames alone. The exact
AutoDex asset flow is:

1. `src/grasp_generation/reorient/gen_all.py` builds a scene for each source
   and target stable-pose pair, including both support surfaces and pillars.
2. BODex optimizes one `T_key_hand` that is valid in the pair scene.
3. `filter_contact_safe_grasps.py` transforms world contacts into the canonical
   key frame; `stage_precision_reorient_candidates.py` applies whole-hand and
   opposing-face gates and writes the runtime cell layout.
4. `src/experiment/reset/reorient.py` classifies the perceived source pose,
   selects a coverage target/release height, and runs the complete Franka
   approach, close, lift, reorient, descend, release, retreat, reperception,
   and target-pose verification chain.

The strict two-contact run produced 37 numerical candidates but all contacts
collapsed onto one handle face. A three-contact, 180-degree force-cone
follow-up produced one allowed-face candidate and zero opposition/whole-hand
passes. That strict-policy result remains a blocker for promotion into the
runtime candidate pool.

For a separate diagnostic ablation, candidate 104 from the 10k full-key run is
used as an object-relative pose-004 grasp seed. It already has a MuJoCo gravity
retention pass. The following command runs the existing AutoDex FR3 runtime
full-chain preflight without copying the seed into the runtime pool:

```bash
conda run -n autodex_bodex python \
  scripts/precision_insertion/export_curobo_reorientation_preview.py \
  --candidate-root \
    ~/shared_data/AutoDex/precision_insertion/experiments/reorientation_from_stable_grasp \
  --cells 4_0 --pickup-yaw-deg 0 --max-candidates 1 \
  --output \
    ~/shared_data/AutoDex/precision_insertion/presentation_assets/06_reorientation/stable_grasp_104/reorientation_pose_004_to_000.npz
```

The result validates continuous cuRobo approach, lift, high reorientation,
preplace, 10 cm vertical place, post-release lift, and retract. The rendered
12 cm drop is still composed rather than a MuJoCo/contact-dynamics result, and
no post-drop tabletop classifier or robot execution has validated it. Read
`presentation_assets/06_reorientation/status.json` for the exact split. Runtime
recovery must still fail closed rather than replay stale world-frame joints.

Every active MP4 has a 960×540, 10 fps, palette-optimized GIF derivative next
to it. Rebuild all GIFs and the machine-readable inventory with:

```bash
python scripts/precision_insertion/render_gif_derivatives.py \
  --force --include-audit
```

`presentation_assets/gif_manifest.json` records the MP4/GIF pair, dimensions,
frame rate, duration, byte size, and whether it belongs to deprecated audit
evidence. Omit `--include-audit` when only active slide assets are needed.

For a close hand/key still or turntable, reuse the generic mesh renderer:

```bash
PYOPENGL_PLATFORM=egl ~/.venvs/autodex-viz/bin/python \
  src/visualization/turntable_grasp.py \
  --hand inspire --version v8 --obj precision_key_1p5mm \
  --scene table/0/78 --obj-root ~/shared_data/object_processing \
  --still --width 1280 --height 960 --no-object-texture \
  --output ~/shared_data/AutoDex/precision_insertion/visualizations/common_grasp_78_hand_key.png
```

### VLM checkpoint storyboards with actual assets

Do not use generative images as robot-geometry evidence. Build the four VLM
checkpoint illustrations from the recorded candidate-40 trajectory, the
combined FR3/Inspire URDF visuals, and the exact key/socket meshes:

```bash
conda run -n autodex_bodex python \
  scripts/precision_insertion/build_vlm_checkpoint_states.py

conda run -n autodex_bodex python \
  scripts/precision_insertion/render_vlm_checkpoint_storyboards.py
```

The output is under
`~/shared_data/AutoDex/precision_insertion/presentation_assets/08_vlm_checkpoints/`:

- `checkpoint_01_lift.png`: retained, miss, and slip;
- `checkpoint_02_preinsert.png`: aligned, grossly misaligned, and occluded;
- `checkpoint_03_insertion.png`: 20 mm verification, partial insertion, and
  staged rim jam;
- `checkpoint_04_finish.png`: seated and staged post-press jam states.

The renderer uses all 41 original FR3/Inspire visual geometries and writes no
text into the 1920x1080 storyboards. Read `actual_asset_states.manifest.json`
and `render_manifest.json` before presenting them. Recorded panels preserve
saved kinematics; miss/slip, IK misalignment/jam, and finish panels are
explicit explanatory counterfactuals. They are not cuRobo, collision,
dynamics, controller, or physical-validation evidence.

## Stage profiles and controller assets

`build_assets.py` writes one profile under
`~/shared_data/AutoDex/precision_insertion/stages` for each gap. Each profile
resolves the exact object, scene, candidate pool, FoundPose path, socket
geometry, session-pose output pattern, and camera-calibration root. Controller entries are
fail-closed with `implementation_status: required`:

- 1.5 mm requests open-loop Cartesian insertion;
- 1.0 mm requests the same motion with accuracy measurement;
- 0.5 and 0.3 mm request force/contact XY-yaw search.

Force limits and search increments are deliberately `null`. Those are not CAD
assets and cannot be chosen safely without Franka force-signal validation and
physical commissioning. Filling them with guessed values would turn an asset
preparation step into an unreviewed robot-control change.

## Socket pose-estimation asset

The builder produces two deliberately separate representations of the same
source STL:

- `AutoDex/precision_insertion/fixtures/unified_socket/socket_shared_bore_1p5.obj`
  is the task/planning fixture mesh;
- `object_processing/precision_socket_unified/raw_mesh/precision_socket_unified.obj`
  is the canonical FoundPose input.

Both retain the source STL frame. The generated
`processed_data/info/frame_contract.json` records
`T_socket_raw_mesh = identity`; therefore a FoundPose estimate for
`precision_socket_unified` is `T_world_socket`, not a pose for a recentered or
rotated derivative. `pose_measurement_asset.json` records the transform
equation, required evidence, and the session-scoped output contract.

The generated `static_collision.obj` and `socket_static_exact.urdf` preserve
the keyed bore. They are static-fixture assets only. Do not generate or use a
single convex hull for insertion: it would fill the cavity. The pose object
also intentionally has no BODex scene or candidate pool because the robot must
not grasp the mounted socket.

The remaining learned asset is:

```text
~/shared_data/AutoDex/foundpose_assets/precision_socket_unified/
  object_repre/v1/precision_socket_unified/1/repre.pth
```

It must be onboarded from the exact raw mesh above after MV-GoTrack is restored.
The builder writes `GENERATION_REQUIRED.json` instead of fabricating or copying
a representation. When capturing the socket, the segmentation prompt should
include the whole red fixture and keyed opening; masking only the nearly
symmetric exterior makes yaw underconstrained.

The previous feature-branch version of `run_pipeline.py` measured the socket,
but those edits have been removed to preserve original AutoDex behavior. The
independent demo will first measure ChArUco, then take multiple multi-view
socket estimates, transform each `T_world_socket` into
`T_robot_socket = inv(C2R) @ T_world_socket`, select an observed SE(3) medoid,
and check its repeatability. The earlier 2 mm / 2 degree thresholds were
bring-up gates, **not** evidence of sub-millimetre absolute accuracy.

The future session record will freeze that measured pose and insert the exact
concave `static_collision.obj` into each planning scene. If the fixture moves,
abort and start a new session. There is currently **no supported command**
to run socket-aware precision insertion on the robot; do not pass the removed
`--socket-preflight` or `--socket-object` options to the original runner.

## AutoDex camera profile

Precision insertion uses the existing AutoDex acquisition path, not the
ZeroDex cameras. The canonical contract is
`assets/precision_insertion/autodex_camera_profile.json`:

- capture PCs `capture1`, `capture2`, `capture3`, `capture5`, and `capture6`;
- camera serials resolved from the robot PC's active
  `paradex/system/current/pc.json`;
- remote FLIR video armed in hardware-sync mode;
- local UTG900 trigger and timestamp camera configured by `network.json`;
- a `~/shared_data/cam_param/<session>` whose intrinsics and extrinsics cover
  every active serial;
- the newest valid hand-eye session proven to belong to Franka.

Audit these read-only inputs before starting camera daemons or connecting the
robot. Pass the exact calibration selected on the AutoDex robot PC:

```bash
python scripts/precision_insertion/verify_autodex_camera_profile.py \
  --calib-dir <AUTODEX_CALIB_DIR>

python scripts/precision_insertion/verify_autodex_camera_profile.py \
  --calib-dir <AUTODEX_CALIB_DIR> --require-runtime \
  --output ~/shared_data/AutoDex/precision_insertion/autodex_camera_runtime_audit.json
```

Do not rely on lexicographic "latest" without the audit. On this development
host, the current ParaDex mapping and newest NAS calibration disagree on two
serials, and the deployed network snapshot lacks the nested timestamp/trigger
entries expected by upstream AutoDex. That observation does not define the
robot PC; it is why the robot PC must generate its own PASS audit. The runtime
now rejects a calibration missing any active camera before hardware startup.

## Assets that cannot be fabricated

- **FoundPose `repre.pth` for the four keys and socket pose object:** the checked-in onboarding wrapper requires
  `autodex/perception/thirdparty/MV-GoTrack/scripts/onboard_custom_mesh_for_foundpose.py`.
  That directory is absent, and the historical `gunhee1113/MV-GoTrack` GitHub
  repository is unavailable even with the authenticated lab GitHub account.
  Copying a representation from another object would violate the mesh frame.
- **absolute `T_robot_socket` accuracy:** CAD cannot determine the bolted
  fixture's pose in `fr3_link0`. The new startup measurement records it per
  session, but physical target/robot metrology must still establish its bias
  and uncertainty before 1.0/0.5/0.3 mm claims.
- **physical grasp trust:** simulation and planner success cannot establish
  real Inspire contact, print tolerance, or cable/fixture clearance.
- **camera/hand-eye runtime audit:** it depends on the AutoDex robot PC's
  active ParaDex profile, installed trigger/timestamp devices, selected camera
  calibration, and Franka hand-eye sessions. It cannot be certified from this
  development checkout.

The ParaDex2 NAS confirms the intended FoundPose asset contract but does not
contain a representation for any `precision_key_*` object. For example,
`/mnt/paradex2/AutoDex/foundpose_assets/pringles_untextured_backup_20260902_active/summary.json`
records the same onboarding settings used by the local wrapper: millimetre
render scale 1000, 57 minimum viewpoints x 14 in-plane rotations = 798
templates, `dinov2_vits14-reg`, PCA-256, and 2048 clusters. Its 720 MB
`repre.pth` is mesh-specific and cannot be renamed or reused for the key.
The recovery path is to obtain the missing MV-GoTrack checkout from a capture
PC/backup or restore access to its private repository, then run the wrapper on
all four meshes with the selected AutoDex camera intrinsics. The old setup script asks
for CUDA 12.8; do not run it unchanged on this RTX 3090 host with driver 535.

## Git policy

Commit canonical STL, builders, validators, configs, manifests, and usage
documentation. Do not commit learned weights, raw BODex pools, calibration
recordings, candidate state, or experiment videos. Keep geometry/schema,
environment/runtime support, and experiment assets in separate commits so a
hardware change can be reverted without rewriting the CAD history.

## NAS handoff

Create a non-overwriting, checksum-addressed transport bundle after committing
the code whose hash should be recorded:

```bash
python scripts/precision_insertion/prepare_handoff.py \
  --output-root /mnt/paradex2/hyunsu \
  --bundle-name autodex_precision_insertion_handoff_YYYYMMDD_<git-short-sha>

cd /mnt/paradex2/hyunsu/autodex_precision_insertion_handoff_YYYYMMDD_<git-short-sha>
sha256sum -c SHA256SUMS
```

The exporter refuses to merge with an existing bundle. `payload/shared_data`
preserves runtime paths; `source`, `fabrication`, `reproducibility`, and
`handoff_docs` keep canonical inputs, printable models, search evidence, and
unresolved physical work separate.
