# Precision-insertion asset workflow

This directory builds the assets for the staged unified-socket experiment:

| gap | experiment stage |
|---:|---|
| 1.5 mm | full pipeline bring-up with one fixed socket pose and one grasp |
| 1.0 mm | pose-accuracy measurement |
| 0.5 mm | contact-search introduction |
| 0.3 mm | final precision condition |

All four conditions now have geometry, frames, scenes, contact policies,
gap-specific proposal proxies, a common simulation-validated grasp, FR3 plan
evidence, and fail-closed stage profiles. None is physically trusted, and none
has a FoundPose representation yet. The staged experiment must not silently
treat simulation, a controller specification, or an unvalidated transfer as
physical success.

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
- a frozen four-camera ZeroDex calibration snapshot under
  `AutoDex/precision_insertion/calibration`;
- the same runtime grasp, candidate `table/0/78`, under each of the four
  `AutoDex/candidates/inspire/v8/precision_key_*` pools.

Candidate 78 was proposed with the 0.3 mm key's handle-only proxy, then passed
the declared-contact rule, cuRobo full-key collision, and MuJoCo
squeeze/gravity stability independently on all four full meshes. The four
declared contacts are lateral handle contacts; none is on the shaft, bevel,
tip, rear-edge margin, or socket-facing shoulder. The same candidate passed a
hardware-free FR3 approach plus 10 cm vertical-lift plan for all four keys at
the same test pose `(x=0.4 m, y=0, yaw=pi)`.

It is **not physically trusted**. Every runtime pool retains
`PHYSICAL_VALIDATION_REQUIRED.json`; no physical grasp, lift, or insertion has
been claimed. The former 1.5 mm candidate 84 was moved, not deleted, to
`~/shared_data/AutoDex/archive/runtime_before_common_grasp_20261005`.

| gap | geometry | common grasp full-key sim | FR3 plan | physical | controller |
|---:|---|---|---|---|---|
| 1.5 mm | ready | seed 78 passed | passed | required | spec only; open-loop implementation required |
| 1.0 mm | ready | seed 78 passed | passed | required | spec only; accuracy instrumentation required |
| 0.5 mm | ready | seed 78 passed | passed | required | force/contact XY-yaw search required |
| 0.3 mm | ready | seed 78 passed | passed | required | final search controller required |

## Why the geometry is generated this way

The STL inputs are binary millimetre meshes. `build_assets.py` parses them
without relying on an implicit unit convention, scales vertices by `1e-3`,
checks watertightness/orientation, and writes the v8 `object_processing`
contract:

- `raw_mesh/<object>.obj` is the canonical perception mesh;
- `processed_data/mesh/simplified.obj` is the planning/simulation mesh;
- `processed_data/urdf/coacd.urdf` references a conservative convex piece;
- `simplified.json` records CoM, OBB, mass proxy, and scale;
- `tabletop/*.npy` stores five controlled stable poses (tip-down is excluded);
- `AutoDex/scene/inspire/<object>/table/*.json` binds those poses to BODex.

The key frame is fixed across all gaps: `+z` runs from the handle rear toward
the insertion tip, the rear plane is `z=0`, the socket-facing shoulder is
`z=45 mm`, and the tip is `z=85.5 mm`. The socket source frame has its entry
at `z=58.5 mm`. Therefore the seated transform is `Rx(pi)` with translation
`z=103.5 mm`: the shoulder lands at the entry plane and the tip lands at
`z=18 mm`. These are CAD-relative transforms only; they do not determine the
physical socket pose in the Franka base frame.

## Contact rule and grasp proposal

The hand may contact only:

- the four lateral faces of the 39 x 33 x 45 mm handle; or
- the handle rear face.

Every contact must remain 2 mm away from an edge. The shaft, bevel, tip, and
the entire socket-facing handle shoulder are forbidden. `contact_allowed.obj`
and `contact_forbidden.obj` make the partition inspectable, while
`contact_regions.json` is the machine-readable source of truth.

Direct BODex optimization on the full key repeatedly spent fingertips on the
shaft. Each handle proxy changes only the **proposal surface**: it keeps its
corresponding full key's frame, CoM, OBB, and mass proxy, but exposes only the
handle box.
Four digits (thumb/index/middle/ring) are optimized; the little finger is
omitted because the stock five-finger solution repeatedly occupied the shaft.
No proxy result enters the runtime pool directly. Each proposal is checked
again against every full key mesh on which it will run for:

1. numerical BODex quality;
2. every declared object contact belonging to an allowed face;
3. full-hand and table collision in cuRobo;
4. squeeze contact and gravity stability in MuJoCo;
5. FR3+Inspire IK, approach trajectory, and vertical lift planning;
6. finally, supervised physical validation.

The current common grasp is deliberately identical across gaps. This avoids
confounding gap difficulty with a changing wrist/finger pose. A gap-specific
proxy search was also run: its 0.5 mm random pool happened to yield no MuJoCo
pass, while the 0.3 mm pool yielded two. Cross-validation established that
seed 78 from the latter passes every full mesh; this is sampling behavior, not
evidence that 0.3 mm insertion is easier than 0.5 mm insertion.

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

The data overlay keeps the ParaDex2 NAS readable while making new precision
assets and experiment output local and writable:

```bash
python scripts/precision_insertion/setup_overlay.py
bash scripts/precision_insertion/setup_asset_env.sh
bash scripts/precision_insertion/setup_bodex_env.sh
```

The full environment is `~/miniconda3/envs/autodex_bodex` (Python 3.10,
PyTorch 2.4.1 CUDA 12.1, cuRobo native extensions, coal, MuJoCo 3.3.7,
OpenCV ArUco, ParaDex, and AutoDex). CUDA 12.1 is intentional: it is compatible
with the host's RTX 3090 and NVIDIA driver 535/CUDA 12.2 maximum. Verify it:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/verify_bodex_env.py
```

The robot host does not need local PySpin in ZeroDex free-run mode. Each remote
capture PC still needs its own working Spinnaker/PySpin installation because
the ParaDex camera daemon opens the FLIR cameras there.

The mesh renderer is isolated from the planning environment because Open3D
pulls in a large notebook/web visualization dependency set. Install it in a
venv that can read, but cannot modify, `autodex_bodex` packages:

```bash
bash scripts/precision_insertion/setup_visualization_env.sh
```

The renderer needs headless EGL/OpenGL access. A successful 3D planning run
does not imply that EGL is available inside a container or restricted shell.

## Build and validate geometry

```bash
source ~/.venvs/autodex-assets/bin/activate
python scripts/precision_insertion/build_assets.py
python scripts/precision_insertion/pin_zerodex_calibration.py
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
    --contact-policy ~/shared_data/object_processing/$object/processed_data/info/contact_regions.json
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

For a close hand/key still or turntable, reuse the generic mesh renderer:

```bash
PYOPENGL_PLATFORM=egl ~/.venvs/autodex-viz/bin/python \
  src/visualization/turntable_grasp.py \
  --hand inspire --version v8 --obj precision_key_1p5mm \
  --scene table/0/78 --obj-root ~/shared_data/object_processing \
  --still --width 1280 --height 960 --no-object-texture \
  --output ~/shared_data/AutoDex/precision_insertion/visualizations/common_grasp_78_hand_key.png
```

## Stage profiles and controller assets

`build_assets.py` writes one profile under
`~/shared_data/AutoDex/precision_insertion/stages` for each gap. Each profile
resolves the exact object, scene, candidate pool, FoundPose path, socket
geometry, fixture pose, and camera-calibration root. Controller entries are
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
equation, required evidence, and the eventual `fixture_pose.json` output.

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

Once the representation and ZeroDex calibration are available, repeated
multi-view estimates supply `T_world_socket`. The existing AutoDex convention
then applies the calibrated world-to-`fr3_link0` transform and writes the
result as `T_robot_socket` in `fixture_pose.json`. Keep the raw per-view poses,
masks, calibration snapshot, and repeatability residuals with that file; do
not mark it calibrated from one uninspected estimate.

## ZeroDex camera profile

The pinned profile is
`assets/precision_insertion/zerodex_camera_profile.json`. It selects serials
`25305462`, `25322639`, `25322642`, and `26053248`, matching the complete
four-camera calibration snapshot. Audit it before any daemon or robot command:

```bash
python scripts/precision_insertion/verify_zerodex_camera_profile.py
python scripts/precision_insertion/verify_zerodex_camera_profile.py --require-runtime
```

The current active ParaDex2 profile does not describe the ZeroDex layout, so
the strict audit fails. A lab-confirmed ParaDex `system/current` must map the
first three serials to `capture4`, `26053248` to `capturenew`, and supply the
real IPs. Do not copy the stale ParaDex2 IPs or infer `capturenew`'s address.

Once the audit passes, the camera arguments are:

```text
--pc_list capture4 capturenew
--calib_dir ~/shared_data/AutoDex/precision_insertion/calibration/zerodex_4cam_20261002_141639_franka_20261002_145508/cam_param
--camera-sync free_run
```

AutoDex must be the sole camera-daemon owner during this baseline. Stop
ZeroDex `run/stream_owner.py` first, then launch the ParaDex daemons with
exactly the four profile serials. Running both owners causes a lock takeover
and interrupts the other pipeline.

## Assets that cannot be fabricated

- **FoundPose `repre.pth` for the four keys and socket pose object:** the checked-in onboarding wrapper requires
  `autodex/perception/thirdparty/MV-GoTrack/scripts/onboard_custom_mesh_for_foundpose.py`.
  That directory is absent, and the historical `gunhee1113/MV-GoTrack` GitHub
  repository is unavailable even with the authenticated lab GitHub account.
  Copying a representation from another object would violate the mesh frame.
- **`T_robot_socket`:** this is the measured pose of the bolted fixture in
  `fr3_link0`; CAD cannot determine it. Fill `fixture_pose.json` only after a
  physical calibration and retain the measurement method/residual.
- **physical grasp trust:** simulation and planner success cannot establish
  real Inspire contact, print tolerance, or cable/fixture clearance.
- **camera/hand-eye confirmation:** the frozen files are internally complete,
  but someone must confirm the physical serial placement and rerun or approve
  hand-eye calibration after the rig is fixed.

The ParaDex2 NAS confirms the intended FoundPose asset contract but does not
contain a representation for any `precision_key_*` object. For example,
`/mnt/paradex2/AutoDex/foundpose_assets/pringles_untextured_backup_20260902_active/summary.json`
records the same onboarding settings used by the local wrapper: millimetre
render scale 1000, 57 minimum viewpoints x 14 in-plane rotations = 798
templates, `dinov2_vits14-reg`, PCA-256, and 2048 clusters. Its 720 MB
`repre.pth` is mesh-specific and cannot be renamed or reused for the key.
The recovery path is to obtain the missing MV-GoTrack checkout from a capture
PC/backup or restore access to its private repository, then run the wrapper on
all four meshes with the pinned ZeroDex intrinsics. The old setup script asks
for CUDA 12.8; do not run it unchanged on this RTX 3090 host with driver 535.

## Git policy

Commit canonical STL, builders, validators, configs, manifests, and usage
documentation. Do not commit learned weights, raw BODex pools, calibration
recordings, candidate state, or experiment videos. Keep geometry/schema,
environment/runtime support, and experiment assets in separate commits so a
hardware change can be reverted without rewriting the CAD history.
