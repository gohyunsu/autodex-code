# Precision-insertion asset workflow

This directory builds the assets for the staged unified-socket experiment:

| gap | experiment stage |
|---:|---|
| 1.5 mm | full pipeline bring-up with one fixed socket pose and one grasp |
| 1.0 mm | pose-accuracy measurement |
| 0.5 mm | contact-search introduction |
| 0.3 mm | final precision condition |

The current runtime target is only `precision_key_1p5mm`. The other three
objects have complete geometry, frame, scene, and contact-policy assets, but
do not yet have grasps or perception representations. This is intentional:
the staged experiment must not silently treat an unvalidated transfer as a
trusted grasp.

## Current status

Generated under `~/shared_data`:

- four metric key objects under `object_processing/precision_key_*`;
- a proposal-only handle proxy under
  `object_processing/precision_key_handle_contact_proxy`;
- the unified socket mesh, CAD-relative insertion transforms, and fixture-pose
  template under `AutoDex/precision_insertion/fixtures/unified_socket`;
- a frozen four-camera ZeroDex calibration snapshot under
  `AutoDex/precision_insertion/calibration`;
- one 1.5 mm runtime grasp, candidate `table/0/84`, under
  `AutoDex/candidates/inspire/v8/precision_key_1p5mm`.

Candidate 84 passed the declared-contact rule, full-key cuRobo collision,
MuJoCo squeeze/gravity stability, and a hardware-free FR3 approach plus 10 cm
vertical-lift plan. It is **not physically trusted**. The runtime pool retains
`PHYSICAL_VALIDATION_REQUIRED.json`, and strict validation remains blocked.

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
shaft. The handle proxy changes only the **proposal surface**: it keeps the
full key's frame, CoM, OBB, and mass proxy, but exposes only the handle box.
Four digits (thumb/index/middle/ring) are optimized; the little finger is
omitted because the stock five-finger solution repeatedly occupied the shaft.
No proxy result enters the runtime pool directly. Each proposal is checked
again against the full 1.5 mm key for:

1. numerical BODex quality;
2. every declared object contact belonging to an allowed face;
3. full-hand and table collision in cuRobo;
4. squeeze contact and gravity stability in MuJoCo;
5. FR3+Inspire IK, approach trajectory, and vertical lift planning;
6. finally, supervised physical validation.

Candidate 84's four declared contacts are all on lateral handle faces. None is
on the shaft, tip, bevel, rear-edge margin, or socket-facing shoulder.

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

## Reproduce the 1.5 mm grasp search

Run from the repository root after activating `autodex_bodex`:

```bash
python src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file assets/precision_insertion/bodex_handle_proxy_objects.txt \
  --obj_root_dir ~/shared_data/object_processing \
  --scene_filter_file assets/precision_insertion/bodex_baseline_scene_filter.json \
  --exp_name precision_insertion_v3_proxy --seed_num 1000 \
  --grasp_threshold 0.2 --distance_threshold 0.01 \
  -o ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy
```

Curate declared contacts into a non-runtime staging pool. If the output exists,
pass a new explicit backup path; the tool never silently replaces it.

```bash
python scripts/precision_insertion/filter_contact_safe_grasps.py \
  --raw-scene ~/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/precision_key_handle_contact_proxy/table/0 \
  --output-scene ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_v3_proxy/precision_key_1p5mm/table/0 \
  --contact-policy ~/shared_data/object_processing/precision_key_1p5mm/processed_data/info/contact_regions.json
```

Then run the full-key collision and simulation filter against the real key, not
the proxy:

```bash
python src/grasp_generation/sim_filter/run_sim_filter.py \
  --hand inspire --version precision_insertion_v3_proxy \
  --obj precision_key_1p5mm \
  --bodex-root ~/shared_data/AutoDex/contact_screen_staging/inspire/precision_insertion_v3_proxy \
  --candidate-root ~/shared_data/AutoDex/sim_filter_pass/inspire \
  --obj_root_dir ~/shared_data/object_processing
```

Only after those results exist should `promote_sim_validated_grasps.py` copy
passing candidates into the v8 runtime pool. It writes simulation provenance
and a physical-validation gate; it never writes physical success.

The FR3 planner-only check used for candidate 84 is:

```bash
python src/execution/plan_test.py \
  --obj precision_key_1p5mm --version v8 \
  --hand fr3_inspire --candidate-hand inspire \
  --pose_idx 000 --x 0.4 --yaw 0
```

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

- **FoundPose `repre.pth`:** the checked-in onboarding wrapper requires
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

## Git policy

Commit canonical STL, builders, validators, configs, manifests, and usage
documentation. Do not commit learned weights, raw BODex pools, calibration
recordings, candidate state, or experiment videos. Keep geometry/schema,
environment/runtime support, and experiment assets in separate commits so a
hardware change can be reverted without rewriting the CAD history.
