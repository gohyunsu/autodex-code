# Precision-insertion asset workflow

## What is generated

For each key, `build_assets.py` converts the millimetre STL to the v8
`object_processing` contract:

- metric raw/planning meshes;
- a conservative convex collision hull and simple URDF;
- OBB, centre-of-mass, non-symmetry, and controlled tabletop poses;
- separate contact-allowed and contact-forbidden meshes;
- Inspire table-scene JSONs;
- explicit markers for FoundPose and grasp assets that still need GPU or robot
  validation.

It also generates the socket collision mesh and the exact CAD-relative seated
and pre-insertion transforms. The printed key must be flipped by `Rx(pi)`;
this mirrors its chamfer into the socket bore orientation. Fully seated, the
45 mm handle shoulder lies on the 58.5 mm socket entry plane and the 85.5 mm
tip ends at socket z=18 mm.

The builder does **not** invent the robot-to-socket fixture pose, a learned
FoundPose descriptor, or an untested grasp.

## One-time local setup

```bash
git clone https://github.com/snuvclab/paradex.git ~/paradex
python scripts/precision_insertion/setup_overlay.py
bash scripts/precision_insertion/setup_asset_env.sh
```

The overlay keeps NAS data readable while placing new objects and all experiment
outputs on local writable storage. It must be `~/shared_data`, because AutoDex
and ParaDex resolve that path at import time.

## Build and validate geometry

```bash
source ~/.venvs/autodex-assets/bin/activate
python scripts/precision_insertion/build_assets.py
python scripts/precision_insertion/pin_zerodex_calibration.py
python scripts/precision_insertion/validate_assets.py
python -m unittest tests.test_precision_insertion_assets -v
```

Strict validation intentionally fails until the learned/physical assets exist:

```bash
python scripts/precision_insertion/validate_assets.py --require-runtime
```

## Runtime assets that require separate stages

1. FoundPose onboarding: generate
   `AutoDex/foundpose_assets/<object>/object_repre/v1/<object>/1/repre.pth`
   with the ZeroDex camera intrinsics selected for the experiment.
2. Inspire grasp: run BODex on the table scene, then reject every candidate
   unless all declared contacts lie on `contact_allowed.obj` and every hand link
   clears `contact_forbidden.obj`. Physically validate the survivor before using
   it as the open-loop baseline.
3. Fixture calibration: measure `T_robot_socket` after bolting the socket down,
   copy `fixture_pose.template.json` to `fixture_pose.json`, fill the 4x4 matrix,
   and set `calibrated: true`.
4. ZeroDex cameras: pin one internally complete `cam_param_jisoo` snapshot and
   its matching Franka `handeye_calibration_jisoo` snapshot. Do not select
   calibration by lexicographic “latest” at runtime.

## Git policy

Commit canonical STL, builders, validators, manifests, and documentation.
Do not commit learned weights, candidate result state, calibration recordings,
or experiment videos. Make separate commits for geometry/schema, environment,
and runtime integration so each stage is reviewable and revertible.
