# Precision insertion demo

This directory is reserved for an independent precision-insertion demo. Its
runner must not call `src.execution.run_auto.main()` or require edits to the
existing AutoDex execution files (`run_auto.py`, `run_pipeline.py`, or
`scene_cfg.py`). AutoDex/ParaDex hardware and perception APIs may be reused
through explicit adapters, while insertion-specific orchestration lives here.

There is no robot-executable insertion runner in this directory yet. The
existing scenario catalog, VLM observer, and retry policy are offline evidence
and decision helpers, not proof of a continuous insertion plan or hardware
readiness. Do not interpret a grasp/lift simulation pass as an insertion pass.
Offline grasp eligibility now means v8 grasp stability **plus** a centered,
axis-aligned 20 mm key-in-socket endpoint at which the full Inspire hand does
not collide with the socket. It deliberately does not screen the Franka arm
or a fixed transfer trajectory. Each observed trial still needs online
collision-checked planning from its live pose and guarded insertion contact.
The original AutoDex source files are kept at the `main` baseline; the
previous feature-branch changes are preserved at
`archive/precision-pre-isolation`. `PLAN.md` maps existing APIs to the demo
modules that will reuse them, so the runner will not duplicate camera,
FoundPose, Franka pickup, or cuRobo primitives.

The first independent helpers are in `precision_insertion/`:

- `geometry.py` validates SE(3) measurements and freezes one **observed**
  socket-pose medoid. For a C∞ round socket it ignores only unobservable axial
  yaw, not axis tilt or a reversed open rim. Repeatability is not a guarantee
  of absolute camera or hand-eye accuracy.
- `symmetry.py` reads the v8 `object_processing/<object>/processed_data/info/`
  symmetry and tabletop poses. The D∞ cylindrical key may exchange identical
  ends; the C∞ socket may not.
- `world.py` adds the frozen socket mesh to a copy of the cuRobo scene. It
  checks the pose and mesh path and does not mutate the source scene. It does
  not by itself authorize key/socket contact or a robot insertion.
- `config.py` resolves explicit square/cylinder key and socket IDs and the
  20 mm verification target; `assets.py` performs a read-only v8 input audit.
- `xy_voting.py` accepts already geometry-screened **absolute socket-frame XY
  offsets** and synchronized per-view VLM choices. It projects candidate
  anchors for overlays and returns `propose`, `abstain`, `stop`, or
  `no_correction` with camera provenance. It rejects ties, single-view
  decisions, stale/asynchronous frames, a possible slip, and step/total
  budget violations; it never averages candidate positions. A `propose`
  result is not motion authorization or evidence of insertion success.

The XY voting module is a pure offline contract, not a VLM API integration.
Its caller must first validate the candidate offsets against the exact
key/socket/hand geometry, match image intrinsics to the undistorted/resized
AutoDex camera frames, and verify the current grasp and calibration. After a
choice, the live planner must still check the Franka/attached-key path and
the guarded insertion controller must independently enforce contact limits.
The older `autodex.tasks.precision_insertion.decide_retry` proposes a
continuous pose-residual correction; it is not the bounded candidate-ID vote
policy and is not imported as this demo's control loop.

Run the read-only asset audit from the repository root, for example:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py audit \
  --shared-root /home/hyunsu/shared_data --mode square --gap-mm 1.5
```

The audit exits `0` only when its file inputs are present, or `2` when any
are missing. Its `robot_ready` field is always false: file availability does
not establish live calibration, online path safety, or guarded contact. For
cylindrical assets, use `--mode cylinder --gap-mm 20` (gap is the radial gap
value used by that asset family). This command does not contact cameras or
the robot and does not write files.

Run the current offline tests from the repository root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests
```

The future runner must measure ChArUco first, then the socket, with the socket
already rigidly fixed. It must explicitly select `--shared-root` and use the
matching v8 `object_processing` assets and Inspire candidates; see `PLAN.md`
for the full execution and evidence gates. Until those gates are implemented,
there is intentionally no robot-mode command to run here.

The path component `precision-insertion` is a directory name, not an importable
Python package name. If helper modules are added, use an importable package
name such as `precision_insertion` inside this directory.
