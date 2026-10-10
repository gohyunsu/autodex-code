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
- `calibration.py` combines already-captured synchronized board images and
  multiple multi-view socket FoundPose observations. It calls AutoDex's
  `measure_tabletop_from_images` first, converts every socket observation with
  the session C2R transform, rejects stale ordering, unsynchronized captures,
  uncalibrated cameras, and non-repeatable socket poses, then adds the exact
  socket collision mesh to a **copy** of the base scene. The caller explicitly
  supplies time/translation/angle limits; no bring-up threshold is silently
  treated as precision accuracy. `write_session_calibration` saves all pose
  observations and the mesh hash to a new JSON file without overwriting a
  previous session. The capture images/masks themselves must be retained
  separately under their capture IDs.
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
- `endpoint.py` evaluates one fixed-grasp candidate using the full metric CAD
  key, exact socket collision mesh, and every Inspire visual link at the
  centered 20 mm insertion pose. It uses Coal triangle-mesh collision and
  minimum surface distance, excludes the Franka arm and all trajectories,
  and records source hashes. This is grasp-level **endpoint** evidence only;
  simulated grasp stability is still a separate v8/MuJoCo gate.
- `candidates.py` scans the selected shared root's Inspire v8 candidate tree,
  reads matching scene `meta.pose_idx` and tabletop assets, requires full-key
  simulation evidence, and applies `endpoint.py` to surviving grasps. The
  catalogue distinguishes a complete finite scan from missing or truncated
  input. Per-trial selection matches the observed tabletop pose, excludes
  session-attempted grasps, optionally ranks by v8 coverage, and rejects
  changed source files. AutoDex's `load_candidate` is reused with an explicit
  root and whitelist to form a planner `candidate_override`; this does not
  extend AutoDex's lift-only planner to transfer or insertion.
- `outcome.py` defines a VLM-led, sensor-vetoed tri-state task label. A
  multi-view `normal_appearance` assessment is required for `true`, together with
  independently cross-checked **key** depth, commissioned alignment limits,
  held-grasp evidence, and no safety abort. Conflicting or occluded evidence
  is `null`; normal force or a commanded wrist stroke is not success proof.
  Older recorded `normal_20mm` assessments remain accepted as an alias, but
  the new VLM prompt deliberately does not ask the model to infer millimetres.
- `observer.py` defines the event-driven lift, per-camera XY, and insertion
  visual prompts; it reuses ZeroDex's Gemini helper or an already-loaded
  ZeroDex local `BaseVLM` through optional adapters. Inputs have explicit
  camera/phase/time order. Closed-set JSON parsing falls back to
  `unobservable` or `abstain` on malformed responses. Per-view XY calls are
  separate; their output still goes through `xy_voting.py`, not directly to
  a motion controller. This module cannot manufacture synchronized images,
  CAD overlays, depth estimates, or physical ground-truth labels.

The XY voting resolver remains a read-only contract even when an optional
ZeroDex-backed observer supplies votes. Its caller must first validate the
candidate offsets against the exact
key/socket/hand geometry, match image intrinsics to the undistorted/resized
AutoDex camera frames, and verify the current grasp and calibration. After a
choice, the live planner must still check the Franka/attached-key path and
the guarded insertion controller must independently enforce contact limits.
The older `autodex.tasks.precision_insertion.decide_retry` proposes a
continuous pose-residual correction; it is not the bounded candidate-ID vote
policy and is not imported as this demo's control loop.

For a **saved-image, read-only** observer replay, place this demo directory
and a compatible ZeroDex checkout on `PYTHONPATH`, then pass already-loaded
PIL images with explicit camera IDs and capture times. For example:

```python
from main.vlm_base import BaseVLM
from precision_insertion.observer import (
    LabeledFrame, ZeroDexLocalBackend, observe_lift,
)

backend = ZeroDexLocalBackend(BaseVLM(model_id="Qwen/Qwen3-VL-2B-Instruct"))
result = observe_lift(backend, [
    LabeledFrame("front", "before_grasp", 1.0, before_pil),
    LabeledFrame("front", "after_lift", 2.0, after_pil),
])
print(result.to_record())  # review only; not a robot command
```

`before_pil` and `after_pil` must be supplied from the same saved trial; the
timestamps above are placeholders. The optional Gemini adapter accepts an
already-configured `google.genai.Client` and a model ID instead. Both
adapters require the ZeroDex package importable at runtime. Neither adapter
chooses cameras, creates image crops/overlays, measures key depth, or executes
Franka commands. A future capture adapter must supply those inputs and log
the raw images alongside every VLM response.

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

Screen one grasp at the nominal 20 mm endpoint (read-only by default):

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py screen-endpoint \
  --shared-root /home/hyunsu/shared_data --mode square --gap-mm 1.5 \
  --candidate-dir /home/hyunsu/shared_data/AutoDex/candidates/inspire/v8/precision_key_1p5mm/table/0/78 \
  --min-hand-clearance-mm 0.2
```

The clearance above is an **example argument, not an approved physical
threshold**; choose it from measured asset, calibration, and grasp errors.
Exit `0` means the CAD endpoint passed; exit `2` means it failed. Add
`--output /path/to/new_report.json` to write a new report exclusively (an
existing path is not overwritten). The current v8 candidate `table/0/78`
fails: its key fits the square socket nominally with about 1.5 mm CAD gap,
but three Inspire visual links intersect the socket. This screen does not
validate the path to that endpoint, contact dynamics, or hardware readiness.

Screen the **whole currently installed v8 pool** into a new, separate
catalogue (the example clearance remains uncommissioned):

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root /home/hyunsu/shared_data --mode square --gap-mm 1.5 \
  --min-hand-clearance-mm 0.2 \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_20261010.json

~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py select-catalog \
  --catalog /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_20261010.json \
  --pose-stem 000
```

`screen-catalog` writes exclusively; a rerun needs a new output name.
`--max-candidates N` is a pilot scan and always marked incomplete. The local
1.5 mm scan currently finds one candidate, `table/0/78`: MuJoCo grasp evidence
passes and nominal key/socket CAD fit has about 1.5 mm gap, but the Inspire
base, index, and middle links intersect the socket at 20 mm. Thus **0 of this
finite 1-candidate pool** are endpoint eligible; this does not prove other
grasps or tabletop poses impossible. `select-catalog` exits 2 for no eligible
grasp or an incomplete/stale catalogue. It does not execute reorientation.
The example catalogue is in local `shared_data`, **not** the read-only
`/mnt/paradex2` NAS mount.
The same scan across all four square gaps yields one `table/0/78` per gap
and zero endpoint-eligible grasps, with the same three Inspire-link
intersections. The cylinder 20 mm-gap catalogue is incomplete because its
runtime v8 grasp directory has no candidate files. A compact index of all
five local reports and two staging-grasp checks is at
`~/shared_data/AutoDex/precision_insertion/endpoint_catalogs/README.md`.

Run the current offline tests from the repository root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests
```

The future runner must acquire ChArUco images first, then several socket
captures, with the socket already rigidly fixed. It will pass the captured
evidence to `calibrate_session`; the calibration helper does **not** acquire
images, assess the SAM3 mask/FoundPose photometric quality, prove hand-eye
accuracy, or authorize robot motion. The runner must explicitly select
`--shared-root` and use the matching v8 `object_processing` assets and Inspire
candidates; see `PLAN.md` for the remaining gates. Until those gates are
implemented, there is intentionally no robot-mode command to run here.

The path component `precision-insertion` is a directory name, not an importable
Python package name. If helper modules are added, use an importable package
name such as `precision_insertion` inside this directory.
