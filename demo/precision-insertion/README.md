# Precision insertion demo

The isolated cylinder BODex 1,000-per-tabletop-scene run, exact filter
sequence, reproducibility commands, and per-socket 20 mm endpoint counts are
in [OFFLINE_CYLINDER_1000.md](OFFLINE_CYLINDER_1000.md). Passing its offline
endpoint is not permission to execute a robot insertion.

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
  The supplied timestamps must denote **image acquisition**, not SAM/FoundPose
  completion or payload publication. The unchanged AutoDex snapshot/init
  orchestrators do not yet return enough per-camera frame provenance for this
  demo's precision gate. A live camera adapter is pending; never fabricate
  synchronized times from one request ID or use payload `ts` as capture time.
- `symmetry.py` reads the v8 `object_processing/<object>/processed_data/info/`
  symmetry and tabletop poses. The D∞ cylindrical key may exchange identical
  ends; the C∞ socket may not.
- `world.py` adds the frozen socket mesh to a copy of the cuRobo scene. It
  checks the pose and mesh path and does not mutate the source scene. After
  ChArUco measurement, `calibrate_session` replaces the base table cuboid
  with that session's measured height. For every trial,
  `build_trial_scene_from_session` calls the unchanged AutoDex v8 scene
  converter with the **fresh** perceived key pose, retains the same measured
  table and frozen socket, and rejects a changed socket mesh or pose. Cylinder
  tabletop snapping uses this demo's local-z symmetry adapter. These helpers
  do not capture images, plan the Franka path, or authorize insertion.
- `config.py` resolves explicit square/cylinder key and socket IDs and the
  20 mm verification target; `assets.py` performs a read-only v8 input audit.
- `xy_voting.py` accepts already geometry-screened **absolute socket-frame XY
  offsets** and synchronized per-view VLM choices. It projects candidate
  anchors for overlays and returns `propose`, `abstain`, `stop`, or
  `no_correction` with camera provenance. It rejects ties, single-view
  decisions, stale/asynchronous frames, a possible slip, and step/total
  budget violations; it never averages candidate positions. The current VLM
  retry policy proposes only `hold` or one **1 mm** socket-frame axial step
  (`+X`, `-X`, `+Y`, `-Y`); diagonal and fractional-step votes are rejected.
  These are raw hypotheses: exact key/socket/hand geometry and the live path
  must screen them before they are offered to a VLM or robot. A `propose`
  result is not motion authorization or evidence of insertion success.
- `endpoint.py` evaluates one fixed-grasp candidate using the full metric CAD
  key, exact socket collision mesh, and every Inspire visual link at the
  centered 20 mm insertion pose. It uses Coal triangle-mesh collision and
  minimum surface distance, excludes the Franka arm and all trajectories,
  and records source hashes. This is grasp-level **endpoint** evidence only;
  simulated grasp stability is still a separate v8/MuJoCo gate.
- `targets.py` validates the selected socket's CAD entry, pre-insertion hold,
  and 20 mm transforms; it composes them with the **session-frozen measured
  socket pose** and one fixed `T_key_hand` grasp relation. An optional XY
  offset is expressed in the socket frame and applied equally at all three
  poses. The result records source hashes and poses but does not solve Franka
  IK, verify an attached-object trajectory, or authorize contact motion.
- `path_audit.py` consumes the **actual planned** 13-DOF held-lift, transfer
  and axial-descent joint samples, using the unchanged AutoDex planner's
  `fk_wrist()` (cuRobo `ee_link: base_link`). It requires fixed Inspire joints,
  continuous segment handoff, sufficiently dense samples, pre-insertion and
  20 mm goal agreement, and a monotone socket-axis descent. At every FK sample
  it attaches the full key CAD and every Inspire visual link with the same
  `T_key_hand`, then checks Coal collision and hand clearance against the
  frozen socket, measured table and other fixed scene obstacles. It records
  input hashes, failures and minimum observed clearances. This is a
  **sampled held-geometry audit**, not a swept-volume proof: collisions
  between samples, cuRobo Franka arm checks, controller force response and
  physical grasp/insertion remain independent gates. A tilted socket can be
  audited only if an axis-following path has already been generated; AutoDex's
  world-Z stroke planner cannot create that path. The first key/table pair
  immediately after grasp is exempt as intended support contact; all later
  lift samples and every hand/socket pair are checked.
- `preflight.py` accepts one already endpoint-screened AutoDex pickup
  `PlanResult`, replans the 10 cm lift with the declared held Inspire pose,
  calls the unchanged cuRobo planner for transfer and <=5 mm socket-axis
  waypoint segments, and rejects missing paths, wrong goal/FK residuals,
  changed finger joints or a failed `path_audit.py` result. This is a
  planning-only composition, **not** a guarded physical insertion command.
  The default AutoDex pickup lift models the grasp pose, whereas the executor
  squeezes farther; reusing its lift without checking the selected hold would
  compare different hand geometries. Every caller must distinguish a nominal
  commanded hold from actual measured hand state.
- `candidates.py` scans the selected shared root's Inspire v8 candidate tree,
  reads matching scene `meta.pose_idx` and tabletop assets, requires full-key
  simulation evidence, and applies `endpoint.py` to surviving grasps. The
  cylindrical key has one shared v8 grasp tree; every socket gap receives a
  separate endpoint catalogue using that socket's exact collision mesh. The
  catalogue distinguishes a complete finite scan from missing or truncated
  input. Per-trial selection matches the observed tabletop pose, excludes
  session-attempted grasps, optionally ranks by v8 coverage, and rejects a
  different selected key/socket/gap or changed source files. AutoDex's
  `load_candidate` is reused with an explicit
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

To inspect saved or freshly planned paths, keep the original planner instance
and the **actual** dense lift/transfer/descent `(N, 13)` joint arrays. The
transfer begins at the verified lift endpoint; descent begins at the same
joint state where transfer ends. With a `SessionCalibration` and
`InsertionTargets` from this demo, call:

```python
from precision_insertion.path_audit import PathAuditLimits, audit_held_joint_paths

limits = PathAuditLimits(
    max_joint_step_rad=0.02,
    max_wrist_step_m=0.005,
    max_wrist_rotation_deg=1.0,
    goal_position_tolerance_m=0.001,
    goal_rotation_tolerance_deg=1.0,
    axial_lateral_tolerance_m=0.001,
    axial_rotation_tolerance_deg=1.0,
    minimum_hand_clearance_m=0.001,
)
report = audit_held_joint_paths(
    shared_root=shared_root, calibration=session, targets=targets,
    planner=planner, transfer_trajectory=transfer_q,
    descent_trajectory=descent_q, lift_trajectory=lift_q,
    held_hand_q=held_hand_q, limits=limits,
)
print(report["sampled_clear"], report["failures"])
```

These numbers are **illustrative API arguments, not calibrated thresholds**.
Supply the measured/selected held hand state; do not treat AutoDex's nominal
commanded grip as measured feedback. `sampled_clear=True` is never a robot
execution permit. The caller must retain cuRobo's arm/world path result and
commission a guarded insertion controller and between-sample swept checks.

To compose the original AutoDex planner calls after selecting **one** eligible
v8 grasp, call `plan_insertion_after_pickup` with the corresponding fresh-key
trial scene, frozen session, rigid targets and successful original pickup
`PlanResult`. For the baseline executor's nominal squeeze command only:

```python
from precision_insertion.endpoint import nominal_inspire_hold_poses
from precision_insertion.preflight import plan_insertion_after_pickup

hold_q = nominal_inspire_hold_poses(
    pickup_plan.pregrasp_pose, pickup_plan.grasp_pose,
)["autodex_default_controller_hold"]
preflight = plan_insertion_after_pickup(
    planner=planner, pickup_plan=pickup_plan, trial_scene=trial_scene,
    shared_root=shared_root, calibration=session, targets=targets,
    held_hand_q=hold_q, held_hand_source="commanded_nominal",
    limits=limits, axial_waypoint_step_m=0.005,
)
print(preflight.to_record())
```

`sampled_planning_pass` is **not** a physical success label or execution
permit. Run this against the actual selected candidate and measured session;
the current local square v8 pool has no 20 mm endpoint-eligible candidate,
and the cylinder runtime pool is still missing. The nominal squeeze pose is
not a measurement; before physical transfer, re-observe the held key and
validate the hand–key relation against the planned one. A guarded contact
controller, true acquisition-timestamped camera adapter, and reset/repose
execution remain to be implemented.
Do not feed this separately replanned `lift_trajectory` straight into the
unchanged `FrankaExecutor.execute(lift_traj_override=...)`: its start-state
gate still refers to the original pickup `lift_preflight` modeled at grasp
joint values. A demo-local execution adapter must compare live arm/hand state
with the **same** held-lift plan before replay, or replan from live state.

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
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_NEW_SCAN.json

~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py select-catalog \
  --mode square --gap-mm 1.5 \
  --catalog /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_NEW_SCAN.json \
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

For the cylindrical family, generate the key's BODex/MuJoCo v8 grasp pool
**once**, then screen that same pool against **each** selected socket. A
recommended per-socket output layout is
`endpoint_catalogs/cylindrical/<socket_object>/<scan_id>.json`. For example,
after the grasp pool is present, select the 20 mm radial-gap variant with
`--mode cylinder --gap-mm 20` for both `screen-catalog` and `select-catalog`;
the latter now rejects a catalogue for any other socket gap. Keep each scan's
exact key/socket/URDF hashes and clearance rule. At live startup the operator
or a validated fixture identifier must independently confirm which physical
socket is mounted; measuring only its pose does not establish its size.
The `gap_01mm` name denotes a **1 mm one-sided radial clearance**, not 0.1 mm.
The same scan across all four square gaps yields one `table/0/78` per gap
and zero endpoint-eligible grasps, with the same three Inspire-link
intersections. The cylinder 20 mm-gap catalogue is incomplete because its
runtime v8 grasp directory has no candidate files. A compact index of all
five local reports and two staging-grasp checks is at
`~/shared_data/AutoDex/precision_insertion/endpoint_catalogs/README.md`.

### Cylinder endpoint diagnostic images (not runtime grasps)

`render_cylinder_endpoint_diagnostics.py` provides a reproducible *visual
diagnostic* while the cylinder's tabletop v8 grasp pool is absent. It reads the
local, trusted raw **reorientation** pilot, applies the documented relaxed
numeric threshold, and uses `endpoint.py` to test the centered 20 mm key fit
and all Inspire visual links against each of the six exact socket meshes.
Only endpoint-geometry passes receive a Blender bundle. The demonstration
0.2 mm hand/socket clearance is **not calibrated**. Native BODex success,
tabletop suitability, MuJoCo stability, Franka planning, continuous insertion,
and physical success are all unverified; no diagnostic image may be imported
into the runtime candidate catalogue as a validated grasp.

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/render_cylinder_endpoint_diagnostics.py \
  --shared-root /home/hyunsu/shared_data \
  --raw-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_reorient_proxy_pilot_100/precision_key_cylinder_r15_h80_grip_proxy/reorient_12 \
  --output-root /home/hyunsu/shared_data/AutoDex/precision_insertion/visualizations/NEW_DIAGNOSTIC_RUN \
  --min-hand-clearance-mm 0.2

blender --background --python \
  scripts/precision_insertion/render_blender_actual_mesh_animation.py -- \
  /home/hyunsu/shared_data/AutoDex/precision_insertion/visualizations/NEW_DIAGNOSTIC_RUN/precision_socket_cylinder_gap_01mm/raw_reorient_0_1_0/endpoint_bundle.npz \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/visualizations/NEW_DIAGNOSTIC_RUN/precision_socket_cylinder_gap_01mm/raw_reorient_0_1_0/key_socket.png \
  --width 1600 --height 900 --view key-socket --still-frame 1
```

The script creates an exclusive output directory, with one folder and
`screen_report.json` per physical socket variant, then a subfolder for every
endpoint-geometry-pass raw seed. Each seed folder contains actual CAD/URDF
meshes and a transform bundle; render it from `key-socket` and `task` views.
The root `manifest.json` records the evidence scope and runtime eligibility.

### Images for fully offline-filtered v8 grasps

Use `render_eligible_grasp_endpoints.py` **only after** a complete, current
`screen-catalog` run. The input must be the *same key/socket mode* that will be
shown. Every eligible grasp is rechecked against current v8/MuJoCo evidence
and the exact 20 mm CAD/Inspire endpoint before any image is made. It produces
four 16:9 PNGs per grasp: oblique and side views for both the MuJoCo squeeze
and the nominal AutoDex controller hold. The key–hand transform stays fixed;
the images use actual key/socket triangle meshes and evaluated Inspire URDF
visual links. They do **not** depict a Franka arm pose, prove continuous
insertion, or claim a physical task success.

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/render_eligible_grasp_endpoints.py \
  --catalog ~/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_NEW_SCAN.json \
  --output-root ~/shared_data/AutoDex/precision_insertion/visualizations/verified_v8_20mm_NEW_RUN/square_1p5mm
```

Use a **new** output root for every run; previous artifacts are never
overwritten. The layout is `pose_<stem>/<type>_<scene>_<grasp>/<hold>/`
with `endpoint-oblique.png`, `endpoint-side.png`, the source 3D transform
bundle, and a fresh endpoint screen report. `manifest.json` records the
catalogue hash, counts, scope, and relative image paths. If a complete scan
has zero eligible grasps, the command writes only a zero-count manifest and
exits with code 2. If the catalogue is incomplete or stale, it rejects it
without making an image. The 2026-10-10 local square scans each have one v8
candidate and **zero** eligible; the cylindrical v8 pool has zero candidates
and its catalogue is incomplete. The older `cylindrical_endpoint_diagnostics`
images are separate raw reorientation-seed illustrations, **not** results of
this verified renderer and not valid insertion-grasp success examples.

### Replay a saved session and fresh key observation without robot motion

`preflight-trial` loads the frozen ChArUco/socket collision-world snapshot,
checks the selected key/socket catalogue, classifies one freshly measured key
tabletop pose, then tries eligible grasps in v8 order. It reuses the original
AutoDex pickup planner and adds demo-local lift, transfer and 20 mm axial
planning plus sampled held-key/hand collision checks. It takes measured
13-DOF start joints and acquisition timestamps from saved inputs. The limits
JSON must contain the eight positive `PathAuditLimits` fields in
`precision_insertion/path_audit.py`; these are **commissioning inputs**, not
universal defaults. The named session must include a saved collision-world
snapshot and hashes from `write_session_calibration`.

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py preflight-trial \
  --shared-root ~/shared_data --mode square --gap-mm 1.5 \
  --session /path/to/session_calibration.json \
  --catalog /path/to/complete_current_endpoint_catalog.json \
  --key-pose-world-npy /path/to/fresh_key_pose_4x4.npy \
  --key-observation-id capture_001 --key-capture-time-s 100.0 \
  --live-start-q-npy /path/to/measured_start_q_13.npy \
  --start-q-time-s 100.0 --max-key-state-skew-s 0.1 \
  --limits-json /path/to/commissioned_path_audit_limits.json \
  --max-pose-error-deg 10 --axial-waypoint-step-mm 5 \
  --output-dir /path/to/new_trial_report_directory
```

The numeric limits in the example are **illustrative only**. The exclusive
output contains `report.json`, `trial_scene.json`, and, if a full plan is
found, `planned_trajectories.npz`; all are offline evidence and have
`robot_ready: false`. A pose with no candidate may return
`repose_required_unplanned`, not an executable reorientation trajectory.

### Socket perception evidence and current camera-timestamp blocker

AutoDex's unchanged `InitOrchestrator.collect_payloads()` supplies one SAM
mask and FoundPose pose **per camera**, so the new
`precision_insertion.perception_evidence.admit_socket_capture()` checks mask
size/border clipping, FoundPose quality/inliers, pose validity, calibrated
camera identity, and cross-camera *acquisition* skew before handing its
`SocketObservation`s to `calibrate_session()`. Use a socket-specific SAM
prompt while the key is absent. Repeat at least twice; calibration then
checks socket-pose repeatability and freezes the collision world. All
thresholds are explicit commissioning inputs, not silently inferred from a
VLM score. `collect_and_admit_socket_capture()` calls that original AutoDex
collector directly and joins its `request_id` to an injected acquisition
metadata provider; it refuses a mismatched request or the wrong initialized
socket model. The provider contract is
`{"request_id": int, "source": "camera_acquisition", "camera_times_s":
{camera_id: timestamp_seconds}}` on one verified clock.

The current AutoDex daemons' `mask["ts"]` and `pose["ts"]` are stamped when
results are **published after** SAM/FoundPose, not when the camera exposed its
frame. `SnapshotOrchestrator.snap()` likewise does not return per-camera
acquisition timestamps. The admission gate deliberately rejects those
publication timestamps; saved session records now state and validate
`camera_acquisition` as their time source. A demo-local acquisition metadata
side channel (or compatible camera/daemon support) is still needed for the
board, socket, and per-trial key frames before a live session can be treated
as synchronized. This is a **live integration blocker**, not a missing CAD
asset or an invitation to pass one request ID as a timestamp. The offline
planner and catalogue renderer remain usable without cameras.

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
