# Precision insertion demo

The isolated cylinder BODex 1,000-per-tabletop-scene run, exact filter
sequence, reproducibility commands, and per-socket 20 mm endpoint counts are
in [OFFLINE_CYLINDER_1000.md](OFFLINE_CYLINDER_1000.md). Passing its offline
endpoint is not permission to execute a robot insertion.
The follow-up [grasp-fidelity audit](OFFLINE_CYLINDER_1000.md#post-squeeze-fidelity-audit-do-not-use-nominal-renders-as-success-evidence)
shows that the existing fixed-key/commanded-squeeze cylinder images are
diagnostic only: they do not depict achieved MuJoCo grasp geometry. Do not
promote the 13 socket-clear candidates to robot trials from those images.

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
The endpoint screen assumes the initial BODex hand/key transform stays rigid;
the cylinder audit shows this assumption is not verified through squeeze.
Consequently `eligible` in an offline endpoint catalogue means only that
these *nominal* gates passed, never that the grasp is runtime-ready.
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
  These are raw hypotheses: exact key/socket/hand endpoint geometry must
  screen them before they are offered to a VLM; the selected live path must
  be checked again before any robot movement. A `propose`
  result is not motion authorization or evidence of insertion success.
- `xy_overlay.py` projects the already screened 1 mm candidate centers into
  calibrated AutoDex views and produces matching raw/annotated 16:9 crops.
  It checks candidate separation in the **original camera pixels** before
  display enlargement. If fewer than two cameras resolve the offsets, the
  VLM is not called. Cropping does not create missing visual information;
  current overlays are center anchors, not rendered CAD silhouettes.
- `xy_retry.py` combines the existing v8 pose/endpoint catalogue gate, fresh
  20 mm geometry screens, camera projections, per-view ZeroDex-style VLM
  choices and strict multi-view consensus. It runs only after an observed
  insertion failure and caller-supplied evidence that guarded withdrawal
  finished while the key remains held. It compares the multiview-key/live-
  wrist-derived `T_key_hand` against the v8 grasp using commissioned drift
  limits, then screens the actual observed rigid relation at each XY target;
  a large drift stops the retry. Inspire finger configuration remains a
  nominal controller model, not measured finger feedback.
  It returns a **proposal requiring new live preflight**, never a Franka
  command or a claim of insertion success. For a gap smaller than 1 mm, all
  1 mm endpoint offsets may be geometrically impossible; that correctly
  produces `no_safe_direction`, not an override from the VLM.
- `endpoint.py` evaluates one fixed-grasp candidate using the full metric CAD
  key, exact socket collision mesh, and every Inspire visual link at the
  centered 20 mm insertion pose. It combines Coal triangle-surface collision
  and minimum distance with solid-volume vertex containment, so a hand link
  fully inside socket material is not called clear. A paired simulated or
  observed hand/key pose may be supplied explicitly; catalogues are bound
  to the screening implementation hash. It excludes the Franka arm and all
  trajectories and records source hashes. This is **endpoint** evidence only;
  simulated grasp stability is still a separate v8/MuJoCo gate.
- `grasp_fidelity.py` and `screen_cylinder_achieved_endpoints.py` audit how
  the cylinder and hand actually moved during stock MuJoCo closure and then
  rescreen those paired achieved states at 20 mm for each socket. These
  diagnostics explain why a fixed-key/commanded-squeeze illustration can
  show false deep penetration. Their full results and reproduction commands
  are in [OFFLINE_CYLINDER_1000.md](OFFLINE_CYLINDER_1000.md).
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
- `reset_candidates.py` reads only provenance-bound, full-key MuJoCo-stable
  directed v8 reset seeds. `repose_policy.py` cross-checks alternative poses
  with insertion-eligible grasps and reset seed availability at each AutoDex
  release height. Both require explicit pose-fidelity limits. They report
  seeds for a later socket-aware Franka preflight, never a runnable reset.
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

### Repose/reorientation proposal assets

The new key object IDs have v8 tabletop poses but no legacy `paradex`
tabletop tree. Therefore the stock `src/experiment/reset/reorient.py`
cannot resolve its usual v8-to-legacy reset-cell mapping. Its carried-object
planning scene also drops the session-frozen socket mesh, so it must **not**
be invoked as this demo's executable repose path. The demo's
`audit-reorient` command reports directed v8 pose-pair scenes separately
from MuJoCo-stable reset grasp seeds and staged diagnostics. An available
BODex scene is *not* an executable reorientation trajectory.

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root /home/hyunsu/shared_data --mode square --gap-mm 1.5

~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py prepare-reorient-scenes \
  --shared-root /home/hyunsu/shared_data --mode cylinder --gap-mm 20 \
  --manifest /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_scene_manifest_v8_NEW.json
```

`prepare-reorient-scenes` reuses AutoDex's v8 BODex scene generator, creates
only missing directed pose-pair scenes at release heights 0, 4, 8 and 12 cm,
verifies existing scenes, and refuses to overwrite its manifest. Each scene
must exist at **both** the BODex path
`object_processing/<key>/scene/reorient_<h>/<i>_<j>.json` and the stock
sim-filter path `AutoDex/scene/inspire/<key>/reorient_<h>/<i>_<j>.json`.
The [cylinder v2 scene manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_scene_manifest_v8_v2_20261010.json)
records eight directed-height pairs and the hashes of both copies. The square
key already has 20 h=12 cm scene pairs.
[The square v4 audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/reorient_square_1p5_audit_v4_20261010.json)
and [cylinder canonical v4 audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_canonical_audit_v4_20261010.json)
find **zero runtime-stable reset seeds** for either family. Earlier square
whole-hand staging manifests also report zero sampled passes; they must not
be counted as runtime reset assets. The v4 audit reports a raw
`sim_eval.json` success separately from a provenance-bound seed the demo
loader will accept; only the latter enters its stable-seed count.

For the cylinder's 12 cm release scene, direct full-key BODex generation
failed in Coal convex-hull construction (`Too many neighbors`), so it did
**not** yield valid raw full-key proposals. The isolated pilot instead reused
100 raw grip-proxy proposals per directed pose pair, after verifying equal
proxy/full-key scene target poses and object frames, and staged their wrist
and finger proposals under the **physical full-key ID**. This is a proposal
transfer, not a grasp-success claim. The original AutoDex full-key scene
collision, squeeze-contact, and MuJoCo filters were then run on a fresh
staging tree with the required compatibility path. The
[completed pilot audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_filter_audit_pilot100_retry_20261010.json)
reports **200 raw → 107 scene-clear → 37 squeeze-contact → 0 MuJoCo-stable**.
The first attempted filter run lacked `rsslib`; its cached collision failures
are unusable evidence and remain isolated from this fresh retry. Do not merge
or select candidates from that first tree. The commands for the valid pilot
are:

```bash
PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/stage_cylinder_reorient_proposals.py \
  --raw-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_reorient_proxy_pilot_100 \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_STAGE \
  --shared-root /home/hyunsu/shared_data --expected-per-cell 100

PYTHONPATH=demo/precision-insertion:demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_cylinder_r15_h80", "/home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_STAGE", "/home/hyunsu/shared_data/AutoDex/precision_insertion/NEW_RESET_SIM_PASS", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'

PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_cylinder_reorient_pilot.py \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_STAGE \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/NEW_RESET_AUDIT.json
```

Use new output paths because staging and audits refuse overwrite and the
stock filter caches each seed result. A zero-yield pilot does **not** prove
reset grasps impossible; it shows this small proxy-proposal sample supplied
none. More robust BODex proposals and socket-aware
Franka pickup–lift–reorient–place–retreat preflight remain required before
automatic repose can replace `repose_required_unplanned`.

The stock sim filter copies only the four grasp arrays to its passing output;
it leaves `sim_eval.json` beside the raw proposal. Therefore that output must
not be treated as a verified runtime reset pool by itself. After each **new**
full-key filter run, `promote_v8_reset_candidates.py` checks the raw pass, the
stock copy, and the staged scene hashes before writing each passing seed to
`AutoDex/candidates/inspire/reset_12/<key>/reorient_12/<v8_i>_<v8_j>/<seed>`
with a bound `sim_eval.json` and `source_evidence.json`. It never copies an
unverified seed or replaces an existing canonical candidate. For this pilot:

```bash
PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/promote_v8_reset_candidates.py \
  --shared-root /home/hyunsu/shared_data \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_reorient_fullkey_eval_pilot100_retry_20261010 \
  --stock-candidate-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_reorient_fullkey_sim_pass_pilot100_retry_20261010 \
  --audit /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_filter_audit_pilot100_retry_20261010.json \
  --manifest /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_promotion_pilot100_retry_20261010.json
```

[The promotion manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_promotion_pilot100_retry_20261010.json)
records **zero** promoted seeds. For a later nonempty pool,
`precision_insertion.reset_candidates.load_v8_reset_seeds` reads the direct
v8 cell, converts each object-frame wrist transform using the *fresh* key
pose, honors the original AutoDex reset-grasp success-rate ordering, and
rejects changed seed/scene/key-mesh evidence. It also recomputes the MuJoCo
post-squeeze key-in-hand displacement and symmetry-reduced axis tilt from the
saved trajectory. The caller **must** supply commissioned maximum drift and
tilt limits; a seed outside either limit is omitted. It is only a seed loader;
it does not
inherit the stock reset runner's legacy pose map, and it does not establish a
socket-aware Franka path or authorize any motor command.

An expanded isolated run generated **1,000 proxy proposals per directed 12 cm
reset cell**, 2,000 total. Native BODex `success` was 0/2,000, as with the
tabletop run; the stock filter does not gate on that strict flag. Full-key
filter counts were **936 scene-clear → 467 squeeze-contact → 2 MuJoCo-stable**:
`0_1/191` and `1_0/631`. The
[filter audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_filter_audit_1000_20261010.json)
records the exact stage totals. Their squeeze-end hand-relative center drifts
are approximately **1.1 mm** and **9.9 mm** respectively; the latter also
has approximately **36°** symmetry-reduced axis tilt. A MuJoCo gravity pass
is thus not evidence of a rigid key/hand transform. Using *illustrative,
uncommissioned* 3 mm center-drift and 8° tilt limits, only `0_1/191` survives
the loader; those values do not approve robot execution.
For the previously used *project diagnostic* BODex thresholds (maximum
`grasp_error ≤ 0.2`, mean absolute `dist_error ≤ 10 mm`), `0_1/191` records
about 0.081 and 8.9 mm, whereas `1_0/631` records about 0.361 and 4.7 mm.
Thus only the first also passes that diagnostic. These numbers are reported
separately: the original AutoDex sim filter does **not** enforce them, and
native BODex `success` remains false for both.

To reproduce the expanded run, use fresh output names in all four steps:

```bash
PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 2 \
  --obj_list_file demo/precision-insertion/configs/cylinder_grip_proxy.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_type reorient_12 --seed_num 1000 --seed 11010 \
  --exp_name NEW_RESET_PROXY_1000 \
  --output_dir /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_PROXY_1000

PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/stage_cylinder_reorient_proposals.py \
  --raw-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_PROXY_1000 \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_FULLKEY_1000 \
  --shared-root /home/hyunsu/shared_data --expected-per-cell 1000

PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_cylinder_r15_h80", "/home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_FULLKEY_1000", "/home/hyunsu/shared_data/AutoDex/precision_insertion/NEW_RESET_STOCK_PASS_1000", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'

PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_cylinder_reorient_pilot.py \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_RESET_FULLKEY_1000 \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/NEW_RESET_FILTER_AUDIT.json
```

The stock filter catches some internal collision errors and caches failures;
check the complete audit before interpreting its process exit code. Its
MuJoCo pass output alone is not a runtime reset pool. The subsequent local
promotion command uses the new stage, stock-pass root and audit together.

`~/shared_data/AutoDex/candidates/inspire/reset_12` is a symlink to the
read-only `/mnt/paradex2` NFS mount on this machine. The expanded passing
seeds were therefore written to the **local, non-runtime**
[handoff tree](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff/reset_12)
with a [promotion manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_promotion_1000_local_handoff_20261010.json).
No candidate was installed into the canonical NAS tree. To reproduce local
staging after a fresh full-key filter run, pass
`--output-candidate-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff/NEW_reset_12`
to `promote_v8_reset_candidates.py`, along with that run's stage, stock
candidate root, audit, and a new promotion manifest path. Do not overlay the
handoff onto an existing NAS candidate cell without checking IDs and hashes.
The [handoff v4 audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_audit_v4_20261010.json)
uses `run_pipeline.py audit-reorient --candidate-root` to count the two
evidence-bound seeds separately from the canonical NAS audit, which still
counts zero.
The 999 KB [local handoff bundle](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_bundle_20261010.tar.gz)
contains both candidates, audit reports, v8 key scenes, key and six cylinder
socket object-processing assets, fixture geometries, and a handoff README.
Its SHA-256 is
`6b83a94026074f925c57567dd754771e01cbfda16491d2e8a10ba647894acc98`.
The archive has **not** been copied to NAS. Scene JSONs embed this host's
absolute mesh/URDF paths; a different AutoDex-host shared root requires
scene regeneration and fresh evidence validation, not a blind path edit.
Even after installation on a writable AutoDex host, both seeds still need
socket-aware full-chain Franka planning and physical validation.

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

For the isolated **1,000-proposal-per-tabletop-scene cylinder run**, use
`render_offline_cylinder_filtered.py`. It independently verifies the complete
2,000-seed stage, the original AutoDex scene/contact/MuJoCo counts, and every
saved 20 mm screen against fresh exact-CAD screens before drawing any image.
It renders all 13 accepted IDs for each of six cylindrical socket gaps
(78 grasp–socket pairs), with oblique and side 16:9 views for both the MuJoCo
squeeze and nominal AutoDex controller hold. The output stays separate from
the empty live v8 grasp tree:

```bash
cd /home/hyunsu/autodex-code
export PYTHONPATH=/home/hyunsu/autodex-code/demo/precision-insertion/compat
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/render_offline_cylinder_filtered.py \
  --summary /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_endpoint_screen_1000_squeeze_20261010/summary.json \
  --shared-root /home/hyunsu/shared_data \
  --output-root /home/hyunsu/shared_data/AutoDex/precision_insertion/visualizations/cylinder_offline_filtered_20mm_NEW_RUN
```

See each `gap_XXmm/pose_N/grasp_ID/` folder and the root `manifest.json` for
evidence paths and images. These are **offline endpoint illustrations only**:
there is no Franka arm pose, full path/contact validation, or robot insertion
success. The 1 µm numerical hand clearance used in this run is not a safe
physical margin. Do not call these runtime-ready grasp candidates.

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
To add a **read-only** reset-seed assessment when that happens, supply both
`--max-reset-drift-mm` and `--max-reset-axis-tilt-deg`. These must be
commissioned, not copied from an illustrative example. Add
`--reset-candidate-root /path/to/handoff_parent` only when auditing a local
handoff rather than the canonical NAS; the parent must contain `reset_<h>/`.
`--attempted-reset HEIGHT/TARGET_STEM/SEED_ID` excludes a previously tried
reset seed. The report's `repose_assessment` distinguishes an insertable
target pose from a pose with an available reset seed; it still cannot claim a
socket-aware full-chain plan, safe release, or physical pose change. With
this optional assessment enabled, missing reset seeds produce
`repose_assets_unavailable`, a local-only handoff produces
`repose_staged_only_unplanned`, and a canonical seed still produces
`repose_required_unplanned` until the whole reset path has passed preflight.

For a **separate planning-only reset attempt**, call
`precision_insertion.repose_transition.preflight_v8_repose_transition(...)`
from a process that already holds the session calibration, fresh trial scene,
complete insertion endpoint catalog and the unchanged
`GraspPlanner(hand="fr3_inspire")`. Supply the observed source and desired
target three-digit v8 tabletop stems, a commissioned release XY in robot
meters, one of the 4/8/12 cm v8 release-height cells, measured 13-joint start
state plus acquisition timestamps, reset-grasp fidelity limits, frozen-socket
clearance, a measured ChArUco interior-edge clearance and path-audit limits.
The entire key footprint must lie inside the measured ChArUco corner hull;
the broad cuRobo table cuboid alone is insufficient. The function first
requires an eligible insertion grasp at the *target* pose; then it loads
provenance-bound reset
seeds, plans one AutoDex pickup per seed and checks the same held key through
lift, transfer and straight-down descent with the socket still present. The
returned `ReposeTransitionPreflight.to_record()` can be saved alongside the
trial report. With no release goal, its best status is
`held_reset_path_available_release_unplanned`. Supplying both an explicit
7-joint `retreat_goal_arm_q` and `minimum_release_key_clearance_m` additionally
preflights opening to AutoDex's pregrasp hand pose, a +10 cm vertical exit and
the original planner's joint-space retract. The conservative world contains
both a key left at the release height and a key at the selected tabletop rest
pose; the hand is screened against both after opening. Its strongest status is
`nominal_reset_preflight_pass_drop_unobserved`, **not** reset success or robot
authorization. The read-only CLI now exposes this as `preflight-repose`; it
still has **no motor mode**. A saved session calibration, full v8 endpoint
catalog, fresh key pose, measured start joints and commissioned numeric limits
are required:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py preflight-repose \
  --shared-root /path/to/shared_data --mode cylinder --gap-mm 20 \
  --session /path/to/session_calibration.json \
  --catalog /path/to/current_endpoint_catalog.json \
  --key-pose-world-npy /path/to/fresh_key_pose_world.npy \
  --key-observation-id capture_001 --key-capture-time-s 100.0 \
  --live-start-q-npy /path/to/measured_fr3_inspire_q13.npy \
  --start-q-time-s 100.0 --max-key-state-skew-s 0.1 \
  --limits-json /path/to/commissioned_path_audit_limits.json \
  --from-pose-stem 000 --to-pose-stem 001 --height-cm 12 \
  --release-x-m 0.40 --release-y-m 0.00 \
  --min-rest-socket-clearance-mm 10 --min-board-edge-clearance-mm 10 \
  --max-pose-error-deg 10 --max-reset-drift-mm 3 \
  --max-reset-axis-tilt-deg 8 \
  --retreat-goal-q-npy /path/to/approved_retreat_q7.npy \
  --min-release-key-clearance-mm 1 \
  --output-dir /path/to/new_repose_preflight_report
```

The numbers and release site shown are **illustrative, not commissioned**.
Do not use them as physical safety limits. Omit both retreat options to stop
at held-key descent. `--reset-candidate-dir`, when used, must point to the
specific `reset_12/` directory for this example, not its parent; this differs
from the assessment CLI's `--reset-candidate-root` parent-of-heights path.
Every run creates a new `report.json`, `trial_scene.json`, and, on selected
planning paths, `planned_trajectories.npz` with source hashes. Exit 0 means
the requested *offline planning scope* passed, not that robot motion is safe;
exit 2 means no qualifying nominal reset path or missing prerequisites.
The 0 cm original AutoDex
release cell, measured post-lift key/hand relation, dynamic drop and landing
verification remain to be implemented. The retract uses the original cuRobo
collision checker; unlike opening and vertical exit, it has no separate
full-visual-mesh sample audit yet. See `tests/test_repose_preflight.py` for a fully offline usage
contract; its fake planner does not demonstrate real Franka reachability.

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

`precision_insertion.session_bootstrap.bootstrap_session()` is the
non-motion session assembly point. Supply the **raw distorted** empty-board
BGR snapshots, their request ID and verified acquisition times, and at least two
`SocketCaptureInput` records (each with a socket-specific SAM prompt,
**undistorted** BGR frames from the *same* FoundPose request, SAM masks,
FoundPose payloads, request ID, and verified acquisition times). It calls the existing
per-view admission gate and `calibrate_session()` in order, then freezes one
socket pose and collision scene. `write_session_bootstrap_artifacts()` creates
a fresh directory containing board frames, every socket frame/mask/pose
payload, the calibration record, and a SHA-256 file manifest. It refuses to
replace a previous session; `verify_session_evidence_bundle()` detects
subsequent file changes. The caller still needs a trusted acquisition-
metadata adapter: this API does not convert AutoDex's publication `ts` into
camera time. The bundle proves the inputs supplied to calibration, **not**
sub-millimetre hand-eye accuracy or robot readiness.

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

`precision_insertion.live_capture` now connects the unchanged AutoDex
`SnapshotOrchestrator.snap(decode=True)` and initialized
`InitOrchestrator.collect_payloads()` to the session inputs. Call
`collect_board_snapshot()` first, then `collect_socket_capture()` at least
twice while the socket is fixed and the key is absent, then pass the returned
frames/payloads and timestamps to `bootstrap_session()`. Socket frames are
the **same-request undistorted PNGs** written by the stock init daemon; the
adapter waits for those asynchronous files and refuses missing images rather
than taking a later snapshot. Use a unique capture root on a filesystem
shared by the robot PC and capture PCs (for example, a common absolute NAS
mount), and preserve those intermediate directories until the evidence bundle
has been verified. The metadata callback must return verified camera
acquisition times keyed by the same request ID. The current stock daemons do
not themselves provide that callback, so these adapters are **not yet a live
calibration command** and cannot authorize robot motion.

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
