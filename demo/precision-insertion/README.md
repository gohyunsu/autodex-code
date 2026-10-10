# Precision insertion demo

## Calibrated VLM point grounding (read-only diagnostic)

`SessionRunner.prepare_grounded_xy_diagnostic(...)` is an alternative to
closed-set XY voting **after an observed insertion failure and a logged guarded
withdrawal**. It leaves the existing `run_auto.py`, `run_pipeline.py`, and
`scene_cfg.py` untouched. It stores original, same-request AutoDex camera
frames, measured Franka/Inspire state, frozen calibration, prompts/responses,
landmarks and a metric report in
`xy_retry_assessments/NNN/`. It does not call `record_retry`, plan a robot path,
or send a motion command. The earlier candidate-ID voting path remains
available for comparison.
The continuous-offset reports use `precision_insertion_grounded_xy_diagnostic_v2`;
the evaluator rejects earlier cardinal-step reports rather than silently
interpreting them as continuous-offset results.

The separate `lateral_preflight.py` can now **read-only preflight** a
nonzero, at-most-1 mm socket-XY hold shift *once its inputs have been
independently source-verified*. It begins at measured 13-DOF joints, requires
the wrist to match the saved withdrawn hold, removes only the carried key
from the frozen cuRobo world, and keeps the socket and table. It calls the
stock FR3/Inspire `plan_cartesian_pose(..., lock_hand=True)`, then audits
every resulting FK sample against the full key/hand meshes, exact socket,
table and supplied commissioned future-trial surface bounds. A planned
vertical detour, rotation, sparse path, changed Inspire pose, collision or
insufficient uncertainty margin rejects the shift. It records the exact
joint path and `insertion_replan_allowed=false`; it does **not** perform a
motion, reobserve the key, or authorize another insertion.

`SessionRunner.prepare_grounded_lateral_hold_preflight(...)` now binds the
saved cylinder diagnostic to this planner **for the first withdrawn hold
only**. Pass its `GroundedXYDiagnostic`, saved
`xy_retry_assessments/NNN/report.json`, a fresh measured Franka/Inspire
sample, and separately commissioned joint-drift, visual tip/axis-error and
lateral-path limits. The binder rechecks source image bytes, attempt,
withdrawal, physical-grasp medoid, task geometry, camera/session hashes,
confidence calculation, key-axis direction and predicted-versus-observed
tip/axis consistency. It saves `lateral_hold_preflights/NNN/` without calling
`record_retry` or changing the attempt state. Its first-step-only source
contract means a second adjustment needs a **new** capture and a separately
verified post-shift hold, not reuse of the earlier diagnostic. Physical
execution of even the first shift is not implemented.

After an **external commissioned controller** executes that first shift,
save a fresh same-request capture with
`write_raw_camera_capture(capture, output_dir, phase="post_lateral_hold")`.
Its frame IDs must advance beyond the diagnostic's per-camera IDs, and all
exposures must follow the motion-completion time. Supply a
`precision_insertion_lateral_hold_execution_v1` JSON log with matching
`attempt_id`, `candidate_id`, absolute `preflight_report_path` and its SHA-256,
`source="commissioned_lateral_controller"`, ordered `started_at_s` and
`completed_at_s`, and
`measurement={"trajectory_complete":true,"safety_abort":false,
"grasp_held":true}`. Its `source_records` must hash and reference three
separate files named `trajectory_feedback`, `safety`, and `grasp_state`.
The schema is an external evidence contract; **this repository does not
produce or authenticate those physical measurements**.

Then call `SessionRunner.assess_postshift_lateral_alignment(...)` with the
saved passing `GroundedLateralPreflight`, its `report.json`, execution log,
new raw capture directory, synchronized measured Franka/Inspire state,
frozen camera calibration, selected VLM backend, and commissioned timing,
tracking and visual-error limits. It re-verifies the exact planned joint
bytes, source hashes, live wrist target, camera timing and unchanged CAD,
then grounds the tip/shaft axis in the *new* frames. The report is stored at
`postshift_checkpoints/NNN/report.json` and has one of:
`visual_alignment_within_budget`, `residual_requires_new_shift`,
`held_relation_inconsistent`, or `visual_abstain`. The visual budget must fit
inside cylinder radial clearance after the future key-surface bound. Even
`visual_alignment_within_budget` sets `insertion_replan_allowed=false`:
the fresh 20 mm key/hand endpoint and axial planning checks below are separate
read-only gates, while guarded contact remains unfinished. There is no
automatic second shift.

Before using such a checkpoint as a held-key hypothesis, call
`verify_postshift_checkpoint(result, report_path, plan=shift_plan)` to
recheck its report, earlier planned joint bytes, execution-source hashes and
new raw camera PNGs. For a cylinder,
`reconstruct_axisymmetric_held_hypothesis(...)` composes the measured wrist
and physical grasp medoid, minimally rotates the predicted **directed** key
axis onto the newly triangulated axis, then translates the selected local
insertion tip onto the triangulated tip centre. Rotation about the key axis is
unobservable: its yaw is inherited from the medoid, **not** measured by the
VLM. A tip/axis disagreement beyond commissioned limits rejects the
hypothesis. `tip_axis_visual_surface_bound(...)` computes the extra possible
CAD-surface displacement from commissioned tip/axis error limits; this must
be added to the physical medoid's surface bound before the 20 mm endpoint
and sampled-path margin check. These functions still do not create an
insertion retry or approve a robot command.

`SessionRunner.prepare_postshift_insertion_preflight(...)` is the next
**read-only** gate for that same saved checkpoint. It rechecks the original
trial binding, failed held attempt, physical grasp-medoid source, completed
lateral-shift report and fresh post-shift camera pixels. Its inputs include
commissioned worst-case visual tip/axis errors, maximum allowed discrepancy
from the medoid, height/orientation drift limits and a future whole-surface
`SurfaceDeviationBounds`. The latter must cover the physical grasp bound
**plus** the key-surface displacement caused by visual tip/axis error; a
95%-confidence covariance alone is not a worst-case bound. It applies one
cylinder yaw gauge consistently to the exact 20 mm key/whole-Inspire endpoint
screen and the key/hand axial targets, uses measured Inspire joints, calls
the existing FR3/Inspire cuRobo transfer and axial planner, then runs the
sampled held-key/hand collision and uncertainty-margin audits. The report
and hashed planned joint paths are stored under the same attempt at
`postshift_20mm_preflights/NNN/`. Square mode does not inherit cylinder yaw
symmetry. A passing status is
`sampled_postshift_20mm_preflight_pass`, but the report still sets
`insertion_replan_allowed=false` and `robot_ready=false`: it is **not** an
execution or guarded-contact controller. The session runner immediately calls
`verify_postshift_insertion_preflight(...)` after saving: it rechecks the
checkpoint/source files, physical medoid, candidate and CAD hashes, sampled
margin, and exact saved transfer/axial joint arrays. A later executor must
recheck those sources again immediately before motion. The unchanged
`FrankaExecutor.follow_joint_trajectory` is **not** a guarded insertion
controller: its generic path does not require contact abort and can attempt
an endpoint landing after a stalled or timed-out stream. Do not replay the
saved axial path with that API as a contact insertion command. The real
15 mm radial-gap cylinder
asset audit on this host currently finds zero v8 grasp candidates and no
key/socket FoundPose representations, so a real cylinder replay is not yet
possible here.

For a commissioned session, reuse the *same* runner and saved checkpoint:

```python
result = runner.prepare_postshift_insertion_preflight(
    planner=planner,
    shift_plan=completed_shift_plan,
    checkpoint=postshift_checkpoint,
    checkpoint_report_path=postshift_report_path,
    bounds=commissioned_future_surface_bounds,
    max_visual_tip_error_m=commissioned_tip_worst_case_m,
    max_visual_axis_error_deg=commissioned_axis_worst_case_deg,
    max_axis_prior_residual_deg=commissioned_prior_axis_limit_deg,
    max_hold_height_delta_m=commissioned_hold_height_limit_m,
    max_preinsert_hand_rotation_deg=commissioned_xy_only_rotation_limit_deg,
)
```

The five error/limit variables above are measurements to commission; do not
copy numbers from the unit tests. This call writes a report but never moves
Franka or changes an attempt label.

The underlying first-shift planning API used by this binder is
`plan_lateral_hold_shift(planner, mode, shared_root, calibration, trial_scene,
start_q, expected_hold_pose, T_key_hand, increment_socket_xy_m, bounds,
limits, max_path_deviation_m, max_hold_height_deviation_m,
max_hold_rotation_deg)`. The three new path tolerances and the
`SurfaceDeviationBounds` must be commissioned, not copied from synthetic
tests. `write_lateral_hold_preflight(result, output_dir)` saves an exclusive
report and the planned trajectory. A positive sampled audit is a necessary
planning check only; fresh post-shift visual geometry, 20 mm endpoint fit,
guarded contact and physical outcome remain mandatory.

The VLM sees each **raw, full-resolution, undistorted** camera image separately.
For the smooth cylindrical key it returns the insertion-tip centre and two
points on the *visible projected shaft centreline*. Different views need not
mark the same physical shaft locations: the image line is the geometric
constraint. It returns `null` if the tip or straight shaft is obscured. The
alternate two-corresponding-landmark solver is retained for objects with an
identifiable CAD cross-section at `axis_reference_key_z_m`; an arbitrary point
on a smooth cylinder is **not** such a correspondence. For the square key,
neither two collinear landmarks nor one axis line resolves yaw, so this
diagnostic abstains until non-collinear CAD features are added.

Given full-resolution undistorted intrinsics `K_i` and frozen
`T_camera_socket,i = T_camera_world,i T_world_robot T_robot_socket`, the
cylinder solver triangulates the tip from multiple views, converts each 2D
axis line to a 3D plane through its camera, and finds their intersection
direction. Pairwise hypotheses and joint tip/line reprojection reject
inconsistent camera views. The marked-landmark variant instead triangulates
two points and checks their CAD spacing. Both compute lateral axis residuals
at the socket rim and at the plane 20 mm deeper. Translation changes both
residuals equally; if their difference or axis tilt is too large, XY alone
cannot resolve the error and the solver abstains. For a cylindrical key yaw is
irrelevant; the same two collinear points **cannot** determine square-key yaw,
so square mode abstains. It never treats a hidden point, a fabricated overlay,
or VLM confidence text as a millimetre measurement.

`AlignmentLimits` requires held-out estimates of VLM pixel error and a
*systematic lateral error floor* that includes camera/board/socket calibration,
time synchronization and correlated visual bias. It also requires explicit
parallax, reprojection, tilt, 20 mm sweep and 95% uncertainty limits. Defaults
are intentionally absent except for three required views. Two can be set
explicitly for a diagnostic but cannot independently reject a bad camera by
consensus. The report estimates the **continuous socket-frame XY correction**
`(ΔX, ΔY)` that aligns the key axis at the socket rim and 20 mm depth. Only
the next motion increment is limited to 1 mm in norm, preserving the earlier
retry-motion bound; it is not quantized to cardinal directions. The increment
is offered only when the 95% lower bound of its reduction in mean squared
rim/depth error is positive; otherwise it abstains. This is a *visual
hypothesis*, not certified physical
accuracy. Actual key/socket fit, whole-hand clearance, live cuRobo preflight,
guarded contact and independent task-success measurement remain separate
requirements. A highly accurate point fit can still be wrong if the VLM
consistently labels the wrong physical feature, or the fixture moves.

For the lateral key-axis offsets `e₀` at the rim and `e₂₀` at 20 mm depth,
minimizing `||e₀ + δ||² + ||e₂₀ + δ||²` gives the unique unconstrained
translation `δ* = -(e₀ + e₂₀)/2`. For the bounded next increment `δ`, its
squared-error improvement is `-2 e·δ - ||δ||²`, where
`e = (e₀ + e₂₀)/2`. Its conservative 95% lower bound subtracts
`2 sqrt(χ²₂(0.95)) sqrt(δᵀΣδ)`, with `Σ` the measured lateral-error
covariance. The 2D ellipse accounts for selecting a correction from the
estimated error itself.
The bound remains conditional on the commissioned pixel-noise and systematic
error model; held-out coverage must be measured before physical use.

Before allowing this diagnostic to influence a physical retry, collect a
held-out AutoDex-camera dataset with independently measured cylinder tip/axis
poses and deliberate ±1 mm socket-frame offsets. Preserve native image sizes,
capture timestamps and per-camera extrinsics. Measure tip/line pixel errors,
3D lateral error, axis error, abstention rate and metric XY-correction accuracy
separately by view and by occlusion. Populate `AlignmentLimits` from those
held-out results; its numbers must not be guessed from VLM text. Check that
the shaft centreline is recoverable in at least three simultaneous views.
Then add an execution adapter that repeats exact endpoint and whole-hand
collision screening with the measured held relation, live cuRobo path
planning and guarded force/contact control. The present point/line report
cannot be passed to `record_retry` and has no robot actuation route.

Use `evaluate_grounded_alignment.py` to compare saved diagnostics against
**independently measured**, same-capture held-key poses. Its input manifest
has schema `precision_insertion_grounded_eval_manifest_v1`, commissioned
`max_reference_capture_skew_s`,
`max_truth_translation_uncertainty_95_m`,
`max_truth_axis_uncertainty_95_deg`, and a nonempty `samples` list. Each sample
names a saved `diagnostic_report` and an `independent_pose` JSON (paths relative
to the manifest). The latter uses schema
`precision_insertion_independent_key_pose_v1`; it records the same
attempt/candidate/request/session/geometry IDs, `capture_time_s`,
`tip_socket_m`, downward unit `insertion_axis_socket`, 95% pose uncertainty,
and hashed absolute `source_files` from external optical metrology or an
independent fiducial tracker. A nominal wrist pose or the VLM's own estimate
is **not** acceptable ground truth. Example invocation:

```bash
python demo/precision-insertion/evaluate_grounded_alignment.py \
  --manifest /path/to/heldout/manifest.json \
  --output /path/to/heldout/new_report.json
```

The exclusive output reports abstentions, lateral/axis errors, empirical
95%-radius coverage and whether the advised bounded XY increment improved or
worsened alignment. It always reports `robot_ready=false`; source-file hashes
and a method label cannot themselves certify the metrology's calibration.

The mathematical pattern follows ZeroDex's multi-view point grounding and
triangulation, but not its rounded projection/20-pixel RANSAC defaults, which
are unsuitable as unvalidated millimetre tolerances. 3D feature-based
insertion servoing motivates using the key and socket axes rather than the
image's lowest pixel. CAD-constrained pose tracking motivates the longer-term
silhouette/depth refinement. See the [ZeroDex paper](https://arxiv.org/abs/2606.19340),
[Hartley–Sturm triangulation](https://doi.org/10.1006/cviu.1997.0547),
[visual servoing fundamentals](https://doi.org/10.1109/MRA.2006.250573),
[3D-feature insertion visual servoing](https://arxiv.org/abs/2405.18830),
[uncertainty-aware triangulation](https://arxiv.org/abs/2008.01258), and
[FoundationPose](https://arxiv.org/abs/2312.08344).
The tip-plus-axis-line reconstruction, its confidence gate, and VLM prompts
are **our adaptation** of these ideas, not a result experimentally validated
by those papers on this AutoDex rig.

The isolated cylinder BODex 1,000-per-tabletop-scene run, exact filter
sequence, reproducibility commands, and per-socket 20 mm endpoint counts are
in [OFFLINE_CYLINDER_1000.md](OFFLINE_CYLINDER_1000.md). Passing its offline
endpoint is not permission to execute a robot insertion.
The follow-up [grasp-fidelity audit](OFFLINE_CYLINDER_1000.md#post-squeeze-fidelity-audit-do-not-use-nominal-renders-as-success-evidence)
shows that the existing fixed-key/commanded-squeeze cylinder images are
diagnostic only: they do not depict achieved MuJoCo grasp geometry. Do not
promote the 13 socket-clear candidates to robot trials from those images.

The NAS handoff at
`/mnt/paradex2/hyunsu/autodex_precision_insertion_handoff_20261010_652ac909`
contains large per-object `pending_foundpose/.../repre.pth` onboarding
artifacts, but its own README labels them **validation pending**: no real
AutoDex key/socket camera images were used to check masks, axes, and open-rim
direction. They have not been promoted into canonical runtime
`foundpose_assets`. No grasp-specific *physical* key–hand calibration record
was found in that handoff. Consequently the current retry route still needs
an observed held-key pose **to reach live retry preflight**. A separate
nominal-key diagnostic can ask the VLM for a direction without that pose, but
cannot authorize a retry. The new bounded post-lift planning route can use a
grasp-specific physical calibration **if one is independently commissioned**;
no such record or future-trial error bound is available here, so this route
is not eligible for robot execution. File presence and synthetic-template counts do
not establish live perception accuracy.

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

## Evidence-only session supervisor

`precision_insertion/session_runner.py` now links the existing pieces into a
single frozen-socket session **without connecting to a robot**. Construct
`SessionRunner(mode=..., calibration=..., catalog=..., shared_root=...,
output_dir=<new directory>, max_xy_retries=...)` only after the board/socket
bootstrap and a complete endpoint catalogue. The constructor binds hashes of
that session and catalogue. For each new key observation:

1. Save the admitted multi-view image/mask/pose bundle with
   `write_key_capture_artifacts`; pass its directory as `key_evidence_dir` to
   `preflight_next_key` along with the measured 13-joint state, its timestamp,
   the existing AutoDex planner and commissioned planning limits. The runner
   verifies the saved frame bundle before calling `plan_admitted_key_trial`.
2. Inspect `current_decision()`. A passing pickup/transfer/20 mm **plan**
   yields `execution_gate_required`, not a motor command. Only then call
   `begin_selected_attempt(attempt_id=..., started_at_s=...)` to create an
   unlabelled attempt record. Actual physical execution requires a separate
   commissioned adapter and the named safety gates.
3. After the physical lift, `postlift_candidate_pose_prior` can use measured
   Franka/Inspire feedback and the selected v8 grasp to form a **loose**
   search prior *before* the grasp-success verdict. If the key remains visible,
   admit a new key capture
   with `admit_postlift_key_capture` (phase `held_postlift`) and save it with
   `write_key_capture_artifacts`. Call `prepare_observed_lift_label` with
   that bundle, the same measured joint sample, a ZeroDex-compatible VLM
   backend, commissioned camera/rise limits, and a completed-lift log. It
   uses the **saved** tabletop and post-lift raw frames, requires two paired
   camera views, and stores prompt/raw response plus source image hashes.
   A decisive visual/observed-key-rise result records `grasp_success`;
   conflict or abstention leaves the label unknown. If the hand occludes
   FoundPose, save a raw `after_lift` bundle with `write_raw_camera_capture`
   and call `prepare_raw_lift_label` instead. It compares the same cameras'
   before/after pixels and records held/miss/slip only when at least two
   views show the key; occlusion is unknown. This raw route supplies **no**
   key–hand transform and cannot authorize transfer by itself. A raw positive
   leads to `held_relation_evidence_required`, not a transfer command. If a
   separately measured *physical* calibration for the exact v8 grasp and a
   commissioned future-trial surface bound are available, call
   `prepare_bounded_postlift_transfer` with fresh stationary measured joints.
   It re-screens the 20 mm endpoint, replans from that state, and audits sampled
   clearance margins without claiming a newly observed 6D key pose. See
   [BOUNDED_POSTLIFT.md](BOUNDED_POSTLIFT.md). The candidate prior
   alone never proves that the key was held. With a pose-observed success, call
   `prepare_postlift_transfer` with that bundle and the *same* measured joint
   sample. It verifies the candidate/prior/frame binding, replaces the
   nominal grasp relation with the observed one, screens the measured hand
   against the centered 20 mm socket endpoint, and saves a fresh held-path
   plan. A rejected observation may be retried only with newer frames. A
   passing report changes the supervisor's next action to
   `transfer_execution_gate_required`; it is a plan, **not** an arrival label
   or motion permit.
   Following separately controlled transfer, run
   `prepare_observed_preinsert_label` on a fresh raw camera bundle, measured
   stationary joints and external transfer log. `preinsert_reached=True`
   requires its saved positive multi-view checkpoint, the exact unchanged
   post-lift plan, and bound trajectory/key/socket/grip evidence. For the
   20 mm outcome,
   collect a later raw full-frame multi-camera capture even if FoundPose fails,
   save it with `write_final_insertion_capture`, and call
   `prepare_observed_insertion_label` with that final bundle, a
   guarded-execution metric record, and a ZeroDex-compatible VLM. The before
   image can be either a saved `held_preinsert` FoundPose bundle or a raw
   same-request capture saved with `write_preinsert_raw_capture`. For a raw
   before image, put its `manifest.json` in the `preinsert_reached` event's
   `preinsert_image` ref; `key_socket_pose` remains a separate kinematic/pose
   evidence ref. For a pose-bound before image, `key_socket_pose` points to
   its `key_observation.json`.
   This hashes the raw frames, prompt/response and numeric source records
   before recording `insertion_success`. Direct `observe_insertion` now
   requires that same verified checkpoint; arbitrary path strings cannot
   create a session task label. A missed grasp excludes that candidate on the next *fresh*
   key observation. Planning-only rejects are skipped only when continuing
   the same budget-limited camera capture; a new pose/state may make them
   viable. Do not call `observe_stage("reset_success", True, ...)` with a
   path string: it is rejected. For a supervised manual return, save a
   recovery log and a **new** admitted tabletop key capture, then call
   `observe_reset_landing`. It checks same tabletop class, a commissioned
   maximum XY center shift from the trial-start pose, measured table support,
   board footprint and socket clearance before recording success or failure.
   After a successful insertion the log must explicitly state that the key
   was removed from the socket. This is observed/manual recovery, not an
   automatic extraction or reset motion.
4. After an observed insertion failure and an externally logged guarded
   withdrawal, call `prepare_observed_xy_retry` with a **held-preinsert** key
   capture, the exact same full-frame VLM images, measured Franka/Inspire
   state and explicit commissioning limits. It derives the key–hand relation,
   screens 1 mm offsets with the measured finger joints, votes across views,
   replans from the withdrawn live state and saves source images/overlays.
   Pass the exact saved **passing post-lift preflight report** referenced by
   the observed pre-insertion stage: its held relation, current wrist FK and
   the held-key admission prior must agree before the VLM is consulted.
   Only a passing result calls `record_retry` to log a pending retry; neither
   method sends the robot an XY command.
   If the withdrawn key is too occluded for another FoundPose estimate,
   `prepare_unobserved_xy_diagnostic` can instead save same-request raw
   camera frames, a measured 13-joint sample, the matching failed attempt,
   guarded withdrawal and a multi-view VLM direction under
   `xy_retry_assessments/NNN/`. It shares the failed-attempt and withdrawal
   checks with the observed retry route, but does **not** produce a live
   preflight, a pending `xy_retry` event or a robot command. Its nominal
   key/socket collision is diagnostic only because squeeze may have shifted
   the actual key relative to the commanded hand target.
   For the cylindrical key, `prepare_grounded_xy_diagnostic` accepts either
   the observed post-lift plan or a re-verified bounded physical-calibration
   plan after the same guarded-withdrawal and camera checks. It asks each
   native-pixel view for a visible tip centre and shaft axis line, triangulates
   the tip/axis, and calculates a continuous socket-frame correction with an
   at-most-1 mm increment. Missing landmarks, weak parallax, tilt, or high
   uncertainty abstain. This is still a **diagnostic**, not a pending retry:
   a passing, saved cylinder diagnostic can now be source-bound to the
   first withdrawn-hold lateral cuRobo/CAD preflight with physical-medoid
   surface bounds. This is still not an execution or renewed contact retry:
   post-shift observation and the new 20 mm endpoint must be checked first.
   The square key additionally needs insertion yaw and therefore abstains
   with only an axial line.
5. If the pose's candidate pool is exhausted, `current_decision()` returns
   `preflight_repose`. Call `preflight_repose` with the **same saved key capture**,
   synchronized measured joints, a listed target tabletop stem, directed v8
   reset height/seed assets, explicit release XY and commissioned geometric
   limits. This reuses `preflight_v8_repose_transition` and stores its input
   files and paths. Only a complete nominal pickup/held/release/retreat plan
   can produce `repose_execution_gate_required`; a held path with no release
   remains planning-only. `begin_repose_attempt` starts a separate, unlabeled
   record. After physical execution, `observe_repose_landing` needs a logged
   release-completion time and a **newer** saved multi-view key capture: it
   checks the target tabletop class, measured
   table support, ChArUco footprint and socket clearance before recording
   `reorient_success`. A wrong class is failure; missing/ambiguous support
   evidence keeps the outcome unknown for review. No result is inferred from
   the reset plan alone.

A supervised reset log is JSON with schema
`precision_insertion_supervised_reset_v1`, the exact `attempt_id`,
`method: "supervised_manual_return"`, a nonempty `reviewed_by`, a
`completed_at_s` later than the attempt's last event, and boolean
`socket_clear`/`hand_open` values of `true`. If insertion was attempted,
`key_removed_from_socket` must also be `true`. These are operator assertions,
not sensor proof; `observe_reset_landing` independently checks the returned
key's fresh multi-view pose and fixed-socket/table geometry.

The lift adapter must provide a JSON execution log with schema
`precision_insertion_lift_execution_v1`, matching `attempt_id` and
`candidate_id`, `trajectory_complete: true`, `force_abort: false`, and
`completed_at_s` between attempt start and the after-lift exposure.
`prepare_observed_lift_label` hashes that log separately from the VLM report.
`prepare_raw_lift_label` uses the same log and an `after_lift` raw capture
manifest, but no post-lift FoundPose. A visible two-view result is a VLM
label, not a 6D key pose or physical contact proof; first physical trials
need human review of false positives and occlusion frequency.
The physical lift completion is the attempt event time; the later VLM
decision time is retained in the report. This is still an external execution
assertion, not a robot-control interface. A clear miss with no admitted
held-key FoundPose cannot yet use this positive-evidence pathway; its
raw-frame failure observation needs a separate binding.

The guarded-insertion adapter must provide a JSON record with schema
`precision_insertion_guarded_execution_v1`, matching `attempt_id` and
`candidate_id`, physical `started_at_s` and `completed_at_s`, and a
`measurement` object containing exactly `key_depth_interval_m` (lower/upper
conservative **key** depth or `null`), `key_depth_source`,
`alignment_within_limits`, `safety_abort`, and `grasp_held`. Its
`source_records` must contain `key_depth`, `alignment`, `force_trace`, and
`grasp_state`, each as `{ "path": "/absolute/path", "sha256": "..." }`.
The check hashes these files but cannot establish that their contents came
from calibrated sensors; the robot-side producer, its timing and uncertainty
model still require commissioning. `FinalInsertionCapture` carries raw frames
for either before or after phase; matching camera request/frame IDs, raw BGR
full frames and camera-acquisition metadata are required;
the stock AutoDex daemons do not currently supply this complete provenance.
Only camera views present in both captures are compared, and a positive
visual class needs at least two supporting views. Hidden/ambiguous images do
not prove insertion. A command stroke alone is not an allowed key-depth
source. The output remains read-only and cannot start guarded contact.
The observed `prepare_postlift_transfer` still insists on a fresh held-key
FoundPose relation. The alternative `prepare_bounded_postlift_transfer` accepts
only an exact-candidate physical calibration, future-trial bound and fresh
measured robot state; it is not a motion permit. The raw insertion checkpoint
does **not** remove the held-key FoundPose requirement in
`prepare_observed_xy_retry`. Hand occlusion remains a retry feasibility risk;
do not silently substitute the BODex nominal relation. MuJoCo's achieved
squeeze relation is a better simulation prior, but hardware drift and camera
visibility need grasp-specific calibration or another bounded relation
estimate before a physical transfer is authorized.

### Occluded key after grasp: calibration route (not yet a transfer permit)

Re-estimating the full key pose after every pickup should **not** be a required
runtime step: Inspire can hide most of it. The intended alternative is to
retain the tabletop key pose for planning the pickup, classify `held`/`miss`/
`slip`/`unknown` from paired pre/post-lift raw views and measured hand state,
and predict the held key from measured wrist FK and a **grasp-specific physical
key–hand calibration**. Visual `held` does not itself measure that transform.
For commissioning, temporarily use a viewpoint, fiducial or external tracker
that independently observes the key during several distinct physical pickups
of the *same* v8 candidate; log paired key pose, measured wrist and Inspire
joints, measurement-error bounds and hashed source evidence. The read-only
`physical_grasp_calibration.calibrate_physical_held_relation` summarizes those
samples using a symmetry-aware measured medoid. It explicitly rejects
MuJoCo/nominal samples and never reports `robot_ready`. Its empirical spread
is descriptive, **not** a guaranteed future-trial error bound. The new
`prepare_bounded_postlift_transfer` uses the verified medoid with an
independently supplied bound for endpoint and sampled-path planning, while
`prepare_postlift_transfer` and the live XY retry still require the existing
independent held-key observation. No real calibration samples
or commissioned error limits have been supplied yet.
`verify_physical_held_relation` can later reconstruct the summary from the
hashed source JSON files and the currently selected v8 `wrist_se3.npy`;
changing either fails. This checks file consistency, not that the data were
truly acquired from a physical robot or that claimed measurement bounds hold.

Before permitting the occlusion route to drive a robot, separately validate
repeatability on held-out physical pickups, bind the exact candidate and
gripper command to that calibration, bound camera/board/socket/FK/grasp/slip
error jointly, and re-screen the full key plus hand at 20 mm and along the
fresh planned path with that uncertainty margin. Reject or pause when measured
hand joints or visual cues differ from the calibrated grasp, the key disappears
without independent grip evidence, or the uncertainty exceeds the socket
clearance. At pre-insertion, VLM can flag gross misalignment or slip; it must
not turn hidden pixels into a millimetre-accurate key pose. Guarded contact
and observed depth/force remain necessary for insertion success.

`uncertainty_margin.py` now provides the **read-only sampled-distance portion**
of that future gate. Given a passing 20 mm endpoint, passing sampled path,
matching key/socket/CAD hashes, and externally commissioned maximum surface
deviations for the key and hand relative to the fixed fixture, it rejects any
nominal clearance smaller than the required baseline plus that deviation. A
helper converts a bounded error in `T_key_hand` into a key-surface displacement
using the *whole key's* distance from the hand frame, rather than just the key
center. The empirical spread in `physical_grasp_calibration.py` is explicitly
**not** an admissible future-trial bound. This audit is not wired into the
transfer gate yet, because no independently justified future-trial bounds or
physical calibration exist; it cannot establish swept-volume or contact
safety from sampled nominal distances.

The supervisor writes a new `session_run.json`, immutable copies of the
frozen calibration and endpoint catalogue, numbered
`trial_preflights/<n>/` reports with hashed key-evidence bindings, and
`repose_preflights/<n>/` reports and
`attempts/<id>/state_<n>.json` immutable label snapshots. A failed process
does not overwrite earlier evidence; there is not yet an automatic resume
loader. Passing a file path as an evidence reference asserts that the caller
actually collected it: the supervisor does **not** establish physical truth
from filenames. Neither this bookkeeping nor a green offline plan replaces
post-lift measured key/hand preflight, guarded contact control, calibrated
camera acquisition, recovery execution or real-robot validation.

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
  demo's precision gate. The demo-local collectors now require the exact
  frame ID, decoded-image digest, acquisition-time method and bounded clock
  error; unchanged AutoDex daemons fail closed. The needed capture-PC and
  robot-PC adapter work is specified in
  [CAMERA_FRAME_HANDOFF.md](CAMERA_FRAME_HANDOFF.md). Never fabricate
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
- `key_perception.py` has separate tabletop and held-key admission. Tabletop
  `admit_key_capture` rejects SAM masks that cover the frozen socket's camera
  projection; this prevents selecting the socket as the key before grasping.
  At pre-insertion hold, overlap is expected, so `admit_held_key_capture`
  instead requires synchronized per-view FoundPose agreement, a sufficiently
  good AutoDex silhouette IoU, and a tightly bounded key-pose prior from
  **measured wrist feedback plus an already observed held relation**. The
  prior source is an asserted caller contract until the live adapter binds
  those inputs. Both phases retain exact frame evidence and the frozen camera
  calibration. A held-key observation cannot start a new tabletop trial.
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
  In the observed-key route, exact key/socket/hand endpoint geometry screens
  them before VLM voting. In the unobserved-key diagnostic route, the key's
  nominal CAD collision is **not** a sound veto on squeeze-induced XY error:
  only the commanded hand/socket endpoint is checked and the VLM may give a
  direction, with no live preflight or motion permit. The selected live path
  and guarded contact still need separate checks before robot movement. A `propose`
  result is not motion authorization or evidence of insertion success.
- `xy_overlay.py` projects the already screened 1 mm candidate centers into
  calibrated AutoDex views and produces matching raw/annotated 16:9 crops.
  It checks candidate separation in the **original camera pixels** before
  display enlargement. If fewer than two cameras resolve the offsets, the
  VLM is not called. Cropping does not create missing visual information;
  current overlays are center anchors, not rendered CAD silhouettes.
- `held_scene_overlay.py` is a separate **read-only diagnostic adapter** for
  the proposed raw-versus-predicted-mesh VLM input. Given synchronized measured
  13-joint feedback, the session-frozen `T_robot_socket`, an explicitly sourced
  `T_key_hand` hypothesis, undistorted raw BGR frames, matching intrinsics and
  `T_camera_robot`, it assembles the existing FR3/Inspire URDF visual meshes
  plus the key/socket OBJ meshes. It reuses AutoDex's
  `src.visualization.overlay_robot_video.RobotOverlayRenderer` for translucent
  multi-view overlays; stock execution code is unchanged. The key projection
  is `T_robot_hand(measured FK) @ inverse(T_key_hand hypothesis)`, **not** a
  fresh observation of a hidden key. Feed both untouched raw and overlay images
  to an observer; never infer a millimetre offset or insertion success from the
  overlay alone. The renderer depth-tests the synthetic meshes against each
  other but has **no measured scene depth**, so real table/hand/object
  occlusions may be drawn incorrectly. The caller must establish same-session
  image/calibration provenance and the physical validity of the held relation.
  `v8_nominal_diagnostic` and `mujoco_achieved_diagnostic` are illustration
  sources, not calibrated transfer estimates. `verified_physical_grasp_calibration`
  is only a source label here, not a proof that uncertainty is acceptable.
  `build_held_scene_comparison` converts same-pixel raw/overlay frames into
  per-camera VLM pairs and records their decoded-pixel hashes. The timestamps
  and calibration remain caller claims until checked against an admitted
  same-session camera capture. `observer.observe_preinsert_hold_views` asks
  each camera separately whether the *visible* key is still held and is
  coarsely consistent with the predicted socket approach. Two synchronized
  views must agree; conflict, occlusion or malformed output returns `unknown`.
  Even `coarse_match` is **not** the `preinsert_reached` label: that also needs
  verified trajectory completion, measured stationary arm/hand feedback,
  bounded key–hand relation and calibrated key/socket pose residual. This
  observer cannot replace the endpoint and live-path preflight gates.
  `preinsert_checkpoint.assess_preinsert_checkpoint` now fuses its visual
  assessment with a matching saved post-lift plan, same-session frozen socket
  and camera calibration, provenance-verified raw preinsert bundle, measured
  stationary Franka/Inspire joints, target hand-pose residual and an external
  transfer execution log. The log schema is
  `precision_insertion_transfer_execution_v1`: matching attempt/candidate IDs,
  `started_at_s`, `completed_at_s`, tri-state `measurement` fields
  `trajectory_complete`, `safety_abort`, `grasp_held`, and hashed absolute
  `source_records` for `trajectory_feedback`, `safety`, `grasp_state`.
  Visible two-view coarse alignment plus passing measured/log gates can yield
  a provisional `preinsert_reached=True` checkpoint; occlusion gives unknown.
  `write_preinsert_checkpoint` saves the exact overlay PNGs and hashes, and
  `verify_preinsert_checkpoint` rechecks saved frames, overlays, transfer
  sources and the plan file. After a separately controlled transfer,
  `SessionRunner.prepare_observed_preinsert_label(...)` creates this report.
  A repeated VLM call must use genuinely newer camera exposures; reusing the
  same frame IDs/pixels or an overlapping exposure interval is rejected even
  if the capture directory has a new name. Unknown assessments may be retried
  only with fresh synchronized images.
  `observe_stage("preinsert_reached", True, ...)` now requires the *same*
  positive report, saved transfer log, raw preinsert manifest, frozen
  post-lift plan and non-early observation time. Use its report path for both
  `preinsert_checkpoint` and `key_socket_pose`, its transfer-log path for
  `trajectory` and `grasp_state`, and its raw bundle's `manifest.json` for
  `preinsert_image`. The subsequent insertion checkpoint compares against
  that same raw before-image; an arbitrary path string no longer creates a
  positive arrival label through `SessionRunner`. This remains a read-only
  evidence gate, not proof that the controller/clock/sensor producer is
  authentic or that the grasp stayed rigid after the post-lift estimate.
  Local CPU tests cover assembly and adapter validation with a fake renderer;
  actual GPU rendering and optical alignment still require the AutoDex PC,
  its nvdiffrast/ParaDex dependencies, and calibrated live frames.
- `xy_retry.py` combines the existing v8 pose/endpoint catalogue gate, fresh
  20 mm geometry screens, camera projections, per-view ZeroDex-style VLM
  choices and strict multi-view consensus. It runs only after an observed
  insertion failure and caller-supplied evidence that guarded withdrawal
  finished while the key remains held. It compares the multiview-key/live-
  wrist-derived `T_key_hand` against the v8 grasp using commissioned drift
  limits, then screens the actual observed rigid relation at each XY target;
  a large drift stops the retry. The live retry screen now uses the measured
  Inspire finger joints paired with that relation, and the withdrawn-state
  preflight rejects a changed hand pose before rescreening the target.
  The multi-view retry now also binds every full-frame VLM image to its
  request/frame ID, decoded-pixel hash and bounded acquisition time. A missing
  capture-side provenance producer or excessive worst-case skew/age prevents
  voting. It also rejects a camera intrinsic/extrinsic different from the
  session-frozen rig; `CAMERA_FRAME_HANDOFF.md` applies to retry images.
  It returns a **proposal requiring new live preflight**, never a Franka
  command or a claim of insertion success. For a gap smaller than 1 mm, all
  1 mm endpoint offsets may be geometrically impossible; that correctly
  produces `no_safe_direction`, not an override from the VLM.
  `assess_xy_retry(..., observed_T_key_hand=None,
  observed_key_hand_source="v8_nominal_unobserved_key")` is a **diagnostic
  alternative** for the squeeze-offset hypothesis. It still requires a
  verified failed insertion, completed guarded withdrawal, held-grasp status,
  measured Inspire joints, current catalogue, fresh camera frames and frozen
  camera calibration. Each candidate must explicitly clear the nominal
  **hand/socket** endpoint, but the nominal key/socket intersection is only
  reported, not used to reject a direction: the actual squeezed key pose is
  unknown. A two-view VLM choice returns
  `diagnostic_xy_hypothesis_only`, never
  `proposal_requires_live_preflight`. The session runner deliberately has no
  route that promotes this unknown-key advice to `record_retry` or robot
  execution. A physical trial needs commissioned uncertainty bounds,
  full-path hand clearance, contact limits and observed 20 mm depth.
- `retry_session.py` binds the preceding failed insertion, caller-logged
  withdrawal, admitted held-key FoundPose, unchanged full-frame VLM pixels,
  measured 13-joint robot feedback and frozen camera calibration. It derives
  the held relation modulo object symmetry, calls `xy_retry.py`, then reuses
  `retry_preflight.py` for the selected target. A no-direction, abstain, slip,
  stale frame, changed hand or failed path leaves the retry unrecorded; the
  assessment is saved for diagnosis. Withdrawal logs are still caller
  assertions until the guarded executor supplies them. The required JSON
  contract is `precision_insertion_guarded_withdrawal_v1` with matching
  attempt/candidate IDs, `completed_at_s`,
  `status=withdrawn_to_preinsert_hold`, `key_still_held=true`,
  `safety_abort=false`, and `source=commissioned_guarded_controller`.
  Its separate `assess_unobserved_xy_diagnostic` path reuses those attempt,
  plan, state, camera and withdrawal checks, but deliberately stops after
  saving the VLM direction. The session runner cannot promote this result
  into `record_retry`.

- Point/axis-grounded alignment is implemented as a **read-only cylinder
  diagnostic** in `grounded_alignment.py` and `retry_session.py`, described
  above. It uses raw synchronized views, a frozen socket coordinate system,
  triangulated tip and multi-view projected shaft lines. It rejects poor
  parallax, reprojection, axis tilt, uncertainty or a continuous XY increment
  that cannot confidently improve the lateral residual. Square-key yaw still lacks a
  non-collinear feature, so square mode abstains. The independent held-out
  evaluator is implemented, but no real AutoDex-camera error dataset or
  externally measured held-key pose has been supplied. CAD silhouette/depth
  refinement and guarded robot execution remain future work; the diagnostic
  is not a motion permit.
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
- `held_relation.py` resolves the lifted key pose against the measured wrist.
  For the cylindrical D∞ key, it removes only axial-yaw and identical-end
  frame ambiguity **about the CAD symmetry center** before measuring drift;
  a real center shift is not erased. The square key keeps its full pose.
- `postlift_preflight.py` binds the selected v8 candidate, frozen session,
  append-only observed grasp label, fresh multi-view key pose and synchronized
  measured 13-DOF Franka/Inspire state. It rejects excessive post-squeeze
  key/hand drift, rechecks the exact centered 20 mm hand/socket endpoint with
  the **measured** finger joints, then reuses `preflight.py` to plan transfer
  and axial descent from the observed lift state. No initial BODex
  `T_key_hand` is silently replayed after physical squeeze. A saved result
  is still a planning report, not contact-control or physical success.
- `bounded_postlift.py` is the separate hidden-key alternative. It verifies
  the raw two-view lift checkpoint and exact-candidate physical calibration,
  rejects an out-of-range measured Inspire pose, then reuses the same endpoint
  screen and held-path planner with a separately commissioned future-trial
  surface bound. Its report and the pre-insertion overlay explicitly label the
  key pose as a calibration hypothesis, not a new observation. No physical
  calibration or such bound is currently available on this PC.
- `live_robot_state.py` reads a fresh Franka state and the Inspire IP
  controller's actual raw motor angles. It reuses AutoDex's
  `convert_inspire_raw` for planner order, rejects stale/asynchronous feedback
  and excessive measured-versus-commanded hand error or arm velocity, then
  supplies a typed `LiveRobotState` to the post-lift preflight. Do **not** use
  `FrankaExecutor.get_hand_qpos()` for this: it returns `commanded_nominal`.
  Franka `get_data()["time"]` is controller uptime, so this adapter waits for
  an advanced state and stamps robot-PC receipt time; Inspire time is a
  software read time. Their skew check does not prove hardware-level
  simultaneity or camera synchronization. Thresholds must be commissioned.
- `live_capture.collect_key_capture()` reuses the socket capture's strict
  same-request image/ID/hash/exposure-time gate after FoundPose has been
  reinitialized for the selected key's v8 representation. Use a key-specific
  SAM prompt; the loose key and fixed socket must not be conflated. Stock
  camera daemons still lack the required provenance output, so this is an
  adapter contract rather than a live runnable capture command.
- `key_perception.admit_key_capture()` reuses the existing AutoDex per-view
  SAM/FoundPose quality gate, rejects disagreeing multi-view key poses, and
  calls unchanged `InitOrchestrator.refine_from_payloads` in IoU/silhouette
  mode. It checks the current camera matrices against the **frozen session
  camera snapshot**, projects the session-frozen exact socket collision mesh
  into each view, and excludes a view when its proposed key mask covers too
  much of that projected socket. The overlap fraction and pixel dilation
  are explicit commissioning limits. A separately commissioned minimum
  selected-mask/refined-CAD IoU is also required for tabletop keys; the stock
  whole-image silhouette loss is weak evidence for a small object. For the D∞
  cylinder, it compares
  physical centers and unoriented axes; the square key retains full
  orientation. It returns the selected
  key pose and the uncertainty-expanded interval of **all** accepted camera
  frames. This is not proof that a plausible segmentation chose the right
  object or that hand-eye calibration is sub-millimetre accurate. The socket
  projection is a conservative convex hull: a genuine key close to/behind
  the socket may be rejected, so this is a veto, not a key segmentor.
  `write_key_capture_artifacts(capture, observation, new_dir)` stores the
  same-request images, masks, poses, frame provenance and selected pose in
  an exclusive per-trial evidence bundle;
  `verify_key_capture_artifacts(new_dir)` checks it before replay/handoff.
  `admit_postlift_key_capture()` uses the selected BODex relation and
  measured wrist only as a bounded first-lift search prior; it checks
  multi-view agreement and mask/silhouette IoU without the tabletop socket
  mask veto. This distinct `held_postlift` phase cannot be substituted for
  `held_preinsert` in the VLM retry. Set prior drift and timing limits from
  measured hardware; a tight limit may reject genuine squeeze drift, while
  a loose limit cannot itself establish that the hand truly holds the key.
- `lift_checkpoint.py` consumes the exact saved tabletop/post-lift key bundles,
  validates paired camera IDs, pixel hashes, exposure order, measured joint
  alignment, the selected candidate prior and observed key-center rise, then
  persists the VLM prompt/raw answer and source PNG hashes. Its verifier
  rejects changed input captures before a positive session label is written.
  An ambiguous result remains unlabelled so newer frames can be assessed.
- `trial_preflight.plan_admitted_key_trial()` first aligns the measured robot
  state to that whole key-capture interval, then calls the existing
  `plan_fresh_key_trial()` candidate filter and pickup-to-20 mm preflight.
  The latter direct API remains available for explicitly marked saved
  offline replays; use the admitted wrapper for a live capture.

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
- `records.py` keeps separate observed grasp, pre-insertion and insertion
  labels. An XY retry record now requires an actual `false` insertion verdict,
  a still-held key, no safety abort, two different voting cameras, and a
  choice ID matching exactly one 1 mm socket-frame cardinal move. `null`
  insertion evidence cannot trigger a retry.
- `session_policy.py` connects existing trial preflight and attempt records to
  the next **evidence gate**. It routes failed grasp to fresh key observation
  and candidate exclusion, failed transfer to recovery, held/no-abort
  insertion failure to guarded withdrawal and 1 mm XY assessment, and pose
  exhaustion to a separate repose preflight. An explicit retry-count limit
  is required. Its `execution_gate_required` result does not authorize a
  robot command; the live capture/executor and commissioned F/T gates are
  still absent.

For a physically stationary post-lift hold, reuse the executor's already-open
ParaDex controller handles; do not construct a second controller. The call is
read-only:

```python
from precision_insertion.live_robot_state import read_live_franka_inspire_state

joint_sample = read_live_franka_inspire_state(
    arm=executor.arm, hand=executor.hand,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_sample_age_s=commissioned_state_age_s,
    max_hand_command_error_raw=commissioned_hand_tracking_error_raw,
    max_arm_update_wait_s=commissioned_state_update_wait_s,
    max_arm_velocity_rad_s=commissioned_stationary_velocity_rad_s,
)
```

The five limits are experiment-specific measurements, not values inferred by
this helper. Pass `joint_sample` to `plan_postlift_observed_transfer` along
with the same arm–hand skew, hand-error and velocity limits for boundary
revalidation. The sample's single wrench vector is diagnostic only; guarded
insertion needs a commissioned continuous trace and abort path.

For a saved `preflight-trial` report, the decision API can be inspected without
connecting to the robot:

```python
import json
from precision_insertion.session_policy import decide_after_trial_preflight

report = json.loads(open("trial/report.json", encoding="utf-8").read())
print(decide_after_trial_preflight(report).to_record())
```

After actual stage observations have been appended to an `AttemptRecord`, use
`decide_after_attempt(attempt, max_xy_retries=<commissioned_limit>)`; this is
an in-memory policy API, not a replay parser or live execution loop.

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
controller, true acquisition-timestamped camera adapter, measured-state
lift/transfer execution, and reset/repose execution remain to be implemented.

After the physical lift, use the session runner's
`postlift_candidate_pose_prior(...)` and `admit_postlift_key_capture(...)`
before assigning a grasp-success label. Only after a separate observed
`AttemptRecord.grasp_success=True` may the runner call
`prepare_postlift_transfer(...)`, or, after a raw positive lift and independent
physical grasp commissioning, `prepare_bounded_postlift_transfer(...)`. The
observed route reuses
`plan_postlift_observed_transfer(...)` and
`write_postlift_preflight(...)`; it saves each rejected/passing observation
under `attempts/<id>/postlift_preflights/<index>/` with a hash-bound key
capture. Only `sampled_postlift_preflight_pass` can proceed to a separately
guarded transfer gate. This in-memory API is not yet a CLI because no live
acquisition adapter or full-task robot executor has been commissioned. The old
nominal preflight remains a candidate-selection estimate, not the motion
plan to replay after squeeze.

The demo-only [pickup execution boundary](PICKUP_EXECUTION.md) accepts the
v8 `execute(..., skip_lift=True, start_from_current=True)` contract and
verifies the saved approach, measured 13-DOF start, live interlock and
post-squeeze feedback. It now **rejects the unchanged stock
`FrankaExecutor`**: its follower may make a blocking final move after a
stream stall. A replacement still requires hardware-side velocity-command
expiry and commissioning. The boundary commands **no lift, transfer or
insertion** and does not set `grasp_success`. It has fake-executor tests only,
not a commissioned live-robot launch path.

After a measured squeeze, [the measured-lift replan](MEASURED_LIFT.md)
re-screens the 20 mm endpoint with achieved finger joints and uses existing
v8 cuRobo planning plus the demo's full held-key/hand path audit from the
achieved FR3/Inspire state. It additionally needs a commissioned bound on
key/hand surface error. This is still a saved plan, **not** a lift command;
the stock lift follower cannot be used as a fail-closed contact/reflex monitor.

Do not feed the nominal `plan_insertion_after_pickup`'s separately replanned
`lift_trajectory` straight into the
unchanged `FrankaExecutor.execute(lift_traj_override=...)`: its start-state
gate still refers to the original pickup `lift_preflight` modeled at grasp
joint values. A demo-local execution adapter must compare live arm/hand state
with the **same** held-lift plan before replay, or replan from live state.

For a **saved-image, read-only** observer replay, choose a backend explicitly.
The local option reuses ZeroDex's `main.vlm_base.BaseVLM`; it does not use a
Gemini key or send images to an API. Install its optional dependencies in an
**isolated** venv, not the production AutoDex Python environment:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m venv --system-site-packages \
  ~/.venvs/precision-vlm
~/.venvs/precision-vlm/bin/python -m pip install \
  -r demo/precision-insertion/requirements-local-vlm.txt
export PYTHONPATH="$PWD/demo/precision-insertion:$HOME/realtime_vlm"
```

This example reuses the existing hardware-compatible PyTorch installation.
Check CUDA visibility outside a sandbox that hides the GPU:

```bash
~/.venvs/precision-vlm/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

The first model load downloads public weights into the Hugging Face
cache. For a semantic-only smoke test on *saved* images (not a real trial):

```bash
~/.venvs/precision-vlm/bin/python demo/precision-insertion/probe_vlm.py \
  --backend local --model-id Qwen/Qwen3-VL-2B-Instruct --task lift \
  --view front /path/to/before.png /path/to/after.png \
  --output /tmp/precision-local-lift-probe.json
```

Repeat `--view CAMERA BEFORE_PNG AFTER_PNG` for more cameras. This probe
records ordered source paths and hashes, the exact prompt and raw response,
and a fail-closed parsed label; it refuses to overwrite an existing report.
Its timestamps are synthetic ordering markers, **not** acquisition times.
It performs no live camera capture, metric XY grounding, motion, or physical
success certification. Initial GPU smoke-test findings and caveats are in
[`LOCAL_VLM_VALIDATION.md`](LOCAL_VLM_VALIDATION.md). `--allow-cpu` only
permits a slow exploratory smoke test. For a programmatic checkpoint using
actual capture timestamps:

```python
from precision_insertion.observer import (
    LabeledFrame, load_vlm_backend, observe_lift,
)

backend = load_vlm_backend(
    mode="local", model_id="Qwen/Qwen3-VL-2B-Instruct",
    max_input_size=(640, 640), require_native_pixels=False,
)
result = observe_lift(backend, [
    LabeledFrame("front", "before_grasp", 1.0, before_pil),
    LabeledFrame("front", "after_lift", 2.0, after_pil),
])
print(result.to_record())  # review only; not a robot command
```

To measure local/API model errors rather than relying on one illustrative
render, use the independent-annotation dataset replay in
[`VLM_BENCHMARK.md`](VLM_BENCHMARK.md).

`before_pil` and `after_pil` must be supplied from the same saved trial; the
timestamps above are placeholders. Semantic labels can use ZeroDex's
aspect-preserving resize. **Metric point/axis grounding cannot:** create a
backend with `require_native_pixels=True` and a `max_input_size` at least as
large as every supplied undistorted frame (or implement an explicit crop-to-
original-pixel transform). The backend rejects a frame that ZeroDex would
silently resize; the point-grounding path also rejects a semantic-only local
backend. This safeguards coordinate bookkeeping, not VLM pixel accuracy:
calibration/held-out grounding error limits remain mandatory. Lift and
insertion comparisons reject unpaired camera IDs, duplicate phase frames,
or reversed timestamps; a front
"before" image cannot be compared to a side "after" image as if it tracked
one object. The optional Gemini adapter accepts an already-configured
`google.genai.Client` and a model ID, or explicitly reads `GEMINI_API_KEY` via
`ZeroDexGeminiBackend.from_env(model="YOUR_MODEL_ID")`. Install the optional
`google-genai` package in the execution environment first with
`python -m pip install -r demo/precision-insertion/requirements-gemini.txt`
(the pin matches the local ZeroDex checkout). Unlike the earlier
adapter, Gemini no longer imports ZeroDex's complete voting script, which
also imports unrelated local-model dependencies; it sends the same ordered
PNG images plus one text prompt through the Google Gen AI SDK. The local
adapter still requires the ZeroDex package and model weights. In either case,
no key is saved in trial reports, and calling Gemini sends camera images to
an external API. The configured backend is passed explicitly to the relevant
`SessionRunner` observation method; there is no implicit model/API call.
Neither adapter chooses cameras, creates image crops/overlays, measures key
depth, or executes Franka commands. A future capture adapter must supply those
inputs and log the raw images alongside every VLM response. The saved-image
probe is therefore not a live local-VLM insertion loop.

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
An optional read-only **pose-fidelity** audit applies the exact drift/axis
gate also used by the direct-v8 reset seed loader. Supply *both* limits;
without them, `stable_reset_seed_counts_by_height_cm` means only an
evidence-bound MuJoCo pass, not a key-in-hand pose repeatability pass:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root /home/hyunsu/shared_data --mode cylinder --gap-mm 20 \
  --candidate-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff \
  --max-reset-drift-mm 3 --max-reset-axis-tilt-deg 8 \
  --output /path/to/new_reset_fidelity_diagnostic.json
```

The 3 mm / 8 degree values are **illustrative, not commissioned robot
limits**. In [this local diagnostic](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_fidelity_diagnostic_example_3mm_8deg_20261011.json),
`0_1/191` passes and `1_0/631` fails the pose-fidelity gate despite both
passing the original MuJoCo gravity test. Neither has a verified socket-aware
Franka reset trajectory or a physical repose result. A runtime with these
limits would therefore have no eligible `1 -> 0` reset seed; it must not
assume that the original two MuJoCo passes provide bidirectional coverage.
The 999 KB [local handoff bundle](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_bundle_20261010.tar.gz)
contains both candidates, audit reports, v8 key scenes, key and six cylinder
socket object-processing assets, fixture geometries, and a handoff README.
Its SHA-256 is
`6b83a94026074f925c57567dd754771e01cbfda16491d2e8a10ba647894acc98`.
The archive has **not** been copied to NAS. Scene JSONs embed this host's
absolute mesh/URDF paths; a different AutoDex-host shared root requires
scene regeneration and fresh evidence validation, not a blind path edit.
The demo-local [reset handoff rehydration tool](RESET_HANDOFF_RELOCATION.md)
does this without changing stock code or overwriting canonical reset seeds;
it has been exercised against the actual archive extracted under a different
root. Its output remains a staged grasp pool, not a Franka reset plan.
The six socket CAD/fixture directories have a separate
[socket handoff relocation tool](CYLINDER_SOCKET_HANDOFF_RELOCATION.md),
because their pose template and task geometry also embed this host's absolute
paths. Its default is a read-only preflight; installation never fabricates a
FoundPose representation or a calibrated socket pose.
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
In particular, these stills combine the initial fixed key–hand transform with
the *commanded* squeeze/hold joint pose, not MuJoCo's jointly achieved key and
finger pose. The saved `sim_traj.json` does contain both actual post-squeeze
poses. `grasp_fidelity.achieved_hand_state` extracts their matched
`T_key_hand` and Inspire joints, and `screen_cylinder_achieved_endpoints.py`
already rechecks that pair against every socket at 20 mm. A faithful achieved
still must export **both** values from the same trajectory index before
rendering; changing only the finger angles repeats the visual penetration
artifact. Even such a still is a counterfactual rigid placement at the socket,
not a simulated transfer/insertion or a measured physical grasp.

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
VLM score. `collect_and_admit_socket_capture()` now reuses the stricter
`collect_socket_capture()` adapter and requires an exclusive `capture_root`.
It checks the saved same-request PNGs, sensor frame IDs on both SAM and
FoundPose outputs, pixel hashes and bounded acquisition-time metadata before
the per-view quality gate. The old `camera_times_s`-only side channel is no
longer accepted. Unchanged AutoDex daemons do not expose all these fields;
the demo-local capture-PC handoff in
[CAMERA_FRAME_HANDOFF.md](CAMERA_FRAME_HANDOFF.md) must be commissioned first.

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

`precision_insertion.live_capture` specifies the demo-local adapters that
must wrap AutoDex's snapshot and initialized FoundPose collectors. Call
`collect_board_snapshot()` first, then `collect_socket_capture()` at least
twice while the socket is fixed and the key is absent, then pass the returned
frames/payloads and timestamps to `bootstrap_session()`. Socket frames are
the **same-request undistorted PNGs** written by the stock init daemon; the
adapter waits for those asynchronous files and refuses missing images rather
than taking a later snapshot. Use a unique capture root on a filesystem
shared by the robot PC and capture PCs (for example, a common absolute NAS
mount), and preserve those intermediate directories until the evidence bundle
has been verified. The metadata callback must return the exact request,
frame IDs, image hashes and bounded exposure times. The unchanged stock
orchestrators discard frame IDs, so they deliberately fail this gate until
demo-specific metadata-preserving adapters are deployed. These capture
helpers alone cannot authorize robot motion.

`precision_insertion.live_session_start.start_precision_session()` now joins
those pieces into **one non-motion startup call** on the AutoDex robot PC. It
first checks the v8 socket raw mesh, exact collision mesh and **canonical**
socket FoundPose `repre.pth` at the explicit `shared_root`; pending
synthetic-only PTH files are not used. Cylinder mode also checks that task
geometry and the uncalibrated
socket pose template reference that root's raw mesh, frame contract and
FoundPose path, and that the fixture/object collision CAD bytes and rim frame
agree. A source-PC absolute path now fails **before any camera capture**.
With an already-streaming AutoDex
camera rig and an independently commissioned `AcquisitionTimeProvider`, it
collects the ChArUco board snapshot **before** initializing socket FoundPose,
takes at least two unique-request socket captures while the key is absent,
runs `bootstrap_session()`, freezes the socket in the collision world, writes
the evidence bundle and reloads the saved calibration to check it. Duplicate
request IDs or an existing output directory fail closed; partial capture
files after a failure are retained for review. It does not connect to or
command Franka/Inspire.

```python
from precision_insertion.config import select_mode
from precision_insertion.live_session_start import start_precision_session
from precision_insertion.perception_evidence import SocketViewLimits

mode = select_mode("square", 1.5)
started = start_precision_session(
    mode=mode, shared_root=shared_root,
    snapshot_orchestrator=board_snap, init_orchestrator=init,
    acquisition_metadata_for_request=acquisition_metadata_for_request,
    capture_root=shared_capture_root, evidence_dir=new_session_dir,
    calibrated_camera_ids=active_serials,
    intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
    image_hw=(H, W), c2r=measured_franka_C2R,
    base_scene=base_curobo_scene,
    view_limits=SocketViewLimits(
        minimum_mask_pixels=commissioned_mask_pixels,
        minimum_foundpose_quality=commissioned_foundpose_quality,
        minimum_foundpose_inliers=commissioned_inliers,
        minimum_border_clearance_px=commissioned_border_px,
        maximum_capture_skew_s=commissioned_camera_skew_s),
    socket_prompt="fixed red socket on the ChArUco board",
    socket_capture_count=2,
    board_timeout_s=commissioned_board_timeout_s,
    socket_timeout_s=commissioned_socket_timeout_s,
    max_socket_translation_mm=commissioned_socket_repeatability_mm,
    max_socket_angle_deg=commissioned_socket_repeatability_deg,
)
# started.evidence_dir/session_calibration.json freezes the measured world.
```

`board_snap` and `init` are the metadata-preserving adapters described in
[CAMERA_FRAME_HANDOFF.md](CAMERA_FRAME_HANDOFF.md); all `commissioned_*`
variables need measured values from the AutoDex rig. A later NAS addendum now
contains 35 complete **real 20-camera shots**, but none has verified
per-camera exposure times, repeated session socket measurements, or promoted
canonical FoundPose PTHs. See [NAS_PERCEPTION_QA.md](NAS_PERCEPTION_QA.md)
for the read-only image/evaluation audit. Those saved shots therefore cannot
yet complete this live startup call. Its tests exercise ordering, missing
assets and duplicate-request rejection, not live hardware accuracy.

After a frozen session and a **complete, session-bound** endpoint catalogue
have been loaded into `SessionRunner`,
`precision_insertion.live_key_trial.prepare_next_live_key()` is the next
non-motion boundary. It only accepts the runner's `capture_fresh_key` or
`reobserve_key_and_preflight` decision. It prechecks the v8 key raw mesh and
canonical FoundPose `repre.pth`, initializes the key in the existing AutoDex
FoundPose orchestrator with silhouette refinement, takes a new
acquisition-bound multi-view capture, rejects masks that include the frozen
socket, saves and verifies that capture, checks **measured** 13-DOF
Franka/Inspire feedback against the camera exposure interval, and calls the
runner's existing pose-conditioned v8 pickup/transfer/20 mm preflight.
It does not use the socket FoundPose result as the key pose, reuse an old key
image, mark a physical grasp successful, or send a robot command.

Precision key OBJs use solid-colour MTL files. The unchanged AutoDex
silhouette loader represents them as `TextureVisuals` despite there being no
texture image, which makes FoundationPose's tensor builder fail. This demo
prepares a [compatible local renderer](precision_insertion/silhouette_compat.py)
*before* key daemon initialization: it converts only that no-image visual to
vertex colour, while preserving the same raw OBJ vertices, faces, units and
object frame. `init_object(load_silhouette=False)` still initializes the
unchanged AutoDex FoundPose daemons; it leaves the demo-prepared local
renderer in place for IoU/silhouette refinement. This has been checked on all
five key raw meshes at the mesh-load level. A full CUDA renderer and live
perception replay still require the AutoDex robot PC's FoundationPose stack
and real-camera validation; this compatibility fix alone does not promote a
pending `repre.pth` or validate a key pose.

```python
from precision_insertion.live_key_trial import prepare_next_live_key

prepared = prepare_next_live_key(
    runner=runner, init_orchestrator=init,
    acquisition_metadata_for_request=acquisition_metadata_for_request,
    state_for_observation=feedback_sampler.state_for_observation,
    capture_root=shared_capture_root,
    key_evidence_dir=new_key_evidence_dir, capture_id="key_001",
    calibrated_camera_ids=active_serials,
    intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
    image_hw=(H, W),
    key_prompt="blue precision key on the board, excluding the fixed red socket",
    view_limits=commissioned_view_limits,
    maximum_multiview_center_error_mm=commissioned_center_mm,
    maximum_multiview_angle_error_deg=commissioned_angle_deg,
    maximum_socket_mask_overlap_fraction=commissioned_socket_overlap,
    socket_projection_dilation_px=commissioned_socket_dilation_px,
    minimum_refinement_iou=commissioned_key_iou,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_hand_command_error_raw=commissioned_hand_error_raw,
    max_arm_velocity_rad_s=commissioned_hold_velocity_rad_s,
    max_key_state_skew_s=commissioned_key_state_skew_s,
    planner=planner, limits=commissioned_path_limits,
    max_pose_error_deg=commissioned_tabletop_pose_error_deg,
    axial_waypoint_step_m=commissioned_axial_step_m,
    capture_timeout_s=commissioned_capture_timeout_s,
    image_write_timeout_s=commissioned_image_write_timeout_s,
)
print(prepared.preflight.status, runner.current_decision().action)
```

Create `feedback_sampler` **after the arm/hand have settled but before the
key capture**. The demo-only `ExposureStateBuffer` and
`RobotFeedbackSampler` can be configured as follows (commission all limits;
keep enough samples to survive the full FoundPose inference time):

```python
from precision_insertion.feedback_buffer import (
    ExposureStateBuffer, RobotFeedbackSampler,
)
from precision_insertion.live_robot_state import read_live_franka_inspire_state

feedback_buffer = ExposureStateBuffer(
    max_samples=commissioned_history_capacity,
    max_bracket_span_s=commissioned_bracket_span_s,
    max_key_state_skew_s=commissioned_key_state_skew_s,
    max_arm_hold_drift_rad=commissioned_arm_drift_rad,
    max_hand_hold_drift_raw=commissioned_hand_drift_raw,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_hand_command_error_raw=commissioned_hand_error_raw,
    max_arm_velocity_rad_s=commissioned_hold_velocity_rad_s,
)
feedback_sampler = RobotFeedbackSampler(
    feedback_buffer,
    lambda: read_live_franka_inspire_state(
        arm=franka, hand=inspire,
        max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
        max_sample_age_s=commissioned_feedback_age_s,
        max_hand_command_error_raw=commissioned_hand_error_raw,
        max_arm_update_wait_s=commissioned_arm_update_wait_s,
        max_arm_velocity_rad_s=commissioned_hold_velocity_rad_s,
    ),
    period_s=commissioned_feedback_period_s,
)
with feedback_sampler:
    feedback_sampler.wait_until_ready(
        timeout_s=commissioned_feedback_warmup_s)
    # Trigger prepare_next_live_key(...) here, passing
    # state_for_observation=feedback_sampler.state_for_observation.
    ...
```

The buffer requires measured feedback samples **before and after every
admitted camera exposure**, checks every intervening sample for arm/hand
motion, and returns an unmodified measured state rather than an invented
interpolation. A missing/wide bracket, stale camera relation, motion or
sampler error rejects preflight. Reading the arm and hand only *after* slow
SAM/FoundPose inference is insufficient. Franka state receipt and Inspire
software-read timestamps are **not hardware-latched joint timestamps**;
their relation to calibrated camera UTC must be commissioned and checked on
the AutoDex rig before using this for physical motion. These read-only
checks never authorize contact control.

When `prepare_next_live_key()` succeeds, `SessionRunner.preflight_next_key()`
also saves `measured_start_state.json` beside the trial report, hashes it in
`key_evidence_binding.json`, and ties both files to the saved camera bundle.
`runner.verify_current_preflight_evidence()` rechecks these hashes, the
session/catalogue identity and, for an actual `TrialPreflight`, the saved
scene/trajectory files. `begin_selected_attempt()` and repose preflight call
that verifier before creating a new attempt or reset plan. This prevents a
changed report from being silently rehashed as a fresh execution input; it
still does **not** certify that a trajectory is safe to command on the live
robot.

An optional `planning_options` mapping accepts only the existing repose
fidelity/root arguments, never runner-owned candidate exclusions. The saved
camera evidence survives a failed state or planner gate for diagnosis. This
API still requires commissioned camera clocks, real key/socket FoundPose
representations, a complete endpoint catalogue and live feedback; there is
no safe robot-execution CLI yet.

Run the current offline tests from the repository root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests
```

The startup call enforces ChArUco before repeated socket captures, with the
socket already rigidly fixed. Its calibration helper does **not** independently
prove the SAM3 mask/FoundPose photometric quality, hand-eye accuracy, or
robot-motion safety. The future trial runner must explicitly select
`--shared-root` and use the matching v8 `object_processing` assets and Inspire
candidates; see `PLAN.md` for the remaining gates. Until those gates are
implemented, there is intentionally no robot-mode command to run here.

The path component `precision-insertion` is a directory name, not an importable
Python package name. If helper modules are added, use an importable package
name such as `precision_insertion` inside this directory.
