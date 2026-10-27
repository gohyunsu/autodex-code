# Precision insertion demo

The current cylinder frame conventions and live gate audit are in
[SYMMETRY_AND_LIVE_GATES.md](SYMMETRY_AND_LIVE_GATES.md).
The current active-route comparison with stock AutoDex and removed
failure thresholds is in
[FAILURE_GATE_AUDIT_20261017.md](FAILURE_GATE_AUDIT_20261017.md).
The post-lift code review and preinsert VLM/overlay overlap are in
[DOWNSTREAM_AUDIT_20261021.md](DOWNSTREAM_AUDIT_20261021.md).
The current pose-transform and saved-key center replay are in
[PIPELINE_CODE_AUDIT_20261026.md](PIPELINE_CODE_AUDIT_20261026.md).
The current fixed-cylinder socket localization and its saved-camera replay
are in [RECOVERY_REVIEW_20261021.md](RECOVERY_REVIEW_20261021.md).

The board-9 robot launcher routes five fixed VLM cameras, one to each
capture PC: capture1/`24122734`, capture2/`25305463`,
capture3/`25322638`, capture5/`25322649`, and capture6/`25305461`.
Classification can finish on the first three consistent replies. All 20
original camera frames remain in the raw capture and FoundPose path.
Classification inputs are 384×288; native-pixel XY grounding uses the
original undistorted frame. The current saved-image probe and active route
are in [TRIAL_20261020.md](TRIAL_20261020.md).
`PRECISION_XY_ESTIMATOR=tracked_tip` selects the separate experimental
key-tip tracker for **failed-trial XY advice only**. It records a short
frame-ID-bound preinsert window from capture2/`25322651` and combines its
temporal tip with the saved multiview snapshot. The normal
lift/preinsert/final VLM labels remain in use. Start its capture-PC service
separately with `bash demo/precision-insertion/manage_robot_pc_services.sh
start-tracker`. See
[LIVE_RUNTIME.md](LIVE_RUNTIME.md) for the exact boundary and limitations.
`PRECISION_XY_ESTIMATOR=sam_half_ellipse` selects failed-trial XY advice
using Qwen key-box grounding and a warmed local AutoDex
SAM3 model to the **same saved preinsert frame**. The ordinary snapshot and
lift/preinsert/final labels remain unchanged. See
[SAM_HALF_ELLIPSE_RUNTIME_20261026.md](SAM_HALF_ELLIPSE_RUNTIME_20261026.md).
With that setting, `run_robot_pc.sh --check` reports the selected estimator
and confirms the SAM Python and checkpoint files exist. It does not load the
model or connect to CUDA, cameras, or the robot.
The operator-stopped approach and its measured timing are recorded in
[STOPPED_TRIAL_20261017.md](STOPPED_TRIAL_20261017.md). The 2026-10-20 live
trial reached transfer; its false positive lift label, transfer stall, and
384×288 saved-image replay are in
[TRIAL_20261020.md](TRIAL_20261020.md). The earlier command-channel wait,
downstream stock comparison, and saved 804-sample geometry replay are in
[LAST_RUN_AUDIT_20261020.md](LAST_RUN_AUDIT_20261020.md).
The later missed-grasp trial, stock-speed restoration, and reset-path repair
are in [TRIAL_20261020_093746.md](TRIAL_20261020_093746.md).
The following trial exposed duplicate FR3 lift slowdown and a stop-go
approach reference; the measured trace, offline replay and controller fix are
in [TRIAL_20261020_094911.md](TRIAL_20261020_094911.md).

## Active cylinder force-stop route (2026-10-13)

The 20 mm cylinder stroke now uses the current AutoDex FR3 executor's
`_follow(..., stop_wrench_z=12.0, abort_on_contact=True)` through
`FrankaForceStopInsertionAdapter`. It keeps the executor's normal
`traj_speed` and held-object speed cap. On detected force, it stops the
stream, follows the already planned socket-axis path backward to preinsert
with the key gripped, records `force_abort`, and sends the held key through
the existing return-to-original-pose action. The FR3 integration now records
that planned held transfer, descent, hand opening and clear-view exit from
measured stock AutoDex waypoints before a fresh landing image. That failure
does not open the hand at the socket. Once the returned key is observed in its
original tabletop pose, the failed attempt reads its saved pre-contact
preinsert frames for next-trial XY diagnosis. The diagnosis never moves the
held key or treats the force stop itself as a measured XY error. Use
`PRECISION_XY_ESTIMATOR=sam_half_ellipse` to select the Qwen-box/SAM3 tip
estimator; the launcher still defaults to `vlm`. If no tip can be measured,
it records an unavailable diagnostic and applies no XY update. The active
cylinder route does not require a separate contact-controller commissioning
record. A normal stroke uses the built-in `StockSocketRelease` callback: it preplans
full hand opening, reversal of the audited socket-axis path, and a stock
clear-view retract using the frozen socket world. It opens the hand only
after a completed non-aborted stroke, checks measured clear-view feedback,
and obtains the same-camera initial/final visual task label. The stock force
record contains
no independent key-penetration measurement; `inside_socket` remains a visual
label, not measured 20 mm depth. The older guarded-contact sections below
describe optional or historical contracts and do not gate this cylinder
route. The FR3 integration also supplies stock v8 restoration/repose callbacks
from the board-9 `robot_pc_stages:configure` motion interlock. Its ten stages
and camera callbacks now assemble offline. The built-in release callback
needs the existing path-audit limits and is not a measured key-depth test.

For the complete AutoDex-PC execution entry point and required integration
module, see [LIVE_RUNTIME.md](LIVE_RUNTIME.md). `run_live.py` keeps all
precision-specific robot-loop code under this demo; stock AutoDex execution
files remain unchanged.
On the board-9 robot PC, manage long-lived daemons separately with
`bash demo/precision-insertion/manage_robot_pc_services.sh start` (or
`check`/`stop`), then set `PRECISION_STAGE_ROOT` to the active cylindrical
recipient stage and run `bash demo/precision-insertion/run_robot_pc.sh --check`
before `--run`. The stage needs `launch.json`, the matching
`recipient_gap{GAP_MM}_endpoint_catalog.json`, and `franka_runtime.json`;
`PRECISION_LAUNCH_TEMPLATE`, `PRECISION_ENDPOINT_CATALOG`, and
`PRECISION_ROBOT_CONFIG` can select explicit current files. No active stage
is installed at the former 2026-10-13 temporary path. A pipeline exit no
longer restarts or stops capture/Franka services.
The default cylinder radial gap is 20 mm; set `PRECISION_GAP_MM` to 1, 3, 5,
10, or 15 for the matching recipient-screened socket catalog after its
file-only `--check`. Each physical socket change needs a new board/socket
capture; the 20 mm session evidence must not be reused for another gap.
Each `--run` copies the board-9 launch template to a uniquely named launch
JSON and points it at a new session-evidence directory, so a prior run's
evidence is never reused or overwritten. `--check` reads the template only.
The key is placed after the fixed socket capture when prompted.
The stock AutoDex default Cartesian planner may put small finger motion in a
held transfer even when its start and goal hand poses match: its default
`lock_hand=True` route still optimizes all 13 joints. The live demo now keeps
the selected held Inspire pose constant in the returned lift, transfer and
socket-axis paths before auditing the exact path it would execute. It no
longer aborts a successful repose plan solely because the raw planner path
has finger jitter (the observed case was 0.014084 rad). Invalid paths, missed
goals and audited collisions remain planning failures. The live CLI applies
the same `quiet_curobo` logging policy as stock AutoDex `run_auto.py`, so
per-solve cuRobo messages do not flood the console; planner failures still
surface through their returned status or exception.
For a board-9, 20 mm key-plus-socket **camera-only** check, use
[`probe_live_key_socket.py`](probe_live_key_socket.py) as documented in
[ROBOT_PC_STAGED_BRINGUP.md](ROBOT_PC_STAGED_BRINGUP.md). It saves every raw
view, SAM mask and FoundPose result without opening the robot controller.
For the separate capture4 ZeroDex profile, `run_zerodex_capture4.sh
--camera-only` additionally runs the frozen-socket session, admitted key
pose, tabletop classification and endpoint-catalog selection without robot
motion; see [ZERODEX_CAPTURE4.md](ZERODEX_CAPTURE4.md).
For an implementation-level comparison with the local stock AutoDex v8
pipeline, source inputs/outputs, active prompts and actual validation scope,
see [CURRENT_V8_PIPELINE_REVIEW.md](CURRENT_V8_PIPELINE_REVIEW.md).
The October 12 FR3/Inspire successful grasp and its recovered right-angle
mount are traced in [ADAPTER_GEOMETRY_AUDIT_20261015.md](ADAPTER_GEOMETRY_AUDIT_20261015.md).
The robot-PC script selects this recovered URDF for planning, live wrist
references, collision geometry and camera overlays. The current NAS URDF
does not reproduce the saved October FK; no NAS or original AutoDex file is
modified by the precision runner.
For run-by-run startup, socket, planning, motion, VLM and NAS-I/O timing,
including the report command and optimization experiments, see
[TIMING_PROFILE.md](TIMING_PROFILE.md).
For the operation-by-operation latency comparison against AutoDex v8 and
remaining avoidable costs, see [LATENCY_COMPARISON.md](LATENCY_COMPARISON.md).

For a concise distinction between **offline-usable**, staging-only and
missing assets, see [OFFLINE_ASSET_STATUS.md](OFFLINE_ASSET_STATUS.md). It
does not treat live-camera/robot commissioning as a missing 3D file.
Presentation-ready 16:9 cylinder animations, their evidence boundary, and
rebuild commands are indexed in
[PRESENTATION_VISUAL_ASSETS.md](PRESENTATION_VISUAL_ASSETS.md).
The active cylinder live loop uses a **next-physical-attempt** cumulative XY
offset after failed insertion. It does not perform lateral correction or
reinsertion within the failed attempt. The current behavior is specified in
[NEXT_TRIAL_XY.md](NEXT_TRIAL_XY.md): 10 mm per-update norm cap, default gain
1.0, no cumulative cap, nominal v8 held relation, and original-pose reset.
Sections below discussing 1 mm same-held-attempt shifts or physical/observed
held relations are historical/optional utilities, **not** the live route.
The older same-attempt lateral-shift tools discussed below are experimental
read-only/opt-in utilities, not the active live retry route.

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
verified post-shift hold, not reuse of the earlier diagnostic. An opt-in
[held XY execution boundary](LATERAL_EXECUTION.md) now exists, but requires an
independently commissioned controller; none is bundled.

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
saved axial path with that API as a contact insertion command. The current
15 mm radial-gap catalogue has 12 **offline simulated** v8 candidates that
pass its nominal 20 mm endpoint screen. Their 4.4–38.5 mm achieved
hand-relative center drift has no commissioned acceptance limit, and the
newly generated local key/socket FoundPose representations still await
real-image pose/mask QA (see [FOUNDPOSE_ONBOARDING.md](FOUNDPOSE_ONBOARDING.md)).
The older NAS `pending_foundpose/` files were not promoted. These are not physical or robot-ready
grasps; a real cylinder replay is still not possible here.

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

The passing post-shift replan contains **two** paths: a non-contact transfer
from the new measured hold, followed by an axial 20 mm path. They cannot be
interchanged: the shifted hold can differ from the axial start. Call
`SessionRunner.prepare_postshift_path_handoff(...)` with that preflight, its
saved report, the same checkpoint/shift plan, and a fresh stationary
`LiveRobotState`. It saves `postshift_path_handoffs/NNN/report.json` with the
hashed archive, both path endpoints, and `transfer_required`. The verifier
replays the checkpoint/CAD/plan source checks and the 20 mm target geometry.
This is a **read-only evidence packet**, not a transfer or contact execution
gate. The existing centered-path `execute_bound_guarded_insertion` must not
be used for this retry. The opt-in
`execute_bound_postshift_transfer(...)` can now run **only the non-contact
transfer** through an injected, independently commissioned controller and
save a replayable execution log. No controller is bundled. The new
`SessionRunner.assess_postshift_transfer_arrival(...)` requires fresh raw
AutoDex views and measured joints *after* that transfer, re-grounds the
cylinder tip/axis against the frozen socket, and saves a read-only arrival
status. If aligned, call
`SessionRunner.prepare_postshift_arrival_axial_replan(...)` with the saved
arrival and prior preflight, the same trial sources, and commissioned
whole-surface and visual-error bounds. It refits the held relation to the
**new** tip/axis, reruns exact 20 mm key/whole-hand endpoint screening and
plans only axial waypoints from the **measured arrival joints**. Its sampled
key/hand and uncertainty audits use the same cylinder yaw gauge; a changed
goal outside the measured pre-insertion tolerance rejects without a second
transfer. The previous axial path is never replayed. This new report remains
read-only. A passing report can be committed through
`SessionRunner.record_grounded_xy_retry(...)`, which checks the saved
multi-view diagnostic, completed shift, fresh arrival and axial preflight,
then records a pending continuous <=1 mm XY retry without moving the robot.
A fresh measured, stationary state can then be bound through
`SessionRunner.prepare_retry_axial_handoff(...)` to the pending event and
the **new axial-only archive**; its exclusive packet is read-only and does
not authorize contact. The opt-in `execute_bound_retry_guarded_insertion(...)`
boundary can use a separately commissioned contact controller after fresh
feedback, watchdog/contact reviews and an explicit live interlock. No such
controller is bundled. `prepare_observed_retry_insertion_label(...)` binds
new paired raw camera frames, a VLM verdict and the retry execution record
to a second insertion event; legacy first-insertion label code rejects retry
records. Physical key-depth admission is still needed before retry success
can be labeled. See
[POSTSHIFT_HANDOFF.md](POSTSHIFT_HANDOFF.md) for the call contract.
Its `verify_retry_guarded_metric(...)` additionally defines the external
contact-record schema and replays the exact retry handoff/trace/source chain;
it does not authenticate physical key penetration or run a robot controller.

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
`/mnt/paradex2/hyunsu/handoff/20261010_handoff_652ac909`
contains large per-object `pending_foundpose/.../repre.pth` onboarding
artifacts, but its own README labels them **validation pending**. A later NAS
addendum and [real-image audit](NAS_PERCEPTION_QA.md) now contain 35 selected
20-camera shots and 61 offline evaluation reports. Those images expose
wrong-object key masks and still lack per-camera exposure timestamps and
independent pose/rim validation. Those older NAS PTHs have therefore **not**
been promoted into canonical runtime `foundpose_assets`. Separate local
798-view representations for the first square and cylinder key/socket pairs
were generated on 2026-10-11, but remain camera-validation pending; see
[FOUNDPOSE_ONBOARDING.md](FOUNDPOSE_ONBOARDING.md). No grasp-specific
*physical* key–hand calibration record was found in that handoff.
Consequently the current retry route still needs
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

`run_live.py` now assembles the opt-in cylinder execution loop. The first-party
[`robot_pc_stages.py`](robot_pc_stages.py) supplies all ten board-9 cylinder
stages and reuses the stock v8 planner, FR3 speed, force-stop,
release and reset callbacks. After socket-only capture it waits for the
operator to place the white key and press Enter. The perception/landing
thresholds beyond the accepted feedback and path proposal are provisional;
the assembled loop has offline test coverage but no physical insertion run.
The first-party noncontact
adapter checks the exact running stock ParaDex daemon identity; no physical
watchdog test has been recorded. The scenario catalog, VLM observer, and
retry policy are evidence and decision helpers, not proof of hardware
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
For promoted cylindrical candidates the catalogue also carries the hashed
MuJoCo `simulated_post_squeeze_fidelity_diagnostic` through pose selection.
It is diagnostic evidence, not a commissioned acceptance threshold or a
measured post-lift key-in-hand relation. Before transfer, the live pipeline
rechecks the endpoint and path from measured post-squeeze joints. The cylinder
route may explicitly use a nominal relation with an unassessed physical
surface margin; the catalogue alone cannot authorize insertion.
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
   and call `prepare_raw_lift_label` instead. The distributed live route
   retains the saved before/after frames and projects the measured hand pose
   through the right-angle adapter URDF and undistorted session cameras to
   crop the after-lift image in two close cameras. One clear `YES` labels
   `held`; two `NO` answers label `miss`; any other pair is unobservable.
   The independent per-camera answers and crop coordinates are recorded.
   The local/API comparison route still uses before/after pairs.
   This raw route supplies **no**
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
   post-lift plan, and bound trajectory/key/socket/grip evidence. The saved
   pre-insertion verifier replays each exact VLM raw answer against its prompt,
   parsed label and image order; a changed JSON label cannot create arrival.
   For the
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
   This binds the raw frames, prompt/response and numeric source records
   before recording a tri-state `insertion_success`. Until a physically
   commissioned key-depth producer and replay verifier are installed, an
   external depth interval is retained for audit but **not admitted** for
   either a positive or depth-driven negative task label; visually normal
   insertion therefore remains unknown. Direct `observe_insertion` now
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
   A verified successful insertion ends this session immediately. The key
   may remain in the socket: release, extraction, reset and a further grasp
   attempt are not dispatched. Any later manual recovery belongs to a new
   session, not `observe_reset_landing` on the successful one. For a failed
   insertion, a supervised return remains evidence-only, not automatic
   extraction or reset motion.
4. In the active cylinder live route, the hand opens and reaches clear view
   before the final visual verdict. After a visible `outside_socket` failure,
   replay the **saved pre-insertion raw views**; do not withdraw or lift the
   key for another image. Multi-view tip/axis reconstruction gives a
   socket-frame residual `d_n`, capped at 10 mm per update. A single-view
   nominal-height estimate and no-tip 2 mm exploration have separate statuses
   and are never labeled as multi-view measurements. Do not move laterally or
   insert again in this attempt. Observe the released key, restore its
   original tabletop pose/position and recognize it again. A new pickup
   of the same grasp and tabletop class uses the accumulated wrist correction
   `p_(n+1)=p_n+w*d_n`, initially zero, with default `w=1.0`. The grasp and
   held-path preflights run again. See [NEXT_TRIAL_XY.md](NEXT_TRIAL_XY.md).
   There is no default retry-count cap or cumulative XY cap. **Square mode has
   no XY error-correction branch**:
   after insertion failure it returns/reobserves the key with effective
   update weight zero. Historical same-held-attempt retry helpers remain in
   source for audit of old experiments but are not assembled into the
   current action loop.
5. If the pose's candidate pool is exhausted, `current_decision()` returns
   `preflight_repose`. Call `preflight_repose` with the **same saved key capture**,
   synchronized measured joints, a listed target tabletop stem, directed v8
   reset height/seed assets, explicit release XY and commissioned geometric
   limits. This reuses `preflight_v8_repose_transition` and stores its input
   files and paths. Only a complete nominal pickup/held/release/retreat plan
   can produce `repose_execution_gate_required`; a held path with no release
   remains planning-only. `begin_repose_attempt` starts a separate, unlabeled
   record. After physical execution, `observe_repose_landing` needs an
   [exact-plan-bound release/exit record](REPOSE_EXECUTION_EVIDENCE.md) and a
   **newer** saved multi-view key capture: it
   checks the target tabletop class, measured
   table support, ChArUco footprint and socket clearance before recording
   `reorient_success`. A wrong but supported class is recorded as failure and
   triggers a fresh key capture and preflight in that *observed* class;
   missing/ambiguous support evidence keeps the outcome unknown for review.
   No result is inferred from
   the reset plan alone. The [synthetic v8 reset diagnostic](SYNTHETIC_REPOSE_VALIDATION.md)
   exercises the staged candidate through the same planner without promoting
   it to a runtime reset asset.

   The active cylinder reset chain uses the **nominal v8** squeeze hand state
   and key-in-hand transform as hypotheses. It does not require or substitute
   a measured post-grasp relation; its preflight and observed landing remain
   separate. The optional `held_hand_source="measured"` planner primitive and
   [reset grasp calibration CLI](RESET_GRASP_CALIBRATION.md) belong to earlier
   experiments and are not assembled into the cylinder live loop.
   For an insertion failure with the key still held, the separate
   [failed-trial held-key return](FAILURE_RETURN.md) preflight starts after
   axial withdrawal from measured joints and reuses the same socket-aware
   transfer/descent and release checks. It requires an explicit tabletop
   destination; opening above the socket is not a reset. Controller execution
   and fresh landing verification are still integration work.

A supervised reset log is JSON with schema
`precision_insertion_supervised_reset_v1`, the exact `attempt_id`,
`method: "supervised_manual_return"`, a nonempty `reviewed_by`, a
`completed_at_s` later than the attempt's last event, and boolean
`socket_clear`/`hand_open` values of `true`. These are operator assertions,
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
`candidate_id` and the frozen `session_calibration_sha256`, physical
`started_at_s` and `completed_at_s`, and a
`measurement` object containing exactly `key_depth_interval_m` (lower/upper
conservative **key** depth or `null`), `key_depth_source`,
`alignment_within_limits`, `safety_abort`, and `grasp_held`. Its
`source_records` must contain `key_depth`, `alignment`, `force_trace`, and
`grasp_state`, each as `{ "path": "/absolute/path", "sha256": "..." }`.
The `force_trace` must now be a replayable `guarded_contact_trace_v1` from
the demo's sample policy, consistent with the metric's attempt, frozen
session, timing and `safety_abort`. The checkpoint target remains exactly
20 mm. This checks internal sample/decision consistency, but **not**
whether those samples came from calibrated sensors; the robot-side producer,
its timing and uncertainty model still require commissioning. The other
three source files must each use schema
`precision_insertion_external_metric_claim_v1` and identify their source,
attempt, candidate, frozen session calibration, producer, acquisition method,
stroke-time measurement and hashed raw evidence. The checkpoint cross-checks
these claims against the guarded summary and referenced files. This is a
consistency/provenance check only: a matching JSON claim and hash do **not**
prove a calibrated physical measurement. In particular, until a commissioned
key-depth replay verifier exists, even a non-null claimed depth is excluded
from the outcome gate and saved as `key_depth_admissibility: not_admitted`.
An optional [exposed-length depth diagnostic](EXPOSED_DEPTH.md) now asks
the VLM for a visible key rear-centre/axis rather than its hidden tip and
calculates a worst-case CAD depth interval from calibrated multi-view
geometry. `saved_exposed_depth.py` binds that diagnostic to verified
final/abort camera frames, frozen calibration and saved VLM text, and can
replay the arithmetic without requerying the model. It is **not** yet
admitted as a physical key-depth source: real landmark/calibration error
bounds and guarded-controller completion binding remain necessary.
`FinalInsertionCapture` carries raw frames
for either before or after phase; matching camera request/frame IDs, raw BGR
full frames and camera-acquisition metadata are required;
the stock AutoDex daemons do not currently supply this complete provenance.
Only camera views present in both captures are compared, and a positive
visual class needs at least two supporting views. Hidden/ambiguous images do
not prove insertion. A command stroke alone is not an allowed key-depth
source. The output remains read-only and cannot start guarded contact.
After a verified positive pre-insertion hold, the read-only
[`guarded_axial_handoff.py`](GUARDED_AXIAL_HANDOFF.md) binds that checkpoint,
the exact observed post-lift 20 mm axial path and a fresh measured
FR3/Inspire start state into one replayable packet. It does **not** command
the arm or prove that the key penetrated the socket; a commissioned guarded
controller and post-stroke evidence remain separate.
The opt-in [guarded insertion execution boundary](GUARDED_AXIAL_HANDOFF.md#opt-in-external-contact-boundary)
now checks this packet, a fresh live start, independent watchdog/contact
commissioning records and an explicit interlock before calling an injected
controller. It requires a replayable v2 metric/trace and terminal robot
feedback afterward. It defaults to **no motion**; no commissioned contact
controller is shipped, and its return never sets insertion success.
A v2 guarded metric and contact trace can cite this exact packet and its
saved trajectory digest; the insertion checkpoint then checks the matching
observed-hold raw capture and records `observed_hold_axial_path`. Legacy v1
metric/trace records remain read-only diagnostics without that path binding.
Neither format authenticates the robot-side producer or admits wrist travel
as physical key depth. See [handoff record contract](GUARDED_AXIAL_HANDOFF.md).
The demo now also has a pure, **non-actuating**
[`GuardedContactMonitor`](precision_insertion/guarded_contact.py) for the
planned 20 mm stroke. With separately commissioned limits and acquisition-
time samples already expressed in the frozen **socket frame**, it latches an
abort on stale/missing samples, Inspire command-tracking loss, axial/lateral
force or moment excess, key-axis/yaw alignment excess, implausible depth
jumps or overshoot. At the nominal 20 mm wrist/FK endpoint it returns
`hold_for_key_depth_and_vlm`, **not** `insertion_success`: nominal progress
does not measure the key after squeeze. The saved VLM checkpoint still needs
an independently bounded physical key-depth source. This monitor does not
transform the FR3 wrench, verify sensor calibration, stop the robot, or
generate the `guarded_execution_v1` metric record. A separate
[`guarded_trace.py`](precision_insertion/guarded_trace.py) serializes and
replays the sample decisions used in that record. Its Python-side sample
deadline is not an independent daemon dead-man. The current recipient's
ParaDex daemon source contains a 1 kHz stream-expiry watchdog and the installed
binary contains its watchdog messages, but the watchdog has not been
commissioned for this insertion controller. An actuator must
meet that robot-side safety prerequisite before interpreting any `continue`
decision as permission to follow a preflighted path. The thresholds in unit
tests are synthetic examples, **not** robot settings. The exact sample,
frame and API contract is in [GUARDED_CONTACT.md](GUARDED_CONTACT.md).
The observed `prepare_postlift_transfer` still insists on a fresh held-key
FoundPose relation. The alternative `prepare_bounded_postlift_transfer` accepts
only an exact-candidate physical calibration, future-trial bound and fresh
measured robot state; it is not a motion permit. The raw insertion checkpoint
does **not** remove the held-key FoundPose requirement in
`prepare_observed_xy_retry`. Hand occlusion remains a retry feasibility risk;
do not silently substitute the BODex nominal relation. The separately
selected experimental `held_relation_route="nominal"` does use the exact v8
candidate relation as an **unmeasured hypothesis** for the cylinder
repeatable-bias/next-trial-XY study; its reports carry distinct provenance
and still require fresh joints, endpoint and held-path preflights. This is
not an observed post-squeeze pose or a physical-calibration claim. MuJoCo's achieved
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
  The active `observer.observe_preinsert_temporal_views` instead compares one
  camera's raw `after_lift` and raw `preinsert` images in time order. It never
  sends the CAD overlay to the VLM. The historical single-image
  `observe_preinsert_raw_views` remains only for older retry diagnostics.
  `preinsert_checkpoint.assess_preinsert_checkpoint` fuses the active visual
  assessment with a matching saved post-lift plan, same-session frozen socket
  and camera calibration, provenance-verified raw after-lift and preinsert
  bundles with matching camera/frame IDs and ordered exposure times, measured
  stationary Franka/Inspire joints, target hand-pose residual and an external
  transfer execution log. The log schema is
  `precision_insertion_transfer_execution_v1`: matching attempt/candidate IDs,
  `started_at_s`, `completed_at_s`, tri-state `measurement` fields
  `trajectory_complete`, `safety_abort`, `grasp_held`, and hashed absolute
  `source_records` for `trajectory_feedback`, `safety`, `grasp_state`.
  Visible two-view coarse alignment plus passing measured/log gates can yield
  a provisional `preinsert_reached=True` checkpoint; occlusion gives unknown.
  `write_preinsert_checkpoint` saves the exact overlay PNGs and hashes, and
  `verify_preinsert_checkpoint` rechecks both saved raw frame sets, VLM prompt
  and answer replay, overlays, transfer
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
  For a fresh post-shift arrival, `start_at_preinsert=True` skips the old
  transfer query and requires measured wrist FK to meet the newly fitted
  pre-insertion target. Its two repeated transfer samples represent a
  **stationary audit hold**, not an executable transfer trajectory. Only the
  subsequent axial waypoints are replanned.
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
  Franka `get_data()["time"]` comes from the local daemon's host monotonic
  clock. This adapter converts that timestamp to robot-PC wall time and
  rejects stale states without waiting for another packet; Inspire time is a
  software read time. Their skew check does not prove hardware-level
  simultaneity or camera synchronization. Thresholds must be commissioned.
- `live_capture.collect_key_capture()` uses the selected key's v8 FoundPose
  representation and a key-specific SAM prompt. The live tabletop path takes
  `pose_only=True`: it keeps same-request mask/pose frame IDs and exposure
  times, and begins v8 planning without waiting for the init daemon's PNGs or
  comparing image hashes. Socket and held-key captures retain their existing
  same-request image path. The separate initial tabletop VLM snapshot is
  taken with the already-open AutoDex cameras while candidate planning runs;
  its own request/frame IDs and timestamps are saved with undistorted images.
- [`registered_key_roi.py`](precision_insertion/registered_key_roi.py) and
  [`registered_key_capture.py`](precision_insertion/registered_key_capture.py)
  provide an **opt-in** ChArUco-registered socket-only baseline → cropped
  SAM3 → full-frame FoundPose producer on the same live key capture. The
  baseline is bound to the opened session; failed ROI views never fall back to
  stock full-frame poses. A [saved AutoDex-camera replay](REGISTERED_KEY_ROI.md)
  rejected a known wrong-socket mask. The ROI thresholds are exploratory,
  the live path needs caller-supplied loaded models, and no real key-pose
  accuracy or robot motion has been commissioned yet.
- `key_perception.admit_key_capture()` reuses the existing AutoDex per-view
  SAM/FoundPose quality gate and calls unchanged
  `InitOrchestrator.refine_from_payloads` in IoU/silhouette mode. For live
  cylinder tabletop keys, raw FoundPose pose spread is recorded but does not
  veto the stock selection; square and held-key captures still require raw
  pose agreement. It checks the current camera matrices against the **frozen session
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
  Live tabletop trials use `write_key_pose_artifacts` to store masks, poses,
  frame metadata and selected pose in an exclusive per-trial bundle, plus an
  independent `initial_tabletop` raw visual bundle. Other captures use
  `write_key_capture_artifacts` for same-request images and pose evidence.
  `verify_key_capture_artifacts` accepts both formats for replay/handoff.
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
  different selected key/socket/gap. Full changed-source SHA checks are an
  explicit offline audit; the live catalog assumes immutable NAS assets.
  For a planner `candidate_override`, the demo reads only the exact
  catalog-selected v8 NPY files. Stock `load_candidate` walks the entire
  object candidate tree even with a one-grasp whitelist, which would repeat
  NAS enumeration for every attempted grasp. The AutoDex planner itself is
  unchanged and still stops at pickup/lift.
- `reset_candidates.py` reads only provenance-bound, full-key MuJoCo-stable
  directed v8 reset seeds. `repose_policy.py` cross-checks alternative poses
  with insertion-eligible grasps and reset seed availability at each AutoDex
  release height. Key-in-hand drift/tilt is recorded; an extra fidelity gate
  is applied only when the caller supplies both limits. They report
  seeds for a later socket-aware Franka preflight, never a runnable reset.
- `outcome.py` defines a VLM-led, sensor-vetoed tri-state task label. A
  multi-view `normal_appearance` assessment is required for `true`, together with
  independently cross-checked **key** depth, commissioned alignment limits,
  held-grasp evidence, and no safety abort. Conflicting or occluded evidence
  is `null`; normal force or a commanded wrist stroke is not success proof.
  Older recorded `normal_20mm` assessments remain accepted as an alias, but
  the new VLM prompt deliberately does not ask the model to infer millimetres.
- `observer.py` defines the event-driven lift, per-camera XY, and insertion
  visual prompts; its local Qwen adapter uses first-party Transformers and
  its optional Gemini adapter uses the Google Gen AI SDK. Inputs have explicit
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
  exhaustion to a separate repose preflight. A retry-count cap is optional;
  the cumulative XY distance has its own configured bound. Its
  `execution_gate_required` result does not authorize a
  robot command; the current cylinder loop uses the first-party stock FR3
  force stop and still needs physical trial evidence.

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
    max_arm_velocity_rad_s=commissioned_stationary_velocity_rad_s,
)
```

The four limits are experiment-specific measurements, not values inferred by
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
The demo-local observer supplies votes. Its caller must first validate the
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

The selected held hand pose is applied to **every** sampled joint of the
lift, cuRobo transfer and demo-local socket-axis Jacobian path before the
full key/hand path audit. The raw cuRobo hand-column jitter alone does not
reject the candidate. A malformed/non-finite path, incorrect starting joint
state, incompatible world or unexpected solver exception still raises rather
than being silently reclassified as ordinary grasp infeasibility.

`sampled_planning_pass` is **not** a physical success label or execution
permit. Run this against the actual selected candidate and measured session;
the 2026-10-11 local square 1.5 mm pool has seven **nominally** eligible
pose-`004` grasps, while the other four square poses have none. The cylinder
pool has 12 nominally eligible grasps, but no physical relation is certified.
The nominal squeeze pose is
not a measurement; before physical transfer, either observe the held key and
validate its hand–key relation or use the separately calibrated bounded
physical-grasp route described below. A guarded contact
controller, true acquisition-timestamped camera adapter, commissioned
measured-state lift/transfer execution, and reset/repose execution remain
missing.

The six local cylinder catalogs were [refreshed and source-verified on
2026-10-11](CYLINDER_ENDPOINT_CATALOGS.md) after the candidate verifier
changed. Earlier staged catalogs are stale; a recipient host must re-screen
or verify its own paths and source hashes before starting a session. The
12-per-gap result remains nominal offline eligibility, not a measured grasp
or insertion success count.

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
stream stall. The [demo-local completion-checked subclass](FRANKA_COMPLETION_CHECK.md)
reuses stock streaming but removes that blocking landing and defers the last
hand ramp until measured arrival. It still requires hardware-side
velocity-command expiry and commissioning. The boundary commands **no lift, transfer or
insertion** and does not set `grasp_success`. It has fake-executor tests only,
not a commissioned live-robot launch path.

After a measured squeeze, [the measured-lift replan](MEASURED_LIFT.md)
re-screens the 20 mm endpoint with achieved finger joints and uses existing
v8 cuRobo planning plus the demo's full held-key/hand path audit from the
achieved FR3/Inspire state. It additionally needs a commissioned bound on
key/hand surface error for a bounded-margin claim. The cylinder route can
explicitly use `bounds=None` for a stock-style nominal CAD plan and records
that margin as unassessed. This is still a saved plan, **not** a lift command.

The separate [measured lift execution boundary](LIFT_EXECUTION.md) can pass
that exact saved lift to the first-party measured adapter after checking the
running local stock daemon identity, fresh matching robot feedback, explicit
motion opt-in and interlock. An external adapter still needs a reviewed
watchdog record. It defaults to no motion and has only fake-adapter execution
tests. Its successful execution record is input to the later paired-camera lift checkpoint, never
itself a grasp-success label.

After the lift checkpoint and a passing observed or uncertainty-bounded
post-lift replan, the separate [held transfer execution boundary](TRANSFER_EXECUTION.md)
can submit only its exact saved non-contact transfer to a commissioned
external controller. It is disabled by default and currently has fake-adapter
tests only. Its controller log is consumed by the later synchronized-camera
pre-insertion checkpoint; neither the controller log nor a successful end
joint state alone sets `preinsert_reached=True`.

At the guarded insertion endpoint, the separate
[conditional wrist-depth diagnostic](CONDITIONAL_WRIST_DEPTH.md) can compute
the key-tip depth interval implied by measured wrist pose, the frozen socket
CAD and a commissioned key-surface error bound. It deliberately emits no
admissible `key_depth_source`: the physical key may have slipped despite
nominal wrist travel, and a 20 mm nominal endpoint with nonzero uncertainty
does not prove at least 20 mm of key penetration.

Do not feed the nominal `plan_insertion_after_pickup`'s separately replanned
`lift_trajectory` straight into the
unchanged `FrankaExecutor.execute(lift_traj_override=...)`: its start-state
gate still refers to the original pickup `lift_preflight` modeled at grasp
joint values. The demo-local measured lift boundary compares live arm/hand
state with the **same** held-lift plan before invoking its external controller;
it does not make the stock follower safe.

For a **saved-image, read-only** observer replay, the default backend is local
`Qwen/Qwen3-VL-2B-Instruct`. Gemini is an explicit opt-in only.
The newly generated key/socket FoundPose representations have a separate
[saved-camera diagnostic](FOUNDPOSE_SAVED_REPLAY.md); it does not validate or
enable the VLM/robot loop.
The local option uses `precision_insertion.local_qwen.LocalQwenModel`; it does
not use a Gemini key, send images to an API, or import ZeroDex. Install its
optional dependencies in an
**isolated** venv, not the production AutoDex Python environment:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m venv --system-site-packages \
  ~/.venvs/precision-vlm
~/.venvs/precision-vlm/bin/python -m pip install \
  -r demo/precision-insertion/requirements-local-vlm.txt
export PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src"
```

This example reuses the existing hardware-compatible PyTorch installation.
Check CUDA visibility outside a sandbox that hides the GPU:

```bash
~/.venvs/precision-vlm/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

The loader uses `local_files_only=True`; point it to the verified NAS bundle
or staged copy. For a semantic-only smoke test on *saved* images (not a real
trial):

```bash
~/.venvs/precision-vlm/bin/python demo/precision-insertion/probe_vlm.py \
  --model-id /absolute/staged/shared_data/AutoDex/models/Qwen3-VL-2B-Instruct \
  --task lift \
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
    max_input_size=(640, 640), require_native_pixels=False,
)
result = observe_lift(backend, [
    LabeledFrame("front", "before_grasp", 1.0, before_pil),
    LabeledFrame("front", "after_lift", 2.0, after_pil),
])
print(result.to_record())  # review only; not a robot command
```

The same `backend` object can be passed to the demo session's
`prepare_observed_lift_label`, `prepare_observed_insertion_label`, and
read-only pre-insertion/XY observation methods. They accept the `ImageVLM`
interface, so choosing local versus Gemini does not alter the physical
motion or safety gates. `run_auto.py` currently opens an evidence-only
session; it does not run a commissioned live VLM/robot loop.

To measure local/API model errors rather than relying on one illustrative
render, use the independent-annotation dataset replay in
[`VLM_BENCHMARK.md`](VLM_BENCHMARK.md).

To exercise **local-model pixel grounding → calibrated multi-view XY
diagnostic** on saved cylinder pre-insertion images, use the separate
[`GROUNDING_PROBE.md`](GROUNDING_PROBE.md) manifest/CLI. It checks native
image geometry and records raw per-view marks, but does not certify camera
provenance, metric accuracy or robot readiness.

`before_pil` and `after_pil` must be supplied from the same saved trial; the
timestamps above are placeholders. Semantic labels may use the Qwen
processor's resizing. **Metric point/axis grounding cannot:** create a
backend with `require_native_pixels=True` and a `max_input_size` at least as
large as every supplied undistorted frame (or implement an explicit crop-to-
original-pixel transform). The backend pads only right/bottom to Qwen's
32-pixel patch grid and rejects a frame if the padded dimensions exceed the
declared ceiling or processor pixel-area limits. For this Qwen3-VL-2B
processor, 640×480 is unchanged; 1280×720 becomes 1280×736 by lossless
padding and requires a ceiling of at least 1280×736. The latest inspected AutoDex NAS camera
calibration records 2048×1536, which passed a single-view local-model smoke
test with `max_input_size=(2048, 1536)`; this is not a multi-view accuracy
result or a guarantee that the next live session uses that calibration.
Unknown local processor geometry also
fails closed. The point-grounding path also requires an explicit verified
native-pixel backend contract: an undeclared adapter or the current Gemini
adapter cannot produce metric coordinates because its internal image
transforms have not been verified. Gemini remains available for semantic
read-only labels. This safeguards coordinate bookkeeping, not VLM pixel accuracy:
calibration/held-out grounding error limits remain mandatory. Lift and
insertion comparisons reject unpaired camera IDs, duplicate phase frames,
or reversed timestamps; a front
"before" image cannot be compared to a side "after" image as if it tracked
one object. The optional Gemini adapter accepts an already-configured
`google.genai.Client` and a model ID, or explicitly reads `GEMINI_API_KEY` via
`ZeroDexGeminiBackend.from_env(model="YOUR_MODEL_ID")`. Install the optional
`google-genai` package in the execution environment first with
`python -m pip install -r demo/precision-insertion/requirements-gemini.txt`
(the pin matches the optional Google SDK adapter). Gemini sends the same
ordered PNG images plus one text prompt through the Google Gen AI SDK. The
local adapter requires Transformers and the verified Qwen snapshot, with no
ZeroDex package or camera dependency. In either case,
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
  --pose-stem 004
```

`screen-catalog` writes exclusively; a rerun needs a new output name.
`--max-candidates N` is a pilot scan and always marked incomplete. The
historical 2026-10-10 1.5 mm scan had only `table/0/78`, whose Inspire links
intersect the socket. The current local 2026-10-11 scan has eight installed
v8 candidates: that old pose-`000` failure plus seven pose-`004` grasps from
the full-key 10,000-proposal pilot. All seven pass the *nominal* exact-CAD
20 mm endpoint at the **uncommissioned** 0.2 mm hand clearance. The other
four square tabletop poses still have no eligible grasp in this finite pool.
The current complete catalogue is
`~/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_20261011_v8_8_nominal_r2.json`.
`select-catalog` exits 2 for no eligible grasp or an incomplete/stale
catalogue. It does not execute reorientation.
The example catalogue is in local `shared_data`, **not** the read-only
`/mnt/paradex2` NAS mount.

The seven candidates were installed with the demo-local
[`promote_square_candidates.py`](promote_square_candidates.py), which checks
the complete original AutoDex sim-filter pass set, original BODex quality,
the 3,000-samples/link pilot screen, and a fresh exact 20 mm endpoint screen
before atomically creating the previously absent `table/4` v8 scene. It
refuses to replace an existing scene; each candidate carries source-file
hashes, an endpoint report, and `robot_ready: false`. The source experiment
used `contact_policy_mode=report-only`, so the old handle-surface restriction
was **not** an acceptance gate. The physical consequence of contact on other
key surfaces is not established. For a fresh host with the complete source
stage and pass pool, inspect the script's `--help` and supply the explicit
source scene, pass scene, high-resolution report, destination and clearance;
do not re-run it over this already installed scene.

On an otherwise identical *fresh* shared-data root, the exact input contract
is:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/promote_square_candidates.py \
  --shared-root /home/hyunsu/shared_data --gap-mm 1.5 \
  --source-scene /home/hyunsu/shared_data/AutoDex/sim_staging/inspire/precision_insertion_unconstrained_20mm_10k_top50/precision_key_1p5mm/table/4 \
  --pass-scene /home/hyunsu/shared_data/AutoDex/sim_filter_pass/inspire/precision_insertion_unconstrained_20mm_10k_top50/precision_key_1p5mm/table/4 \
  --highres-report /home/hyunsu/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/highres_task_screen_pose_004.json \
  --output-scene /home/hyunsu/shared_data/AutoDex/candidates/inspire/v8/precision_key_1p5mm/table/4 \
  --simulation-version precision_insertion_unconstrained_20mm_10k_top50 \
  --min-hand-clearance-mm 0.2
```

The destination must be absent. A different host/root requires source
scenes and their absolute CAD references to be regenerated/verified for that
host; copying the JSON path strings alone is not sufficient.
The hash-verified local [square tabletop handoff](SQUARE_TABLETOP_HANDOFF.md)
packages the necessary offline files for transfer. Its documented
non-overwriting recipient installer verifies the source bundle, rebinds the
known scene/fixture paths and seven candidate scene hashes, and requires a
new endpoint catalog at the recipient root before candidate selection.

The separate
[`audit_square_grasp_fidelity.py`](audit_square_grasp_fidelity.py) reads
the selected catalogue and records the MuJoCo achieved key-in-hand change.
Its local report is
`~/shared_data/AutoDex/precision_insertion/experiments/unconstrained_20mm_10k/square_pose004_fidelity_20261011_r2.json`.
To regenerate it, use a new output filename:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_square_grasp_fidelity.py \
  --catalog /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_20261011_v8_8_nominal_r2.json \
  --gap-mm 1.5 --pose-stem 004 \
  --output /tmp/square_pose004_fidelity_NEW.json
```

At squeeze end, the seven key centers move **2.9–21.1 mm** in the hand and
rotate **7.0–17.0°** relative to the initial pose. These are diagnostics,
not calibrated acceptance limits. The nominal endpoint result must not be
used as a measured held-key endpoint or a robot execution permit.

### Repose/reorientation proposal assets

The new key object IDs have v8 tabletop poses but no legacy `paradex`
tabletop tree. Therefore the stock `src/experiment/reset/reorient.py`
cannot resolve its usual v8-to-legacy reset-cell mapping. Its carried-object
planning scene also drops the session-frozen socket mesh, so it must **not**
be invoked as this demo's executable repose path. The demo's
`audit-reorient` command reports directed v8 pose-pair scenes separately
from MuJoCo-stable reset grasp seeds and staged diagnostics. An available
BODex scene is *not* an executable reorientation trajectory.

The newer [curated staged handoff](REORIENT_CURATED_HANDOFF.md) combines
individually verified offline reset seeds from selected square and cylinder
directed cells without touching the canonical tree. It is also copied to the
personal ParaDex2 NAS for transfer. It does **not** fill the missing pose
directions, commission pose-fidelity limits, or supply a socket-aware Franka
repose trajectory. The older scan counts below describe their respective
experiments; they are not the contents of that curated staging root.

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

#### Square key: one directed 12 cm reset-cell pilot

The earlier *canonical* square audit above still has zero installed reset
seeds. A separate, non-runtime pilot now covers `000 -> 004` for the 1.5 mm
key. `configs/square_full_key.txt` selects the **full key**, and
`configs/square_reorient_0_to_4.json` selects only that directed v8 scene.
With `--seed_num 1000 --seed 11011`, BODex wrote 1,000 raw proposals. The
unchanged AutoDex filter reported **843 scene-clear, 251 squeeze-contact,
7 MuJoCo-gravity-stable** (`2, 9, 194, 226, 229, 489, 813`). Native BODex
`success=0/1000` is a different, stricter proposal flag and is not this
stock filter's acceptance gate. The stock filter copies only grasp arrays;
[`stage_square_reorient_passes.py`](stage_square_reorient_passes.py) verifies
all 1,000 results, matching copied arrays and scene/key hashes before
creating the [local square handoff](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_0_4_1000_20261011/manifest.json).
It does not install anything into NAS.

Unlike a symmetric cylinder, a square key must preserve **full key-in-hand
rotation**, including yaw. Its reset loader and audit now recompute full
relative rotation and key-center drift from MuJoCo `sim_traj.json`. Among the
seven stock passes, squeeze-end drift spans **4.5–18.6 mm** and full rotation
spans **7.9–43.7°**. With illustrative, **uncommissioned** 3 mm / 5° gates,
the [square handoff audit](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_0_4_1000_20261011/audit_provisional_3mm_5deg.json)
accepts **0/7** for reset planning. A gravity-stable grasp is not proof that
the assumed rigid key–hand transform survived squeeze. No socket-aware Franka
reset path or physical repose has been validated.

A later **separate** `000 -> 004` square reset pilot (`--seed_num 1000`,
`--seed 11013`) produced [six evidence-bound MuJoCo-stable staged seeds](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_gamma180_handoff_0_4_1000_seed11013_20261011/manifest.json).
The direct-v8 loader recognizes all six under that explicit handoff root, but
the canonical NAS candidate tree still has zero installed square reset seeds.
Their saved end-gravity key-center drift spans about 3.6–21.9 mm, and the
square key's full relative rotation about 4.3–33.0°. No pose-fidelity limit
has been commissioned, and even a limit-passing staged seed would still need
socket-aware Franka pickup/place/release preflight and observed landing.
Do not reinterpret the earlier `--seed 11011` result as the current entire
staged pool, or treat these six as an executable reset action.

To reproduce on a fresh output tree, from the repository root:

```bash
PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file demo/precision-insertion/configs/square_full_key.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_type reorient_12 \
  --scene_filter_file demo/precision-insertion/configs/square_reorient_0_to_4.json \
  --seed_num 1000 --seed 11011 --exp_name NEW_SQUARE_RESET_RAW \
  --output_dir /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_SQUARE_RESET_RAW

PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_1p5mm", "/home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_SQUARE_RESET_RAW", "/home/hyunsu/shared_data/AutoDex/precision_insertion/NEW_SQUARE_STOCK_PASS", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'

~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/stage_square_reorient_passes.py \
  --shared-root /home/hyunsu/shared_data --gap-mm 1.5 \
  --raw-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_SQUARE_RESET_RAW \
  --stock-candidate-root /home/hyunsu/shared_data/AutoDex/precision_insertion/NEW_SQUARE_STOCK_PASS \
  --cell 0_4 --expected-seed-count 1000 \
  --output-root /home/hyunsu/shared_data/AutoDex/precision_insertion/square/NEW_SQUARE_HANDOFF
```

`audit-reorient --candidate-root /path/to/NEW_SQUARE_HANDOFF` counts
evidence-bound stock passes. Add both `--max-reset-drift-mm` and
`--max-reset-rotation-deg` only with an explicitly stated experimental or
commissioned policy. For square keys, the latter tests full relative
rotation; for cylinders, it tests symmetry-reduced axis tilt. The old
`--max-reset-axis-tilt-deg` spelling remains an alias for existing commands.

The other three directed cells toward the only currently insertion-eligible
square tabletop class `004` were then sampled at the same **1,000 proposals
per scene** using
[`square_reorient_remaining_to_4.json`](configs/square_reorient_remaining_to_4.json)
and the same unchanged stock filter. Results are per cell, not evidence that
other samples or a different grasp policy cannot work:

| Directed cell | Scene-clear | Squeeze contact | MuJoCo stable | Handoff evidence |
| --- | ---: | ---: | ---: | --- |
| `000→004` | 843 | 251 | 7 | [manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_0_4_1000_20261011/manifest.json) |
| `001→004` | 871 | 170 | 0 | [manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_1_4_1000_20261011/manifest.json) |
| `002→004` | 891 | 180 | 0 | [manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_2_4_1000_20261011/manifest.json) |
| `003→004` | 750 | 17 | 0 | [manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_handoff_3_4_1000_20261011/manifest.json) |

Overall, **4,000 raw → 3,355 scene-clear → 618 squeeze-contact → 7
MuJoCo-gravity-stable**, and **0/7** satisfy the provisional 3 mm / 5°
key-in-hand pose gate. The catalog has seven insertion-endpoint-eligible
grasps at pose `004`, none at poses `000–003`; the one stock grasp at `000`
fails the insertion endpoint. Thus *no automatic repose route to a known
insertable pose is established*. These are local handoff reports, not
canonical NAS reset seeds, and do not include fixture-aware Franka plans.
The very low `003→004` squeeze-contact count (17/1,000) motivates inspecting
proposal placement and scene geometry before merely increasing sampling.

### Square reorient proposal-policy checks (offline only)

The preceding 30° wrench cone came from the insertion proposal config, not
from AutoDex's 12 cm reorient proposal setting. Holding the key mesh, scene,
stock filter and 1,000-proposal budget fixed, a 180° cone gave the following
independent local batches. `003→004` remains particularly difficult; the
extra squeeze contacts do not establish a reset path.

| Cell and setting | Scene-clear | Squeeze contact | MuJoCo stable | Provisional 3 mm / 5° key–hand fidelity |
| --- | ---: | ---: | ---: | ---: |
| `000→004`, 50 mm seed inflation, 30°, seed 11011 | 843 | 251 | 7 | 0 |
| `000→004`, 50 mm seed inflation, 180°, seed 11011 | 852 | 291 | 10 | 1 |
| `000→004`, 50 mm seed inflation, 180°, seed 11013 | 859 | 279 | 6 | 0 |
| `001→004`, 50 mm seed inflation, 180°, seed 11014 | 852 | 229 | 0 | 0 |
| `002→004`, 50 mm seed inflation, 180°, seed 11014 | 869 | 216 | 0 | 0 |
| `003→004`, 50 mm seed inflation, 30°, seed 11012 | 750 | 17 | 0 | 0 |
| `003→004`, 50 mm seed inflation, 180°, seed 11012 | 745 | 25 | 1 | 0 |
| `000→004`, 10 mm seed inflation, 180°, seed 11011 | 780 | 191 | 0 | 0 |

The 10 mm row changes only BODex's *proposal-surface sampling distance* via
[`square_reorient_gamma180_inflate10mm.yml`](configs/square_reorient_gamma180_inflate10mm.yml),
not the final contact tolerance or stock simulation criteria. It is a
negative ablation, not a replacement config. The raw/filter outputs and
non-runtime [handoff manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_inflate10_gamma180_handoff_0_4_1000_20261011/manifest.json)
are kept separate from canonical candidates. All 1,000 10 mm proposals were
written, and the unchanged AutoDex collision, squeeze-contact and MuJoCo
filters reported 780/191/0 respectively. BODex's native `success` was zero
for this batch, as it was for the original; the staged MuJoCo count is a
separate criterion.

The single 180° `000→004` seed satisfying the **illustrative, uncommissioned**
pose-fidelity bound is seed `184` in the [local gamma180 handoff](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_gamma180_handoff_0_4_1000_20261011/manifest.json).
Its MuJoCo end-of-gravity key–hand center drift is 0.20 mm and full rotation
change is 4.56°. However, its maximum BODex `grasp_error` is 0.2095, just
above the separate project diagnostic threshold of 0.2. The stock AutoDex
MuJoCo filter does **not** apply that diagnostic threshold; neither a stock
pass nor an illustrative fidelity pass means this seed is physically reset-
ready. It still lacks a commissioned fidelity bound, fixture-aware Franka
path and real-robot test. Do not install these handoffs into the canonical
runtime reset tree.

The 180° scans now cover all four directed cells toward pose `004`, each with
1,000 proposals: **4,000 raw → 3,318 scene-clear → 761 squeeze-contact →
11 MuJoCo-stable → 1 provisional pose-fidelity pass**. The `001→004` and
`002→004` results are bound in separate local [cell 1 manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_gamma180_handoff_1_4_1000_20261011/manifest.json)
and [cell 2 manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_gamma180_handoff_2_4_1000_20261011/manifest.json).
They used [`square_reorient_1_2_to_4.json`](configs/square_reorient_1_2_to_4.json)
with unchanged `precision_insertion.yml` plus `--task_gamma 180 --seed 11014`;
the earlier `003→004` scan used seed 11012. The stock `sim_eval.json` records
no detailed failure reason after squeeze, but its evaluator stops when the
key drifts more than 50 mm or rotates more than 15° under simulated gravity.
For the `001→004` and `002→004` contact subsets, **none** completed all 50
gravity steps; their median recorded force-test lengths were 5 and 6 steps.
Thus these are early stability failures, not merely a shortage of contact.
They do not justify labeling either pose impossible, nor do they establish
automatic repose to pose `004`.

An additional [8 cm release-height scan](SQUARE_H8_REORIENT_SCAN.md) uses
the same v8 scene generator, 1,000 proposals per scene, and stock filter for
`001→004` and `002→004`. It found one MuJoCo-stable `001→004` seed but no
fidelity-eligible or robot-ready reset; `002→004` again had no stable seed.
The staging command accepts `--height-cm 8` without changing its historical
12 cm default.

The 10 mm ablation is reproducible by replacing `-c` in the BODex command
above with the absolute path to the demo-only YAML, adding
`--task_gamma 180`, and using fresh `--exp_name`, raw, filter and handoff
output roots. `square_reorient_3_to_4.json` restricts the corresponding
180° comparison to `003→004`. Never reuse an existing output root for a
different generator setting.

To repeat only these remaining cells, use the preceding BODex and stock-filter
commands with `--scene_filter_file
demo/precision-insertion/configs/square_reorient_remaining_to_4.json`,
`--seed 11012`, and **new** raw/filter output roots. Then call
`stage_square_reorient_passes.py` separately with `--cell 1_4`, `2_4`, and
`3_4`, `--expected-seed-count 1000`, and a distinct `--output-root` for each.
An empty stable pool still gets a complete per-cell manifest; it is not an
executable reset candidate. Do not copy those empty or uncommissioned pools
to `AutoDex/candidates/inspire/reset_12`.

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
saved trajectory. Drift/tilt limits are **optional**, not part of the stock
v8 acceptance rule; when supplied, a seed outside either limit is omitted.
Without them the values remain diagnostics. It is only a seed loader;
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

The following proxy-based reproduction command is **historical** and must
not be used for new unrestricted grasp proposals: the current staging script
expects the full-length proposal object. For the active 80 mm surface and
matching staging steps, follow [FULL_LENGTH_CYLINDER_PROPOSALS.md](FULL_LENGTH_CYLINDER_PROPOSALS.md).
The old command is retained only to identify the archived experiment:

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
`/mnt/paradex2` NFS mount on this machine. At the time of this historical
run, the sandbox exposed it as read-only, so the expanded passing seeds were
written to the **local, non-runtime**
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
evidence-bound seeds separately from the *then-empty* canonical NAS audit.
Since that audit, `0_1/191` has been installed in the canonical tree; the
current state is summarized in [OFFLINE_ASSET_STATUS.md](OFFLINE_ASSET_STATUS.md).
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
limits would therefore have no eligible `1 -> 0` seed *in that historical
two-seed pool*. The default loader now imposes no uncommissioned drift gate.
The later full-key run and source-verified `1_0/1687` canonical installation
are documented in [OFFLINE_ASSET_STATUS.md](OFFLINE_ASSET_STATUS.md); neither
seed yet has a verified Franka repose trajectory or physical landing.
The 999 KB [local handoff bundle](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_bundle_20261010.tar.gz)
contains both candidates, audit reports, v8 key scenes, key and six cylinder
socket object-processing assets, fixture geometries, and a handoff README.
Its SHA-256 is
`6b83a94026074f925c57567dd754771e01cbfda16491d2e8a10ba647894acc98`.
The byte-identical archive is now in the [ParaDex2 NAS handoff](/mnt/paradex2/hyunsu/handoff/20261011_cylinder_reset_v8_6b83a940/cylinder_reset_v8_20261011_6b83a940.tar.gz).
The archive itself is a staging handoff, not an automatic canonical-tree
installation; `0_1/191` was installed separately after fidelity screening.
Scene JSONs embed this host's
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
new [full-length proposal workflow](FULL_LENGTH_CYLINDER_PROPOSALS.md)
removes the historical 25 mm proposal restriction without changing the
exact physical key used for AutoDex filtering. The current source-bound
zero-added-margin runtime catalogues are listed in
[COLLISION_ONLY_CATALOGS.md](COLLISION_ONLY_CATALOGS.md); endpoint screening
and promotion of the new full-length pool have finished, but the candidates
remain **offline evidence**, not physical insertion successes. A
recommended per-socket output layout is
`endpoint_catalogs/cylindrical/<socket_object>/<scan_id>.json`. For example,
after the grasp pool is present, select the 20 mm radial-gap variant with
`--mode cylinder --gap-mm 20` for both `screen-catalog` and `select-catalog`;
the latter now rejects a catalogue for any other socket gap. Keep each scan's
exact key/socket/URDF hashes and clearance rule. At live startup the operator
or a validated fixture identifier must independently confirm which physical
socket is mounted; measuring only its pose does not establish its size.
The `gap_01mm` name denotes a **1 mm one-sided radial clearance**, not 0.1 mm.
The historical scan across all four square gaps yielded one `table/0/78`
per gap and zero endpoint-eligible grasps. A later full-key pilot supplied
seven nominally endpoint-clear 1.5 mm grasps at pose `004`; it did not
populate the 0.3, 0.5, or 1.0 mm square pools. The historical
`cylinder_gap20mm_20261010.json` is incomplete
because it was built before the cylinder v8 candidates were promoted. The
2026-10-11 scan screens all **12 currently installed** cylinder candidates
separately against each of the six socket meshes. All 12 pass each *nominal*
20 mm endpoint at the example 0.2 mm hand clearance (11 `end_down`, one
`side_down`). This is not a physical grasp or insertion success count: the
same candidates' MuJoCo end-squeeze key-in-hand center drift spans about
4.4–35.2 mm, and no post-squeeze fidelity threshold has been commissioned.
The new catalogues are under `endpoint_catalogs/cylindrical/<socket_object>/`
and retain a `robot_ready: false` marker. A compact index of the local
reports and two square staging-grasp checks is at
`~/shared_data/AutoDex/precision_insertion/endpoint_catalogs/README.md`.

### Cylinder endpoint diagnostic images (not runtime grasps)

`render_cylinder_endpoint_diagnostics.py` provides a reproducible *historical
visual diagnostic* from before the cylinder's tabletop v8 pool was promoted.
It reads the local, trusted raw **reorientation** pilot, applies the documented relaxed
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
`robot_ready: false`. The report records `cartesian_planner_mode`. The default
AutoDex endpoint-approximation route may move Inspire joints during transfer;
the demo rejects such a path. For an **offline experiment only**, prefix the
command with `AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1` and add
`--planner-mode native-locked-experimental`. A mismatch fails before reading
session assets. This selects AutoDex's original locked-hand option, but it is
not approved for robot motion and must still pass the fixed-hand and sampled
collision audits. A pose with no candidate may return
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
state plus acquisition timestamps, optional reset-grasp fidelity limits, frozen-socket
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
preflights fully opening Inspire to `INSPIRE_INIT`, a +10 cm vertical exit and
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

AutoDex's `InitOrchestrator.collect_payloads()` returns one SAM mask and
one pose envelope per camera. The board-9 cylinder route uses the mask, not
the FoundPose estimate: `admit_socket_capture()` checks mask size/border,
calibrated camera identity and acquisition skew, then `calibrate_session()`
freezes the board/body result. An updated demo capture-PC adapter skips the
unused FoundPose inference for that request and publishes a frame-bound
`ok=false` pose envelope. Stock socket modes and key perception still check
FoundPose quality, inliers and pose validity. Use a socket-specific SAM
prompt while the key is absent. The current 20 mm launch takes one socket
capture and freezes its board/body pose in the collision world. This
single observation does not establish pose repeatability. All
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
BGR snapshots, their request ID and verified acquisition times, and at least one
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
side channel (or compatible camera/daemon support) is needed to bind board,
socket and per-trial key frames to exposure intervals. The current board-9
launch uses the demo-local frame-ID-preserving adapters and bounded request
interval metadata. This provides a bounded exposure timestamp, not a
sub-frame synchronized camera measurement. A publication timestamp or one
request ID alone is not an exposure timestamp.

`precision_insertion.live_capture` specifies the demo-local adapters that
must wrap AutoDex's snapshot and initialized FoundPose collectors. Call
`collect_board_snapshot()` first, then `collect_socket_capture()` once for the
current board-9 launch while the socket is fixed and the key is absent, then pass the returned
frames/payloads and timestamps to `bootstrap_session()`. Socket frames are
the **same-request undistorted PNGs** written by the stock init daemon; the
adapter waits for those asynchronous files and refuses missing images rather
than taking a later snapshot. Use a unique capture root on a filesystem
shared by the robot PC and capture PCs (for example, a common absolute NAS
mount), and preserve those intermediate directories until the evidence bundle
has been verified. The metadata callback must return the exact request,
frame IDs, image hashes and bounded exposure times. The unchanged stock
orchestrators discard frame IDs; the live demo adapter retains them. These
capture helpers alone cannot authorize robot motion.

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
takes the configured number of unique-request socket captures while the key is absent
(one for the current board-9 / 20 mm launch),
takes only admitted SAM-mask views from each board/body capture. The current
board-9/20 mm launcher uses the measured board and undistorted SAM masks to
fit the fixed cylindrical wall on CPU; its mean projected-body IoU must meet
the configured limit. The stock IoU/silhouette route remains available for
comparison and for square mode. This fit does not independently measure a
physically tilted socket.
Each capture contributes exactly one refined pose. With one capture, its
pose is frozen directly and the record says `repeatability_checked: false`;
the raw per-camera pose envelopes remain saved evidence; in mask-only mode
they explicitly have no FoundPose pose. It runs
`bootstrap_session()`, freezes the socket in the collision world, writes
the evidence bundle and reloads the saved calibration to check it. Duplicate
request IDs or an existing output directory fail closed; partial capture
files after a failure are retained for review. It does not connect to or
command Franka/Inspire. Neither silhouette loss nor per-view IoU proves
absolute socket accuracy; compare the refined pose with independently
measured fixture truth before a robot trial.

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
    socket_prompt="fixed white square socket on the ChArUco board",
    socket_capture_count=1,
    board_id="11", board_capture_phase="empty_before_socket",
    confirm_socket_placement=lambda: input(
        "Place the white socket after the empty-board capture; Enter=ready: ") == "",
    board_timeout_s=commissioned_board_timeout_s,
    socket_timeout_s=commissioned_socket_timeout_s,
    max_socket_translation_mm=commissioned_socket_repeatability_mm,
    max_socket_angle_deg=commissioned_socket_repeatability_deg,
    minimum_socket_refined_iou=commissioned_per_view_iou,
)
# started.evidence_dir/session_calibration.json freezes the measured world.
```

The same startup now has a demo-local CLI. Copy
[`configs/live_session_start.example.json`](configs/live_session_start.example.json)
to a new configuration file and replace **every placeholder** with the current
AutoDex PC's absolute paths, active capture-PC/IP/camera inventory, measured
Franka C2R, fixed-obstacle cuRobo scene and independently commissioned
per-camera exposure-time calibrations. The example's threshold numbers are
structural only; commission them on the actual rig. An empty or stale base
scene is not a safe motion world. Use the precision init daemon on the selected
capture PCs instead of the stock init daemon, and leave the stock snapshot
daemon running; see [camera handoff](CAMERA_FRAME_HANDOFF.md).

```bash
# Read-only file/asset/calibration preflight; no camera connection:
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/path/to/commissioned_session_launch.json

# Explicit non-motion board/socket capture after the above succeeds:
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/path/to/commissioned_session_launch.json --capture
```

The CLI validates the timing files, camera IDs, CAD, C2R and world before
opening sockets. `--capture` joins the stock AutoDex orchestrators with the
demo's frame-provenance adapters, executes `start_precision_session()`, closes
the camera connections even on failure, and prints the immutable evidence
directory. It never commands Franka/Inspire. The CLI has synthetic tests but
has **not** been deployed on the robot PC; this change did not obtain or
verify real exposure-time calibration files for that rig. Do not
replace them with PUB timestamps merely to make the command pass.

Once that startup evidence and a current **complete** endpoint catalog exist
on the same AutoDex PC, the demo's independent `run_auto.py` can open an
evidence-only trial ledger. It checks both canonical key/socket FoundPose
files, the saved acquisition-bound board/socket frames, the frozen collision
world, each pose-indexed v8 catalog row and its source hashes **before**
creating the output directory. It never moves Franka/Inspire:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_auto.py \
  --shared-root /path/to/shared_data \
  --mode square --gap-mm 1.5 \
  --session-evidence /path/to/new_session_evidence \
  --endpoint-catalog /path/to/current_recipient_catalog.json \
  --output-dir /path/to/new_trial_run
```

The output contains frozen session/catalog snapshots and
`source_evidence_binding.json` with exact source hashes. The printed next
decision is initially `capture_fresh_key`. A present PTH is not a QA-approved
pose estimator, and a positive endpoint catalog is not motion authorization;
the first-party board-9 live binding and failed-key return/repose callbacks
still require physical verification on this rig.

`run_auto.py` opens a ledger and **exits**; its in-memory runner is not
resumable by another process. For a long-running program use the existing
[`live_key_trial.prepare_next_live_key()`](precision_insertion/live_key_trial.py)
after `open_verified_session()`, as documented below. That path requires
measured robot feedback buffered across the actual camera exposures; a
single joint read after SAM/FoundPose inference is not equivalent. On this
workstation, canonical FoundPose files are now present for the square 1.5 mm
first-test pair. A complete offline endpoint catalogue is also present for
square 1.5 mm, but physical pose accuracy, commissioned camera timestamps and
the robot motion path are still missing; file presence does not make the
session runnable.

For a **single long-running, non-motion process**, use
[`open_live_precision_session`](precision_insertion/live_orchestration.py).
It validates the launch files and the complete v8 endpoint catalogue,
including every indexed pose's source hashes and screening-implementation
hashes, before opening camera connections. It then collects ChArUco followed
by the configured socket captures,
opens the source-bound `SessionRunner`, and keeps the **same** initialized
AutoDex camera transport and acquisition-time provider alive for fresh key
captures until the context exits:

```python
from pathlib import Path
from precision_insertion.live_orchestration import open_live_precision_session

with open_live_precision_session(
    launch_config_path=Path("/absolute/commissioned_session_launch.json"),
    endpoint_catalog_path=Path("/absolute/complete_endpoint_catalog.json"),
    run_output_dir=Path("/absolute/new_trial_run"),
    max_xy_retries=2,
    clear_view=clear_view,  # commissioned ClearViewForPerception adapter
) as live:
    runner = live.opened.runner
    init = live.cameras.init_orchestrator
    acquisition_metadata = live.cameras.acquisition_metadata_for_request
    # Supply runner, init and acquisition_metadata to prepare_next_live_key()
    # together with commissioned feedback sampling, planner and limits below.
```

This joins existing boundaries; it does not duplicate FoundPose, v8 filtering,
or trial planning. The catalogue is checked again after capture, then bound
to the measured socket CAD; a file changed mid-capture cannot bypass that
check. The context closes every camera connection on normal exit
or failure, retaining already written capture evidence for diagnosis. The
required `clear_view` adapter calls the supplied stock Franka executor to
open Inspire and park before the first board/socket exposure; it does not
construct a controller. After any trial, fresh key perception requires a
measured open hand and arm at clear view. See the
[failure-return sequence](FAILURE_RETURN.md). A successful camera/catalogue
preflight is not motor authorization for pickup or insertion.

For a judged grasp miss, the live recovery handler reuses stock AutoDex
`release(..., open_to_init=True)` and `reset()` with the selected v8 pickup
plan. The copied collision world retains the fixed socket and original
tabletop key; after measured clear-view feedback it recaptures the key at
its actual supported pose and excludes the missed grasp. It does not pick up
and drop the key simply to restore its former XY. In the live 20 mm cylinder
mode, a lying `001` key first enters the directed v8 `001→000` reorientation
route. That route and held-key failure returns now plan release at the board
support pose (`release_height_m=0`), using the stock force-aware vertical
descent before opening. The v8 `reset_12` name still identifies the candidate
seed family; it is not the physical drop height. A freshly observed supported
reset within 30 mm of the original CAD center is accepted for the next trial.
For a saved preinsert XY correction, the next `000` trial plans only the same
v8 grasp. If that grasp cannot be planned, it reports
`repeat_grasp_unavailable` instead of trying a different grasp under the
same correction. The saved preinsert diagnostic decodes only the configured
assessment cameras while retaining all original frame identities in its report.
A still-held insertion failure follows the distinct held-key return and
planned release/exit path.

For a repeated session, [`run_live_precision_session()`](precision_insertion/live_orchestration.py)
now keeps this same camera context open while the demo-local
[`run_session_actions()`](precision_insertion/action_loop.py) dispatches the
runner's existing decisions. See [action-loop integration](ACTION_LOOP.md)
for the handler contract and routing. The new
[`LiveKeyPlanningHandlers`](precision_insertion/live_planning_handlers.py)
connects fresh key capture, complete v8 preflight, same-capture candidate
continuation; the recovery handler owns stock AutoDex clear-view reset after
a judged grasp miss.
[`LivePickupHandlers`](precision_insertion/live_pickup_handlers.py) connects
the selected attempt to the existing pickup execution boundary and a
measured-squeeze lift preflight. It requires a commissioned pickup executor;
the unmodified stock follower is explicitly rejected by that execution
boundary. A passing lift preflight is not a lift command or grasp label.
[`LiveLiftHandlers`](precision_insertion/live_lift_handlers.py) connects the
measured lift to a fresh raw capture on the same AutoDex cameras and the
existing read-only Qwen lift checkpoint. It requires a commissioned lift
adapter, camera-side exposure metadata and exposure-aligned robot feedback;
unknown VLM outcomes do not become positive grasp labels.
[`LiveHeldRelationHandlers`](precision_insertion/live_held_relation_handlers.py)
handles the likely occluded-key case after visual grasp success: it reads a
grasp-specific *physical* key/hand calibration, measures current joints and
invokes the existing uncertainty-bounded held-transfer preflight. No such
physical calibration artifact is present locally, so a MuJoCo squeeze pose
cannot be used to advance this branch.
[`LiveTaskHandlers`](precision_insertion/live_task_handlers.py) connects the
passing held-transfer plan to execution, a same-session Qwen pre-insertion
hold checkpoint using raw images only, then connects the bound guarded 20 mm
stroke to a required full-hand-open/clear-view exit. Each camera's initial
tabletop raw image is paired with its final clear-view raw image; the released
key is classified as inside/outside/unobservable. The first successful label
ends the session with the key released in the socket.
[`LiveReposePlanningHandlers`](precision_insertion/live_repose_handlers.py)
reuses the exhausted tabletop pose's saved key capture for the existing
directed v8 reset preflight, including seed/cell continuation. It needs a
target-pose/release-cell selector.
[`LiveReposeExecutionHandlers`](precision_insertion/live_repose_execution.py)
connects that plan to an injected physical reset executor and then reuses
the AutoDex key capture and clear-view gate to label the observed landing.
The physical reset controller and its independent execution log remain
AutoDex-PC inputs.
[`LiveHeldReturnHandlers`](precision_insertion/live_recovery_handlers.py)
handles the released-key failure route. From the already measured open-hand
clear view it runs the shared SAM/FoundPose key perception, restores a key
that is not at the trial-start pose/position with the existing v8 reset
planner, and recaptures it before recording reset success.
[`LiveGroundedRetryHandlers`](precision_insertion/live_retry_handlers.py)
now replays the multi-view raw snapshot saved at the pre-insertion hold,
uses Qwen cylinder-tip/axis grounding, and saves a socket-XY correction for the
**next physical pickup**. It does not command lateral motion or another
insertion in the same attempt and does not acquire a post-failure image. After
released-key perception/restoration, clear view and fresh key
perception, a same-pose/same-grasp new attempt uses the cumulative wrist
command offset `p_(n+1)=p_n+w*d_n` (default `w=1.0`); each diagnostic is
bounded to 10 mm and there is no separate cumulative offset bound.
Insufficient image evidence produces a labeled single-view estimate or
exploratory search step; it does not silently invent a multi-view measurement.
[`LiveTransferRecoveryHandlers`](precision_insertion/live_transfer_recovery.py)
now reuses the same held-return/release plan after a completed transfer only
when controller feedback still reports a held key and the pre-insertion VLM
sees the key matching the held prediction, with a measured hand-pose error.
The five return phases are logged without inventing an axial withdrawal.
Suspected slip, incomplete transfer and other uncertain key states still
require a supervised tabletop recovery record. Both routes end in a fresh
key capture; only its observed landing label permits the next grasp.
Stopped pickup/lift commands and unknown post-lift or pre-insertion VLM
verdicts also use that supervised route. Their grasp/arrival labels remain
unknown where appropriate; an unknown pre-insertion verdict is recorded so
the session never repeats the already executed transfer.
[`compose_live_handlers()`](precision_insertion/live_orchestration.py)
requires exactly one handler for every nonterminal action before the
session begins, so an absent robot-PC callback is visible before pickup.
A successful 20 mm label still requires admitted key depth. The optional
[exposed-depth admission path](DEPTH_ADMISSION.md) now connects replayable
multi-view/CAD key-depth estimation to both first and retry insertion labels,
provided a matching reviewed held-out error audit exists. Unverified
controller depth claims remain masked. This checkout has no real-camera
commissioning audit, so its synthetic positive-label tests are not evidence
of physical insertion success.
A missing handler returns `needs_handler` at the exact pending phase; no
trial result is invented. The guarded pickup/lift hardware adapters, raw
post-lift camera producer, physical held-relation calibration data, real
transfer/insertion controllers and camera callbacks,
guarded withdrawal, independent controller logging, later VLM observations
and the real repose executor are **not bundled**, so this is not yet an unattended
robot loop.

`board_snap` and `init` are the metadata-preserving adapters described in
[CAMERA_FRAME_HANDOFF.md](CAMERA_FRAME_HANDOFF.md); all `commissioned_*`
variables need measured values from the AutoDex rig. A later NAS addendum now
contains 35 complete **real 20-camera shots**, but none has verified
per-camera exposure times or repeated session socket measurements. Newly
generated canonical FoundPose PTHs have not passed those images' pose QA.
See [NAS_PERCEPTION_QA.md](NAS_PERCEPTION_QA.md)
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
acquisition-bound multi-view pose capture, rejects masks that include the
frozen socket, saves that pose and checks **measured** 13-DOF Franka/Inspire
feedback against its exposure interval. The runner's existing pose-conditioned
v8 pickup/transfer/20 mm preflight overlaps a new raw initial-tabletop VLM
snapshot. The snapshot exposure must finish before this non-motion handler
returns; PNG persistence continues in the background and is joined by the
first VLM reader or run cleanup.
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
    key_prompt="white plastic cylindrical peg",  # cylinder mode; square needs its own prompt
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
API still requires commissioned camera clocks, real-image-validated key/socket FoundPose
representations, a complete endpoint catalogue and live feedback; there is
no safe robot-execution CLI yet.

Run the current offline tests from the repository root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests
```

An explicitly **synthetic**, read-only full-chain planner diagnostic is
available when the calibrated camera session is not yet captured. It uses
real v8 candidate files and actual Franka/Inspire planner queries, but the
table/key/socket placements and robot start are hypothetical:

```bash
AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1 \
  PYTHONPATH="$PWD/demo/precision-insertion/compat" \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/diagnose_synthetic_full_chain.py \
  --shared-root "$HOME/shared_data" \
  --catalog "$HOME/shared_data/AutoDex/precision_insertion/endpoint_catalogs/square_1p5mm_20261011_v8_8_nominal_r2.json" \
  --mode square --gap-mm 1.5 --pose-stem 004 \
  --table-z-m 0.04 --key-x-m 0.4 --key-y-m 0 \
  --socket-x-m 0.6 --socket-y-m 0 --max-candidates 7 \
  --planner-mode native-locked-experimental \
  --output-dir /tmp/precision-synthetic-square-pose004-NEW
```

Use a **new** output directory each time. The report records source hashes,
the hypothetical transforms, planner output, candidate status or a preserved
code-failure traceback, and `robot_ready=false`; it must never be replayed on
hardware. The sibling `<output-dir>.timing/timing_summary.json` records the
same per-candidate pickup, insertion and held-path audit spans as a live run;
planner construction occurs before this trace and is not included. The demo
defaults to `use_cuda_graph=False` for historical diagnostic compatibility;
add `--use-cuda-graph` when comparing with the live planner default. The demo
installs two narrowly guarded, process-local compatibility
adapters for this checkout: vendored cuRobo's default sample-count typo and
the existing AutoDex singleton-world call to cuRobo's batch IK updater. It
does not change either source tree. In the initial square 1.5 mm synthetic
scene, the default Cartesian route produced an actual `0.03725 rad`
Inspire-joint motion in a supposedly held transfer trajectory. The fixed-hand
contract rejected that candidate and now records the deviation before
examining the next eligible grasp. With AutoDex's explicit **experimental** native
locked-hand mode, `table/4/5102` passed the sampled offline chain in this
same hypothetical scene. Its hand drift was below `3.6e-8 rad` in lift,
transfer and axial paths. The diagnostic records the selected planner mode
and rejects a mismatched environment flag. The native mode has **not** passed
hardware-stack validation, and neither result is a physical grasp/insertion
outcome or a motion authorization. See [synthetic planner validation](SYNTHETIC_PLANNER_VALIDATION.md).

To inspect a saved passing report after transferring it to another PC, run
the independent **read-only** artifact/trajectory verifier:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py verify-saved-preflight \
  --report /path/to/saved_trial/report.json
```

It rejects changed scene/NPZ bytes, disagreement between the selected grasp
and sampled audit, nonfinite/oversize joint steps, hand motion while the key
is supposedly held, and disconnected pickup/lift/transfer/axial segments.
It does **not** rerun collision planning, verify camera calibration, measure
the actual post-squeeze key pose, or authorize Franka motion. A report from
`diagnose_synthetic_full_chain.py` remains synthetic even if this verifier
passes.

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

# Board-free cylinder trial

For a session with no physical ChArUco board, see [BOARD_FREE_MODE.md](BOARD_FREE_MODE.md).
It uses the existing camera and hand-eye calibration, requires an explicit
working-surface height, and remeasures the socket with FoundPose.
