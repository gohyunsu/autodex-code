# Current code audit: key tip, transforms, and recovery latency

This audit compares the local stock `/home/robot/AutoDex` checkout at
`448732d` with `feat/precision-insertion` through `8af32ec` and the changes
described below. It is a source and saved-evidence review. No camera,
controller, or robot was started.

## What is actually connected

| Step | Live source and result | Stock relationship |
| --- | --- | --- |
| Tabletop key | Capture-PC SAM3/FoundPose, admitted by `key_perception.py`, yields a fresh `T_robot_key`. | Reuses AutoDex's multi-view object processing. |
| Candidate wrist | `candidates.planner_candidate_override` reads object-local `wrist_se3.npy` as `T_key_hand` and passes `T_robot_key @ T_key_hand` to the stock planner. | This is the convention in stock `run_auto.py`: the candidate wrist is transformed by the observed object pose before planning. |
| Post-lift held relation | `nominal_postlift.py` uses that selected candidate relation; other explicit routes can use an independently observed or bounded relation. `compensated_key_hand_relation` maps a next-trial socket XY correction into the relation's key-frame translation. | Stock `run_auto.py` records both world `wrist_se3.npy` and `inv(T_robot_key) @ wrist_se3` for later object-local reuse. The demo's nominal relation follows the latter convention. |
| Preinsert / entry / verification | `targets.build_rigid_insertion_targets` composes `T_robot_socket @ T_socket_key @ T_key_hand`. The socket-frame XY offset is applied before this composition; the insertion direction is rotated by `T_robot_socket`. | Stock's world-vertical Jacobian stroke cannot directly represent a tilted socket axis. The demo's axial path and sampled held-key audit are insertion-specific. |
| Failed return / reorientation | `original_pose_restoration.py` measures a fresh tabletop key, classifies its v8 pose, loads the selected local wrist relation, plans pickup and a CAD rest target, then checks release and exit. `repose_transition.py` uses saved v8 reset seeds for pose changes. | The candidate-to-world wrist multiplication and wrist-to-object inversion match stock `run_auto.py` and `executor/real.py`. A planned landing is not proof of the actual landed pose; the next trial recaptures the key. |

For original-pose restoration, `reset_landing_geometry` transforms both the
trial-start and newly observed world key poses through `inv(c2r)` and compares
the same CAD mesh center in robot coordinates. `original_center_release_xy`
subtracts the target v8 pose's rotated local mesh center from the desired
robot-frame center, so a lying-to-upright reset does not shift the physical
center by the key's half-length. The placement wrist is composed as
`T_robot_key_rest @ T_key_hand`, equivalent to stock's
`T_obj_target @ inv(T_obj_in_wrist)` when the grasp relation is the selected
v8 object-local wrist. A fresh key capture follows each release; the live
handler retries restoration up to three times and accepts a supported key
within 30 mm of the original center. These are planned/observed transforms,
not proof that the nominal post-squeeze key-to-hand relation is physically
exact. The restoration, target-selection, and live-recovery tests pass without
connecting hardware.

A read-only replay of the actual `20261021_140856_019116` attempt
`attempt_d729c8e4814a401abc219bb3371c2747` checks the easy-to-miss
end-for-end cylinder case. The original observed key's local +Z axis points
almost opposite to the canonical `001` rest pose (axis dot product `-0.9085`).
The two pose **origins** differ by `78.145 mm` in XY. With the nominal
80 mm cylinder's local midpoint `[0, 0, 0.040] m`, however, the planned rest
and original physical centers agree in XY to floating-point precision. This
supports the center-compensated transform in the current code; comparing pose
origins alone would incorrectly call this a large return error. The saved
controller record contains the pickup, lift, transfer, supported descent,
release, exit, and acknowledged stop, but this historical attempt has no
`reset_landing.json` or later key image. It therefore proves the planned
coordinate arithmetic and recorded robot waypoints, not where the key
physically landed. The nominal midpoint has not been rechecked against the
historical mesh because its stage was moved; the current runtime must not
depend on that archived stage. The source hashes, matrices and calculation are
in `/home/robot/shared_data/hyunsu/tmp/20261026_restoration_pose_replay/`.

`run_auto.py`, `run_pipeline.py`, `scene_cfg.py`, and `executor/real.py` are byte-identical in the two local
checkouts. `autodex/planner/planner.py` differs by one constructor option:
the precision checkout may inject a robot URDF path for the right-angle
FR3/Inspire mount. This makes the planning model selection explicit; it does
not change stock planning algorithms. Tests in `test_candidates.py`,
`test_targets.py`, and `test_next_trial_xy.py` cover the object-local wrist
convention, a tilted socket axis, cylinder yaw, and socket-frame correction.
Stock `GraspPlanner.plan` expands cylinder candidates over its yaw grid only
when it loads candidates itself. The precision pickup passes one explicit
`candidate_override`, which stock deliberately leaves unexpanded. This keeps
the selected v8 grasp fixed across controlled retries and preserves the same
`T_robot_key @ T_key_hand` convention, but can miss a feasible yaw-equivalent
pickup that stock's broader search would find. The later cylinder target yaw
gauge is a separate held-key alignment step; it does not silently add pickup
grasps. The tilted-socket correction test now also verifies that a socket-XY
increment becomes `R_robot_socket @ [dx, dy, 0]` in robot coordinates.

The FR3 executor is a separate comparison: `src/execution/franka_executor.py`
is **not** byte-identical to stock. Its precision-branch changes inject the
right-angle `O_T_EE`/link8-to-hand transform, retain stock's 0.40 near-object/held-object
speed scale as one configured value, and advance its streamed joint reference
using measured tick time. The streaming command period remains the stock
default of 20 ms (50 Hz). `follow_progress.py` keeps the reference moving
continuously when the arm lags or is already farther along the planned path;
`stream_timing.py` keeps filtering and command validity tied to wall time.
The current `_follow` computes free-hand playback as `speed * profile_scale`,
where `profile_scale` is 1.0 beyond 30 cm, falls linearly to 0.40 from 30 to
15 cm, and stays 0.40 nearer the object. While holding a key it instead uses
`min(speed, 0.40)` and does not multiply by the proximity profile. The live
pickup and held-transfer callers use the default `speed=1.0`, so their near
and held playback scales are both 0.40. The branch also sleeps only for the
remainder of each 20 ms tick, whereas stock sleeps a full 20 ms *after* state,
FK, and command work; actual command rate still depends on robot-PC load.
This source comparison finds no second executor-side near/held slowdown to
remove. It does not measure the achieved joint speed or smoothness.
The mount-frame comparison uncovered a separate correctness error: ParaDex
`position` is libfranka `O_T_EE`, which matches recovered-URDF `fr3_link8` in
three saved v8 episodes. The previous precision factory passed a
`fr3_link7 → base_link` matrix to the executor and double-applied the 107 mm
`fr3_joint8` segment in live wrist feedback. It now passes
`fr3_link8 → base_link`; the branch's generic FR3 default is converted to
the same EE frame. The saved replay maxima are 0.259 mm for `O_T_EE` versus
URDF link8 and about 107 mm for the old hand-base estimate; see
`ADAPTER_GEOMETRY_AUDIT_20261015.md` and the hashed report under
`/home/robot/shared_data/hyunsu/tmp/20261026_fr3_otee_mount_frame_audit/`.
The planner's fixed-joint URDF did not change. This fixes the feedback pose
used by endpoint checks and fallback proximity, without loosening a motion
failure threshold or changing a planned target.
`reset_scene.py` keeps a visually missed key at its tabletop scene pose
instead of projecting a phantom carried key from the wrist. That tabletop
pose can be stale if the missed grasp physically moved the key; a fresh key
observation is still required before a new attempt. These changes are covered
by `test_follow_progress.py`, `test_reset_scene.py`, and the integration
tests, but no saved-image replay can prove the robot's actual tracking or
contact behavior.

`autodex/planner/jacobian_stroke.py` also differs from stock: its FR3
`held_object_speed_scale` default is 1.0 instead of 0.40. The numerical
Jacobian, collision checks, and retimer are otherwise unchanged. This is a
speed-policy correction in this separate checkout, not an insertion-specific
geometry rule: both the stock vertical stroke used by lift/repose and the
demo's socket-axis continuation use the same default. The FR3 executor still
applies the 0.40 near-object/held playback scale once. The saved
`20261020_094911_916554` run showed the previous planner-plus-executor
reduction and the read-only replay in `TRIAL_20261020_094911.md` compares
reference lengths; it does not measure post-change physical speed. The
`test_axis_jacobian.py` retiming test checks that the socket-axis path reaches
the retimer at the base joint limits. This deliberate stock divergence
should remain visible when comparing timings or rebasing the checkout.

As an independent arithmetic replay, the saved post-lift reports from
`20261022_101456_545928`, `20261022_103429_752726`, and
`20261022_105828_126084` were checked against their frozen socket records.
In all three, `T_robot_key_stage @ T_key_hand` equals the reported
`T_robot_hand_stage` for preinsert, entry, and verification to displayed
floating-point precision. Entry-to-verification translation equals
`0.020 m * insertion_axis_robot` to displayed precision. Their shared axis
is approximately `[0.0092, -0.0016, -1.0000]` in robot coordinates. This
checks internal transform composition only; it cannot establish that the
measured socket or hand-eye pose was physically correct.
The same calculation was repeated on six saved **20 mm** post-lift reports
from `20261020_090836_567804` through `20261021_121211_290331`.
All six compose `T_robot_key_stage @ T_key_hand` exactly to their recorded
wrist targets, place the entry key origin on the recorded socket XY axis,
and move entry to verification by `0.020 m` along the recorded insertion
axis (floating-point residual below `1.1e-16 m`). These records have zero
retry XY correction, so they do not validate a physically applied correction.

A separate read-only arithmetic replay applied a synthetic socket-frame
`(+2, -1) mm` wrist correction to all ten saved 10 mm and 20 mm post-lift
reports. The copied task geometry was used only when its SHA-256 matched each
saved report. For preinsert, entry and verification, the current
`compensated_key_hand_relation` moved the wrist by the expected
`R_robot_socket @ [0.002, -0.001, 0]` with maximum residual `9.63e-17 m`
and no rotation change. The saved nominal compositions were exact at JSON
precision. All ten historical yaw gauges and corrections were zero. The
[per-attempt arithmetic report](/home/robot/shared_data/hyunsu/tmp/20261026_xy_correction_saved_replay/xy_correction_report.json)
tests the implemented frame conversion against real saved plans; it cannot
establish physical execution or a nonzero live retry.

## SAM3 half-ellipse: from offline result to optional live XY path

At `568107f`, `held_key_tip.distal_key_mask_half_ellipse` was called only by
saved-image tools. The live CLI had only `vlm` and `tracked_tip`. This review
added an explicit `sam_half_ellipse` option; the ordinary `vlm` default and
the stock AutoDex snapshot producer remain unchanged:

- `vlm`: `retry_session._compute_grounded_alignment` asks Qwen for raw-pixel
  key landmarks/axis and computes socket-frame XY advice.
- `tracked_tip`: `retry_session._compute_tracked_alignment` uses the initial
  tabletop key mask as an appearance reference, then estimates a tip from
  saved preinsert RGB frames and optional temporal frames. This is **not**
  SAM3 half-ellipse fitting.
- `sam_half_ellipse`: the saved, frame-bound undistorted preinsert PNG is
  cropped around the planned key axis. The existing distributed Qwen backend
  returns a coarse key box. A separate, warmed local process running the
  installed AutoDex SAM3 model segments **that box's ROI of the same PNG**.
  `distal_key_mask_half_ellipse` fits the mask boundary; the existing
  multiview tip XY estimator converts the approximate point to bounded
  next-trial advice. The attempt report records the image frame ID, camera
  timestamp/hash, prompts, response, ROI, fitted center, mask PNG hash,
  per-view model time and CUDA peak memory. The small mask PNGs are saved in
  `xy_retry_assessments/NNN/sam_roi_masks/`. The raw preinsert image bundle is
  referenced by its saved manifest and source PNG hashes; the retry writer
  no longer re-encodes all 20 camera PNGs into a second bundle.

The live SAM route passes its diagnostic through `LiveGroundedRetryHandlers`
to the real `SessionRunner` ledger. A same-grasp, same-tabletop-pose preflight
then consumes the bounded socket-frame increment as a cumulative wrist
correction; an intermediate key reorientation retains the pending advice.
The corrected key-to-hand translation uses `R_socket_key.T @ [dx, dy, 0]`,
so composition through `T_robot_socket @ T_socket_key` moves the wrist by
`[dx, dy]` in the socket plane while leaving the CAD key target on the socket
axis. The session regression now runs the SAM handler into the real ledger
and through the fresh same-grasp attempt for both a nonzero and zero metric
increment. It confirms the software seam, not the physical tip location.

The local SAM3 worker is request-bound to one camera at a time. If its reply
times out, is malformed, or carries another request/camera ID, the client now
terminates that worker. The affected view is recorded as unavailable; a later
camera cannot consume a stale reply as its own mask. Without enough remaining
views, the retry produces no metric XY advice and follows the ordinary
recovery path. Subprocess protocol tests cover timeout, wrong ID, and bad
JSON without contacting a camera or robot. The worker's ordinary error reply
now includes its camera ID, matching the client protocol: a segmentation
error on one view can be recorded and the healthy worker can process another
view instead of being mistaken for a stale response.
The current SAM3 `model_builder._load_checkpoint` can print missing-key
diagnostics to stdout. The worker now reserves stdout for JSON replies and
redirects model/library fd-1 output to stderr before loading the checkpoint.
A subprocess regression confirms that a diagnostic print cannot precede or
corrupt the `ready` reply. The checkpoint and AutoDex SAM3 source are present
on this PC, but CUDA is currently unavailable, so model loading and live
inference were not exercised in this audit.
The same per-view handling now covers a malformed result payload (wrong image
size, unreadable mask, or missing inference metadata): the client retires the
worker and reports that camera as unavailable. A mask with the wrong ROI shape
is omitted as one invalid observation. Neither is allowed to supply a pixel
point, and healthy views can still yield an XY diagnostic. If the backend is
unavailable for every view, the live retry handler records an unavailable
diagnostic and continues recovery without a measured XY update.

On the real preinsert bundle of `20261022_103429_752726`, three read-only
repetitions of reading and re-encoding 20 PNGs to memory took
`12.502 / 11.599 / 12.102 s`. Reading and hashing the five selected original
PNG files took `0.009 / 0.008 / 0.007 s`. This isolates the redundant image
work and excludes filesystem writes, inference, and robot time; it is not a
measured end-to-end episode speedup.

Full-frame SAM3 with `white cylinder` was rejected for this route: on six
saved accepted-socket views it found the key in only two ROIs; changing the
full-frame text prompt did not make all six reliable. The warmed ROI worker
reproduced the saved half-ellipse centers in all six, with maximum delta
`0.000017 px` (numerical replay). Its model load took 12.14 s with cached
weights and its per-ROI inference took 0.25–0.62 s; one earlier cold load
took 56.38 s. The results and mask files are under
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/`.
The saved comparison still has no physical face-center ground truth, and the
new mode has not run a live insertion. It is opt-in only.
The five configured views of accepted 10 mm run `20261022_103429_752726`
were also replayed through the live alignment function using saved Qwen
answers and source PNGs. All five were admitted, and the result differed
from the earlier offline 3D proxy by at most `0.042 mm` per coordinate.
This is a same-evidence numerical consistency check, not physical validation.
That replay produced `diagnostic_metric_xy_correction` with source
`multiview_tip_xy_vertical_axis_assumption` and a bounded socket-frame
increment of `[-9.6561, -2.6000] mm`. The session test now exercises this
SAM/tracked-tip source through saved advice, fresh same-grasp preflight and
the next attempt's wrist-correction field, alongside the Qwen line source.
It verifies the wiring and arithmetic only; no historical robot trial
applied this value.
On the same five real saved observations, removing any one camera changes the
fitted socket-frame XY by at most `0.79 mm`. Across all ten three-camera
subsets, the largest change from the five-view fit is `2.52 mm`. Camera
`25322649` has a `6.83 px` residual in the five-view fit; it remains within
the tip proxy's `15 px` observation tolerance. This sensitivity check supports
using the five selected views for the optional SAM retry and retaining each
per-camera residual in the report. It does **not** establish the true physical
key face center or justify a tighter failure gate. The source-frame replay is
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/tip_subset_sensitivity_saved_replay.json`.
The saved mask/component review is at
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/mask_component_association_saved_replay.json`.
In the visually checked `24122734` and `25322638` raw frames, the largest
mask component covers the held key while a smaller disconnected component
covers a white hand part; choosing the largest component removes that hand
part in these examples. Camera `25322649` shows a narrow key silhouette and
the highest five-view residual. These checks do not establish that the largest
component will always be the key when the socket overlaps the ROI.

An end-on camera can project the planned key's rear and distal tip to the
same pixel. The SAM mask can be valid while its distal-edge direction is
undefined. That view now records `key_mask_end_unavailable` with the source
mask; other cameras still contribute to the XY estimate. A
three-camera geometry test supplies one end-on view and verifies that the two
remaining views yield a metric result. This avoids discarding independent
camera observations without loosening their geometric fit.

A centered measured tip exposed a retry seam that the nonzero saved runs could
not exercise. The two-view fallback treated a valid socket-frame XY point near
`(0, 0)` as missing and invented a 2 mm exploration step. The three-plus-view
tip fit could return a zero step, but the attempt ledger rejected it and ended
the retry. Both now keep a measured zero correction as zero. The same-grasp
next attempt records the diagnostic source even when its cumulative wrist
correction remains `(0, 0)`; a zero exploratory step with no measured tip is
still invalid. Synthetic centered-tip tests and the full session-ledger test
cover this path. This avoids turning a valid centered observation into an
unmeasured motion and does not assert physical insertion success.

The shared tracked-tip/SAM multiview fit previously identified outlier
cameras but returned the point fitted using **all** cameras. In a five-camera
synthetic calibration check, one key-tip point displaced by 100 px left four
inliers yet moved the fitted XY point by about 6.4 mm and its height by
44.7 mm. With the live 10 mm nominal-height bound, that point became an
exploratory fallback despite four exact inlier points. The estimator now
refits the point using its existing inlier set when at least one camera is
rejected; it changes no admission limit or next-trial step bound. The same
five real saved observations from `20261022_103429_752726` all remain inliers
and produce the identical point. The frame IDs, timestamps, source PNG byte
hashes, and resulting residuals are in
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/tip_inlier_refit_saved_replay.json`.

The same estimator also bypassed its configured minimum camera parallax when
at least three tip views were present. Three synthetic cameras separated by
only 0.011 degrees produced a `diagnostic_metric_xy_correction` despite a
5-degree limit. It now applies that existing limit before triangulation and
again to the accepted inlier cameras; insufficient parallax uses the existing
nonmetric XY fallback. Replaying the five saved 10 mm views with production
limits gives 137.25 degrees of parallax and the unchanged point
`[0.016843, 0.004535, 0.083036] m` in the socket frame. This checks the
geometry and saved data only, not physical endpoint accuracy.

The three-plus-view tip fitter also used algebraic projection for residuals,
which gave a zero-error metric XY correction for a point **behind every
camera**. The two-view fallback already rejects nonpositive camera depth.
The multiview fitter now reuses that positive-depth projection: a synthetic
three-camera backward-ray case produces only explicitly unmeasured search
advice, while the saved five-view 10 mm case retains all five inliers and its
tip changes by `2.12e-11 m`. The saved point is `0.793–1.372 m` in front of
those cameras. The source PNG byte and decoded-pixel hashes, frame IDs,
timestamps, calibration hash, camera depths, and replay parameters are in
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/positive_depth_saved_replay.json`.
This is a physical visibility requirement already used by the existing
two-view code, not a new insertion tolerance.

There was one more outlier failure before the inlier refit: a grossly wrong
view could spoil the **initial** all-view SVD so badly that no three inliers
were found. In a five-camera synthetic case with four exact points, moving
the fifth by `(500, -500) px` previously produced unmeasured exploratory
advice. The fitter now tries deterministic two-camera seeds only if its
ordinary all-view fit is unusable. It ranks candidates by the number of
camera projections agreeing under the existing pixel tolerance, refits those
views, and still requires positive depth, three inliers, sufficient parallax,
and the existing nominal-height bound. That case recovers the exact four-view
point. On the five real saved 10 mm observations, the ordinary path remains
selected with all five inliers; its point changes by less than `2.2e-11 m`.
The conditional pair search took about `8.5 ms` per warm synthetic call here,
versus `1.7–2.5 ms` for the ordinary five-view replay, excluding image/model
I/O. The five saved masks have long-axis agreement above `0.9989` with the
planned direction, although that agreement alone does not prove face-center
accuracy. No insertion tolerance or next-trial step limit changed. The source
frame IDs, PNG and mask hashes, calibration hash, observed pixel points, and
recomputed result are in
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/tip_pair_seed_recovery_saved_replay.json`.
The available saved frames still lack an independently measured physical face
center.

The current XY function was also replayed against **all three** visually
accepted 10 mm socket episodes in the saved SAM half-ellipse comparison,
using each episode's original point pixels, frozen intrinsics/extrinsics,
socket pose and post-lift target. It returned metric XY advice for the two-,
five- and three-view sets, respectively. The reconstructed socket-frame tips
were `[0.016486, 0.003040, 0.083239]`,
`[0.016843, 0.004535, 0.083036]` and
`[0.031266, 0.009530, 0.093847] m`; the commanded preinsert tip height was
`0.085 m` in all three. The two-view case has no independent third-view
cross-check. The three-view case is `8.85 mm` above nominal and therefore
passes the current `10 mm` height limit narrowly; its saved offline
leave-one-view-out error reached `10.74 px`. This is a reason to retain the
per-view observations and avoid presenting its XY advice as a measured true
key center. The [read-only replay](</home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/accepted_runs_current_xy_replay.json>)
records each original PNG and mask hash, frame ID, timestamp, calibration and
post-lift report hash, limits and result. It does not rerun SAM or validate a
physical contact outcome.

The ten saved ROI-worker masks behind those three episodes were then replayed
through the **current** `projected_key_axis_crop` and
`distal_key_mask_half_ellipse` functions, using each saved Qwen box crop/ROI,
its frozen camera calibration, and its post-lift target. All ten recomputed
full-image centers exactly matched the recorded worker-replay centers at
JSON precision. The [mask-to-point replay](</home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/roi_mask_to_tip_current_code_replay.json>)
binds each mask to its source PNG hash, frame ID, timestamp, calibration and
post-lift report. This checks the pixel-coordinate mapping and fit using
existing masks; it does not rerun Qwen/SAM or independently measure the
physical distal face. The full `demo/precision-insertion/tests` suite passed
with `1204 passed, 5 skipped` in the planner environment without camera or
robot access.

## Extra criteria and their purpose

The stock planner supplies IK, world/self collision, and trajectory
generation. The stock executor detects controller errors and tracks motion.
The insertion demo additionally checks the fixed socket and held-key geometry
at sampled path points, same-request image identity, exposure-aligned robot
state, source asset identity, and task labels. These checks have no stock
insertion equivalent because stock AutoDex does not perform this socket task.
They should be evaluated against observed false stops and physical evidence,
not removed solely because the stock pipeline lacks them. In particular,
unknown key location after a controller abort cannot be turned into a
measured `outside_socket` XY correction.

The active board-9 FR3 stage previously limited the first key preflight to
one v8 grasp. A failed first candidate returned `candidate_budget_exhausted`;
`LiveKeyPlanningHandlers.continue_same_capture` then immediately planned the
remaining candidates from the same saved key pose and measured joint state.
Among 34 saved cylinder preflight reports from October 20–22 (24 with a
20 mm gap and 10 with a 10 mm gap), 15 first reports ended at this artificial
budget (10 with a 20 mm gap and five with a 10 mm gap). The stage now uses
`prepare_next_live_key`'s existing unlimited default, so one preflight
evaluates eligible grasps until one passes
or the pose is exhausted, as stock AutoDex's candidate search does. The
general same-capture continuation remains available for an explicitly
budgeted stage. This removes an intermediate report and decision lookup, not
the real IK or held-key checks. Historical timing cannot establish a measured
live speedup for this configuration change.

The five sampled-path rejections in this set were all nominal held-key mesh
collisions with the table during transfer, rather than an added clearance
margin or path-sampling threshold. Those collision checks remain in place.

The optional `tracked_tip` retry path can use a capture-PC JPEG window around
the exact preinsert frame. The per-camera record now distinguishes the raw
PNG observation from the effective temporal-median pixel used for multiview
alignment, and its `admitted` field reports whether that effective view was
actually included. Both sources retain their frame IDs and source hashes in
their respective saved manifests. The ordinary `vlm` and optional
`sam_half_ellipse` modes continue to use the same saved undistorted PNG for
their per-view observation.

The opt-in `tracked_tip` route previously re-read all 20 preinsert PNGs even
when the live run configured five assessment cameras. It now uses those five
and adds the temporal tracking camera only when the saved window identifies
the exact preinsert frame. An offline replay with no camera selection still
uses every available view. The raw 20-camera capture and its original frame
IDs are unchanged. On one real saved preinsert bundle, warmed read, decode,
decoded-pixel-hash and PNG-byte-hash loops took 0.868/0.893 s for 20 views
and 0.266/0.291 s for the six configured-plus-temporal views. This isolates
selected-view I/O; it excludes full raw-capture verification, inference and
robot time. A public diagnostic test binds the selected frames into the saved
retry report. None of the ten stored nominal post-lift reports in this
experiment tree contains a nonzero `wrist_correction_socket_m`, so actual
next-attempt application remains unobserved in these episodes.

The old `max_total_offset_m` retry budget remains an optional legacy field;
the active FR3 stage factory sets it to `None`. The live launcher defaults to
the VLM estimator. Existing `BOARD_FREE_MODE.md` code is statically tested
but the former recipient stage was moved, so `run_robot_pc.sh --check` cannot
currently certify an active robot-PC asset installation. The launcher now
requires an explicit current `PRECISION_STAGE_ROOT` and checks its three
input files before opening any camera or robot connection; it no longer
defaults to the moved 2026-10-13 temporary path.

## Recovery-path latency change in this review

The live action loop queries `SessionRunner.current_decision()` more than once
between a passing post-lift plan and transfer. The decision previously ran the
full nominal/bounded post-lift verifier on every lookup, including candidate
and calibration reads plus both trajectory arrays in the saved NPZ. The plan
is already verified when saved, and `execute_bound_held_transfer` repeats the
full source/archive verification before it permits a motor command. Decision
lookup now checks the bound report hash only. If a source file changes after
planning, lookup can still return `transfer_execution_gate_required`, but the
executor rejects the changed source before motion. Tests cover repeated
lookup, report mutation, source mutation, and archive mutation. This removes
duplicate pre-command I/O; no live timing gain has been measured.

The saved preinsert checkpoint was fully reverified at creation, then again
at insertion handoff and failed-trial XY diagnosis. The latter calls decoded
the selected source PNGs and replayed the visual record even during the same
live session. It now follows the existing live lift-checkpoint cache: the
first verification remains complete, while later calls in that session reuse
the validated report only if its byte hash is unchanged. Offline verification
still takes the full path, and the saved XY diagnostic independently reads
and checks its selected undistorted frames. The reuse is timed as
`evidence.live_preinsert_report_reuse`. A fixture test confirms that a changed
report invalidates the cache and that it does not survive the session scope.
No live-session speedup is claimed because the active stage is absent.
In the saved `20261022_103429_752726` timing trace, 35
`evidence.live_manifest_reuse` calls took `0.002 s` in total; no
`evidence.png_read_verify` or `evidence.undistort_replay` span was recorded.
The failed-trial XY diagnosis in `20261022_105828_126084` took `2.105 s`,
while its two `audit.geometry_samples` calls took `65.716 s` cumulatively.
These are historical timings before the current geometry-pruning changes.
They do not support removing more frame provenance to address the measured
planning bottleneck.

`repose_path_audit.py` previously recomputed a fixed-pose inverse and called
the exact Coal mesh-pair report at every FK sample. It now uses the same
conservative AABB distance lower bound already present in
`path_audit.py`. A sample is skipped only after an exact distance for that
pair exists and the lower bound exceeds both that minimum and required
clearance. The initial key/table and planned zero-height support-contact
exceptions are unchanged. On the synthetic 33-sample test fixture, exact
pair queries fell from 131 to 40; a single warmed local timing fell from
36.4 ms to 20.4 ms. This is **not** a measured live-session time saving.
The regression compares the entire audit result with and without pruning.

The saved run `20261022_105828_126084` ended after a released-key
`outside_socket` verdict: five `grounded_axis` VLM calls finished, then
`action.assess_saved_preinsert_xy` raised `invalid crop mark was promoted to
full pixels` at 173.07 s. The line parser had already rejected a VLM answer
and produced a null observation with `parse_error`, but the
crop mapper still wrote a dictionary of null full-frame marks. The evidence
verifier interpreted that dictionary as a promoted invalid mark and aborted
the session. The mapper now records `mapped_full_pixels=None` whenever the
parser rejected the answer. The raw answer and parse error remain in the
per-camera record; that view supplies no metric point, while other views can
still be used. The verifier continues to reject actual mapping tampering.
A mixed valid/invalid three-view regression reproduces this path without
robot or camera connections. The aborted attempt has no saved per-view XY
diagnostic report, so the exact rejected answer is not established by that
artifact.

Saved failed-trial XY diagnosis previously used `map_view_calls` in its Qwen
landmark and Qwen-box/SAM3 paths, so one independent camera worker raising a
timeout or runtime error discarded every completed view. Those two retry
paths now record that camera as unavailable and fit from the remaining valid
views. A total model outage remains distinct from a visual abstention: it
records unavailable and cannot generate a synthetic exploratory XY step.
Malformed SAM response identity or payload now retires the worker and marks
that view unavailable; an invalid ROI mask is also omitted as one view.
Neither becomes a measured pixel. Synthetic three-camera tests retain a
metric two-view result after one failed Qwen or SAM worker, and test complete
Qwen, SAM, and mixed outages. The saved
`20261021_111900_186891` grounding run reported CUDA OOM on all five Qwen
views, so this change would not have supplied a metric result for that run.
No live latency or insertion outcome is claimed from these offline tests.

The local SAM3 ROI subprocess is shared by those camera calls. A timed-out
request previously left the process alive, so its late response could be read
as the next camera's response and raise an identity error after otherwise
valid views had completed. The backend now retires that subprocess on timeout;
waiting camera calls fail fast as unavailable while already completed views
remain usable. A subprocess protocol test reproduces the timeout followed by
a second camera request without starting SAM3 or robot hardware.

The five capture-PC Qwen workers have the same request/reply risk. A timed-out
or malformed response previously left its SSH worker open, and an identity
mismatch surfaced as an uncaught `ValueError` in the saved XY path. The robot-PC
client now retires only that camera's channel on timeout, malformed JSON, or
wrong request/camera identity while holding the per-camera request lock. The
worker also reserves stdout for JSON before loading Qwen; library prints and
native writes go to stderr. CPU-only protocol tests cover channel retirement,
other-camera isolation, and stdout separation. This is a demo-specific remote
inference transport; stock AutoDex does not use these SSH workers. It changes
failure containment, not the VLM prompt, mask fit, or XY command value.

The saved 10 mm run `20261022_103429_752726` completed its 20 mm stock-force
stroke, then failed 6.87 s into `trial.release_and_clear` with the generic
`FR3 stream did not finish at a measured stationary endpoint` exception.
Its failure record confirms that a later stop was acknowledged and the
measured maximum joint velocity was 0.0188 rad/s. It does **not** contain the
failed segment or terminal joint error, so this evidence cannot distinguish
an overstrict endpoint check from a real short motion or FR3 fault. The stock
follower's normal streaming convergence is 0.04 rad L2 and may make a final
blocking correction; the precision wrapper omits that correction near the
socket and while holding a key, then checks the measured endpoint before a
hand ramp or next motion. Saved transfer failures show why a stopped stream
alone is not proof of arrival. The endpoint condition is retained pending
measured evidence. New failures now include their release phase and the
measured L2 joint error, velocity, limits, and FR3 error flag in the exception
record. `trial.release_and_clear` now has nested open-hand, socket-axis exit,
and clear-view retract spans. These changes add diagnosis and timing without
changing motion or admission thresholds. `test_socket_release.py` covers both
exit failure phases and terminal error reporting without connecting a robot.
The stock `FrankaExecutor.release()` marks `_holding_object=False` after its
hand ramp, so the axis exit and clear-view retract use the empty-hand follower
profile. The daemon's `stop_streaming` command also accepts a repeated stop;
the demo's measured stop acknowledgement does not introduce a daemon error.
The release callback now retains the verified report and joint paths from its
pre-insertion `prepare()` call. It no longer reloads the key mesh and path
archives after the insertion stroke while the arm is at the socket. The saved
release record still uses the report hash calculated at preparation. This
removes duplicate file validation from the motion path; it does not alter the
planned geometry, stop behavior, or endpoint criterion.

A read-only inventory of root-level saved execution failure records found two
lift, four transfer, and one socket-release failures. The October 17 lift
traces reported an Inspire hold-change gate and a force-norm gate; current
`franka_measured_lift.py` records those measurements without using either as
a hard failure. Two later transfer records with saved post-state joints had
target residuals of 0.0183 and 0.0279 rad L2 (0.0134 and 0.0233 rad maximum
per joint), both within the stock follower's 0.04 rad L2 convergence limit.
Current `transfer_execution.py` no longer applies its separate tighter
Python endpoint threshold to the first-party `FrankaMeasuredLiftAdapter`.
The remaining saved transfer trace records a stopped-short stream; the
release trace lacks a segment and terminal error. Neither can be dismissed
as a false software stop from the saved evidence alone.

The saved 10 mm session `20261022_105828_126084` spent 22.99 s auditing
498 nominal samples before pickup and 42.72 s auditing 263 new transfer/axial
samples after lift. The second audit checks a newly planned path from the
measured post-lift state, so reusing the first path result would be incorrect.
Each audit now also omits Coal's surface-collision query and solid-occupancy
classification when the already computed enclosing-box gap proves the meshes
are disjoint by more than 2 nm. It still computes the exact mesh distance for
the recorded pair minimum and uncertainty margin; overlapping and touching
boxes still use both collision and distance queries. On a local 263-sample
fixture, the complete audit result was identical, Coal collision calls fell
from 1,052 to 38, and three warmed timings changed from 0.117/0.102/0.102 s
to 0.058/0.052/0.053 s. This fixture is much smaller than the real Inspire
meshes, so these timings do not predict the live session's speedup.

## Current offline verification

On October 26, the current worktree's non-moving regressions passed in the
local `planner` Python environment: 44 candidate/stage/policy tests passed,
102 SAM-tip/XY/target/restoration tests passed (three skipped), and 40
live-assembly and retry-handler tests passed. The restoration code computes
the observed and original key poses as `inv(c2r) @ T_world_key`, preserves
the original CAD mesh center when choosing a new v8 rest origin, and
composes the placement wrist as `T_robot_key_rest @ T_key_hand`. The same
branch-local `FR3_CLEAR_VIEW` joint target is used by clear-view, release,
restoration, and repose paths; stock's unmodified clear-view target is not
mixed into these plans. These tests and algebraic checks do not establish physical
placement accuracy. An earlier full test collection stopped while importing
cuRobo because CUDA was unavailable to PyTorch in this environment at that
time; the subsequent full-suite result appears below.
The October 26 retry confirmed the boundary: `nvidia-smi` lists an RTX 3090,
but this execution environment exposes no `/dev/nvidia*` device nodes and
both `autodex` and `planner` PyTorch environments report zero CUDA devices.
This does not establish that the host GPU or deployed robot runtime is down.
The release, completion, clear-view, force insertion, live task, FR3 assembly,
SAM ROI, XY transform, and session-runner regressions were rerun after caching
the release plan: 119 passed, four skipped. One skipped test imports the stock
executor and requires CUDA; it now imports stock only when executed, so the
completion wrapper's CPU-only regressions still run on this PC. Later on
October 26, the full demo suite completed with
`PYTHONPATH=$PWD/demo/precision-insertion /home/robot/anaconda3/envs/planner/bin/python -m pytest -q demo/precision-insertion/tests`:
**1204 passed, 5 skipped in 44.75 s** at local HEAD `56ce93f`. The earlier
CUDA import failure above is historical and is not the current suite result.
These tests cover the SAM diagnostic-to-ledger-to-next-wrist software route,
but the optional SAM mode has no live robot execution or physical tip ground
truth in this audit. After the measured-EE mount correction and its saved
raw-state regression, the same complete command passed again:
**1205 passed, 5 skipped in 42.98 s**. The mount-focused tests passed
**46/46**. A static
`run_robot_pc.sh --check` attempt stopped before any camera or robot connection
because `PRECISION_STAGE_ROOT` is unset and the former recipient stage is in
`hyunsu/tmp/20261023_prev`, not an active installation.

## Remaining evidence needed

1. Replay additional rejected-socket and failed-grasp episodes with this
   optional route; distinguish bad fixture pose from key-mask error.
2. Restore and rehydrate an active recipient stage, then run static
   `run_robot_pc.sh --check` without connecting cameras or the robot.
3. Measure actual planning and recovery times after the AABB change. Local
   unit-fixture timings cannot establish an end-to-end gain.
4. Independently check physical mount/hand-eye and held-key motion on the
   robot before treating planned pose transforms as measured accuracy.
