# Current code audit: key tip, transforms, and recovery latency

This audit compares the local stock `/home/robot/AutoDex` checkout at
`448732d` with `feat/precision-insertion` through `bf3b880` and the changes
described below. It is a source and saved-evidence review. No camera,
controller, or robot was started.

## What is actually connected

| Step | Live source and result | Stock relationship |
| --- | --- | --- |
| Tabletop key | Capture-PC SAM3/FoundPose, admitted by `key_perception.py`, yields a fresh `T_robot_key`. | Reuses AutoDex's multi-view object processing. |
| Candidate wrist | `candidates.planner_candidate_override` reads object-local `wrist_se3.npy` as `T_key_hand` and passes `T_robot_key @ T_key_hand` to the stock planner. | This is the convention in stock `run_auto.py`: the candidate wrist is transformed by the observed object pose before planning. |
| Post-lift held relation | The current cylinder live assembly uses `nominal_postlift.py` and the selected v8 candidate relation. `compensated_key_hand_relation` maps a next-trial socket XY correction into the relation's key-frame translation. The cylinder live assembly does not select the other held-relation implementations. | Stock `run_auto.py` records both world `wrist_se3.npy` and `inv(T_robot_key) @ wrist_se3` for later object-local reuse. The demo's nominal relation follows the latter convention. |
| Preinsert / entry / verification | `targets.build_rigid_insertion_targets` composes `T_robot_socket @ T_socket_key @ T_key_hand`. The socket-frame XY offset is applied before this composition; the insertion direction is rotated by `T_robot_socket`. | Stock's world-vertical Jacobian stroke cannot directly represent a tilted socket axis. The demo's axial path and sampled held-key audit are insertion-specific. |
| Failed return / reorientation | `original_pose_restoration.py` measures a fresh tabletop key, classifies its v8 pose, loads the selected local wrist relation, plans pickup and a CAD rest target, then checks release and exit. `repose_transition.py` uses saved v8 reset seeds for pose changes. | The candidate-to-world wrist multiplication and wrist-to-object inversion match stock `run_auto.py` and `executor/real.py`. A planned landing is not proof of the actual landed pose; the next trial recaptures the key. |

The stock AutoDex checkout still uses its original FR3 mount and clear-view
configuration. The precision runner uses the October 12 right-angle Inspire
mount. `run_robot_pc.sh` already exported that URDF, but a direct
`run_live.py` invocation without the environment variable could use the
planner's default URDF and the executor's fallback hand offset while
`AssetPaths.robot_urdf` audited the old NAS URDF. These are different geometry
sources for one physical hand. Cylinder precision code now selects the same
checkout right-angle URDF by default for planner construction, executor
`fr3_link8 → base_link` feedback and geometry assets. An explicit
`PRECISION_FR3_INSPIRE_URDF` still selects one alternate file for all three;
the square offline asset fallback remains unchanged. This removes an
inconsistent direct-entry path, not a robot-motion gate. Static and injected
factory tests check both default and explicit selection without opening FR3.
The cylinder test fixtures now write synthetic URDF bytes only under pytest's
temporary directory. Previously they could overwrite the checkout's selected
mount, making later geometry tests fail on invalid XML. The restored checkout
URDF is unchanged from HEAD. The complete non-moving suite passes
**1228 passed, 5 skipped**; `run_robot_pc.sh --check` passes against the
current static 20 mm stage with 98 eligible candidates. Its
`static_preflight_only` scope remains a file check, not a camera or robot test.

For a stock 12 N force stop, `live_task_handlers` records the completed
reverse-axis ascent and `force_abort`. `session_policy` first routes the held
key through return and a fresh tabletop observation. Only after
`reset_success=true` does it select `assess_saved_preinsert_xy`; the selected
`vlm`, `tracked_tip`, or `sam_half_ellipse` estimator reads the saved preinsert
frames and records advice for the **next** pickup. An unavailable observation
records abstention and proceeds without a new XY increment. The insertion
contact is never converted directly into a measured image-space error.
`test_session_policy.py`, `test_session_runner.py`, and
`test_live_retry_handlers.py` cover this recovery-to-diagnostic route.
The stock force-stop return previously also required the fresh state sample
to arrive within one second of the completed ascent log. That duplicated the
live FR3 reader's current-packet check and could reject a stationary arm
after bookkeeping took longer than one second. The stock ascent return now
uses the fresh measured state and the existing joint-drift check without a
log-to-state age cap. The separate commissioned-withdrawal route retains its
configured gap bound. A two-source regression checks both cases; neither
change alters the axial ascent or the return trajectory planner.
The complete non-moving demo suite after this change passes
**1230 passed, 5 skipped**.

At HEAD `091292e`, the force-stop policy/runner/retry regressions were rerun:
**93 passed, 3 skipped**. A separate tip, target, nominal post-lift, and
restoration subset passed **34/34**. The full non-moving suite passed again:
**1230 passed, 5 skipped in 45.18 s**. The file-only 20 mm `--check` also
passed with `PRECISION_XY_ESTIMATOR=sam_half_ellipse`; its recipient stage
remains `static_preflight_only`, so this check does not start or validate a
SAM CUDA worker. Current-head arithmetic replay of the three saved 10 mm SAM
increments had a maximum wrist-shift residual of `6.85e-17 m`; a saved 10 mm
and 20 mm yaw/XY counterfactual had a maximum residual of `1.26e-10 m`.
The source-bound [current-head replay](</home/robot/shared_data/hyunsu/tmp/20261026_saved_sam_advice_pose_replay/current_head_091292e/README.md>)
records inputs and results. These are software-frame checks, not physical
tip-center or robot-motion measurements.

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
physically landed. The archived stage is now available under
`hyunsu/tmp/20261023_prev`. Its raw key mesh and selected v8 wrist relation
are byte-identical to the current 20 mm recipient stage: the mesh bounds are
`[-15, -15, 0]` to `[15, 15, 80]` mm and its actual CAD center is
`[0, 0, 40]` mm. Recomputing with that mesh yields the same sub-nanometer XY
center difference between the saved original and planned rest pose. The
hash-bound calculation is in
`/home/robot/shared_data/hyunsu/tmp/20261026_restoration_pose_replay/current_stage_mesh_transform_check.json`.
It verifies saved geometry and stock-equivalent wrist composition, not the
physical landing.

The same replay exposed a separate cylinder symmetry error before pickup
planning. `snap_axisymmetric_tabletop_pose` previously replaced the measured
rotation with the nearest v8 tabletop orientation about the **mesh origin**.
It compensated the D∞ end flip about the CAD center, but not the smaller
axis snap. The measured physical center therefore moved in the generated key
scene. Among 43 saved key observations with frozen calibration, the largest
artificial XY shift was **0.876 mm**. The snap now preserves the observed CAD
center for both orientation operations; stock `pose_world_to_scene_cfg`
continues to raise any resulting tabletop mesh penetration to the measured
surface. A synthetic tilt and nonidentity camera-to-robot transform test
exercise the live scene path. The saved-pose comparison and source hashes are
in `/home/robot/shared_data/hyunsu/tmp/20261026_restoration_pose_replay/saved_key_snap_center_comparison.json`.
This is a coordinate correction, not evidence that FoundPose's measured center
or a future grasp is physically accurate. The full non-moving demo suite after
this change passes **1226 passed, 5 skipped**.

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
One stock-inherited terminal fall-through was unsafe for this task: when a
stream exhausted its timeout, stopped advancing for the existing 2.5 s stall
window, or exhausted its existing tick budget, `_follow` still ran its final
hand ramp and optional direct `arm.move` to the last joint target. A stalled
stream has not traversed the planned collision-checked path. The precision
checkout now stops/ends the stream as before, then reports that existing
incomplete condition before either terminal command. No timing, speed, force,
or endpoint threshold was added. A CPU-only test executes the actual follower
with inert arm/hand objects and covers normal arrival, deliberate 12 N force
stop, timeout, stall, and tick-budget exhaustion. The original AutoDex checkout
was not changed.
The same stock follower also printed a failed `stop_streaming` call but could
still enter its terminal hand ramp or direct landing move. This checkout now
checks the daemon response and propagates a failed stop before those commands.
For an intentional force stop, a failed stop also prevents the reverse-axis
ascent. This uses the daemon's existing success/error result, not a new
position, force, or timing threshold. The inert-controller regression covers
both a rejected normal stop and a rejected force stop.
The precision wrapper then sent `stop_streaming` again immediately after the
follower's accepted stop before checking stationary feedback. It now waits
for that feedback directly. Explicit `stop_and_acknowledge()` calls at separate
motion boundaries still issue their own stop. This removes one redundant
daemon command per normal `_follow` without changing the velocity limit,
wait window, held-hand ramp order, or force-stop ascent gate. A CPU-only
regression executes the actual composed follower and checks exactly one stop
command. The stationary feedback sample is now reused for the endpoint check,
removing one more state read and its duplicate velocity comparison. No robot-PC
round-trip time was measured, so the latency gain is two removed requests per
normal `_follow` rather than a claimed number of seconds.
The full non-moving demo suite passes **1231 passed, 5 skipped in 46.03 s**
with the `planner` Python environment. The file-only robot-PC `--check`
also passes; it reports `static_preflight_only` and `robot_ready=false`.
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
The `20261022_103429_752726` socket-release exception below is a different
check: `CompletionCheckedFrankaMixin` compares **measured joints** with the
last joint waypoint. It does not use `O_T_EE`, so the mount correction cannot
be claimed to resolve that historical stop. The saved failure lacks its
terminal joint error and segment name; that failure remains undiagnosed.
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

The saved SAM diagnostic increments from three 10 mm episodes were also
applied counterfactually to those episodes' saved nominal post-lift relations
using the current staged fixture. The staged verification transform exactly
reproduced each saved key target. Across preinsert, entry and verification,
the corrected wrist shifted by `R_robot_socket @ [dx, dy, 0]` with maximum
arithmetic residual `6.85e-17 m`; key targets and wrist rotations did not
change. Frame IDs, timestamps, source PNG/mask hashes and each calculation
are in the [saved SAM advice replay](/home/robot/shared_data/hyunsu/tmp/20261026_saved_sam_advice_pose_replay/README.md).
These saved attempts had zero applied XY correction, so the replay proves
software coordinate arithmetic, not a physical corrected trial.

The same next-trial XY composition was checked with a nonzero cylinder yaw
gauge. `build_rigid_insertion_targets` rotates a cylinder key target by that
gauge, while `compensated_key_hand_relation` previously inverted only the
ungauged CAD key rotation. That would rotate the requested socket-XY wrist
increment by the yaw. The helper now inverts the **effective**
`R_gauge @ R_socket_key` rotation and accepts the same yaw gauge as the target
builder. A tilted-socket regression covers zero, positive and negative yaw
through preinsert, entry and verification. On one hash-matched saved 10 mm
and one 20 mm post-lift plan, a counterfactual 0.8 rad yaw and (+2, -1) mm
increment yielded a maximum 1.26e-10 m composition residual; the old formula
would miss by 1.742 mm. [Replay script and results](/home/robot/shared_data/hyunsu/tmp/20261026_cylinder_yaw_xy_replay/README.md)
record the source hashes. The active nominal path uses yaw zero and those
historical plans also used yaw zero, so this finding does not explain a
previous physical trial's offset. The post-shift route uses a newly
reconstructed held relation; this fix is the general compensation primitive
for a future nonzero-yaw next-trial route.

## SAM3 half-ellipse: from offline result to optional live XY path

At `568107f`, `held_key_tip.distal_key_mask_half_ellipse` was called only by
saved-image tools. The live CLI had only `vlm` and `tracked_tip`. This review
added an explicit `sam_half_ellipse` option; the generic CLI and robot-PC
launcher retain their `vlm` default. The stock AutoDex snapshot producer
remains unchanged:

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
The synthetic same-frame ROI regression additionally checks both full-image
pixel coordinates and the sign of the triangulated socket-frame X correction.
It composes that nonzero increment through the next-attempt key-to-hand
command as well, covering crop-origin addition, camera projection, the advice
sign, and socket-frame wrist translation together. The accepted 10 mm
saved-image proxies remain visual estimates rather than
independent measurements of the key's physical end.

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
An [inventory of all saved 20 mm attempts](/home/robot/shared_data/hyunsu/tmp/20261026_tip_20mm_evidence_inventory/README.md)
found 26 insertion attempt records, only three with preinsert raw captures.
All three were excluded by the earlier socket-rim visual QC; the others have no
preinsert images. Thus the accepted socket-pose replay here covers 10 mm
only. There is currently no socket-valid saved 20 mm image set that can
establish this end-point method's physical accuracy. CUDA is unavailable in
this execution environment, so no new 20 mm Qwen/SAM inference was run.
On the accepted 10 mm bundle, five-frame PNG decode, decoded-pixel SHA-256,
color conversion and PIL construction took 0.259/0.273/0.300 s serially and
0.106/0.094/0.098 s with four workers. The ~0.18 s warm difference is small
next to model loading and inference; this audit did not add a more complex
live frame-loader just for that difference.
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
the tip proxy's `15 px` observation tolerance. The largest difference in the
bounded next-trial wrist step was `0.63 mm`. The distributed live path starts
all five views and now returns after three consistent metric SAM points;
otherwise it waits for later views. All source frames remain bound, while the
diagnostic stores the views completed before its decision. This saved-case
sensitivity quantifies the latency/precision tradeoff but does **not**
establish the true physical key face center or justify a tighter failure gate.
The source-frame replay is
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
the exact preinsert frame. The per-camera record distinguishes the raw PNG
observation from the effective temporal-median pixel passed to multiview
alignment. Its `admitted` field means that a usable **2D point** was supplied;
the fit's `inlier_cameras` identifies the views that actually contributed to
the accepted **3D point**. A view can be admitted locally and rejected by the
multiview fit. Both sources retain their frame IDs and source hashes in their
respective saved manifests. The ordinary `vlm` and optional
`sam_half_ellipse` modes continue to use the same saved undistorted PNG for
their per-view observation. The SAM view records use the same local meaning
for `admitted` and also report final `inlier_cameras` separately.

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
**46/46**. A later complete run in the `flir_env` environment passed
**1210 tests with 4 skipped and 1 deselected**; the deselected test imports
stock cuRobo, which is absent from that environment. In `planner` it skipped
because this execution environment exposes no CUDA device. The current
recipient stage at `/home/robot/precision_stage_20261026` passed the
file-only `run_robot_pc.sh --check` without camera or robot access. Its launch
scope is `static_preflight_only`, so it cannot start a physical trial.
The current runtime config selects the nominal v8 held relation. The failed
trial handler no longer accepts four unused legacy motion/capture adapter
fields, and the integration factory no longer creates the unused withdrawn
capture callback. Its non-moving handler, assembly and session-runner tests
pass (56 passed, 3 skipped); the factory/stage/handler focused tests pass
55/55 after this removal.

The post-failure policy had a separate decision-loop bug: after a stock
force-stop ascent or a slipped-key recovery, the recovery handler could save
a fresh, successful key landing, but the attempt retained its original
`force_abort` or `slip` failure code. `decide_after_attempt` checked that
code before the observed `reset_success` label and requested the same
recovery again. The action loop then stopped with `decision_unchanged`.
It now gives a successful, freshly observed landing precedence over the
earlier failure code; a failed landing still stops. A regression drives the
actual action loop from force stop through an observed reset, and another
checks the slipped-key route. The immediate axis withdrawal and held-key
return remain the stock-aligned response. After the return is observed, the
same saved preinsert image may produce next-trial XY advice before fresh-key
preflight; it does not classify the force contact itself as a visually
measured XY error.
The complete non-moving demo suite after this policy fix passed in the
`planner` environment: **1212 passed, 5 skipped in 45.51 s**. Skips require
hardware/CUDA or optional runtime dependencies; this result does not verify
physical recovery.
An additional runner-level regression now follows the force-stop record
through `observe_reset_landing` and into a new key preflight, for both the
automatic held return and a supervised return. This exposed an unrelated
manual-log gate: it required `key_removed_from_socket=true` after **any**
insertion attempt, including a force stop where the key remained in the
hand. The manual log already requires `socket_clear=true` and `hand_open=true`,
and the runner independently requires a fresh supported tabletop key pose.
The redundant field is no longer required; neither the automatic return nor
the fresh landing checks changed.
The full non-moving suite after the runner-level regression and gate removal
passed: **1214 passed, 5 skipped in 41.39 s** (`planner` environment).
After the cylinder-yaw XY composition fix, the same full suite passed:
**1216 passed, 5 skipped in 42.54 s**.

The saved-preinsert diagnostic now also follows a **recovered** stock force
stop. The order is recorded reverse-axis ascent, held-key tabletop return,
fresh `reset_success=true` observation, saved-image XY diagnosis, then fresh
key preflight. The diagnosis reads the pre-contact image and updates only the
next trial's socket-frame XY command. A failed return stops; a missing tip
records unavailability and continues without a new XY step. The change also
aligns the ledger's accepted unavailable reasons with the live SAM/tracker
handler, which previously raised on a legitimate abstention. The ordinary
released-key failure route and stock force-stop motion are unchanged.
The non-moving suite passes **1221 passed, 5 skipped in 43.80 s**; `bash -n`
and `git diff --check` pass. No force-stop SAM diagnosis has been run on a
robot: this host currently exposes no CUDA device to the AutoDex environment,
so `run_robot_pc.sh` retains the existing `vlm` default. Select
`PRECISION_XY_ESTIMATOR=sam_half_ellipse` only when its local CUDA worker is
available.

The recovery key capture and the next trial's key capture currently run as
separate actions. `capture_returned_key_for_reset` saves a pose-only result
to check the landing; `prepare_next_live_key` also starts the trial's raw
`initial_tabletop` image evidence used by the final VLM pair. Passing the
returned pose directly to planning would leave that pair without its bound
initial image and would bypass the current post-outcome exposure ordering.
In the saved `20261021_092625_546425` timing report, the recovery key capture
took **64.015 s**. That run did not reach the later reobservation action, so
it cannot quantify a second capture's cost. A future reuse path needs to
preserve the initial image/frame binding and check the measured robot state;
simply removing the freshness condition is not a valid latency fix.

The held and released return handlers also read clear-view feedback immediately
before calling `capture_returned_key_for_reset`, which reads and validates the
same state again before capture. The duplicate outer read has been removed;
the capture function now owns the single timed `recovery.clear_view` feedback
check. Released-key recovery still records its separate post-release measured
state before the fresh key capture. This removes only a redundant feedback
read, with no change to robot motion or the force-stop recovery order.

An additional saved-image trace found that the shared VLM fallback can emit an
explicitly exploratory 2 mm step when no tip is visible. The optional
SAM/tracked-tip path had inherited that branch, so an empty SAM mask could
become next-trial wrist advice despite having no observed endpoint. It now
records `abstain` without an XY increment in that case. The existing
single-view planar and two-or-more-view metric paths remain available. The
live handler records `sam_half_ellipse_unavailable` or
`tracked_tip_unavailable` and continues recovery; it issues no correction
from a missing or inconsistent tip. The VLM route's separately labeled
exploratory policy is unchanged. Synthetic no-tip, behind-camera, SAM-mask,
and live-handler tests cover this distinction; they do not establish the
physical accuracy of an accepted tip.

The SAM half-ellipse factory now loads its CUDA worker before opening the
command-capable FR3 core. Previously it checked only the Python/checkpoint
paths up front, then loaded the model during handler assembly after the
initial robot and camera session. A CUDA/model-load failure could therefore
terminate that session after its clear-view move. The worker now warms first,
closes if FR3 startup fails, and moves its stderr log into the run directory
on orderly exit. Startup-order/resource tests and the full non-moving suite
pass (**1233 passed, 5 skipped**). The file-only 20 mm
`run_robot_pc.sh --check` passes with the SAM mode, but reports
`robot_ready=false`; it still does not load CUDA or contact a robot.
The [current-head saved SAM-to-wrist replay](</home/robot/shared_data/hyunsu/tmp/20261026_saved_sam_advice_pose_replay/current_head_8f1b0f0/README.md>)
checks three saved 10 mm advice steps and a nonzero-yaw 10/20 mm
counterfactual against `8f1b0f0`; maximum wrist-shift residuals were
`6.85e-17 m` and `1.26e-10 m`, respectively. These are arithmetic checks,
not measured motion or physical tip-center accuracy.

## Retry preflight uses the advised wrist relation

The next-trial XY audit found one planning mismatch after the saved advice
had already been accepted: `plan_fresh_key_trial` screened the next pickup's
insertion path with the nominal v8 wrist relation, whereas
`nominal_postlift.py` later screened and commanded the advised wrist
correction. The retry preflight now uses `compensated_key_hand_relation` for
the same candidate and tabletop stem, records the correction in its report,
and binds it to the correction consumed when that physical attempt begins.
The first pickup and square-mode paths remain nominal. This changes which
path is screened before pickup, without adding a clearance, timing, or
motion threshold; the measured post-lift replan remains necessary. A
synthetic candidate-path test and the saved tip-to-wrist arithmetic replay
cover the software transform, not actual held-key accuracy. The complete
non-moving suite passes **1234 passed, 5 skipped**; the 20 mm file-only
`run_robot_pc.sh --check` also passes with the SAM estimator and still reports
`robot_ready=false`.

## Failure recovery and XY advice after a force stop

The stock 12 N force-stop follower immediately ascends its executed axis path
with the hand closed. Precision insertion then returns the held key to the
tabletop, observes the landing, and only after `reset_success=true` reads the
saved pre-contact `preinsert` images for an optional next-trial XY estimate.
The accepted estimate changes the next pickup's wrist target; neither the
contact event nor a missing tip becomes a measured XY offset. This route was
already connected in `session_policy`, `live_retry_handlers`, and
`session_runner`; the operator's requested ordering needs no new robot motion
branch. The released-key route follows the same saved-image diagnostic but
has already opened the hand before its landing capture.

One released-key recovery gate was redundant: it rejected a fresh measured
clear-view state if its packet timestamp preceded the just-written XY advice
event. Asynchronous state publishing can produce that order while the arm is
stationary. The handler still verifies fresh measured feedback, open hand,
and clear-view joints. It now records the time of that verification as the
recovery completion, retaining the packet's own timestamp separately. The
runner accepts a packet sampled before verification and continues to require
the recovery confirmation and new key capture after the failed attempt.
This removes no controller force stop, path collision check, or measured
clear-view check. A regression exercises the packet/advice ordering through
the live handler and another through the reset ledger; these tests do not
establish physical recovery success. The full non-moving suite after this
change passes **1234 passed, 5 skipped**; the 20 mm file-only `--check`
passes with `robot_ready=false`.

## Reused force-stop ascent and measured finger settling

The stock FR3 force-stop follower holds Inspire closed during its reverse-axis
ascent. The demo's held-return handler reuses that ascent instead of issuing a
second withdrawal, then plans the key's return from fresh measured joints.
Its extra 0.02 rad unchanged-pose check previously compared all 13 joints.
Measured Inspire fingers can settle against the held object even when the arm
has not moved; stock AutoDex does not use finger settling to reject a return.
The first revision limited that comparison to the seven FR3 arm joints. A
further stock comparison showed that even this 0.02 rad fixed difference
gate was redundant: `plan_repose_held_from_state` takes the newly measured
13-joint state, computes the held key pose by wrist FK, and plans and audits
the return path from that state. The handler now checks that the new feedback
follows the logged ascent in time and belongs to the same attempt, then lets
the existing measured-state validation and path planner decide whether a
return is possible. It does not command a second withdrawal. A regression
confirms a 0.1 rad changed arm state reaches planning as the actual start;
another rejects feedback that did not advance after the ascent. Neither test
proves that a physical key remains in the hand.
After removing the fixed arm-difference gate, the full non-moving suite
passes **1235 passed, 5 skipped**. The 20 mm file-only `--check` reports 98
eligible candidates and `robot_ready=false`; it does not open cameras or
command the FR3.

The saved `20261021_092625_546425` recovery-key capture took 64.015 s,
including approximately 50.08 s waiting for the first stock PNG writes.
That timing predates the current pose-only recovery producer, which skips the
same-request PNG wait when registered ROI assistance is absent. A separate
saved `20261022_103429_752726` initial key capture took 2.873 s and its
following raw visual snapshot took 1.117 s; neither proves the duration of a
new live recovery. Five actual 10 mm preinsert PNGs cost 8.38 s on an initial
cold read but only about 0.011 s to hash after they were cached. The XY
diagnostic necessarily reads those selected images for inference, so removing
its later byte-hash check would not remove that cold read and offers no
material measured latency gain here. The source-hash check remains.
The earlier arm-only comparison passed **1234 passed, 5 skipped** before
the fixed arm-difference gate was removed. The current full-suite result is
reported above.

## Remaining evidence needed

The SAM static readiness path now passes the requested estimator through
`run_robot_pc.sh --check` to `run_live.py --preflight`. The report names the
selected mode and the resolved SAM Python/checkpoint files and size. A missing
checkpoint fails at this read-only stage instead of first appearing at live
startup. With the current 20 mm recipient stage it reports 98 eligible
candidates and `robot_ready=false`; this does not verify model loading or
physical tip accuracy. `test_live_runtime.py` checks both present and missing
SAM inputs without opening FR3 or importing stage code. The full non-moving
suite passes **1235 passed, 5 skipped**; the selected SAM `--check` reports
the model file and `robot_ready=false`.

1. Replay additional rejected-socket and failed-grasp episodes with this
   optional route; distinguish bad fixture pose from key-mask error.
2. Commission an active live recipient stage from the checked file-only
   stage, including a current socket scene and hardware readiness. The
   static `--check` has passed, but it does not test physical execution.
3. Measure actual planning and recovery times after the AABB change. Local
   unit-fixture timings cannot establish an end-to-end gain.
4. Independently check physical mount/hand-eye and held-key motion on the
   robot before treating planned pose transforms as measured accuracy.
