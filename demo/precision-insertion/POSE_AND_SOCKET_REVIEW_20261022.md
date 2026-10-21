# Socket localization and pose flow review (2026-10-22)

Scope: read-only source review against stock `/home/robot/AutoDex` at
`448732d`, and replay of three real 2026-10-21 socket-only captures. No new
camera frame, robot movement, or physical ground truth was used. The stock
checkout was not edited. The branch's `src/execution/run_auto.py` is byte
identical to stock; its FR3 executor and planner are not, so control/path
equivalence is limited to the functions identified below.

## Frames and pose-changing stages

`T_AB` maps coordinates in B into A. `R` is the FR3 robot base, `W` the
calibrated camera world, `S` the measured socket, `K` the key CAD frame, and
`H` the Inspire `base_link` hand frame. The v8 `wrist_se3.npy` is `T_KH`.
The measured session transform `C2R` is `T_WR`, so a FoundPose world pose is
converted by `T_RW = inv(C2R)`. Stock `run_auto.py` uses the same grasp
relation: `T_HK = inv(T_RH_grasp) @ T_RK_grasp` and later
`T_RH_goal = T_RK_goal @ inv(T_HK)` (around lines 2235 and 2341).

| Situation | Input and calculation in this demo | Result and stock relationship |
| --- | --- | --- |
| Board/socket measurement | `live_session_start.py` calls stock `measure_tabletop_from_images`, then captures SAM masks and FoundPose hypotheses with acquisition metadata. `board_socket_fit.py` projects mask-centroid rays to the measured board plane at the CAD wall midpoint, takes median board XY, and uses the board normal for the open socket axis. `calibration.py` transforms W→R and freezes one pose. | `T_RS` plus a fixed socket collision mesh in every later scene. Stock uses ChArUco and the same capture stack but has no fixed insertion socket; its unconstrained whole-mesh silhouette objective is not the active socket pose. |
| Fresh tabletop key | `key_perception.py` admits the frame-bound FoundPose result; `world.py` computes `T_RK = inv(C2R) @ T_WK`, builds the key target plus frozen socket and measured table. `pose_selection.py` matches a v8 tabletop class under cylinder symmetry. | A new observed key pose, not the last episode's old position. Stock `run_auto.py` likewise observes and classifies before choosing a v8 grasp. |
| Pickup and first lift | `candidates.py` loads the selected v8 `T_KH` and supplies `T_RH_grasp = T_RK @ T_KH` to the unchanged stock v8 pickup planner. `pickup_execution.py` and `franka_measured_lift.py` use the planned approach/squeeze and measured FR3 state. | The physical grasp relation after squeeze is not directly measured by this step. The active cylinder route uses the selected v8 relation as a named hypothesis, as stock place does. |
| Post-lift transfer | `nominal_postlift.py` reads measured joints, optionally changes only the commanded hand translation by `next_trial_xy.py`, and replans from those joints. `targets.py` forms `T_RK_goal = T_RS @ T_SK_CAD`, then `T_RH_goal = T_RK_goal @ T_KH_command`. | The current attempt's commanded wrist goal. Stock also composes object and hand transforms, but has no measured socket/20 mm insertion target. The original CAD key goal stays centered when a retry shifts the wrist command; this avoids applying the XY correction twice. |
| Preinsert and insertion | `preflight.py` plans held transfer to the CAD preinsert pose. `axis_jacobian.py` continues with fixed hand orientation along `R_RS @ insertion_direction_socket`, with sampled collision and endpoint checks. `stock_force_insertion.py` follows that trajectory and reverses the executed corridor on force stop. | A nominal 20 mm axis stroke or force-stopped shorter stroke, not an independently measured key depth. Stock `jacobian_stroke.py` is world-Z only and cannot directly implement a tilted socket axis. |
| Release and final image | `stock_socket_release.py` opens Inspire and follows a planned exit/clear-view path; `live_task_handlers.py` takes final images. | The VLM `inside_socket` label is visual evidence, separate from the motor/force record. Stock place also opens and retracts, but has no insertion label. |
| Failure XY for next attempt | `live_retry_handlers.py` reuses the failed attempt's *saved* preinsert frame; `grounded_alignment.py` estimates a tip/socket axis relation or records a clearly named fallback. `session_runner.py` accumulates the bounded socket-XY increment. `next_trial_xy.py` maps socket XY into key-frame hand translation by `R_SK.T @ delta_S`. | A correction to the next wrist command, with source attempt and frame provenance. It does not alter frozen `T_RS` or relocate the key in software. Stock has no task-specific XY feedback. |
| Failed grasp or released-key recovery | `clear_view.py` reuses stock release/reset for a judged missed grasp. For a released key, `live_recovery_handlers.py` recaptures its **new** pose; `original_pose_restoration.py` chooses the old CAD center as a target, not as the new observed source. | A fresh pickup/repose plan if the tabletop class or position requires it. Stock also re-observes after place/reset, but has no fixed socket exclusion. |
| Lying-key reorientation/restoration | `repose_transition.py` loads a directed v8 `reset_12` seed for `001→000`. `repose_preflight.py` composes a new rest pose from the exact v8 tabletop orientation, measured table height and chosen XY, then `T_RH_rest = T_RK_rest @ T_KH`. `stock_repose_execution.py` uses stock pickup, held transfer, support descent, opens, exits and recaptures. | The key's **observed current pose** is the pickup source; the chosen upright rest pose is the placement goal. The 12 cm seed is a grasp asset, not the physical drop height: active restoration requests `release_height_m=0` and a force-aware supported descent. |
| Clear view | `clear_view.py` uses the branch's `FR3_CLEAR_VIEW` for initial `home(clear_view=True)` and checks measured arrival. A later release uses its planned exit. | Clear-view is a joint-space camera/next-planning posture, not a key/socket pose update. The current target was visually checked on the robot; no new clearance test was performed in this review. |

`xy_retry.py`, observed post-lift relation, and post-shift lateral modules
remain separate experimental/offline paths. The current cylinder live
assembly selects the nominal post-lift route and saved-preinsert next-trial
diagnostic; their stricter physical-relation and image-age criteria must not
be attributed to the active path.

## Why the socket axis is currently tied to the board

The socket has a flat 60 mm-radius flange intended to lie on the board; its
outer cylindrical wall is 40 mm radius for the 20 mm radial-gap part, from
local z=5 to 55 mm. SAM masks often include the wall but omit the flange;
some overhead masks include the flange instead. The current CPU fit uses
each accepted mask's centroid, undistorted `K`, camera extrinsic and `C2R`
to intersect a ray with the plane through the wall midpoint (30 mm above
the board). It takes the coordinate-wise median of those board XY estimates,
sets the socket base on the measured board, and fixes its axis to `-board z`.
Axial yaw is unobservable for this round part. It then renders the wall-only
convex hull in every view and records per-view/mean mask IoU. The configured
0.8 bound applies to the mean, not to every overhead mask. The full exact
socket collision mesh is still used for planning. FoundPose hypotheses are
still captured for stock evidence compatibility, but do not determine the
active final socket pose.

The board is **not** a universal axis prior. It is a physical support
constraint for this flat, fixed flange. If the flange is visibly tilted,
unseated, or on a shim, the current assumption is invalid and the board
normal should not be forced. A free silhouette fit alone is not the right
fallback: stock `silhouette.py` optimizes the *whole mesh* in 6 DoF against
masks that often cover only the wall. It can improve that mismatched loss by
moving or tilting a physically impossible flange. A future tilt estimator
should segment the same body part in independent views, include flange/board
nonpenetration and support height, and validate against held-out views and
physical tilt evidence. Board-aligned initialization before that optimizer
would not preserve the support constraint.

## Replay on saved real frames

The three capture bundles are under `_pipeline_runs/` in the
`precision_key_cylinder_r15_h80` experiment, with IDs below. Each uses its
own frozen `frozen_session_calibration.json`,
`source_evidence_binding.json`, and socket-only undistorted PNG masks. The
mask grid at `/tmp/socket_masks_20261022.png` and the old/new overlay at
`/tmp/precision_socket_body_fit_20261021.png` are diagnostics made from those
real frames, not generated camera evidence or a new calibration.

| Capture ID | Admitted views | Old whole-mesh mean IoU | Board-aligned wall mean IoU | 0.8+ wall views | Free-tilt held-out mean vs board-aligned held-out mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261021_090945_053190` | 18 | about 0.697 | 0.943175 | 17 | 0.930045 vs 0.933561 |
| `20261021_104732_950124` | 19 | about 0.697 | 0.924748 | 17 | 0.911090 vs 0.917928 |
| `20261021_140856_019116` | 19 | old fit 2/19 at 0.8+ | 0.921415 | 17 | 0.950105 vs 0.948518 |

The old column scores the saved whole-mesh path and the new column scores
the body-only wall objective, so their IoUs are not a like-for-like accuracy
comparison. The saved-frame overlays show where the projected socket axis
and wall land; neither score supplies a surveyed socket ground truth.

For the held-out comparison, quarter-resolution masks from alternating
camera IDs fit board XY plus two axis-tilt parameters with a trimmed IoU
objective; the remaining camera IDs were scored at full resolution. The
free fit improved one held-out set by 0.0016 IoU and worsened two. A soft
3° tilt prior and a flange-support-height variant likewise did not improve
all three held-out sets. Full-resolution XY-only optimization shifted the
estimated center by under 0.2 mm and gained only 0.0002–0.0004 mean IoU.
On the last capture a free tilt gave a 0.0016 in-sample gain, but at a
board-fixed base its about 1.2° tilt would put the flange about 1.2 mm
through the board. These are model-versus-mask comparisons, not metric
ground truth, and do not justify an additional runtime optimizer.

## Gate and latency judgment

The previous active-route audits already removed image-age vetoes, extra
held-lift force limits and redundant hand-transition checks. This review
found no saved case where a remaining active numeric **motion** gate caused
an unwarranted stop. The measured endpoint, joint/path collision, support
placement and force-stop reverse corridor have direct geometric or observed
failure reasons, so they remain. Static source hashes and replay verifiers
are evidence-integrity checks; the live catalog selector no longer rescans
the candidate tree on every trial. The board-constrained socket CPU fit also
skips the stock silhouette renderer warmup and 100-iteration socket fit.
The remaining nominal 20 mm endpoint screen is about 5 s per selected
candidate on this PC (single-candidate profile: about 3.1 s of 28 Coal BVH
builds and 1.2 s of two URDF loads). Changing that collision implementation
would invalidate all recipient catalogs, so no unmeasured shortcut was
introduced during the six-gap catalog rebuild.
