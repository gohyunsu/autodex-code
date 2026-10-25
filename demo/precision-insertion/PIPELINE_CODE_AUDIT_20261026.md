# Current code audit: key tip, transforms, and recovery latency

This audit compares the local stock `/home/robot/AutoDex` checkout at
`448732d` with `feat/precision-insertion` at `568107f`. It is a source and
saved-evidence review. No camera, controller, or robot was started.

## What is actually connected

| Step | Live source and result | Stock relationship |
| --- | --- | --- |
| Tabletop key | Capture-PC SAM3/FoundPose, admitted by `key_perception.py`, yields a fresh `T_robot_key`. | Reuses AutoDex's multi-view object processing. |
| Candidate wrist | `candidates.planner_candidate_override` reads object-local `wrist_se3.npy` as `T_key_hand` and passes `T_robot_key @ T_key_hand` to the stock planner. | This is the convention in stock `run_auto.py`: the candidate wrist is transformed by the observed object pose before planning. |
| Post-lift held relation | `nominal_postlift.py` uses that selected candidate relation; other explicit routes can use an independently observed or bounded relation. `compensated_key_hand_relation` maps a next-trial socket XY correction into the relation's key-frame translation. | Stock `run_auto.py` records both world `wrist_se3.npy` and `inv(T_robot_key) @ wrist_se3` for later object-local reuse. The demo's nominal relation follows the latter convention. |
| Preinsert / entry / verification | `targets.build_rigid_insertion_targets` composes `T_robot_socket @ T_socket_key @ T_key_hand`. The socket-frame XY offset is applied before this composition; the insertion direction is rotated by `T_robot_socket`. | Stock's world-vertical Jacobian stroke cannot directly represent a tilted socket axis. The demo's axial path and sampled held-key audit are insertion-specific. |
| Failed return / reorientation | `original_pose_restoration.py` measures a fresh tabletop key, classifies its v8 pose, loads the selected local wrist relation, plans pickup and a CAD rest target, then checks release and exit. `repose_transition.py` uses saved v8 reset seeds for pose changes. | The candidate-to-world wrist multiplication and wrist-to-object inversion match stock `run_auto.py` and `executor/real.py`. A planned landing is not proof of the actual landed pose; the next trial recaptures the key. |

`run_auto.py` and `executor/real.py` are byte-identical in the two local
checkouts. `autodex/planner/planner.py` differs by one constructor option:
the precision checkout may inject a robot URDF path for the right-angle
FR3/Inspire mount. This makes the planning model selection explicit; it does
not change stock planning algorithms. Tests in `test_candidates.py`,
`test_targets.py`, and `test_next_trial_xy.py` cover the object-local wrist
convention, a tilted socket axis, cylinder yaw, and socket-frame correction.

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

The old `max_total_offset_m` retry budget remains an optional legacy field;
the active FR3 stage factory sets it to `None`. The live launcher defaults to
the VLM estimator. Existing `BOARD_FREE_MODE.md` code is statically tested
but the former recipient stage was moved, so `run_robot_pc.sh --check` cannot
currently certify an active robot-PC asset installation.

## Recovery-path latency change in this review

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

## Remaining evidence needed

1. Replay additional rejected-socket and failed-grasp episodes with this
   optional route; distinguish bad fixture pose from key-mask error.
2. Restore and rehydrate an active recipient stage, then run static
   `run_robot_pc.sh --check` without connecting cameras or the robot.
3. Measure actual planning and recovery times after the AABB change. Local
   unit-fixture timings cannot establish an end-to-end gain.
4. Independently check physical mount/hand-eye and held-key motion on the
   robot before treating planned pose transforms as measured accuracy.
