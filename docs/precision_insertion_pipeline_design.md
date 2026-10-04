# AutoDex precision-insertion pipeline design

## Intended system

The research target is not merely a more difficult grasp. It is an autonomous
trial loop in which AutoDex acquires a key, brings it to a known socket,
attempts a precision insertion, explains the outcome, and selects a useful
next trial. Lift success and insertion success are separate variables:

- `grasp_success` says whether the selected grasp acquired and retained the key;
- `task_success` says whether the key reached the insertion goal;
- candidate `result.json` and grasp coverage continue to learn only from the
  former, so an alignment or controller failure does not poison a good grasp;
- an insertion episode record and the future adaptive policy learn from both.

The four gaps form a gated experiment, not four interchangeable test objects:

1. **1.5 mm:** integrate the full data/control path with one stable pose and
   candidate 78, then commission a guarded straight insertion;
2. **1.0 mm:** measure the cumulative pose, hand-eye, robot, and grasp-held-pose
   error without changing the grasp;
3. **0.5 mm:** introduce force/contact-guided XY and yaw correction;
4. **0.3 mm:** evaluate the final policy only after the previous gates pass.

## Frame and transform contract

`T_A_B` maps points expressed in frame B into frame A. The required transforms
are:

- `T_world_key`: FoundPose estimate before pickup;
- `T_robot_world = inv(C2R)`: hand-eye/world registration;
- `T_robot_socket = T_robot_world @ T_world_socket`: measured at session start;
- `T_socket_key_preinsert` and `T_socket_key_seated`: CAD transforms in
  `task_geometry.json`;
- `T_hand_key`: key pose relative to the grasped hand, initially derived from
  the selected object-relative grasp and later corrected by physical
  repeatability measurements.

The nominal pre-insertion hand goal is therefore:

```text
T_robot_hand_goal = T_robot_socket
                   @ T_socket_key_preinsert
                   @ inv(T_hand_key)
```

This is why socket localization alone is insufficient. A grasp can lift while
placing the key several millimetres or degrees away from its nominal
`T_hand_key` because of compliance, finger calibration, or slip.

## Session startup

The intended startup order is:

1. AutoDex exclusively claims the two ZeroDex capture PCs and four calibrated
   cameras; ZeroDex's own stream owner must be stopped.
2. Load the pinned intrinsics/extrinsics and current Franka `C2R`.
3. With no loose key in the workspace, run socket FoundPose three times.
4. Reject a missing pose, silhouette failure, invalid SE(3), or excessive
   repeatability residual. Select an observed SE(3) medoid; do not average
   rotation matrices componentwise.
5. Save raw images/masks, per-sample poses, perception diagnostics, `C2R`, and
   `fixture_pose.session.json` under
   `~/shared_data/AutoDex/experiment/<exp_name>/<hand>/<key>/`.
6. Freeze `T_robot_socket` for the process and add the exact concave socket
   mesh to every normal and recovery cuRobo scene.
7. Measure the empty ChArUco tabletop, then ask the operator to place the key.

Freezing the transform prevents trial-to-trial perception noise from looking
like controller adaptation. It is valid only while the socket is rigidly
bolted. A moved fixture, camera, robot base, or invalidated hand-eye calibration
requires a new session. Repeatability is not absolute accuracy: the same model
can return a stable biased pose, particularly if the keyed opening is poorly
segmented.

## Episode state machine to implement

The current repository reaches `GRASP_LIFT`; the remaining states must be
implemented as a precision task rather than embedded in grasp-label code.

```text
OBSERVE_KEY -> SELECT_GRASP -> PLAN_PICK -> GRASP_LIFT -> VERIFY_HOLD
  -> PLAN_TRANSFER_WITH_HELD_KEY -> MOVE_PREINSERT -> VERIFY_PREINSERT
  -> INSERT_GUARDED -> VERIFY_INSERTION
  -> {RELEASE_SUCCESS | RETRACT_RECOVERABLE | SAFE_ABORT}
```

Each transition writes its inputs, decision, robot state, force/torque trace,
camera evidence, and reason to the existing append-only pipeline timeline.
The held key must be present as an attached collision object during transfer;
the socket remains a static mesh. Planning only the wrist while omitting the
shaft would defeat the contact restriction and can collide before insertion.

### 1.5 mm controller

"Open loop" should mean no alignment search, not no safety feedback. Plan to a
pre-insertion pose using the frame equation, move at commissioned low speed,
then execute a straight socket-axis stroke with force/torque, joint-limit, and
Cartesian-deviation aborts. Success requires measured insertion depth plus
visual/post-motion confirmation. A force spike without depth is a rim jam.

### 1.0 mm instrumentation

Keep the same candidate and motion. Repeatedly estimate these error terms
separately: socket pose repeatability and bias, key pose repeatability, physical
`T_hand_key` repeatability after pickup, Franka/TCP repeatability, and printed
part dimensions. Report distributions, not a single success rate. The useful
quantity is the pre-insertion key-to-socket translation/yaw error.

### 0.5 and 0.3 mm contact search

Add an explicit controller interface that consumes force/torque and Cartesian
state and returns a bounded correction or abort. A practical progression is
low axial preload, bounded XY pattern, then small yaw hypotheses, always
returning to the last safe pose between hypotheses. Search bounds, increments,
force limits, filtering, and dwell times remain `null` until commissioned on
the physical Franka. The 0.3 mm condition must reuse a controller frozen at
the 0.5 mm stage; tuning directly on the final test condition would confound
evaluation.

## Grasp coverage and reorientation

Candidate `table/0/78` is intentionally a controlled rear-down baseline. Five
non-damaging stable key poses are represented, but scenes 1--4 have no runtime
grasp candidates. Each needs its own BODex proposal because the table blocks a
different handle face. Every candidate must pass the handle contact policy,
full-key collision and MuJoCo validation, FR3 pickup planning, and physical
validation.

Insertion is also an object-reorientation problem, but not necessarily in-hand
reorientation. If a rigidly held candidate permits a collision-free arm path
to `T_robot_hand_goal`, wrist/arm transport performs the reorientation. If the
grasp blocks the socket, violates joint limits, or makes that goal unreachable,
the system needs another pickup grasp or an explicit regrasp. Future ranking
must therefore score pickup feasibility and pre-insertion reachability jointly.

## ZeroDex-style VLM role

The VLM should initially be a read-only observer running after a stopped or
held motion. A single final image is not enough. Give it synchronized,
phase-labelled multi-view evidence: before pickup, after lift, pre-insertion,
peak-contact/abort, and final hold/retract. Attach deterministic metadata such
as intended gap, selected grasp, commanded/measured depth, force event, and
camera IDs. Require a strict result schema:

```json
{
  "verdict": "success | failure | unknown",
  "phase_reached": "grasp_lift | preinsert | insertion | seated",
  "failure_class": "none | grasp_miss | grasp_slip | pose_error | rim_jam | partial_insertion | collision | occluded | system_error",
  "evidence_views": ["camera_serial/frame_id"],
  "confidence": 0.0,
  "reason": "short evidence-grounded explanation"
}
```

Parse and validate this schema; malformed or contradictory answers become
`unknown`. Use multiple camera views and, if needed, independent prompt/model
votes, but preserve disagreement instead of majority-forcing a success. The
VLM must not be the force-limit authority and must not command search motions.
During shadow mode, compare it against human labels and geometric/sensor
checks. Only after a confusion-matrix audit should it replace the unreliable
ChArUco grasp heuristic as an outcome labeler.

For grasp validation specifically, combine evidence rather than asking only
"did it succeed?": object visible on the table before pickup, object moves with
the hand after lift, board visibility change, finger state/effort, and (when
possible) a post-lift key pose. This distinguishes a covered board, occlusion,
and an actually retained key.

## Failure attribution

Every failed episode should identify the earliest failed phase, then record
lower-level evidence:

| phase | likely causes | required evidence |
|---|---|---|
| socket startup | segmentation/representation, calibration, symmetry | masks, overlays, per-view poses, residuals |
| key perception | prompt, occlusion, pose ambiguity | masks, candidates, silhouette loss |
| grasp/hold | unreachable grasp, forbidden contact, miss, slip | candidate, plan, finger/force state, lift views |
| transfer | held-key collision, wrist/joint reachability | attached-object plan and collision report |
| pre-insertion | socket/key/calibration/held-pose error | relative pose residual and multi-view evidence |
| insertion | rim jam, excessive friction, incorrect yaw, controller limit | Cartesian path, depth, forces, search actions |
| verification | occlusion or VLM/sensor disagreement | all raw evidence and `unknown` verdict |

Do not adapt from a generic `failure` bit. A failed grasp should affect grasp
selection; a rim jam should affect alignment/search; a camera/system failure
should affect neither physical policy.

## Current capability and blockers

Implemented now:

- metric key/socket geometry and exact socket static collision mesh;
- machine-readable handle-only contact regions and common candidate 78;
- MuJoCo/full-key/FR3 planning evidence and physical-validation gates;
- separate grasp/task result semantics;
- four-camera ZeroDex profile and calibration snapshot;
- per-session repeated socket pose measurement, freeze, evidence, and scene
  injection for normal and reorientation planning;
- reproducible NAS handoff exporter.

Still blocking a physical insertion run:

1. mesh-specific FoundPose `repre.pth` for all four keys and the socket;
2. physical confirmation of the four camera serials/PC mapping and `C2R`;
3. physical validation of candidate 78 and measurement of `T_hand_key`
   repeatability;
4. an attached-key transfer/pre-insertion planner and precision task motion hook;
5. commissioned 1.5 mm guarded insertion limits and success sensors;
6. 1.0 mm metrology protocol and actual printed-gap measurements;
7. 0.5/0.3 mm contact-search controller and physical safety parameters;
8. multi-pose candidates for tabletop scenes 1--4;
9. phase-aligned external video capture and a schema-validated VLM evaluator;
10. physically labelled evaluation data for grasp/task verdict calibration.

Consequently, the current pipeline can measure the socket and plan pickup with
it in the collision world once the learned representations exist, but it
cannot yet perform or honestly score insertion.

## Recommended implementation order

1. Restore/generate the five FoundPose representations and pass the strict
   ZeroDex camera-profile audit.
2. Run perception-only socket sessions and quantify yaw/translation stability
   with independent physical ground truth; tighten the startup thresholds from
   their 2 mm/2 degree bring-up defaults.
3. Physically validate candidate 78 at 1.5 mm and measure held-key pose scatter.
4. Add an `InsertionTask` execution/evaluation hook and attached-key transfer
   plan, initially stopping at pre-insertion.
5. Commission the guarded 1.5 mm straight stroke and deterministic depth/force
   result fields; keep manual task labels as ground truth.
6. Collect the 1.0 mm error budget before designing search bounds.
7. Implement and commission 0.5 mm XY/yaw search, then freeze it for 0.3 mm.
8. Run the VLM in shadow mode on phase-labelled views, publish its confusion
   matrix and unknown rate, then decide whether it may label trials.
9. Expand stable-pose grasp coverage and only then add adaptive next-trial
   selection conditioned on failure class and task context.
