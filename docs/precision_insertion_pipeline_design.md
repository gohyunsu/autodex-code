# AutoDex precision-insertion pipeline design

## Decision summary

The final system is an **AutoDex task extension**, not a second pipeline and
not a ZeroDex camera port. It retains AutoDex acquisition, FoundPose, BODex
candidate storage, Franka/Inspire planning and execution, coverage, recovery,
and append-only experiment records. It adds a precision-task runtime after a
candidate has passed pick/lift planning.

One physical trial is:

```text
observe exact key pose
  -> select and fully preflight one grasp
  -> pick and lift
  -> verify retained grasp
  -> rigid transfer to the measured fixed socket
  -> guarded insertion and verification
  -> guarded extraction
  -> reverse transfer to the original tabletop pose
  -> open, retract, and verify reset
```

No candidate may be physically picked merely because its lift is feasible.
The pick, attached-key transfer, pre-insertion target, insertion corridor, and
return path must form one immutable `TaskPlanBundle` first. The insertion
stroke remains guarded at runtime because free-space planning cannot certify
contact behavior.

Three outcomes are intentionally independent:

- `grasp_success`: the key was acquired and retained after lift;
- `task_success`: the key reached the seated insertion goal;
- `reset_success`: the key was safely extracted and restored for another
  trial.

AutoDex candidate `result.json` and grasp coverage learn only from
`grasp_success`. The insertion episode learns from all three. A rim jam must
not poison an otherwise good grasp, and a successful insertion followed by a
failed reset must stop the session without relabeling the insertion.

## Non-negotiable manipulation invariants

1. **Handle-only contact.** Declared BODex contacts and the complete Inspire
   visual/collision geometry may touch only the handle's axis-aligned lateral
   faces or rear face. The shaft, tip/bevel, diagonal handle chamfers, and
   socket-facing handle shoulder are forbidden.
2. **Two-millimetre proposal margin.** A declared contact must be at least
   2 mm from an edge of its permitted face. This margin applies to sparse
   proposal contact points; the complete-hand gate separately checks actual
   mesh penetration at the boundary.
3. **Rigid attachment.** At closure, save one candidate-specific
   `T_hand_key`. Until the hand opens, every key pose is derived from hand FK:
   `T_robot_key(t) = T_robot_hand(t) @ T_hand_key`. There is no hidden
   in-hand interpolation or unplanned regrasp.
4. **Exact geometry after proposal.** A handle proxy may generate BODex
   proposals, but all promotion and planning use the full key, full hand,
   table, and exact socket mesh.
5. **Frozen fixture within a session.** The socket is measured once at
   startup, accepted only after repeated-pose checks, and then reused as one
   immutable collision fixture. Movement of the socket, cameras, robot base,
   or hand-eye calibration invalidates the session.
6. **Intentional-contact boundary.** The socket is a normal obstacle during
   pickup, transfer, recovery, and reorientation. Only the constrained axial
   insertion/extraction primitive may enter the declared bore corridor; socket
   collision is never disabled globally.
7. **Fail closed.** Missing perception assets, inconsistent frames,
   unjudgeable grasp state, VLM/sensor disagreement, a failed reset, or a
   poisoned CUDA planner context stops autonomous continuation.

## What remains AutoDex and what is new

| subsystem | responsibility | provenance | status on this branch |
|---|---|---|---|
| FLIR capture PCs, RCC, UTG900 trigger, timestamp camera | synchronized evidence | AutoDex/ParaDex | retained; ZeroDex camera topology is out of scope |
| camera intrinsics/extrinsics and Franka `C2R` | world/robot registration | AutoDex | retained; robot-PC audit still required |
| empty-board ChArUco preflight | table plane and board centre | AutoDex | retained |
| key FoundPose per episode | exact `T_world_key` and tabletop pose | AutoDex | orchestration retained; new learned representations are missing |
| socket FoundPose at session start | fixed `T_world_socket` | AutoDex FoundPose plus new startup orchestration | repeated measurement, SE(3) medoid, freeze, and evidence saving implemented; representation missing |
| BODex candidate format | object-relative Inspire proposal | AutoDex/BODex | retained with a handle proxy and contact screen |
| full-key candidate gates | contact, table, socket, attached-object feasibility | new precision logic using AutoDex assets/planners | sampled screens exist; continuous end-to-end planner gate is not integrated |
| grasp selection and coverage | choose useful physical grasp trials | AutoDex | retained, but must be conditioned on a task-plan pass before execution |
| fixed socket in planning scenes | protect the fixture | new scene composition using AutoDex cuRobo format | implemented for normal and recovery scenes |
| grasp/lift execution | acquire and retain key | AutoDex Franka/Inspire executor | retained |
| task action runtime | transfer, insertion, extraction, return | new | not implemented in `run_pipeline.py` |
| outcome semantics | separate grasp/task/reset evidence | new | grasp/task fields exist; reset field and physical insertion evaluator remain |
| VLM reasoning | phase-aware visual verdict and failure explanation | ZeroDex-style idea on AutoDex images | design only; must begin read-only |
| adaptive next trial | avoid redundant physical trials | future AutoDex extension | deferred until reliable outcome/failure evidence exists |

The existing `TaskInterface` is currently an **outcome evaluator only**.
Passing a custom task into `run_pipeline.main(task=...)` does not add insertion
motion. Today the program still executes AutoDex pick/lift and ordinary table
placement, then calls `task.evaluate`. The task action runtime described below
must be added before any command is described as an insertion run.

## Frame contract

`T_A_B` maps coordinates from frame B into frame A.

| transform | source | lifetime |
|---|---|---|
| `T_world_key` | key FoundPose | refreshed every episode and after reorientation |
| `C2R = T_world_robot` | AutoDex Franka hand-eye calibration | calibration session |
| `T_robot_world = inv(C2R)` | derived | calibration session |
| `T_world_socket` | socket FoundPose | measured at session startup |
| `T_robot_socket = T_robot_world @ T_world_socket` | derived | frozen experiment session |
| `T_socket_key_preinsert` | key/socket CAD | immutable asset |
| `T_socket_key_seated` | key/socket CAD | immutable asset |
| `T_hand_key` | grasp-time FK and perceived key pose | immutable while grasp remains closed |

`T_robot_socket` therefore means the measured socket frame expressed in the
Franka base frame. It is not a second calibration and not a claim that the
socket is at a globally hard-coded location.

The nominal goals are:

```text
T_robot_key_preinsert = T_robot_socket @ T_socket_key_preinsert
T_robot_key_seated    = T_robot_socket @ T_socket_key_seated

T_robot_hand_preinsert = T_robot_key_preinsert @ inv(T_hand_key)
T_robot_hand_seated    = T_robot_key_seated    @ inv(T_hand_key)
```

The currently stored BODex wrist transform is `T_key_hand` (the hand pose
expressed in the key frame); its inverse is the nominal `T_hand_key`. After a
real grasp, `T_hand_key` should be reconstructed from robot FK and the last
trusted key pose, then kept fixed. A later visual held-key estimate is an error
measurement, not permission to make the rendered or planned key slide inside
the hand.

## Symmetry policy

There are two different symmetry questions, and conflating them is unsafe.

- The complete key has **identity-only task symmetry**. Its keyed shaft fixes
  insertion yaw. The socket likewise has identity-only pose symmetry because
  the keyed bore matters even if its outer body looks nearly symmetric.
- The rectangular handle admits a **C2 grasp-proposal symmetry** about key Z.
  It can group tabletop proposal/coverage classes as `000`, `{001,002}` with
  representative `002`, and `{003,004}` with representative `004`.

The runtime still retains all five observed exact tabletop poses. A candidate
transferred from one member of a proposal class to another is a new candidate
and must pass full-key contact, table, arm, transfer, and socket validation.
The representative is allowed to rank or generate candidates; it is not
allowed to replace `T_world_key` or fold insertion yaw. This contract is
machine-readable in `assets/precision_insertion/task_symmetry.json`.

The mounted socket has one operational fixture pose. Geometric stable poses
of a loose socket are irrelevant to this experiment.

## Session startup state machine

```text
BOOT
  -> AUDIT_AUTODEX_CAMERAS_AND_CALIBRATION
  -> MEASURE_SOCKET_WITH_KEY_REMOVED
  -> CHECK_SOCKET_REPEATABILITY_AND_SELECT_SE3_MEDOID
  -> FREEZE_T_ROBOT_SOCKET_AND_BUILD_STATIC_COLLISION_WORLD
  -> MEASURE_EMPTY_CHARUCO_TABLE
  -> OPERATOR_PLACES_KEY
  -> READY
```

The current `_session_preflight` already runs socket measurement before the
empty-table ChArUco measurement. Socket preflight must save raw synchronized
images, masks, individual poses, residuals, chosen medoid, `C2R`, mesh/frame
identity, and the final `fixture_pose.session.json`. It rejects invalid SE(3),
silhouette failure, missing views, and excessive translational or rotational
spread.

Repeatability alone is insufficient: a stable biased socket pose is possible.
Before 1.0 mm testing, compare the FoundPose result with an independent fixture
or metrology reference and measure yaw as well as translation. Fixed means
rigidly bolted for a session, not perfectly known without measurement.

## Candidate lifecycle and full-path planning

### Offline generation and promotion

```text
handle-proxy BODex proposal
  -> declared-contact quality gate (including 2 mm margin)
  -> complete Inspire/full-key contact-policy gate
  -> exact tabletop pickup/pregrasp clearance
  -> MuJoCo grasp and lift validation
  -> store candidate geometry without claiming insertion success
```

### Online candidate preflight

For every ranked candidate compatible with the observed exact pose:

1. Plan approach, closure, and clearance lift with full key/table/socket scene.
2. Construct the immutable candidate-specific `T_hand_key` and attach the full
   key collision mesh.
3. Plan a clearance transfer waypoint; never sweep directly through the
   socket or table.
4. Solve and plan to the exact CAD pre-insertion hand goal.
5. Validate complete hand/socket clearance and shaft/bore corridor geometry.
6. Validate the constrained insertion stroke for nominal geometry. This is a
   geometric gate only, not a promise of physical seating.
7. Validate axial extraction to pre-insertion and the reverse transfer to the
   observed original tabletop key pose.
8. Save the result as one immutable `TaskPlanBundle` containing candidate and
   asset hashes, transforms, joint paths, collision-scene hash, and validation
   reports.

Only a candidate passing all eight may be executed. If the best pick grasp
cannot reach the socket, select another grasp; do not insert an unvalidated
in-hand transition. A deliberate regrasp would be a separate future task with
its own fixture, perception, planning, and success criteria.

The planner integration should be a candidate acceptance callback after the
existing AutoDex pick/lift preflight:

```text
GraspPlanner.plan(..., candidate_acceptance_hook=precision_task.plan_candidate)
```

The hook returns either a complete `TaskPlanBundle` or a typed rejection such
as `table_collision`, `hand_socket_collision`, `preinsert_ik`,
`transfer_collision`, or `reset_unreachable`. This preserves AutoDex ranking,
candidate keys, and coverage while preventing a pick-only candidate from
reaching physical execution.

## Physical episode state machine

```text
OBSERVE_KEY
  -> CLASSIFY_EXACT_POSE_AND_PROPOSAL_CLASS
  -> SELECT_FULLY_PREFLIGHTED_CANDIDATE
  -> EXECUTE_PICK_AND_LIFT
  -> VERIFY_HOLD
       failure/unknown -> SAFE_PICK_RECOVERY -> END
  -> EXECUTE_RIGID_TRANSFER
  -> HOLD_AT_PREINSERT_AND_CAPTURE
  -> INSERT_GUARDED
  -> VERIFY_INSERTION
       success/failure/unknown
  -> EXTRACT_GUARDED_TO_PREINSERT
  -> EXECUTE_REVERSE_TRANSFER
  -> PLACE_AT_ORIGINAL_EXACT_TABLETOP_POSE
  -> OPEN_AND_RETRACT
  -> VERIFY_RESET
  -> UPDATE_RESULTS_AND_COVERAGE
```

The task action hook belongs immediately after lift/hold verification and
before the current coverage-driven table reposition/place branch. A new
`PhysicalTaskRuntime` should expose at least:

```text
plan_candidate(context, pick_plan) -> TaskPlanBundle | TaskPlanRejection
execute_after_lift(context, task_plan) -> TaskExecutionRecord
evaluate(context, grasp_evidence, task_execution) -> TaskOutcome
recover_or_reset(context, task_execution) -> ResetOutcome
```

`LiftTask` remains the default and follows the current code path unchanged.
`PrecisionInsertionTask` owns transfer, insertion, extraction, and return, so
`run_auto.py` does not accumulate object-specific controller logic.

### Guarded insertion and extraction

For 1.5 mm, “open loop” means no alignment search. It does not mean open-loop
safety. Execute a low-speed socket-axis stroke while logging Cartesian pose,
joints, commanded depth, measured depth, force/torque, velocity, and abort
reason. Hard safety limits are deterministic and commissioned on the robot;
they are never supplied by a VLM.

The task-specific contact model divides socket space into:

- allowed bore-corridor proximity/contact for the shaft during axial motion;
- forbidden outer rim/body penetration by the hand or handle;
- forbidden lateral or yaw motion after force exceeds the safe preload;
- seated depth interval, which must agree with force and visual evidence.

On a jam or partial insertion, hold the grasp, retract along the last verified
axial path, and only then use the reverse free-space plan. Never open the hand
inside or above the socket as a generic recovery.

## Successful reset and pose reorientation are different operations

### Reset after every attempted insertion

Reset restores the exact initial tabletop pose of that episode:

```text
seated/contact pose -> guarded axial extraction -> preinsert
  -> reverse transfer -> original observed tabletop pose
  -> open -> vertical retract -> verify key support and pose
```

The reverse transfer may reuse the validated path only after the key has
returned to the nominal pre-insertion attachment state. If contact search
changed XY/yaw, first unwind or re-establish that state under force control.
`reset_success=false` makes `session_continuable=false` even when
`task_success=true`.

### Reorientation after pose coverage is exhausted

Reorientation is not part of a normal successful trial. It runs when every
eligible grasp for the current exact pose has been attempted or the current
pose has no full-task-feasible candidate:

```text
choose a previously validated acquisition grasp
  -> pick/lift -> rigid wrist/arm reorientation
  -> place in a target exact stable pose with remaining coverage
  -> open/retract -> re-run key FoundPose
```

The frozen socket remains in the collision world. The next episode must not
reuse the commanded placement pose as truth; it observes the actual key again.
The existing AutoDex rotate/reorient handlers provide the resource-sharing and
coverage pattern, but their target selection must use precision task
feasibility rather than pick coverage alone.

## Outcome and data contracts

The episode record should contain:

```json
{
  "grasp_success": true,
  "task_success": false,
  "reset_success": true,
  "session_continuable": true,
  "episode_status": "task_failure_reset_ok",
  "failure_phase": "insert_guarded",
  "failure_class": "rim_jam"
}
```

The existing top-level `success` remains a compatibility alias for
`task_success`; consumers must use the explicit fields. Candidate state is
written as soon as lift verification is judged:

- judged retained grasp -> candidate `success=true`, regardless of later jam;
- judged miss/slip -> candidate `success=false`;
- unjudgeable evidence or system failure -> no candidate label;
- planning rejection before motion -> no physical candidate label, but keep a
  typed planning record;
- task/reset results never overwrite candidate grasp success.

Every phase writes synchronized evidence to the existing pipeline trace:
input transforms and covariance/repeatability, selected candidate key,
`T_hand_key`, plan/asset hashes, robot states, force/torque trace, camera frame
IDs, deterministic verdict, VLM verdict, and recovery outcome.

## ZeroDex-style VLM reasoning on AutoDex cameras

The useful ZeroDex element is structured visual reasoning, not its camera
setup. Begin with a read-only shadow observer at motion holds:

| checkpoint | visual question | deterministic evidence fused with it |
|---|---|---|
| after lift | retained key, miss, or slip? | hand state, motion completion, optional held-key pose |
| pre-insertion hold | gross key/socket alignment or occlusion? | CAD target residual, calibrated projections |
| insertion abort/final hold | seated, partial, rim jam, or unknown? | depth, F/T trace, controller state |
| after reset | key supported on table in intended pose? | FoundPose, table plane, hand-open state |
| after reorientation | did a new exact stable pose result? | fresh FoundPose and tabletop classifier |

Supply synchronized multi-view before/after frames, phase labels, camera IDs,
and deterministic metadata. Require schema-validated output:

```json
{
  "verdict": "success | failure | unknown",
  "phase_reached": "lift | preinsert | insertion | reset | reorient",
  "failure_class": "none | grasp_miss | grasp_slip | pose_error | rim_jam | partial_insertion | collision | occluded | system_error",
  "evidence_views": ["camera_serial/frame_id"],
  "confidence": 0.0,
  "reason": "short evidence-grounded explanation"
}
```

Malformed output, insufficient visibility, or disagreement with deterministic
depth/force constraints becomes `unknown`. The VLM does not command motion,
relax collision checks, or override a safety stop. Human labels remain ground
truth until a held-out confusion matrix and unknown rate show that the VLM can
replace the unreliable ChArUco-corner heuristic for grasp outcome labeling.

## Stage gates for the four clearances

| clearance | purpose | required gate before progression |
|---|---|---|
| 1.5 mm | full integration and safe straight insertion | one full-task-feasible grasp, repeatable socket pose, guarded stroke, successful extraction/reset, labelled end-to-end repetitions |
| 1.0 mm | quantify accumulated accuracy | distributions for socket bias/repeatability, key pose, `T_hand_key`, Franka/TCP, printed dimensions, and preinsert XY/yaw residual |
| 0.5 mm | add contact search | bounded XY then yaw controller, safe return between hypotheses, commissioned force/depth limits, better result than straight baseline |
| 0.3 mm | final precision condition | freeze the 0.5 mm controller and thresholds before evaluation; no tuning on the final test set |

The 0.1 mm part is useful as a fabrication/metrology stress sample but is not a
runtime stage because its physical fit is already unreliable.

## Failure attribution and adaptation

Always assign the earliest failed phase and preserve raw evidence.

| phase | representative classes | policy update allowed |
|---|---|---|
| perception/startup | missing representation, segmentation, calibration, pose ambiguity | perception/calibration only |
| candidate planning | forbidden contact, table/socket collision, IK, reset unreachable | candidate feasibility model |
| grasp/hold | miss, slip, unstable closure | grasp success model and coverage |
| transfer/preinsert | attached-object collision, joint limit, pose residual | task-feasibility ranking |
| insertion | rim jam, yaw/XY error, friction, controller limit | alignment/search controller |
| verification | occluded or sensor/VLM disagreement | no physical policy update |
| reset/reorient | extraction, return, placement, or reacquisition failure | recovery policy; stop session if unsafe |
| system | camera, network, CUDA, robot SDK | no physical policy update |

Adaptive selection should use typed outcomes and grasp/task context, not a
single failure bit. It is scientifically premature until outcome labels and
the full-task feasibility gate are reliable.

## Verified status and blockers (2026-10-06)

Verified in this workspace:

- four metric runtime keys (1.5/1.0/0.5/0.3 mm), handle proxies, exact unified
  socket pose/collision object, CAD preinsert/seated transforms, stage files,
  and NAS handoff tooling exist;
- contact policy encodes side/rear-only contact and the 2 mm proposal margin;
- AutoDex camera contract, ChArUco startup, socket measurement/medoid/freeze,
  and socket injection into normal/recovery scenes exist;
- lift and task result semantics are separated;
- all five tabletop scenes now have BODex generation assets, while proposal
  symmetry exposes three presentation/proposal classes;
- scene `002` candidate `511` and scene `004` candidate `290` pass the current
  sampled exact-mesh geometric preview while preserving rigid attachment; this
  is not cuRobo/MuJoCo or physical certification;
- scene `000` has 5,000 BODex seeds and 24 declared-contact/quality survivors.
  Seven survive the sampled complete-hand contact gate, but every one fails
  the socket gate with about 6.2--6.8 mm sampled penetration. There is
  currently no honest rigid insertion candidate for representative `000`.

Blocking physical execution:

1. FoundPose `repre.pth` is absent for all four keys and the socket.
2. The robot PC still needs a passing audit of active AutoDex serials,
   intrinsics/extrinsics, hardware sync, timestamp camera, and Franka `C2R`.
3. Scene `000` needs a new insertion-clear side/rear grasp family or a
   deliberately designed regrasp fixture; the current candidates cannot be
   used.
4. Candidates `511` and `290` still need continuous cuRobo attached-object
   planning, MuJoCo validation, reset planning, and physical validation.
5. `PhysicalTaskRuntime`, `TaskPlanBundle`, insertion/extraction controller,
   reset outcome, and task-aware candidate hook are not implemented.
6. Force/torque, velocity, depth, search, and abort limits remain uncommissioned
   and must not be guessed offline.
7. Phase-aligned VLM evaluator and physically labelled validation dataset do
   not exist.

Therefore the current `run_pipeline.py` is safe to use for AutoDex-style
pick/lift bring-up with a measured socket in the collision world once learned
pose assets are restored. It is **not yet an executable precision-insertion
pipeline**.

## Implementation order

1. Generate/restore the five FoundPose representations and pass the robot-PC
   AutoDex camera/calibration audit.
2. Add `TaskPlanBundle` and a task-aware candidate acceptance hook; stop at a
   visualized pre-insertion hold with no contact.
3. Continuously validate candidates `511`/`290`, generate an honest scene-000
   grasp, and add reverse-path/reset validation.
4. Add `PhysicalTaskRuntime` after lift verification while retaining
   `LiftTask` as a byte-for-byte behavioral default.
5. Commission guarded 1.5 mm insertion and extraction with manual ground-truth
   labels; require reset before automated repetition.
6. Measure the 1.0 mm error budget, then commission bounded 0.5 mm XY/yaw
   search and freeze it for 0.3 mm evaluation.
7. Run the VLM in shadow mode, audit it, and only then consider replacing the
   ChArUco grasp heuristic.
8. After trustworthy typed outcomes exist, add adaptive next-trial selection
   over candidate similarity, task feasibility, and failure class.
