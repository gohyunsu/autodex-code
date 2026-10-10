# Precision insertion demo implementation plan

The demo will run on the AutoDex Franka and Inspire rig, with its current
ParaDex camera system, v8 grasp contract, and `object_processing` geometry.
It will have its own trial loop under `demo/precision-insertion/`; it will not
call `src.execution.run_auto.main()` or add insertion behavior to the existing
AutoDex entry points. A simulated grasp pass is not an insertion pass.

## Legacy code boundary

`main` at `448732d` is the original-file reference. The 18 pre-existing files
changed by this feature branch have been restored to that tree; the previous
implementation remains recoverable at `archive/precision-pre-isolation`.
The duplicated fixture test and its old execution-side helper were replaced
by the local `tests/test_fixture_world.py`, `geometry.py`, and `world.py`.
An isolation check must continue to assert an empty `git diff main...HEAD`
for **all 18** original paths, not just the three execution entry points.
New feature files elsewhere in the branch, including `autodex/tasks` and the
asset-generation scripts, are still historical/offline utilities; they are
not the new robot runner. Any feature-specific runtime policy they contain
must be migrated into this demo before robot mode is exposed.

| Former shared-file modification | Demo ownership / migration gate |
| --- | --- |
| `run_auto.py`, `run_pipeline.py`, `scene_cfg.py` | Independent CLI, state machine, ChArUco-then-socket calibration, and local scene builder; never import their entry-point side effects. |
| `autodex/planner/planner.py` and `autodex/utils/symmetry.py` | Use the unchanged public planner with explicit candidates; keep cylinder symmetry local. The changed cuRobo `update_world` list contract must be tested against the installed version and handled in a demo adapter or a pinned compatible environment, not patched globally. |
| `reorient.py`, `view_reorient.py`, `gen_all.py`, `gen_scene.py` | Demo-owned native-v8 directed transition catalog, preflight, execution and visualization. Do not treat old legacy-index cells as native-v8. |
| `BODex/generate.py`, `run_sim_filter.py`, and the five vendored cuRobo files | Keep their original behavior. Place precision-specific generation configuration, evidence normalization, and any needed compatibility shim under the demo; re-run simulator parity tests before promoting a candidate. |
| `plan_test.py`, `exp.py` | Demo-only debugging and visual evidence; not a physical-success gate. |

The `scene_cfg.py` change contains two useful behaviors, but neither must live
in that shared file: (1) the cylindrical key is authored along local z, not
the historical can local y, so tabletop snapping must use asset symmetry;
(2) the measured socket must be a fixed collision mesh. The new demo's local
`symmetry.py` and `world.py` own those behaviors. Calling the original
`pose_world_to_scene_cfg()` for the cylindrical key without this local
adaptation would be incorrect. Existing AutoDex cylinder behavior should not
change merely because this demo exists.

The demo may import unchanged low-level ParaDex/AutoDex APIs for camera
capture, FoundPose, Franka/Inspire control, cuRobo, and MuJoCo. It must not
call `run_auto.main()` or `run_pipeline.main()`. Pure functions in the
unchanged `scene_cfg.py` may be reused when their contract fits; they must
not be copied wholesale. A small adapter should make each dependency
explicit and testable. No implicit fallback to another camera profile or a
legacy ParaDex object root is allowed.

## Reuse map, with the boundaries that require new task logic

| Existing API to call unchanged | Demo-specific glue; reason it cannot be reused verbatim |
| --- | --- |
| ParaDex camera controller, trigger/timestamp handling, `SnapshotOrchestrator` | Own the order and lifetime of ChArUco, socket, then per-trial key captures; assert active serials match calibration. Do not implement another camera transport. |
| `src.execution.charuco_tabletop.measure_tabletop_from_images` / `save_tabletop_measurement`, `src.execution.handeye.save_arm_C2R` | Retain one session calibration record and its quality gates. No second board detector or hand-eye solver. |
| `InitOrchestrator.init_object` / `trigger_init` | Swap the initialized FoundPose target from socket to key; keep the same cameras and explicit v8 mesh/representation frame. Do not implement another pose estimator. |
| `src.execution.scene_cfg.pose_world_to_scene_cfg` and `autodex.planner.obstacles.add_obstacles` | Use unchanged table/target construction for the square key. Locally correct the cylindrical key's local-z axial tabletop handling and add the session-frozen socket mesh; the original `scene_cfg` hardcodes older local-y cylinder IDs and has no fixture insertion. |
| `autodex.utils.symmetry.get_asset_symmetry` and v8 tabletop assets | Read the same symmetry metadata but enforce task-specific D∞ key end equivalence versus C∞ socket open-rim direction. For square-key pose classes, reuse `classify_tabletop_pose`; for the cylinder, its current global axis lookup lacks an explicit v8 root, so a small local classifier is necessary. |
| `autodex.utils.path.load_candidate` with explicit `candidates_root` and whitelist, plus `GraspPlanner.plan(candidate_override=...)` | Read scene metadata and `openpose_<pose>.npy` from the explicit shared root, then pass only pose-matching, insertion-evidenced candidates. Avoid `load_candidate(tabletop_pose_stem=...)` because its scene lookup uses a global path; avoid changing `autodex.utils.path` globals. |
| `GraspPlanner` IK, approach, held-object lift and `plan_vertical_stroke`; cuRobo world conversion | Chain these from the measured live state, then add transfer/hold/20 mm whole-hand and whole-key checks. The baseline planner returns after the *first lift-feasible* grasp, so its `PlanResult.success` cannot be the insertion preflight verdict. |
| `src.execution.franka_executor.FrankaExecutor.execute`, `get_arm_qpos`, `get_wrist_pose`, `follow_joint_trajectory`, and `PipelineTrace` | Reuse the tested Franka/Inspire pickup, safe arm path playback, and event stream. `get_hand_qpos()` reports `commanded_nominal`, not measured grip; vision/pose evidence must confirm the hold. The public joint playback has no insertion-specific contact guard, so it must never be used as the sole guarded-insertion controller. |
| Existing reset/reorient candidate file format and MuJoCo tools | Reuse the v8 file/hand model contracts, but implement task-specific fixture-aware directed transitions, repose, and verification locally. The original reorient logic assumes legacy-index candidates unless modified, and its old trajectories were not checked against this socket. |

For every imported API, an adapter test must prove units, frame convention,
arm DOF (Franka 7 + Inspire 6), asset root, and failure behavior. Where an
unchanged API cannot meet the task safely, add only the narrow missing logic
under this demo. In particular, the FR3 executor's private `_follow` has a
single `wrench[2]` stop threshold used for placing; it is **not** an
insertion controller with lateral-force, moment, depth, and jam diagnosis.
Commission a dedicated guarded controller or a vetted external control
interface before contact trials rather than copying the entire executor or
using unguarded `follow_joint_trajectory` for the contact stroke.

## What the existing v8 candidate pool proves

`BODex/generate.py` writes *every* optimized seed under its generated scene,
including `wrist_se3.npy` in the object frame, finger poses, and `bodex_info`.
Its `success`, `grasp_error`, and `dist_error` are numerical solver evidence,
not physical outcomes; the standard filter does not automatically discard
every seed whose BODex `success` flag is false. The standard
`run_sim_filter.py` then checks pregrasp scene collision, whether the squeeze
pose touches the object, and MuJoCo grasp stability under gravity before
copying passes to `AutoDex/candidates/inspire/v8/<object>/...`. Cached
`--sim-only` execution assumes a prior valid contact screen, so its cache
provenance must be verified. None of these gates tests a Franka transfer to a
socket or insertion contact. The v8 coverage precompute means collision-free
in each same-tabletop deployment scene; it does not include IK or motion
planning.

At runtime AutoDex classifies the observed tabletop pose from
`object_processing` poses, loads candidates whose scene JSON
`meta.pose_idx` matches that pose, and optionally filters by scene type,
scene ID, prior success, session-attempted candidate, and remaining coverage.
`candidate_order` is a whitelist ordered by uncovered-scene gain when
coverage is enabled; zero-gain entries are excluded. The planner then maps
each object-frame wrist pose into the current scene, optionally expands
axial symmetry, rejects backward/world/self-colliding pregrasps, solves arm
IK, and tests approach trajectory plus a held-object 10 cm lift. It returns
the **first** candidate that passes those gates. `result.json` success and
coverage are historically grasp-oriented; they must not be reused as an
insertion-success label. The new demo retains exact candidate IDs and every
gate's rejection reason, then adds the full transfer/hold/20 mm preflight
*before* choosing a physical trial. It does not inherit AutoDex's
one-success-per-tabletop stopping rule: multiple insertion scenarios may be
needed at one pose.

## Offline scenario promotion

For each exact `(key, socket, clearance, tabletop pose, grasp id)`, record an
immutable scenario with asset and source hashes. A physical trial can select
it only after these gates, in order:

1. BODex candidate and source tabletop scene are internally consistent with
   v8 `object_processing`; hand joints and object-frame wrist transform exist.
2. Simulated squeeze contacts and gravity stability pass. Contact location is
   a diagnostic or preference, not a substitute for whole-hand clearance.
3. At the **20 mm inserted pose**, every hand link clears the socket; the key
   and socket have the intended fit. The intentional key/socket interaction
   must not be treated as a forbidden hand/socket collision.
4. The **continuous** pick, lift, held-key transfer, pre-insertion alignment,
   20 mm descent, and safe retreat pass the correct collision/contact models.
   An endpoint-only screen is necessary but insufficient. Keep hand joints
   fixed from verified grasp to deliberate release; maintain one explicit
   `T_wrist_key`. Check its uncertainty after lift.
5. cuRobo validates collision, IK, joint limits, and the arm motion with the
   fixed fixture. Use swept or sufficiently sampled *whole-key and whole-hand*
   checks during transfer. Use an exact or conservative bore/rim geometry
   check and MuJoCo for the contact phase; a generic static collision mesh
   alone cannot certify an insertion that intentionally contacts the socket.
6. Simulation and physical commissioning records are separate. A simulation
   pass never silently becomes `hardware_ready`.

The cylinder's axial yaw is symmetric; square-key yaw is task-relevant. Each
mode keeps its own key/socket identifiers, pose classes, collision geometry,
and calibration uncertainty budget. The measured uncertainty plus execution
error must be small relative to the chosen clearance; otherwise refuse the
trial instead of relying on VLM confidence.

## Session and trial state machine

Session startup order is deliberately **ChArUco, then socket**. The archived
feature-branch hook measured the socket first; original AutoDex measures no
socket at all. The
fixture must already be rigidly fixed while the board is measured, and
removing the board must not move it.

1. `BOOT`: verify AutoDex camera IDs/calibration, current Franka hand-eye
   transform, robot identity, force sensing, emergency stop, exact key/socket
   assets, and writable output. No robot motion if a required input is absent.
2. `MEASURE_TABLE`: capture ChArUco once; save table plane, transforms,
   residuals, frame IDs, and uncertainty for all trials in the session.
3. `MEASURE_SOCKET`: with the key absent from the view, capture several
   synchronized socket observations, reject inconsistent measurements, and
   freeze `T_robot_socket` and its uncertainty. Ignore only unobservable
   axial yaw for the cylinder. Save all samples. Build one immutable base
   collision world containing table and socket.
4. `PERCEIVE_KEY`: each trial captures fresh synchronized views, estimates
   `T_robot_key`, classifies its v8 tabletop pose, and checks segmentation,
   visibility, timestamp, calibration, and pose uncertainty.
5. `SELECT_AND_PREFLIGHT`: select only scenarios matching the observed pose
   and geometry. Revalidate against this session's key and socket poses.
   Plan the full chain for each candidate in priority order, not just pickup.
   Log a separate rejection reason at every gate. If the pool is empty, do
   not infer that all possible grasps are impossible.
6. `PICK_AND_LIFT`: execute the chosen approach and grasp, then observe the
   held key. If visible, update `T_wrist_key` from key pose and wrist FK;
   compare it with the planned rigid transform. A slip or uncertain hold
   prevents transfer.
7. `TRANSFER_AND_HOLD`: keep hand joints fixed while moving to a collision-
   checked pre-insertion pose. Measure the actual key-to-socket residual;
   record whether the required transfer endpoint was reached.
8. `GUARDED_INSERT`: descend along the measured socket axis toward a **measured**
   20 mm insertion depth with commissioned force/torque, speed, workspace,
   and timeout limits. Stop and retreat on jam or sensor disagreement. A
   commanded 20 mm stroke alone is not success.
9. `VERIFY_AND_RECOVER`: fuse depth, pose, grip, F/T, abort code, and visual
   evidence. If the key remains safely held after an alignment failure,
   replan a bounded retry for the same grasp. Otherwise try another grasp,
   reset, or stop according to the observed state.

The base table/socket world stays fixed, but the target key must be removed
from free-world obstacles and represented as an attached object after grasp.
Each candidate and each XY retry creates a **new plan** from live robot state;
freezing the socket does not make old trajectories reusable.

For any grasp `g`, define `T_wrist_key(g)` from the simulated candidate and
re-estimate it after lift when possible. The pre-insertion wrist goal is

`T_robot_wrist_goal = T_robot_socket * T_socket_key_goal(dx, dy) * inverse(T_wrist_key)`.

`dx, dy` are metric offsets in the **socket frame**, never raw image pixel
motions. The first experiment adjusts only these two parameters. Square-key
yaw error is measured and gated; it is not silently corrected by XY updates.

## Labels and evidence

Store each milestone independently as `true`, `false`, or `null` (unjudgeable):

| Field | Meaning and decisive evidence |
| --- | --- |
| `grasp_success` | Key is held through lift; image sequence plus key/wrist pose and hand state. |
| `preinsert_reached` | Held key reaches the specified pose above the socket; measured pose residual, grip state, and trajectory completion. |
| `insertion_success` | Observed key/socket penetration reaches at least 20 mm within alignment and force limits, with no abort; corroborated by images when visible. |
| `reset_success` | A safe verified return or controlled recovery makes the next trial possible. |
| `reorient_success` | A fresh tabletop estimate confirms the requested new pose after a validated transition. |

Do not collapse these fields to one `success`. Persist the phase reached,
failure category, candidate ID, frozen fixture record, calibration and asset
hashes, trajectory checks, robot state, F/T/depth trace, synchronized camera
frames, VLM raw/parsed answer, and any human override in a per-attempt record.
Candidate grasp statistics update from `grasp_success`; insertion retry
statistics update from `insertion_success` and failure cause. Unknown is not
success and must not erase a known earlier milestone. Use a controlled failure
taxonomy, initially `perception_unreliable`, `grasp_miss`, `slip`,
`transfer_unreachable`, `transfer_collision`, `preinsert_misaligned`,
`rim_jam`, `depth_shortfall`, `force_abort`, and `reset_failed`. Preserve the
raw evidence so a later review can correct the category without rewriting
the measured stage outcomes.

Each attempt record has one immutable identity and ordered phase events. Its
summary follows this contract (all stage values start at `null`):

```json
{
  "schema": "precision_insertion_attempt_v1",
  "mode": "square",
  "candidate_id": "table/4/84",
  "xy_offset_socket_m": [0.0, 0.0],
  "grasp_success": null,
  "preinsert_reached": null,
  "insertion_success": null,
  "reset_success": null,
  "reorient_success": null,
  "failure_code": null,
  "events": []
}
```

`preinsert_reached=true` does not imply `insertion_success=true`.
`grasp_success=false` leaves later stages `null` rather than falsely
declaring that insertion was attempted and failed. `insertion_success=true`
requires prior grasp and transfer success, measured depth, and no safety
abort. A VLM response alone cannot set any stage to `true`.

## ZeroDex-style visual adjustment

The local ZeroDex implementation uses a high-level multi-image view selector
and subtask/role grounding, then projects candidate 3D points into calibrated
views and asks a VLM to select among numbered candidates. It aggregates
per-view votes, or triangulates independently grounded per-view points with
RANSAC. Its task-completion checker can infer success from an object becoming
occluded at a destination. **That occlusion rule is unsafe for a narrow
socket** and will not be imported as an insertion-success rule.

Specifically, `view_selector.py` makes one multi-image call for role/visibility
and view selection; `grounding_subtasks.py` localizes 2D subtask points;
`mv_grounding_depth_voting.py` samples metric 3D candidates and projects them
into other cameras for numbered per-view choices; and
`triangulation_subtasks.py` offers independent per-view point grounding with
RANSAC. `phase_verifier.py` performs per-view Qwen yes/no completion votes.
The demo will borrow the **view-aware, projected-choice, multi-view
consistency** pattern, not its open-ended waypoint generation, averaged tied
candidates, or majority-vote completion rule.

For this fixed, known task, high-level task planning is unnecessary. At
`POST_LIFT`, `PREINSERT`, and `INSERT_ABORT`, capture synchronized AutoDex
views and project the CAD key axis, socket axis/rim, target pose, and a small
set of *prevalidated socket-frame XY offsets* onto each image. The VLM returns
a structured class (`held`, `slip`, `misaligned`, `rim_jam`, `occluded`, etc.),
visible evidence, used camera IDs, and either one offset **ID** per view or
`abstain`. A resolver combines view votes with measured pose residual and
uncertainty; ties, occlusion, disagreement, or force-limit abort mean no
automatic motion. It never averages tied offsets into an unvalidated action.

The local geometric estimator and F/T controller supply metric units. The
VLM can rank or reject candidate corrections but cannot directly command
Franka or invent a millimeter offset. For an accepted retry, retreat to a
known clear pre-insertion pose, bound the cumulative offset and attempt count,
replan from the live state, and rerun collision and force gates. Store an
ablation of geometry-only, VLM-gated, and VLM-ranked XY selection with the
same trial conditions.

For the initial professor-requested experiment, expose exactly two learnable
execution parameters, `dx` and `dy` in the frozen socket frame. Build a small
calibration-derived metric offset set inside the validated free space (for
example center and signed axis/diagonal choices), render each proposed key
outline and socket rim in every calibrated view, and request one choice ID or
`abstain` with visibility and failure evidence. The resolver checks camera
timestamps, calibration, matching choice IDs across informative views, the
measured key/socket residual, previous trial direction/depth/F/T response,
and a maximum step/total budget. Repeated disagreement or no observable
improvement stops the search; it does not expand the grid. The model may run
locally or through an API behind one read-only interface; image retention,
latency, and validation accuracy decide which backend is used. Start in
shadow mode with no robot command, and quantify direction-choice accuracy
on labeled failure images before enabling bounded retries.

## Reset and repose policy

`repose` means restoring a usable key position while preserving its tabletop
pose class when possible. `reorient` intentionally changes that class. They
need different plans and labels.

- A failed pickup can move the key. Re-perceive it; do not replay a trajectory
  to the old pose without confirming the current state.
- After a transfer failure while still holding the key, plan a safe return to
  the saved tabletop location and pose. If that plan or sensing fails, stop
  for supervised recovery.
- After an insertion jam, stop contact, retreat axially while grip and force
  permit, then retry from a clear hold pose or return the key. Do not shift
  laterally while engaged in the bore.
- If insertion succeeds, **automatic repeated trials are blocked until a
  validated extraction exists**. Extraction is lower priority; initially end
  the session or use an explicitly supervised manual reset. A "reset every
  trial" guarantee cannot be claimed without this step.
- If no candidate passes for the current tabletop pose, choose a target pose
  that has an insertion-ready scenario and a validated directed `i -> j`
  transition. Generate the matching v8 pair scene and reset grasp seeds,
  screen contact and MuJoCo stability, preflight approach/lift/reorient/
  descent/release/exit against the fixed socket, then verify the new pose by
  fresh perception. Current pose/reorientation catalogs are inventories, not
  automatic-reorient-ready policies.

## Code and data layout

All new or ported execution code and its tests belong below
`demo/precision-insertion/`:

```text
demo/precision-insertion/
  run_pipeline.py                 # independent CLI and state machine
  precision_insertion/
    config.py                     # explicit paths, modes, limits
    assets.py                     # v8/object_processing and evidence checks
    camera.py                     # unchanged AutoDex camera API adapter
    calibration.py                # ChArUco/socket measurement and freeze
    symmetry.py                   # local square/cylinder pose handling
    world.py                      # fixed fixture and attached-key worlds
    candidates.py                 # pose-conditioned scenario selection
    planner.py                    # full-chain preflight and XY replanning
    execution.py                  # Franka/Inspire and guarded stroke adapter
    observer.py                   # multi-view VLM and sensor evidence
    recovery.py                   # retreat, repose, reorient, stop
    records.py                    # tri-state results and immutable provenance
  tests/                          # offline contracts and replay fixtures
```

The CLI will require explicit `--shared-root`, `--mode`, `--gap-mm`,
`--output-root`, and dry-run/robot mode. The default is **dry-run**; robot mode
is fail-closed until commissioned. Read key/socket objects under
`<shared-root>/object_processing/<object_id>` and v8 candidates under
`<shared-root>/AutoDex/candidates/inspire/v8/<key>`. Read fixture and scenario
metadata under `<shared-root>/AutoDex/precision_insertion`. Save demo trial
records under a dedicated output root, not inside candidate geometry trees.
Do not switch to legacy `AutoDex/object/paradex` for v8 tabletop indexing.
Some existing loaders hardcode `~/shared_data`; the demo's asset adapter must
resolve the explicit root and pass exact candidates to `GraspPlanner.plan()`
through `candidate_override`, or provide an equivalent verified adapter.
Changing a global path constant in the shared AutoDex package is not allowed.

## Current asset evidence and missing gates

In the local `~/shared_data` snapshot, metric v8 key/socket meshes and
metadata exist for the square and cylindrical families. The FoundPose asset
directories checked for the 1.5 mm square key, unified socket, cylindrical
key, and 20 mm-gap cylindrical socket contain `GENERATION_REQUIRED.json`,
not `object_repre/v1/<object>/1/repre.pth`. The cylindrical Inspire v8
candidate directory likewise contains only a generation marker. The square
1.5 mm v8 directory has at least one grasp-simulation candidate, but that is
not full-task evidence. No precision-key native-v8 reset candidate directory
was found under the expected Inspire `reset_{0,4,8,12}` roots. These are
local-file observations, not a claim about a different AutoDex host or NAS
mount. A startup validator must report each missing exact path before any
camera or robot lease is taken.

Other required robot-mode evidence is still missing: measured ChArUco and
socket calibration for the live session; certified fixture rigidity and
calibration uncertainty; continuous Franka/Inspire pick-to-20 mm trajectory
with whole-hand/whole-key checks; contact-dynamics and F/T abort validation;
physical success measurement; and commissioned extraction/reset/reorient
policies. Presentation animations and endpoint-only geometry do not fill
these gaps.

## Implementation sequence and acceptance gates

1. Verify the original-file diff against `main` is empty and keep it empty.
   Build the independent CLI, explicit path configuration, camera/robot
   adapters, and tri-state record schema under this demo. Do not resurrect
   the old execution-side fixture helper or import the historical
   `autodex/tasks` package as the new runner's control policy. Test that
   default dry-run cannot claim hardware readiness or write motion commands.
2. Replay saved camera captures to test ChArUco-then-socket startup, pose
   repeatability, symmetry, table/socket world construction, and mismatch
   rejection without robot motion. Test socket movement during the session,
   stale frames, switched camera IDs, bad hand-eye, and asymmetric rim flips.
3. Generate missing FoundPose representations on the actual AutoDex camera
   host, then audit exact pose-conditioned v8 candidate pools. Re-run BODex,
   collision/squeeze/MuJoCo filtering and v8 coverage when needed, keeping
   the generation recipe and hashes. Promote at least one scenario through
   endpoint **and continuous** full-task simulation; record all failure
   gates. No pass means robot insertion stays disabled.
4. Run supervised 1.5 mm square bring-up with calibrated camera/hand-eye and
   approved force limits. Verify the three milestone labels independently,
   then run repeated trials only after safe reset is commissioned.
5. Add the projected multi-view offset-choice experiment, first as logged
   advice, then as bounded same-grasp replanning after safety review. Compare
   against geometry-only correction and no correction.
6. Commission reorientation transitions and post-success extraction as
   separate recovery milestones. Progress to tighter clearances only after
   measured uncertainty and repeatability support them.
