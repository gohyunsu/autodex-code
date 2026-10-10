# Precision insertion demo implementation plan

The demo will run on the AutoDex Franka and Inspire rig, with its current
ParaDex camera system, v8 grasp contract, and `object_processing` geometry.
It will have its own trial loop under `demo/precision-insertion/`; it will not
call `src.execution.run_auto.main()` or add insertion behavior to the existing
AutoDex entry points. A simulated grasp pass is not an insertion pass.

## Legacy code boundary

`main` at `448732d` is the reference for the original execution files. The
current feature branch still changes `src/execution/run_auto.py`,
`run_pipeline.py`, and `scene_cfg.py`; those changes have **not** yet been
removed. Before declaring the demo isolated, migrate any required behavior to
this directory and verify that these three files match `main`. Do not discard
unrelated worktree edits while doing so.

The `scene_cfg.py` change contains two useful behaviors, but neither must live
in that shared file: (1) the cylindrical key is authored along local z, not
the historical can local y, so tabletop snapping must use asset symmetry;
(2) the measured socket must be a fixed collision mesh. The new demo's local
`symmetry.py` and `world.py` will own those behaviors. Calling the original
`pose_world_to_scene_cfg()` for the cylindrical key without this local
adaptation would be incorrect. Existing AutoDex cylinder behavior should not
change merely because this demo exists.

The demo may import unchanged low-level ParaDex/AutoDex APIs for camera
capture, FoundPose, Franka/Inspire control, cuRobo, and MuJoCo. It must not
import execution side effects from `run_auto.py`, `run_pipeline.py`, or
`scene_cfg.py`. A small adapter should make each such dependency explicit and
testable. No implicit fallback to another camera profile or a legacy ParaDex
object root is allowed.

## What the existing v8 candidate pool proves

`BODex/generate.py` saves every seed and its `success`, `grasp_error`, and
`dist_error` metadata; saving a seed alone is not a quality pass. The standard
`run_sim_filter.py` then checks pregrasp scene collision, squeeze contact with
the object, and MuJoCo grasp stability under gravity before copying passing
seeds to `AutoDex/candidates/inspire/v8/<object>/...`. It does **not** test a
Franka transfer to a socket or insertion contact. The v8 coverage precompute
means collision-free in each same-tabletop deployment scene; it does not
include IK or motion planning.

At runtime AutoDex classifies the observed tabletop pose, loads matching
candidate scene metadata, excludes attempted or covered candidates, ranks by
remaining scene coverage, then checks collision, IK, arm/finger approach, and
a held-object 10 cm lift preflight. Its selected grasp is therefore a
pick-and-lift candidate, not a complete insertion scenario. The new demo will
preserve this distinction in evidence records and names.

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

Session startup order is deliberately **ChArUco, then socket**, unlike the
current modified `run_pipeline.py`, which measures the socket first. The
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

## ZeroDex-style visual adjustment

The local ZeroDex implementation uses a high-level multi-image view selector
and subtask/role grounding, then projects candidate 3D points into calibrated
views and asks a VLM to select among numbered candidates. It aggregates
per-view votes, or triangulates independently grounded per-view points with
RANSAC. Its task-completion checker can infer success from an object becoming
occluded at a destination. **That occlusion rule is unsafe for a narrow
socket** and will not be imported as an insertion-success rule.

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

## Implementation sequence and acceptance gates

1. Build the independent CLI, explicit path configuration, adapters, and
   record schema. Port only needed startup, symmetry, and fixture behavior;
   then restore the three legacy execution files to `main` and verify their
   exact diff is empty. Baseline AutoDex tests must still pass.
2. Replay saved camera captures to test ChArUco-then-socket startup, pose
   repeatability, symmetry, table/socket world construction, and mismatch
   rejection without robot motion.
3. Generate and audit exact pose-conditioned candidate pools. Promote at
   least one scenario through endpoint **and continuous** full-task
   simulation; record all failure gates. No pass means robot insertion stays
   disabled.
4. Run supervised 1.5 mm square bring-up with calibrated camera/hand-eye and
   approved force limits. Verify the three milestone labels independently,
   then run repeated trials only after safe reset is commissioned.
5. Add the projected multi-view offset-choice experiment, first as logged
   advice, then as bounded same-grasp replanning after safety review. Compare
   against geometry-only correction and no correction.
6. Commission reorientation transitions and post-success extraction as
   separate recovery milestones. Progress to tighter clearances only after
   measured uncertainty and repeatability support them.
