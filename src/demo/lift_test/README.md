# Jacobian lift test

`lift_test` is an executor-free experiment for the proposed 5 mm Jacobian
continuation lift. It does not modify or invoke the normal pickup executor.
It can, however, obtain the board and object pose from the normal camera
stack. The current scope is the production `v8` + right-Inspire + `table`
contract. Select either arm model without changing the candidate, scene, or
Jacobian-lift flow: `--arm franka` uses `fr3_inspire` (the default) and
`--arm xarm` uses `xarm_inspire`.

## Start a session

The default measures an empty Charuco board, then repeats this dialogue:

```text
XY (Enter = board proxy center) -> object -> tabletop pose -> plan
```

```bash
python src/demo/lift_test/run_session.py

# Run the identical experiment against the XArm6 + Inspire model.
python src/demo/lift_test/run_session.py --arm xarm
```

`q` at every prompt exits. After a trial, choose `n` for another scenario,
`v` to launch Viser for the last successful animation, or `b` to remeasure the
empty board. Object and XY are deliberately not command-line arguments: every
trial records the inputs it actually used.

The four proxy vertices are the outer boundary of the complete Charuco
checkerboard. The normal all-corner fit measures its 9 x 6 internal-corner
lattice, and the code extends that fitted frame by one 5 cm square on every
side to recover board 11's 10 x 7 (50 cm x 35 cm) checkerboard rectangle.
The proxy is still an XY experiment domain, not a claim about any extra white
paper margin around the printed board.

## Candidate policy

The test retains the production candidate funnel up to, but not including,
the legacy lift preflight:

```text
v8 tabletop classification + coverage order
→ candidate collision filter + endpoint IK
→ production approach plan / squeeze q
→ 5 mm Jacobian lift (legacy preflight replacement)
```

The default policy is intentionally a fresh experiment state:

```bash
# Default: do not let past legacy result.json outcomes hide candidates.
python src/demo/lift_test/run_session.py --candidate-policy clean-state

# Read the shared production candidate completion state, without writing it.
# This answers: “what candidate pool would the pipeline see right now?”
python src/demo/lift_test/run_session.py --candidate-policy current-state

# Restrict to candidate records with an existing result.json whose
# success field is true.  This is a separate comparison mode, not the
# production "what remains to collect?" policy.
python src/demo/lift_test/run_session.py --candidate-policy verified-only

# Live, executor-free replacement preflight for the normal table pipeline.
# Only legacy lift-preflight is replaced by the Jacobian backend.
python src/demo/lift_test/run_session.py \
  --object-source perception \
  --candidate-policy pipeline-parity \
  --scene table

# The same restricted-pool comparison is available for an XY map.
python src/demo/lift_test/run_grid.py --candidate-policy verified-only
```

`clean-state` uses the immutable v8 coverage catalogue and an empty,
session-local state path. `current-state` reads shared candidate `result.json`
files exactly as production candidate loading does, so it skips every
completed candidate record — both a prior `success=true` and a recorded
failure. It therefore means “what remains for collection now?”, **not**
“known-good grasps.” `verified-only` does the opposite: it uses the same
immutable coverage ranking as `clean-state`, then retains only candidate
directories whose existing `result.json` contains `"success": true`.
Candidate result state is hand-scoped in the existing asset layout, so this
mode preserves that production convention rather than claiming arm-specific
hardware evidence. No mode writes under `candidates/`.

`pipeline-parity` is deliberately only available from `run_session.py`, and
requires `--object-source perception`. It adds the contracts that candidate
state alone cannot express:

```text
live FoundPose raw pose
→ production tabletop classification on that raw pose
→ pose_world_to_scene_cfg + run_auto add_obstacles(scene, parameters)
→ production shared completion state / coverage ordering
→ production early scene-skip or reorient decision
→ production collision + endpoint IK + approach planning
→ Jacobian lift in place of legacy lift-preflight
```

It accepts the same scene controls as `run_auto.py`, for example
`--scene wall --wall-gap 0.04 --wall-angle 15` or the shelf/clutter controls.
Every result records a `pipeline_parity.contract` and an optional early gate,
so `scene_already_done` and `reorient_needed` are not misreported as lift
failures. It remains executor-free: it proves replacement planning parity,
not live q tracking, gripper contact, or final label success.

By default every IK-valid candidate in coverage order is attempted. Limit a
diagnostic run explicitly; a limited failure is recorded as
`candidate_search_truncated`, not as global infeasibility.

```bash
python src/demo/lift_test/run_session.py --max-candidates 12
```

Each accepted 5 mm endpoint and every interpolated joint-space sample between
adjacent endpoints is collision-checked. The default interpolation spacing is
at most `0.02 rad` per arm joint; adjust it only for diagnostic cost studies:

```bash
python src/demo/lift_test/run_session.py \
  --max-segment-joint-delta-rad 0.02
```

The samples of each Jacobian chord, and the final timestamped C2 trajectory,
are evaluated as cuRobo GPU batches.  This changes only scheduling: every q
row, its original order, the target-present approach world, and the
target-removed squeeze/lift world are retained.  Final validation also uses
batched wrist FK and vectorized attached-object/table clearance.  Result
metadata records `collision_backend=curobo_rollout_batch`, sample/chunk counts,
and the collision/FK/clearance timing split.

## Live perception input

```bash
python src/demo/lift_test/run_session.py --object-source perception
```

This mode asks only for the object name. After the object is placed, it calls
the normal `InitOrchestrator.init_object()` and `trigger_init()` and uses
`pose_world_to_scene_cfg()` to form the planning scene. Thus it retains the
current production behavior: if a measured tabletop is available and the
planning mesh bottom is below it, the object is raised in Z only. Raw and
post-snap robot poses plus the snap delta are saved for each trial.

No `FrankaExecutor`, `RealExecutor`, or arm-motion call is created in either
mode.

## Replay

Each successful episode contains a canonical `plan/execution_trajectory.npz`:
one uniformly timestamped `qpos[T, dof]`, `time_s[T]` reference which stitches
the MotionGen approach, a sampled squeeze, and a C2-retimed lift.  Lift hand
q is fixed to the squeeze q.  The exact saved samples are collision/FK checked
before the candidate is accepted. `animation.npz` is a render companion made
from that same `q(t)`, including its timestamps and attached-object poses.
Replay it without a camera, planner, or CUDA context:

```bash
python src/demo/lift_test/viser_view.py --episode \
  ~/shared_data/AutoDex/experiment/lift_test/inspire/<object>/<episode>
```

The viewer reads the selected arm from the episode's `request.json` and loads
the matching FR3 or XArm6 URDF automatically. Older episodes without that
field remain FR3 by default; override it only when needed with `--arm xarm`.

## Output

Episodes remain compatible with the regular experiment hierarchy:

```text
~/shared_data/AutoDex/experiment/lift_test/inspire/<object>/<timestamp>_<trial>/
  request.json
  result.json
  scene_cfg.json
  pose_robot_raw.npy
  pose_robot_scene.npy
  candidate_source.json
  candidate_metadata.json
  candidate_attempts.json
  jacobian_steps.csv
  animation.npz
  plan/
    traj.npy                 # production-compatible approach trajectory
    wrist_se3.npy            # selected grasp wrist target
    lift_jacobian.npz        # never a legacy lift_preflight artifact
    execution_trajectory.npz # canonical 10 ms q(t), phases, limits, validation
    timing.json              # canonical TimingRecorder tree
```

The session-level board measurement and a cross-object `trial_index.jsonl`
are under `experiment/lift_test/inspire/_sessions/<timestamp>/`.

`candidate_attempts.json` contains the detailed failure taxonomy and per-step
Jacobian records for every attempted candidate. `result.json` contains only a
compact summary and links to the canonical artifacts. Viser advances according
to `animation.npz:time_s` (not a waypoint FPS slider), so it visualizes the
saved execution reference rather than the old 21-node geometric lift. This
remains an executor-free test: a live run must still check live start-state
drift and stream the saved artifact through the arm-specific safety adapter.

## XY feasibility grid

`run_grid.py` is a separate constructed-scenario experiment: it holds one
object and tabletop pose fixed, then tests the full candidate funnel over a
dense Charuco-proxy XY grid. It never moves the robot and does **not** create a
Viser animation by default.

It intentionally has no `pipeline-parity` option: a constructed XY grid is
useful for reachability mapping but cannot represent one actual `run_auto`
perception pose and scene decision.

When object/pose flags are omitted, it first prints valid objects and asks for
one, then prints only that object's tabletop-pose stems and asks for one:

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_grid.py \
  --candidate-policy clean-state --cuda-graph on
```

The default is the FR3 model (`--arm franka`). The grid uses the same arm
switch as `run_session.py`; XArm6 uses the same right-Inspire candidate pool
but its own XArm kinematics, collision model, and initial q:

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_grid.py \
  --arm xarm --candidate-policy clean-state --cuda-graph on
```

For a reproducible non-interactive run, provide both values explicitly. The
pose argument is a filename stem, not a zero-based menu index.

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_grid.py \
  --object apple --tabletop-pose 004 \
  --grid-step-m 0.05 \
  --candidate-policy clean-state --cuda-graph on
```

## Table-only coverage training

`run_training.py` builds an experiment-private, ranked grasp library at the
measured Charuco-proxy centre.  It reads the immutable v8 geometry/coverage
assets, but writes outcomes only below `experiment/<exp-name>`, following the
same `candidate_state/<hand>/<version>/<object>` convention as
`run_auto.py --isolate_experiment`.

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_training.py \
  --exp-name lift_training_franka \
  --arm franka --hand inspire --grasp-version v8 \
  --max-consecutive-failures 20 --cuda-graph off
```

The object XY is not prompted: it is fixed to the measured board centre.  A
coverage candidate succeeds when any of its cylinder-symmetry variants passes
the approach, 5 mm Jacobian lift, and dense execution-reference validation.
All variants failing increments the consecutive-failure streak once.  A
success resets the streak.  Training stops with one of:

- `coverage_complete`: every v8 scene for the selected tabletop pose is covered;
- `training_stalled`: the configured consecutive-failure limit was reached;
- `coverage_unresolved`: no untried positive-gain candidate remains.

The success label is deliberately `planning_feasible`; this executable does
not instantiate a robot executor and is not a hardware verification.

## Ranked-library inference maps

Use the ranked grasps saved by training over the full Charuco-proxy grid:

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_grid.py \
  --exp-name lift_training_franka \
  --candidate-policy campaign-verified --verified-count all \
  --arm franka --hand inspire --grasp-version v8 \
  --grid-step-m 0.05 --cuda-graph off
```

The grid spacing default is **0.05 m**. Each cell tests verified base grasps in
success-rank order and stops at the first feasible rank. Within one rank, the
direction that succeeded during training is tried first and the object's full
cylinder/discrete symmetry grid is also exposed to collision filtering, IK,
approach planning, Jacobian lift, and dense execution-reference validation.
`--verified-count N` continues to mean N base grasps, not N expanded direction
hypotheses. The saved minimum base rank therefore produces exact `N_001.png`,
`N_002.png`, ... prefix maps without planning the same cell again for every N.

The source campaign arm does not have to match the inference arm.  Verified
artifacts contain object-frame wrist and hand geometry, not source-arm joint
trajectories.  For example, this reuses the ranked geometry discovered by a
Franka campaign but reruns collision filtering, endpoint IK, approach,
Jacobian lift, C2 retiming, and final validation entirely with XArm:

```bash
CUDA_LAUNCH_BLOCKING=1 python src/demo/lift_test/run_grid.py \
  --exp-name lift_training_franka \
  --candidate-policy campaign-verified --verified-count all \
  --arm xarm --hand inspire --grasp-version v8 \
  --grid-step-m 0.05 --cuda-graph off
```

Object, hand, grasp version, tabletop pose, and table scene must still match
the campaign. Outputs explicitly record the source arm, target arm, whether
the run is cross-arm, which fields were reused, and which target-arm products
were recomputed. Thus “verified” means ranked on the source arm; each map cell
must independently pass full planning validation on the target arm.

Campaign artifacts follow the production experiment layout:

```text
experiment/<exp-name>/
  <hand>/_sessions/<stamp>/timing.json
  <hand>/<object>/<stamp>_<attempt>/result.json
  candidate_state/<hand>/v8/<object>/<type>/<sid>/<gid>/result.json
  coverage/<hand>/v8/<object>.json
  analysis/lift_grid/<hand>/<object>/<stamp>/
```

Candidate-trial `result.json` files contain canonical `TimingRecorder` trees.
The coverage progress aggregates planning wall time and time-to-success-rank;
grid runs additionally save per-cell candidate traces in `cell_timing.jsonl`.
Robot trajectory duration is recorded separately and is never added to
planning wall time.

The default `center-only` domain plans every grid point whose **object centre**
is inside the complete Charuco proxy. This is the appropriate default because
the board is an experiment-position proxy rather than an object-support safety
boundary. Use `--domain footprint-inside` when a stricter study should require
the full planning-mesh XY footprint, plus any explicitly requested
`--edge-clearance-m`, to remain inside the checkerboard. The clearance default
is 0 m so the map covers the full proxy boundary.
For a camera-free rerun, use a previously measured proxy:

```bash
python src/demo/lift_test/run_grid.py \
  --board-source file --board-json <episode-or-session>/board_proxy.json \
  --object apple --tabletop-pose 004 --grid-step-m 0.05
```

Results are written to:

```text
~/shared_data/AutoDex/experiment/lift_grid/inspire/<object>/<timestamp>/
  request.json
  board_proxy.json
  grid_spec.json
  candidate_source.json
  cells.csv                 # one compact record per XY cell
  cells.npz                 # array form for quantitative analysis
  feasibility_map.png/.pdf  # status/failure map + planning-time map
  progress.json
  result.json
```

The map's green cells have a complete existing-style approach plus 5 mm
Jacobian lift. Other colours distinguish candidate filtering, endpoint IK,
approach, and Jacobian-lift failures. Gray cells are deliberately unplanned:
the object footprint would exceed the requested board-proxy domain.
