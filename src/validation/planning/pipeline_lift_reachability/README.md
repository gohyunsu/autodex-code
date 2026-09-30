# Pipeline lift reachability

Offline, executor-free reachability analysis that uses the public
`GraspPlanner.plan()` result as its authoritative success decision.

For each base grasp, every registered object-symmetry hypothesis is submitted
as one group.  The planner therefore executes the production collision filter,
endpoint IK/far-wrap policy, approach `plan_single_js`, squeeze-state
substitution, and 10 cm Jacobian lift preflight.  Independent bottom/top IK
checks are saved afterward as diagnostics and never change planner success.

## Candidate pools

- `all`: all v8 records matching the selected tabletop pose; ignores prior
  `result.json` files and coverage exhaustion.
- `training-remaining`: current shared production completion state and current
  remaining-coverage ordering.
- `verified-only`: records whose existing `result.json` has `success: true`,
  ordered by immutable coverage when that metadata exists.

## Evaluation modes

- `per-grasp`: one base grasp plus all its symmetry variants per planner call.
- `pipeline-replay`: the complete ordered pool in one planner call, stopping at
  the first approach+lift success as production does.
- `both`: run `pipeline-replay` once and then every per-grasp call per cell.

## Output

```text
outputs/reachability/<hand>/<object>/pipeline_lift/
  <arm>/<version>/<tabletop_pose>/<run_id>/
```

Every completed run contains the exact candidate snapshot, resumable JSONL
records, boolean coverage matrices, a summary, greedy/verified-prefix curves,
and three-panel maps for each base grasp.  Successful trajectories are omitted
unless `--save-trajectories` is requested.

## Quick XArm + Inspire smoke test

```bash
CUDA_LAUNCH_BLOCKING=1 python \
  src/validation/planning/pipeline_lift_reachability/run.py \
  --obj pepsi --arm xarm --hand inspire --version v8 \
  --tabletop-pose 008 --candidate-pool all --evaluation per-grasp \
  --r-min 0.40 --r-max 0.40 --theta-step 360 \
  --max-grasps 1 --cuda-graph off --run-id pepsi_008_smoke
```

## Full Pepsi 008 polar map

```bash
python src/validation/planning/pipeline_lift_reachability/run.py \
  --obj pepsi --arm xarm --hand inspire --version v8 \
  --tabletop-pose 008 --candidate-pool all --evaluation both \
  --r-min 0.20 --r-max 0.60 --r-step 0.05 --theta-step 30 \
  --cuda-graph on --run-id pepsi_008_xarm_all
```

Recreate reports from an interrupted or completed run without CUDA:

```bash
python src/validation/planning/pipeline_lift_reachability/analyze.py \
  --run-dir outputs/reachability/inspire/pepsi/pipeline_lift/xarm/v8/008/pepsi_008_xarm_all
```

Omit `--obj` to process all candidate objects, and omit `--tabletop-pose` to
process every tabletop pose available to each object.  This can be a very long
run; use `--max-grasps`, a single radius, and `--theta-step 360` for a smoke
test before launching the complete sweep.
