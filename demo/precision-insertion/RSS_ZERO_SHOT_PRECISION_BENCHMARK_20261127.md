# RSS benchmark: task-agnostic zero-shot precision execution

Date: 2026-11-27

Status: prospective protocol; no empirical result is claimed

Method hypothesis: `RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`

## 1. Claim under test

This benchmark does not organize manipulation into insertion, extraction or
placement policies.  It tests one narrower and falsifiable claim:

> Given a frozen nominal manipulation policy, known test-time geometry and a
> program composed from a fixed vocabulary of geometric/contact constraints,
> can one frozen execution layer improve physical precision on held-out
> geometry, layout, constraint composition and upstream policy?

The layer does not invent the task, replace long-horizon planning, or promise
universal manipulation.  It improves local execution after a nominal policy
has produced a meaningful action or short trajectory.

## 2. Frozen deployment contract

Before final testing, freeze:

1. the constraint vocabulary and evaluator implementations;
2. the estimator, calibration parameters and residual controller;
3. the base-policy adapter interface;
4. train/validation/test generators and seeds;
5. real fixtures, tolerance levels, outcome metrology and trial budget.

CAD or a calibrated reconstruction and the constraint program are available
at deployment.  Online state estimation and deterministic optimization are
allowed.  Test-specific demonstrations, human corrections, gradient updates,
residual RL, fitted contact manifolds and test-fixture parameter tuning are
forbidden in the primary zero-shot result.

## 3. Unit of generalization

The unit is a **constraint program**, not a task label.  Every episode is a
graph of entities and stages with predicates drawn from:

- pose equality;
- signed clearance;
- required contact;
- forbidden contact;
- disturbance budget;
- path/manifold membership.

`config/precision_constraint_program.schema.json` is the machine-readable
contract.  A human-readable archetype may be logged for analysis but cannot
select a model head, controller, reward or code path.

Every predicate must expose a signed residual `rho`, a declared physical
tolerance `tau`, its entity scope and reference frame.  Evaluation operates on
the normalized margin `rho/tau`, posterior violation probability and measured
physical violation.  This common numeric contract permits comparison across
visually and semantically different manipulation manifestations.

## 4. Independent zero-shot axes

| Axis | Development | Primary test |
| --- | --- | --- |
| Geometry | instances from training generators | held-out geometry generators |
| Layout | instances from training layout generators | held-out layout generators |
| Composition | seen predicate graphs | held-out combinations and graph topology |
| Upstream policy | policies used while training the layer | held-out policy and policy class |

Report Z1--Z4 independently and jointly.  The strongest test combines a held-
out geometry family, layout generator, predicate composition and base-policy
class.  A new random seed from a training generator is not zero-shot
generalization.

An unseen predicate type is excluded from the primary claim.  That is open-
world relation learning and requires a separate method.

## 5. Procedural program generator

Generate scenes by sampling entities, geometry, layout and constraint graphs,
not by writing one environment per task.  Each sample contains:

- one or more manipulated rigid bodies;
- fixed or movable environment entities;
- a nominal goal manifold;
- staged activation of equality, inequality and contact predicates;
- an independently generated nominal trajectory with controlled local error;
- friction, compliance, calibration and observation nuisance variables;
- privileged simulator labels for predicate residual, activation, contact
  identity, object motion and final outcome.

The simulator must resolve exact or validated approximate clearances.  The
actor never receives privileged contact or ground-truth pose labels.

## 6. Physical manifestations, not task-specific methods

The real benchmark should contain at least six manifestations spanning at
least four constraint motifs:

1. tight terminal alignment or engagement;
2. narrow free-space passage;
3. required-contact tangent motion;
4. disturbance-sensitive motion near movable neighbours;
5. path/manifold-constrained motion.

Possible manifestations include mating parts, moving a payload through a
close aperture, sliding along a guide, following a surface, extracting one
object from a dense arrangement, or placing into a narrow free region.  They
provide physical coverage rather than named task families.  The same
estimator, controller and predicate evaluators must run in all cases.

Use a rigidly mounted payload in the primary causal study so grasp error does
not dominate the precision question.  A gripper condition is a secondary
robustness test with object-in-hand uncertainty.

## 7. Upstream policy interface

Every adapter exposes:

```text
observation timestamp
nominal Cartesian action or short trajectory
action frame, rate and horizon
optional policy confidence
```

The precision layer returns a residual in the same declared frame.  The
adapter may transform coordinates but cannot provide simulator privilege or a
task-specific error signal.  Evaluate at least:

1. a model-based planner;
2. an imitation or diffusion policy;
3. a language-conditioned/VLA policy.

For policy-agnostic evidence, train without at least one policy class and test
the frozen layer on it.

## 8. Estimator experiment before control

Construct candidate latent states containing:

- the true active constraint and close geometric aliases;
- different contact identities that explain similar early joint response;
- states near inequality activation boundaries;
- symmetry-equivalent states according to the entity contract;
- hard negatives with the same nominal action but different constraint
  outcomes.

Compare point state, geometry-conditioned point state, raw-history recurrent
state, explicit task-specific belief, and the proposed variable-size active-
constraint belief.

Required metrics are signed-margin error, active-predicate/contact-identity
accuracy, posterior NLL, Brier score, credible-set coverage, violation
probability calibration and latency.  Do not proceed to a learned residual
controller if the shared estimator cannot identify active constraints on held-
out graph compositions.

## 9. Controller experiment

Use the same action budget, observations and nominal trajectories for:

1. frozen base policy only;
2. visual point-pose servoing;
3. classical constrained MPC from a point state;
4. matched-capacity recurrent residual policy;
5. posterior-mean constrained control;
6. proposed posterior-aware residual execution;
7. task-specific residual adaptation as a non-zero-shot upper bound.

First use analytical posterior-aware constrained MPC with a frozen estimator.
Only add a learned controller after belief decision value is established.  If
point-state MPC matches the proposed method, the explicit belief is not
justified under the tested conditions.

## 10. Cross-manifestation metrics

- program success versus physical tolerance;
- probability, duration and magnitude of predicate violations;
- environment disturbance magnitude;
- intervention count, rate and residual norm;
- execution time and path overhead;
- improvement or degradation per frozen base policy;
- zero-shot gap for each held-out axis and their joint split;
- calibration and active-constraint diagnosis metrics from Section 8.

A single success threshold can hide whether precision improved.  Tolerance-
sweep curves and violation magnitude are therefore mandatory.

The headline plot should sweep one or more physical tolerances and show the
base policy, point-state correction and proposed belief-aware layer on the
same axes.  A method that only changes success at one conveniently chosen
threshold has not established a general precision improvement.

## 11. Leakage audit

For every final result, automatically verify:

- no shared mesh or geometry-generator family across forbidden splits;
- no shared layout generator in the Z2 split;
- no isomorphic predicate graph under renamed entity IDs in the Z4 split;
- no base-policy checkpoint or policy-class leakage in the Z3 split;
- no test-scene rollout in training, calibration or hyperparameter tuning;
- no task/archetype string passed to the model or controller;
- frozen code, config, model, geometry and program digests.

`config/precision_layer_episode.schema.json` records split and evidence
without insertion-specific coordinates or phases.  The older
`config/research_episode.schema.json` remains the undergraduate/cylinder
record and is not the RSS benchmark contract.

The authoritative list of train/validation/test cases follows
`config/precision_benchmark_index.schema.json`.  Audit it before training and
again before producing final tables:

```bash
python -m precision_layer.audit path/to/precision_benchmark_index.json
```

The command returns a nonzero status for reused held-out geometry/layout or
policy families, and for a test constraint graph that is exactly isomorphic
to a training graph after arbitrary entity, stage and predicate ID renaming.

## 12. Real-trial protocol

1. Freeze the physical manifestations and tolerance levels before final
   trials.
2. Use evaluation-only metrology for program success, predicate violations
   and environment disturbance.
3. Randomize permitted initial state and nuisance variables with a recorded
   seed.
4. Pair methods on the same condition order where hardware reset permits.
5. Determine trial count through a preregistered pilot and power analysis;
   never stop when the desired success rate appears.
6. Publish every attempted episode, exclusion reason and artifact digest.

## 13. Go/no-go gates

- **Identifiability:** active constraint/contact identity is recoverable on
  held-out composition above matched implicit and point-state baselines.
- **Calibration:** credible-set coverage and violation probabilities remain
  calibrated under the held-out axes.
- **Decision value:** the distributional controller beats point-state MPC at
  matched action and time budgets.
- **Composition:** gains persist on held-out predicate combinations/topology.
- **Policy transfer:** the same frozen layer improves a held-out base-policy
  class.
- **Physical value:** gains persist on real tolerance sweeps without test-
  specific learning.

Failure at a gate narrows the claim.  It must not be hidden by adding a task-
specific model head or tuning the test fixture.

## 14. Evidence record

Every episode stores:

- the digested constraint program and geometry;
- base-policy and precision-layer identities;
- all four split labels;
- generic stage/event timeline;
- predicate-belief snapshots and contact-identity probabilities;
- nominal and residual actions;
- independent outcome and violation metrology;
- synchronized sensor, belief, action and video evidence.

This makes a result auditable without assuming that all manipulation problems
share insertion-like coordinates or phases.
