# RSS benchmark protocol: cross-task contact-constraint inference

Date: 2026-11-26  
Status: prospective two-motif pilot protocol; no result is claimed
Method specification: `RSS_RESEARCH_DIRECTION_20261126.md`
Implementation reuse/gap audit: `RSS_IMPLEMENTATION_GAP_20261126.md`

The method-level generalization contract is superseded by
`RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`.  Mating and narrow passage here
are two benchmark motifs, not the full scope of precision manipulation.

## 1. Purpose

This protocol tests whether one frozen, geometry-conditioned observation model
can convert ordinary Franka joint/action histories into a calibrated posterior
over object pose and active contact identity for held-out geometries/layouts,
and whether a task constraint graph makes that posterior useful in both
contact-seeking mating and contact-avoiding narrow-passage manipulation.

The benchmark is designed to separate five questions:

1. Does contact history identify relative pose at all?
2. Does the observation model transfer across geometry?
3. Is an explicit distribution better than a point or implicit state?
4. Does the improved belief change goal success and forbidden disturbance on
   a real robot?
5. Is the same physical observation model reusable across opposing contact
   semantics, or are task-specific models still required?

It does not test grasp selection, tabletop pickup, semantic task planning, or
language understanding in the primary experiment.

## 2. Frozen primary task families

### 2.1 Family A: mating/engagement

- A part cartridge is rigidly attached to the Franka flange.
- A mating fixture cartridge is recessed in a flat plate.
- The goal is vertical mating to a geometry-specific seated depth.
- The part starts above the fixture; a shared low-force probe/correction
  controller may establish contact before final insertion.
- The primary run ends in `seated`, `jammed`, `escaped workspace`, or
  `time/action budget exhausted`.

The physical success detector must be independent of the evaluated estimator.
A surveyed final depth plus passive retention or a geometry-specific seating
switch may be used.  The exact detector is frozen before final trials.

### 2.2 Family B: selective narrow-passage manipulation

- A payload cartridge is rigidly attached to the Franka flange.
- Interchangeable neighbouring objects define a dense local scene.
- **Extraction** moves a designated central payload out of the row without
  moving neighbours beyond frozen tolerances.
- **Placement** moves a payload into a designated gap without moving
  neighbours beyond frozen tolerances.
- Extraction and placement are separate conditions and are never pooled into
  one success rate.
- A run ends in `goal_reached`, `neighbour_disturbed`, `payload_lost`,
  `escaped workspace`, or `time/action budget exhausted`.

This family instantiates the Pringles-like example but does not depend on a
can category.  Payload/neighbour geometry and row/gap layout are varied.  The
physical outcome is measured independently from the evaluated estimator using
surveyed start/goal regions and neighbour pose metrology.

### 2.3 Task constraint graph

Every episode supplies a machine-readable graph containing:

- manipulated object and goal region;
- required, allowed, and forbidden contact relations;
- translation/rotation disturbance tolerances for each neighbour;
- terminal success/failure predicates;
- available primitive types.

The controlled benchmark provides this graph directly.  A language/VLA
compiler is a separate system-level experiment with graph accuracy reported
before task success.

### 2.4 Sensor contract

Primary method input:

- test-time object and local-environment CAD or calibrated scans;
- the frozen task constraint graph;
- an initial visual pose distribution or particle set;
- Franka `q`, `dq`, commanded motion and command-tracking history;
- action/probe history.

Excluded from the primary method:

- wrist force/torque measurements;
- fingertip tactile measurements;
- fixture-stage ground truth;
- simulator pose/contact labels;
- object-specific learned parameters.

If estimated joint torque is added, it is a separately named sensor ablation,
not silently included in the primary contract.  The primary likelihood is
task independent: the constraint graph affects action value, not the physics
of the measured joint response.

## 3. Apparatus and ground truth

### 3.1 Shared mechanical datum

- All payload cartridges share one flange-side kinematic datum.
- All fixture cartridges share one plate-side dowel/screw datum.
- All narrow-passage scene elements use surveyed base datums.
- Cartridge transforms are surveyed once and versioned.
- No per-trial hand measurement is used as policy input.

### 3.2 Controlled fixture perturbation

Mount the fixture plate on a calibrated XY--yaw stage.  Add roll/pitch through
surveyed shims or a 6-DoF stage for designated experiments.  The stage setting
is evaluation-only ground truth and is hidden from all policies.

This creates repeatable pose residuals without conflating them with camera or
grasp errors.

### 3.3 Two initial-prior conditions

1. **Controlled prior:** sample a declared pose distribution around the stage
   ground truth.  This is the primary estimator experiment.
2. **Perception prior:** replace the controlled distribution with the output
   of the frozen multi-view pose/tracking front end.  All downstream code is
   unchanged.  This is the complete-system experiment.

Results from the two conditions are reported separately.

### 3.4 Neighbour-disturbance ground truth

For narrow-passage tasks, record every neighbour's pre/post pose using
evaluation-only metrology and synchronized video.  Report both maximum and
sum displacement/rotation, number of neighbours crossing tolerance, and task
completion.  Binary success alone cannot distinguish a gentle extraction from
one that rearranges the row.

## 4. Geometry universe and leakage control

### 4.1 Geometry tiers

Use both mating and narrow-passage tiers so the result is not only
cross-section matching or only one clutter layout.

- **Tier P (prismatic):** circular/elliptical, regular polygonal, asymmetric
  keyed, convex spline, star/gear-like and freeform extruded profiles.
- **Tier S (spatial):** tapered, stepped, multi-lobe/multi-peg, or another
  genuinely three-dimensional mating family whose contact sequence cannot be
  reduced to one planar cross-section.
- **Tier N (narrow passage):** cylindrical, prismatic, tapered and asymmetric
  payload/neighbour families under row, staggered and nonuniform layouts.

Each generator records task family, geometry family, parameters, scale,
clearance, layout, manufacturing process and mesh digests.  Mating additionally
records seated depth; narrow passage records goal region and neighbour
disturbance tolerances.

Mating pairs use `config/geometry_pair_manifest.schema.json`; both task
families use the superseding scene contract
`config/constraint_scene_manifest.schema.json`.

### 4.2 Splits

- `train_instance`: training samples from training families.
- `validation_instance`: unseen parameters from training families; used for
  architecture, calibration and stopping decisions.
- `test_instance`: frozen unseen parameters from training families.
- `test_family`: entire geometry generator families excluded from all model
  and controller training.
- `test_layout`: narrow-passage layout families excluded from all model and
  controller training.
- `test_physical`: manufactured parts selected from the frozen test manifests
  before real evaluation.

The primary generalization claim uses `test_family` and `test_layout`.  New
random seeds from a training generator are not sufficient evidence.

### 4.3 Test-time geometry contract

Primary contract: `held_out_family_known_geometry`.

- CAD/scan is available to deterministic geometry preprocessing.
- No contact rollout, reference contact manifold, learned embedding fitting,
  policy fine-tuning, or temperature calibration is allowed on a test part.
- Unknown-geometry-from-images is reported only as a separate stretch study.

## 5. Observation model benchmark

### 5.1 Episodes

For scene geometry `G`, initial object/contact state `x`, and standardized
action/probe sequence `U`, record

```text
H = history(q, dq, command, tracking_error, U).
```

The simulator additionally records privileged object/environment poses,
contact pairs, contact mode, material parameters, neighbour motion and task
outcome, none of which are actor inputs.

### 5.2 Candidate set

For each observation, construct a candidate particle set containing:

- the nearest particle to ground-truth pose;
- samples from the same visual prior;
- local translation and rotation aliases;
- contact-compatible hard cases that share early contact but diverge under
  later probes;
- active-contact aliases that explain the same tracking error with different
  neighbouring objects/surfaces;
- symmetry-equivalent poses when the geometry has a nontrivial symmetry group.

The symmetry contract is attached to each geometry.  Equivalent poses count
as one state only for that geometry; asymmetric parts retain full SE(3).

### 5.3 Ratio estimation

Train a classifier on joint samples versus independent pose--history pairs to
estimate

```text
r_theta(H, x, U, G) ~= p(H | x, U, G) / p(H | U, G).
```

Hard-negative proposals require recorded proposal probabilities and importance
correction.  Calibration uses validation geometries only.  The task graph is
not an input to this physics likelihood; it is applied by the controller to
the posterior.

### 5.4 Estimator metrics

- true-pose particle rank and top-k recall;
- translation and rotation error of posterior mean/MAP;
- negative log likelihood;
- empirical coverage of 50%, 80%, 90% and 95% credible sets;
- expected calibration error;
- posterior entropy reduction per probe;
- contact-mode accuracy and Brier score;
- active object/surface contact-identity accuracy;
- neighbour-motion posterior error;
- inference latency and memory;
- held-out-instance, held-out-family, held-out-layout and cross-task gaps.

## 6. Control benchmark

### 6.1 Frozen probe library

Begin with standardized libraries defined in the current task frame:

- guarded axial approach;
- small positive/negative translations along two plate axes;
- small positive/negative rotations about task-relevant axes.
- narrow-passage lift/translate/backoff motions along local clearance axes.

Amplitudes, speed, compliance and maximum count are frozen from development
conditions before test evaluation.  A learned active probe policy is added
only after the fixed library exposes repeatable posterior multimodality.

### 6.2 Shared controller

The preferred first controller uses the same constraint-aware objective for
every estimator while exposing family-appropriate primitives.  Compare:

1. no contact update;
2. posterior mean/MAP correction;
3. full posterior-aware constrained receding-horizon action selection.

A learned controller is allowed only as a second experiment with estimator
weights frozen and matched controller capacity across baselines.

### 6.3 Control metrics

- family-specific goal success;
- success conditioned on first contact;
- correction/probe count;
- time and Cartesian path length;
- jam and recovery rate;
- peak available interaction proxy under the declared sensor contract;
- posterior entropy at action time;
- number and magnitude of neighbour disturbances;
- success stratified by task, geometry family, layout, clearance and initial
  residual.

## 7. Required baselines

### Estimation

1. visual prior without contact update;
2. geometry-agnostic joint-history encoder;
3. geometry-conditioned point regressor;
4. implicit recurrent fusion state;
5. geometry-specific contact-manifold matching;
6. simulation-per-particle likelihood where computationally feasible;
7. proposed amortized likelihood-ratio posterior.
8. task-specific estimator heads with matched total capacity.

### Control

1. nominal open loop;
2. frozen visual alignment only;
3. Blind-style proprioception-only policy from the same contact-start prior;
4. fixed visual-to-proprioception handoff;
5. implicit visual-proprioceptive recurrent/world-model policy;
6. posterior mean controller;
7. full posterior-aware controller;
8. geometry-specific specialist upper bound.
9. task-specific specialist policies with matched training data.

A direct VLA is a system-level baseline under a separately reported
demonstration and compute budget.  It is not an estimator baseline.

## 8. Ablations

- remove geometry conditioning;
- replace paired geometry with object-only or fixture-only geometry;
- current state versus temporal history;
- remove action/command history;
- point estimate versus calibrated posterior;
- prismatic-only training versus mixed prismatic/spatial training;
- validation-instance versus test-family;
- seen versus held-out narrow-passage layouts;
- task constraint graph versus task-ID token;
- shared likelihood model versus task-specific estimator heads;
- train on both families versus cross-family transfer diagnostics;
- controlled prior versus actual perception prior;
- fixed probes versus information-driven probes, only if implemented;
- symmetric subset with and without correct quotient treatment.

## 9. Real-robot evaluation

### 9.1 Freeze order

1. freeze geometry generators and family split;
2. freeze simulator randomization and sensor contract;
3. freeze estimator architecture and validation calibration;
4. freeze probe/controller settings;
5. manufacture the already selected physical test manifest;
6. run an engineering pilot that does not enter the final count;
7. use the pilot effect size to preregister repetitions;
8. run the final evaluation without test-object learning.

### 9.2 Minimum evidence package

- held-out physical mating pairs plus narrow-passage payload/layouts spanning
  multiple families;
- at least three clearance levels across the set;
- repeated trials and confidence intervals;
- synchronized robot history, actions, posterior snapshots and outcome video;
- estimator and task metrics, not success alone;
- all failures assigned one primary phase/category;
- separate results for controlled and perception priors.

The final trial count is chosen by a preregistered power analysis using pilot
effect size, not by stopping when the desired success rate appears.

## 10. Go/no-go gates

### Gate A: identifiability

On `test_family`, the true pose ranks above contact-compatible aliases and the
posterior is more calibrated than a geometry-agnostic history encoder.

Failure action: revise the observation/probe design; do not train a larger
controller to hide the failed estimator.

### Gate B: distributional value

At matched observations and controller logic, the full posterior improves
success, probe efficiency or recovery over its point summary.

Failure action: remove the belief-distribution contribution and report a
geometry-conditioned point estimator if that result remains meaningful.

### Gate C: cross-geometry amortization

The frozen model transfers to `test_family` and avoids per-test reference
manifold generation or policy fitting, with measurable preprocessing or
online-compute advantage over geometry-specific alternatives.

Failure action: do not claim zero-shot amortization.

### Gate D: cross-semantics reuse

The same frozen physical observation model improves calibrated state/contact
inference in mating and narrow passage.  A structured task constraint graph
improves held-out relation/layout decisions over a task-ID token.

Failure action: remove the cross-task claim.  Report two family-specific
methods only if each is independently publishable.

### Gate E: real decision value

The estimator improvement survives sim-to-real and changes mating success or
narrow-passage disturbance on manufactured held-out scenes.

Failure action: target an estimator/simulation paper only if the remaining
evidence is independently strong; otherwise continue method development.

## 11. Publication boundary

The primary paper claim is supported only if all five gates pass.  The
following are valuable engineering results but insufficient on their own:

- a successful cylinder demo;
- a successful Pringles extraction demo;
- improved visual tracking;
- a learned sensor gate;
- simulation-only unseen-shape success;
- a posterior visualization without calibration;
- success improvements without estimator baselines;
- a gripper-based end-to-end demo with uncontrolled in-hand pose.
