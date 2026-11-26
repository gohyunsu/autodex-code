# Method specification: active-constraint belief for precision execution

Date: 2026-11-27  
Status: prospective method specification; no empirical result is claimed

This document makes the scientific core of
`RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md` implementable. It defines the
belief, observation model, controller and VLA boundary without introducing
task IDs or named-task control branches.

## 1. Scope

The method refines a meaningful nominal Cartesian action or short trajectory
for rigid-body manipulation. It does not discover the long-horizon task,
replace grasp planning, or claim open-world physics. At test time it receives:

- a frozen base-policy action `u_base`;
- known CAD or calibrated reconstruction of relevant geometry;
- a valid constraint program from the frozen vocabulary;
- synchronized multi-view RGB and ordinary robot proprioception;
- no test-task demonstrations, gradient updates or residual RL.

The primary claim is zero-shot transfer across held-out geometry, layout,
constraint composition and upstream policy, not across an undefined new
constraint primitive.

## 2. Constraint token

For each program predicate `k`, construct one token:

```text
c_k = [
  predicate-type embedding,
  ordered entity-role embeddings,
  local geometry embeddings,
  reference-frame encoding,
  declared tolerance tau_k,
  nominal signed margin rho_k(x_hat),
  local Jacobian J_k,
  activation dependencies,
  hard/soft and terminal roles
].
```

Entity and predicate IDs are excluded. The encoder is permutation equivariant
over entity and predicate sets while preserving ordered predicate arguments and
the directed stage/activation graph. The same vocabulary evaluator computes
`rho_k` in all physical manifestations.

Examples of the signed convention are:

- clearance: separation minus required clearance;
- pose equality: tolerance minus weighted SE(3) residual;
- disturbance: allowed displacement minus measured displacement;
- path manifold: tolerance minus distance from the manifold;
- required contact: contact tolerance minus separation when active;
- forbidden contact: signed separation from the forbidden pair.

Positive means satisfied, zero is the boundary and negative is violated.

## 3. Latent state and belief

The latent state contains only variables that can change a constraint decision:

```text
y_t = {
  task-relevant relative transforms,
  signed margins rho_1:K,
  active flags a_1:K,
  contact entity and mode m_1:K,
  relevant environment displacement,
  nuisance contact parameters eta
}.
```

The filter maintains

```text
b_t = p(y_t | z_vision,0:t, z_prop,0:t, u_0:t-1, G, C).
```

Continuous transforms and margins use particles or a mixture distribution;
activation and contact modes are categorical. Symmetry is represented by
equivalence classes in the geometry contract rather than duplicated as
conflicting pose modes.

For continuous axial symmetry, pose equality compares the task-relevant axis
and translation while quotienting axial yaw. If the two object ends are also
physically interchangeable, axis sign is quotiented explicitly; otherwise the
axis remains directional. This declaration belongs to geometry/predicate
parameters and never follows from a task name.

The exported belief token for constraint `k` is

```text
b_k = [
  E[rho_k], Var[rho_k], quantiles(rho_k),
  P(rho_k < 0), P(a_k = 1), P(m_k),
  contact-entity probabilities,
  local sensitivity E[J_k]
].
```

An unconstrained recurrent feature is allowed inside the observation encoder,
but it is not itself the claimed belief and cannot replace these calibrated
outputs.

## 4. Observation interfaces

### 4.1 Multi-view vision

The current multi-view 6D stack supplies timestamped object/environment pose
hypotheses and measurement quality in a common calibrated frame. The RSS
method consumes a distribution or multiple hypotheses, not only one pose.
Object symmetry is reduced before constraint evaluation.

Vision is expected to dominate free-space relative pose and environment
displacement. Occlusion, motion blur and tracker disagreement increase the
posterior uncertainty; they do not select a task-specific fallback.

### 4.2 Proprioception and contact evidence

Use a fixed-rate history of signals genuinely available on the Franka:

```text
q, dq, commanded Cartesian/joint action,
tracking error, measured/estimated joint torque when available.
```

The history encoder is conditioned on candidate geometry, the commanded local
motion and candidate contact state. It estimates a likelihood ratio

```text
r_phi(H_t, y, U, G) ~= p(H_t | y, U, G) / p(H_t | U, G).
```

This asks whether the physical response supports a candidate constraint/contact
state. It does not directly predict a task action. The ratio updates the same
belief initialized by vision.

### 4.3 VLA/task reasoning

The VLA consumes language and scene/entity descriptions and emits a constraint
program in the frozen JSON schema. A deterministic validator rejects malformed
references or unsupported primitives. The VLA never emits millimetre offsets
for the primary method and cannot choose a hidden task-specific controller.

Evaluate separately:

1. program/entity grounding accuracy;
2. predicate type, scope and stage-graph accuracy;
3. execution given an oracle program;
4. end-to-end execution given the compiled program.

This separation prevents semantic parsing errors from being reported as
low-level precision failures.

## 5. Bayesian update

At every synchronized control step:

1. propagate continuous hypotheses through measured robot motion and nuisance
   dynamics;
2. evaluate every program margin and local Jacobian for each hypothesis;
3. apply the multi-view pose/environment likelihood;
4. apply the proprioceptive likelihood ratio for candidate contact modes;
5. normalize, resample or merge equivalent hypotheses;
6. calibrate exported violation probabilities using a frozen validation-only
   mapping.

The estimator processes variable-size entity and predicate graphs. Task name,
human description and benchmark archetype are unavailable model inputs.

## 6. Learning objective

Training data come from procedurally generated geometry/constraint programs and
contain privileged state only for supervision. The deployment actor never
receives privileged labels.

Use a composite objective:

```text
L = lambda_pose    * NLL(relative transform)
  + lambda_margin  * NLL(signed constraint margins)
  + lambda_mode    * CE(active/contact identity and mode)
  + lambda_ratio   * binary ratio-estimation loss
  + lambda_cal     * calibration loss
  + lambda_cons    * graph/geometry consistency loss.
```

Hard negatives must share the same nominal action and similar current image or
joint response while differing in the decision-relevant contact identity or
margin sign. Otherwise a large encoder can solve the dataset from task or
layout shortcuts.

Calibration is selected on validation families only. Test geometry,
constraint graphs and policy rollouts are never used for temperature scaling,
threshold selection or early stopping.

## 7. First controller: analytical and posterior-aware

Do not initially train a large residual policy. Linearize each active margin
under a Cartesian correction `delta_u`:

```text
rho_k(x_next) ~= rho_k(x) + J_k B (u_base + delta_u).
```

Solve a short-horizon constrained optimization:

```text
min_delta_u
    E_b[L_goal(x_next)]
  + lambda_r ||delta_u||^2
  + lambda_i intervention_cost

subject to, for every hard constraint k,
    P_b(rho_k(x_next) >= 0) >= 1 - alpha_k,
    delta_u in actuator/action limits.
```

For an approximately Gaussian margin, the chance constraint becomes

```text
mu_rho,k - z_(1-alpha_k) sigma_rho,k >= 0.
```

The controller remains close to the base policy when margins are comfortably
positive and intervenes when the posterior predicts violation. Information-
gathering actions are a later ablation and must use expected *decision* value,
not entropy reduction alone.

## 8. Required estimator experiment before control

Compare under exactly the same generated histories:

1. vision point state;
2. geometry-conditioned point state;
3. raw-history recurrent state;
4. task-specific belief upper bound;
5. proposed shared active-constraint belief;
6. privileged state oracle.

Report:

- signed-margin MAE and NLL;
- active predicate/contact identity accuracy;
- Brier score and expected calibration error;
- credible-set coverage;
- false-safe rate `P(predicted safe | physically violated)`;
- inference latency;
- gaps for Z1 geometry, Z2 layout and Z4 composition.

Do not proceed to a learned residual controller if the proposed belief is not
better calibrated or more decision-useful than the matched raw-history and
point-state baselines.

## 9. Required controller experiment

Use identical nominal trajectories, observations, action limits and time
budgets for:

1. base policy only;
2. visual point-pose servo;
3. point-state constrained MPC;
4. recurrent residual policy with matched capacity;
5. posterior-mean constrained MPC;
6. proposed posterior-aware constrained MPC;
7. task-specific residual adaptation as a non-zero-shot upper bound.

The headline result is a tolerance curve and violation magnitude, not success
at one threshold. Repeat it for seen and held-out policy classes.

## 10. Undergraduate-to-RSS data bridge

The cylinder study uses a fixed visual-to-contact handoff rather than this full
filter. Its synchronized records provide:

- real visual residual and tracker disagreement before contact;
- real joint/action histories around contact and jam;
- phase-resolved failure labels;
- latency and update-rate requirements;
- examples where visual and proprioceptive evidence disagree.

These records may validate observation corruption and select history length.
They may not enter RSS test-task training or substitute for held-out geometry
evidence.

## 11. Falsification boundaries

- If point-state MPC matches posterior-aware MPC, drop the belief-control claim.
- If raw history matches calibrated contact/margin inference, explicit belief
  has no demonstrated value.
- If a task identifier or task-specific estimator is required, drop the
  shared-representation claim.
- If the method needs real adaptation on every manifestation, drop zero-shot.
- If the VLA compiler is the only novel component, report semantic planning
  rather than precision execution.
- If CAD/program authoring dominates deployment, state the result as a
  known-geometry research system rather than open-world manipulation.

## 12. Implementation order

1. physics-backed generic constraint evaluator and trajectory generator;
2. point-state and privileged-oracle baselines;
3. visual/proprioceptive likelihood modules and calibrated belief export;
4. estimator-only held-out evaluation;
5. analytical posterior-aware controller;
6. held-out base-policy adapter;
7. modular real benchmark;
8. VLA compiler and end-to-end semantic evaluation.

The current `precision_layer.evaluator` implements the backend-independent
signed-margin vocabulary. It consumes metric evidence supplied by a simulator,
CAD engine or real metrology backend; it is not itself a physics engine. This
keeps the predicate semantics frozen while the physics backend is replaced.

`precision_layer.mujoco_adapter` provides the first declarative backend: named
MuJoCo frames and geom pairs are mapped into the same relative-transform,
signed-separation, displacement and manifold-distance evidence without task
dispatch. The existing cylinder reproduction can therefore be used as a pilot
data source, while RSS geometry diversity still requires a new procedural
world generator.
