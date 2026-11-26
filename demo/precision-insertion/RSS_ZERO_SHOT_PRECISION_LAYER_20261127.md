# RSS direction: a zero-shot precision execution layer

Date: 2026-11-27  
Status: authoritative long-term research scope  
Relation to the cylinder thesis: separate scientific claim and apparatus

This document corrects an overly narrow framing around insertion or a fixed
pair of task families.  Insertion, dense extraction, gap placement and other
examples are evaluation instances, not the method definition.

## 1. Recommended research question

> Can one policy-agnostic execution layer improve the precision of unseen
> rigid-body manipulation problems by maintaining a calibrated belief over
> active geometric/contact constraints and applying zero-shot residual
> corrections to an arbitrary nominal policy?

Provisional descriptive title:

> **Belief over Active Constraints for Zero-Shot Precision Execution**

Do not claim universal manipulation.  The intended domain is rigid or
approximately rigid manipulation whose task-relevant goals, clearances,
contacts and disturbance budgets can be represented geometrically.  It does
not initially include deformable objects, granular media, liquid dynamics, or
tasks whose success is primarily semantic rather than geometric.

## 2. Problem contract

At deployment the layer receives:

- a frozen upstream controller or policy `pi_base`;
- a nominal action or short trajectory from that policy;
- object/environment geometry `G`, from CAD or calibrated reconstruction;
- a compositional constraint program `C`;
- multi-view visual evidence and ordinary robot proprioception.

It outputs a local correction without retraining the base policy:

```text
u_t = u_base,t + delta_u_t.
```

The primary zero-shot contract forbids test-scene demonstrations, human
corrections, task-specific residual RL, object-specific contact manifolds and
test-time parameter fitting.  Deterministic geometry processing and online
state estimation are allowed.

The method is policy agnostic only if the same interface works with at least
three meaningfully different upstream sources, for example:

1. a model-based motion planner;
2. an imitation/diffusion policy;
3. a VLA or language-conditioned policy.

## 3. Constraint program rather than task ID

`C` is a variable-size graph or set of predicates.  It contains entities,
task frames, terminal goals, and constraints such as:

```text
goal equality:          g_i(x, G) = 0
clearance inequality:   c_j(x, G) >= margin_j
required contact:       contact(a, b, region) = true
forbidden contact:      contact(a, b) = false
disturbance budget:     ||Log(T_b,0^-1 T_b,t)|| <= epsilon_b
path/tangent rule:      v belongs_to tangent_space(M_k)
```

Examples compose the same vocabulary differently:

- insertion combines pose equality, clearance and required target contact;
- extracting one tightly packed container combines a goal region, narrow
  clearance and neighbour-disturbance constraints;
- placing into an empty gap changes the goal and path constraints but reuses
  the same clearance/disturbance predicates;
- surface following combines required contact and tangent-motion constraints;
- constrained sliding or an articulated mechanism adds a path-manifold
  constraint.

The graph is supplied directly in the controlled experiment.  A VLM/VLA that
compiles language and scene observations into `C` is a later module and is
evaluated separately.  Language parsing must not obscure low-level precision.

## 4. Scientific core: belief over active constraints

Let each predicate expose a signed residual or margin `rho_k(x,G,C)`.  The
latent execution state is represented as a variable-size set:

```text
y_t = {
  object/environment pose residuals,
  rho_k and its local gradient,
  active/inactive state a_k,
  contact identity and mode m_k,
  task-relevant environment motion delta_E,
  nuisance physics eta
}.
```

The filter maintains

```text
b_t = p(y_t | z_vision,0:t, z_prop,0:t, u_0:t-1, G).
```

`z_prop` should include the Franka signals genuinely available at runtime:
`q`, `dq`, commanded action, tracking error, and measured/estimated joint
torque if available.  The no-torque version is an ablation, not an artificial
restriction on the primary method.  No dedicated wrist F/T or tactile sensor
is required for the primary contract.

The learned physical observation model is task independent.  For candidate
state `y`, commanded local motion `U`, and observed history `H`, estimate

```text
r_theta(H, y, U, G)
  ~= p(H | y, U, G) / p(H | U, G).
```

The task program does not change the physics likelihood of a collision.  It
changes how the posterior evidence is valued.  A contact may satisfy one
constraint, violate another, or simply provide information.

This distinction is the intended novelty boundary:

- **constraint specification** says what must hold;
- **active-constraint belief** estimates what is physically happening;
- **residual execution** decides how to correct the nominal action under that
  uncertainty.

## 5. Zero-shot residual control

The residual controller consumes the base action, the posterior and `C`:

```text
delta_u_t = pi_precision(u_base,t, b_t, G, C).
```

A clean first implementation uses posterior-aware constrained MPC or a small
learned proposal followed by analytical projection.  One possible objective
is

```text
min_delta_u  E_b[
    goal_residual(x, u_base + delta_u, C)
  + lambda_v * violation_risk(x, u_base + delta_u, C)
  + lambda_d * disturbance(x, u_base + delta_u, C)
  + lambda_r * ||delta_u||^2
] - beta * expected_information_gain.
```

The layer stays close to the base policy when the belief indicates adequate
margin and intervenes only where predicted constraint satisfaction is poor.
This is not a learned safety shield claim; it is an uncertainty-aware precision
adapter whose output and residual magnitude are evaluated directly.

## 6. What would actually be novel

Each component alone is already occupied:

- geometric constraints are already used as a general manipulation interface;
- environmental-constraint primitives already support one-shot task
  generalization;
- contact-point conditioning already improves zero-shot policies;
- belief-conditioned policies and belief-space manipulation already exist;
- task-specific and policy-agnostic residual adapters already improve
  precision after real interaction or human correction;
- precision-weighted VLA training already improves several assembly tasks.

The paper is defensible only if it supports all four linked claims:

1. **Generic representation:** a variable-size active-constraint belief is
   shared across tasks instead of using task IDs or hand-written state vectors.
2. **Zero-shot composition:** the frozen model handles held-out geometry,
   layout and unseen combinations of known constraint primitives.
3. **Policy-agnostic decision value:** the same layer improves multiple frozen
   upstream policy classes without per-task real data or fine-tuning.
4. **Calibrated physical evidence:** posterior coverage/contact identity
   predicts when and how to correct better than a point state, raw history, or
   implicit residual policy.

If only action residuals improve, the work overlaps residual RL.  If only the
constraint language is new, it overlaps geometric-constraint interfaces.  If
only a belief is added, it overlaps belief-space control and BEACON.  The
combination must produce measurable zero-shot compositional transfer.

## 7. Generalization hierarchy

Report each level separately:

| Level | Held out at test | Meaning |
| --- | --- | --- |
| Z0 | pose/noise only | engineering robustness, not generalization |
| Z1 | object geometry and scale | geometry transfer |
| Z2 | environment layout and clearance | scene transfer |
| Z3 | upstream policy/backbone | policy-agnostic transfer |
| Z4 | constraint-graph composition | zero-shot task composition |
| Z5 | entire constraint primitive type | open-world task transfer; stretch only |

The primary RSS claim should target Z1--Z4.  Z5 cannot be claimed when the
system has no definition or supervision for the new physical relation.

## 8. Benchmark design without hard-coding task families

Build a procedural **constraint-program generator**, not separate insertion
and extraction datasets.  It samples:

- manipulated and environment geometry;
- entity count and layout;
- goal manifolds;
- equality, inequality, contact and disturbance predicates;
- constraint-graph topology and sequence;
- nominal trajectories with controlled local errors;
- friction, compliance, calibration and observation corruption.

Training and testing split by geometry family, layout generator and graph
composition.  A whole predicate combination is withheld, not merely a random
seed.

The real evaluation should instantiate a finite basis of constraint motifs,
not claim that examples exhaust manipulation:

1. tight goal alignment/engagement;
2. narrow free-space passage;
3. required-contact motion such as sliding or surface following;
4. disturbance-sensitive manipulation around movable neighbours;
5. path/manifold-constrained motion such as a guide or articulation.

Use at least six real tasks spanning at least four motifs.  Include insertion
and the tightly packed-container example if useful, but neither receives
special model code.  ContactBench assets can provide an external reference
suite; custom modular fixtures provide controlled metrology and tolerance
sweeps.

## 9. Research questions

- **RQ1 — representation:** does a constraint-set encoder trained on random
  programs identify active constraints and residual margins on held-out graph
  compositions?
- **RQ2 — calibration:** does multimodal filtering improve posterior coverage,
  contact identity and violation prediction over visual point state, raw
  history and implicit recurrent state?
- **RQ3 — intervention:** does uncertainty-aware residual execution improve
  the base policy's success-versus-tolerance curve while using smaller and
  fewer interventions than task-specific adapters?
- **RQ4 — compositional zero shot:** does one frozen layer transfer across new
  geometry, layout, constraint composition and upstream policy without
  task-specific data?
- **RQ5 — modality value:** when do multi-view tracking, joint torque/history,
  or their combination provide unique information about active constraints?

## 10. Required baselines

1. frozen base policy alone;
2. base policy plus visual point-pose servoing;
3. classical constrained trajectory optimization/MPC from a point state;
4. raw-history recurrent residual policy with matched capacity;
5. task-specific residual RL upper bound;
6. ARC-style human-correction adapter under its declared data budget;
7. OmniTacTune-style policy-agnostic residual adaptation where sensors permit;
8. BEACON-style belief-conditioned policy with task-specific belief;
9. geometric-constraint trajectory generation/interface baseline;
10. CAP/contact-conditioned policy baseline where the contact goal is given;
11. PRISM/direct VLA precision baseline under matched demonstrations;
12. proposed active-constraint belief and zero-shot precision layer.

Not every published baseline can be reproduced under an identical sensor
contract.  Conceptual comparisons must not be presented as empirical wins.

## 11. Metrics

### Belief and diagnosis

- signed constraint-margin error;
- active-constraint/contact-identity top-k accuracy;
- posterior NLL, Brier score and credible-set coverage;
- violation probability calibration;
- information gain by modality and action;
- held-out geometry/layout/graph generalization gaps.

### Execution

- task success as a function of physical tolerance;
- number, duration and magnitude of constraint violations;
- neighbour/environment displacement where relevant;
- intervention rate and residual magnitude;
- degradation or gain for each upstream policy;
- recovery rate, path/time overhead and inference latency;
- zero-shot real transfer without test-task updates.

Success at one tolerance is insufficient.  Plotting success versus tolerance
shows whether the layer truly moves the precision boundary.

## 12. Clean experimental apparatus

- Franka with a surveyed rigid payload mount for the primary causal study;
- modular plate/base with interchangeable geometric elements and datums;
- evaluation-only metrology for object/environment pose and constraint margin;
- multi-view RGB plus Franka proprioception for the method;
- optional gripper experiment after the fixed-mount result.

The rigid mount removes grasp and in-hand slip from the primary question.  It
does not restrict the method to insertion; the mounted payload is moved through
different constraint programs.  A secondary experiment restores grasping to
test whether the layer tolerates object-in-hand uncertainty.

## 13. Staged research plan

### Stage 0: undergraduate thesis

Finish the cylinder study as a narrow, auditable result.  Do not retrofit the
RSS claim into it.

### Stage 1: constraint-program simulator

- implement variable entity/constraint graphs;
- generate exact geometry and privileged active-constraint labels;
- inject nominal trajectories from several upstream controllers;
- verify leakage-free Z1--Z4 splits.

### Stage 2: identifiability before control

Train point, implicit-history and likelihood-ratio estimators.  Stop if a
shared model cannot rank the true active constraint on held-out compositions.

### Stage 3: zero-shot precision layer

Attach the frozen estimator to analytical constrained residual control.  Show
improvement across upstream policies before training a more expressive
controller.

### Stage 4: real modular benchmark

Run preregistered tasks spanning the constraint motifs, tolerance levels,
geometries, layouts and base policies.  Use no test-task learning.

### Stage 5: semantic compiler

Only after metric execution works, evaluate a VLA that compiles language and
scene observations into `C`.  Report graph-generation error separately from
execution error.

## 14. Go/no-go and falsification

- If the shared estimator needs task-specific heads, remove the general
  active-constraint representation claim.
- If classical point-state MPC matches the proposed method, uncertainty
  modeling is unnecessary under the tested conditions.
- If a raw-history residual policy matches calibration and decision value,
  the explicit belief is not justified.
- If task-specific residual RL is required on every new scene, zero-shot
  precision adaptation has failed.
- If gains appear only for one upstream policy or one constraint motif, do not
  claim a policy-agnostic manipulation layer.
- If geometry/constraint authoring dominates deployment, report the system as
  a known-geometry research prototype rather than a general household robot.
- If joint history cannot localize active contact in multi-object scenes,
  promote joint torque or an additional contact sensor and narrow the sensor
  claim instead of hiding the failure in a larger policy.

## 15. Closest-work boundary

- Geometric Constraint as General Interface for Robot Manipulation:
  <https://geoconstraintmanip.github.io/>
- From a Single Demonstration to a General Policy for Contact-Rich
  Manipulation: <https://arxiv.org/abs/2605.17601>
- Contact-Anchored Policies: <https://arxiv.org/abs/2602.09017>
- Constraint-Preserving Data Generation:
  <https://proceedings.mlr.press/v305/lin25b.html>
- BEACON: <https://arxiv.org/abs/2609.22730>
- Adaptive Residual Correction: <https://arxiv.org/abs/2610.37547>
- OmniTacTune: <https://arxiv.org/abs/2607.03723>
- PRISM: <https://seen-e.github.io/PRISM/>
- ContactBench 1.0: <https://arxiv.org/abs/2610.24512>
- InsertionWM: <https://arxiv.org/abs/2609.28258>
- Contact-rich control under partial observability:
  <https://roboticsproceedings.org/rss16/p023.html>

This document defines a research hypothesis and evidence bar.  It is not a
claim that the method has been implemented or validated.
