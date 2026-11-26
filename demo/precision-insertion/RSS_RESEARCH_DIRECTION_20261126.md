# RSS direction: geometry-conditioned contact-constraint belief for precision manipulation

Date: 2026-11-26  
Status: proposed clean-room research design  
Relation to undergraduate work: separate apparatus and scientific claim

Scope correction: `RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md` is authoritative
for the long-term method.  The mating and narrow-passage scenes in this file
are examples for a finite evaluation basis, not hard-coded task families.

The prospective experiment and leakage-control contract is frozen separately
in `RSS_BENCHMARK_PROTOCOL_20261126.md`.  Current-code reuse and missing core
modules are audited in `RSS_IMPLEMENTATION_GAP_20261126.md`.

## 1. Recommendation in one sentence

Learn one **amortized, geometry-conditioned proprioceptive observation
model** that updates an explicit belief over object pose and active contact
identity, then use a task constraint graph to interpret the same physical
evidence as useful, tolerable, or forbidden contact across both mating and
dense narrow-passage manipulation.

Provisional descriptive title:

> **Geometry-Conditioned Contact-Constraint Beliefs for Generalizable
> Precision Manipulation**

Do not lock a branded method name before the likelihood-ratio formulation and
closest-work boundary are experimentally validated.

### 1.1 Scope boundary

Do not claim "universal precise manipulation."  The defensible RSS scope is a
zero-shot precision execution layer for rigid-body tasks expressible as a
compositional geometric/contact constraint program.  The following are two
useful examples with opposite contact semantics:

1. **mating/engagement**, where contact with the target fixture is expected
   and may be informative;
2. **selective narrow passage**, where contact that displaces neighbouring
   objects is forbidden while extracting a target or placing into a gap.

The claim concerns a shared physical observation model and constraint
interface, not a single motion policy for every task.  Primary evidence must
hold out geometry, layout, upstream policy and constraint-graph composition;
holding out an undefined new constraint primitive is a stretch experiment.

## 2. Why this is cleaner than extending the current demo

The undergraduate system is intentionally pragmatic:

- protruding cylindrical socket;
- existing key/socket assets;
- tabletop pose estimation and grasping;
- current gripper/hand integration;
- cylinder-specific symmetry and residual coordinates.

Those choices are reasonable for finishing a thesis quickly, but they create
four confounds for an RSS paper: grasp failure, in-hand slip, finger/fixture
collision, and cylinder-specific geometry.  Improving success in that system
would not reveal whether the contribution came from perception, contact
inference, grasping, or task-specific engineering.

The RSS apparatus should therefore remove grasping from the primary claim.
The manipulated payload is rigidly attached to a repeatable Franka flange
mount.  Interchangeable scenes expose two deliberately different contact
semantics: a recessed mating fixture where target contact is expected, and a
dense narrow-passage scene where displacement of neighbouring objects is
forbidden.  Grasping or automatic target acquisition can be restored only as
a secondary system experiment.

## 3. Critical assessment of candidate research directions

| Candidate | Scientific value | Main problem | Decision |
| --- | --- | --- | --- |
| Accumulated XY correction across trials | Useful engineering diagnostic | Cylinder/task specific; calibration learning is well known; does not address unseen geometry | Thesis auxiliary only |
| Learned modality gate between vision and proprioception | Tests phase dependence | Fixed/learned slow-fast modality division is already crowded by Force Policy, RDP, FACTR and related fusion work | Insufficient alone |
| Direct VLA for precise manipulation | Important baseline and may generalize semantically | Data hungry, weak metric-state interpretability, hard to isolate why it works, crowded topic | Baseline, not initial core method |
| End-to-end RGB+joint-history recurrent policy | Strong baseline | Implicit state gives no calibrated uncertainty or contact interpretation; difficult to audit | Required baseline |
| Particle belief with per-hypothesis physics simulation | Principled and interpretable | Closely overlaps BayesContact/BILBA; expensive per geometry and sensor substitution alone is weak novelty | Comparison/inspiration |
| InsertionWM-style visual-proprioceptive world model | Already demonstrates zero-shot insertion on unseen unknown geometry in simulation | An end-to-end RGB/depth+proprioception controller is no longer a novelty claim | Required baseline/boundary |
| Geometry-specific contact manifold matching | Gives accurate explicit SE(3) estimates for complex tight-clearance parts | Requires an offline manifold per geometry and starts partially inserted | Required boundary |
| Geometry-conditioned amortized likelihood-ratio plus task constraint graph | Explicit uncertainty, reusable physical evidence, and different contact semantics without separate low-level policies | Contact belief in clutter, safe extraction, belief-space assembly, and likelihood amortization all exist separately; cross-task transfer must be decisive | **Recommended core, very high risk** |

## 4. Exact research gap

Existing work establishes the following:

- *Blind Precise Assembly* shows that proprioceptive history can control
  diverse assemblies from contact, but its task state remains implicit and it
  does not use visual evidence to reduce global uncertainty.
- *InsertionWM* already combines raw wrist depth and proprioception in a
  learned world model and reports zero-shot insertion on unseen, unknown
  geometry in simulation.  Therefore RGB/depth--proprioception fusion,
  geometry diversity, and an end-to-end learned controller are not sufficient
  novelty by themselves.
- AutoMate learns specialist and generalist assembly policies over diverse
  geometries, but is not a calibrated multimodal state estimator for the
  free-to-contact observability transition.
- Force Policy and Reactive Diffusion Policy divide global vision from local
  force/tactile control, so a visual/local hierarchy alone is not new.
- BayesContact explicitly infers a particle pose belief from depth and
  force/torque likelihoods with simulation-based inference, including active
  probes.
- BILBA and older belief-space manipulation work already use uncertainty and
  contact to plan assembly.
- CaPTURe calibrates contact-dependent particle prediction regions.
- Contact-manifold sampling already estimates full SE(3) pose for complex
  low-clearance parts from primitive contact motions, but constructs a
  geometry-specific reference manifold and begins from partial insertion.
- Amortized neural likelihood-to-evidence estimation has already been used for
  Bayesian robotic grasp inference.  Learning a surrogate likelihood is
  therefore not independently novel.
- *A Contact-Driven Framework for Manipulating in the Blind* already uses
  joint-torque contact particles to estimate occupancy and plan in unknown
  clutter.  A contact particle filter or a clutter-retrieval demonstration is
  therefore not independently novel.
- *SafePicking*, selective rearrangement, and gentle object retraction already
  study extracting a target from dense clutter while limiting disturbance.
  The Pringles task is a useful benchmark, not a novelty claim by itself.

The remaining defensible gap is narrower:

> Can one task-independent, amortized physical observation model turn ordinary
> robot joint/action histories into calibrated evidence over object pose and
> active contact identity for unseen geometry and layout, while a declarative
> task constraint graph lets the same belief support tasks in which contact
> with one surface is useful and contact with another object is forbidden?

This is not merely sensor fusion, clutter avoidance, or language-conditioned
control.  The candidate contribution is a **cross-geometry proprioceptive
likelihood-ratio over relative scene hypotheses**, amortized over training
geometries and shared across contact semantics.  A separate constraint graph
declares goals, allowed contact, forbidden contact, and disturbance
tolerances.  The physics likelihood must not depend on whether a contact is
desirable; that semantics belongs in control and evaluation.  The method must
replace per-geometry contact manifolds and per-particle online physics
simulation with one queryable observation model while retaining an explicit
posterior.  Neither "belief," "Bayesian," "amortized," nor "safe extraction"
is novel alone.

## 5. Central hypothesis and research questions

### Central hypothesis

Given test-time geometry for an unseen manipulated object and local
environment, a learned likelihood field can reuse contact signatures learned
across simulated geometries to score pose/contact hypotheses from joint/action
histories.  Bayesian fusion with a multi-object visual prior should produce a
calibrated posterior that supports both (i) intentional target contact during
mating and (ii) avoidance or recovery from forbidden neighbour contact during
narrow-passage extraction/placement.  It should outperform point estimates,
implicit recurrent fusion, geometry-specific contact matching, and separate
task-specific low-level policies under matched data and action budgets.

### Research questions

1. **Likelihood generalization:** can one learned observation model rank the
   true residual pose above hard negative hypotheses for held-out geometry
   families?
2. **Belief calibration:** after contact observations, does the posterior put
   the declared probability mass on the true pose and contact mode?
3. **Constraint actionability:** at matched observation and control budgets,
   does the posterior improve goal progress while reducing forbidden contact
   and neighbour disturbance over its point summary and implicit state?
4. **Cross-semantics reuse:** can the same observation model support both
   contact-seeking mating and contact-avoiding narrow-passage manipulation
   without geometry- or task-specific inference heads?
5. **Real zero-shot transfer:** can the frozen observation model and
   constraint-conditioned controller manipulate held-out objects, fixtures,
   and layouts without object-specific learning?

Across-trial memory is not part of the core hypothesis.  It may be added only
after repeated-trial data show a stable calibration or mounting bias.

## 6. Generalization contract

The primary claim is:

```text
known_geometry_zero_shot
```

The test object/environment geometry and layout are held out from
likelihood-model and controller training.  Their CAD meshes or calibrated
scans are available at deployment.  Deterministic geometry preprocessing and
task-graph construction are allowed, but no learned parameters, reference
contact manifold, or policy may be fitted on the test scene.

This contract is intentionally different from:

- a new pose of a training cylinder;
- a new object ID with the same cross-section;
- fine-tuning on the test object;
- `unknown_geometry_zero_shot`, where geometry must be inferred only from live
  RGB.  InsertionWM already makes this stronger geometry-availability claim in
  simulation, so it is a stretch comparison rather than the proposed novelty.

## 7. Clean physical apparatus

### 7.1 Robot and payload mount

- Franka arm with no gripper in the primary experiment.
- Kinematically surveyed, stiff flange adapter.
- Interchangeable payload cartridges with a common mechanical datum.
- Payload-to-flange transform recorded for evaluation; the policy receives only
  the observation contract declared for its baseline.

### 7.2 Scene A: mating/engagement

- Flat plate with interchangeable recessed fixture cartridges.
- Common dowel/screw datum makes fixture exchange repeatable.
- No protruding socket wall in the primary experiment.
- Mount the plate on a metrology-grade XY--yaw stage, with calibrated tilt
  shims or a 6-DoF stage for designated tests.  Stage settings provide
  evaluation-only ground truth for imposed residuals and do not enter the
  estimator.
- Plate pose varies across a declared workspace distribution.

### 7.3 Scene B: selective narrow passage

- A common base holds interchangeable rigid or instrumented neighbouring
  objects with surveyed datums.
- The primary manipulation is either extraction from a dense row or placement
  into an empty gap; these are reported separately.
- The manipulated payload begins after a repeatable flange coupling, so grasp
  failure does not confound the precision claim.
- Neighbour pose changes are measured independently for evaluation.  A
  compliant or breakaway mount may protect hardware, but its motion must not
  enter the policy unless declared as an observation.
- Layout varies in gap width, neighbour geometry, height, and target path.

### 7.4 Geometry and layout families

For mating, include:

- circular and elliptical profiles;
- regular polygonal profiles;
- keyed/asymmetric profiles;
- spline/star/gear-like profiles;
- freeform profiles generated under printability and clearance constraints.

For narrow passage, include cylindrical, prismatic, tapered, and asymmetric
payload/neighbour families, plus row, staggered, and nonuniform layouts.  The
gap and route must be represented geometrically rather than by a fixed
Pringles-can landmark vector.

Train/validation/test splitting occurs by **geometry generator family,
parameters, and layout family**.  Holding out only random seeds from the same
narrow generator is insufficient.

### 7.5 Observation and evaluation metrology

- Policy input: calibrated multi-view RGB, robot `q`, `dq`, commanded action,
  and forward-kinematic history; no wrist F/T or fingertip tactile.
- Test-time geometry input: CAD/scan descriptors and a machine-readable task
  constraint graph for the primary contract.
- Primary estimator experiment: initialize a controlled visual prior around
  stage-measured ground truth.  This cleanly measures how much contact
  evidence changes pose uncertainty without confounding the result with one
  perception front end.
- Full-system experiment: replace that controlled prior with calibrated
  multi-view pose/tracking evidence and keep all downstream code fixed.
- Evaluation-only ground truth: stage setting, surveyed fixture/scene and
  flange datums, and independent optical metrology of neighbour motion on a
  subset.  It must not enter the policy.
- Contact labels in simulation: exact geometry/contact state.
- Real success: task-specific goal completion plus the declared forbidden
  contact/disturbance contract.  Seating depth is used for mating; target
  extraction/placement and neighbour displacement are used for narrow
  passage.

## 8. Task-independent physical belief and task-conditioned constraints

Let `G_O` be manipulated-object geometry and `G_E = {G_j}` the local
environment geometry.  A machine-readable task constraint graph `C` contains:

- the manipulated object and target/goal region;
- allowed or required contact relations;
- forbidden contact relations;
- per-object displacement/rotation tolerances;
- terminal success and failure predicates.

The geometry encoder receives the posed local scene, but the physical
observation model does **not** receive a reward label saying whether a contact
is good or bad:

```text
h_G = Enc_geometry(G_O, G_E).
```

The hidden state contains

```text
x_t = (T_WO,t, a_t, m_t, delta_E,t, eta_t),
```

where `T_WO,t` is object pose; `a_t` identifies the active object/surface
constraint; `m_t` is free, approaching, one-sided contact, multi-contact,
sliding, jam, or goal engagement; `delta_E,t` represents task-relevant motion
of neighbouring objects; and `eta_t` contains nuisance compliance/friction.
The implementation may factor this state rather than form one enormous joint
particle set.  A fixed vector of cylinder landmarks or a fixed number of
neighbours is not allowed.

The observations are

```text
z^v_t = multi-object pose/tracking evidence
z^p_t = history(q, dq, FK, commanded_action, tracking_error)
```

and the physical filter maintains

```text
b_t = p(x_t | z^v_0:t, z^p_0:t, u_0:t-1, h_G).
```

For candidate state `x`, short action/probe `U`, and measured history `H`, the
core model estimates

```text
r_theta(H, x, U, G)
  approximately p(H | x, U, G) / p(H | U, G),
where G = (G_O, G_E).
```

The absence of `C` is intentional: the same collision produces the same
physical evidence whether the task calls it alignment, tolerable incidental
contact, or failure.  `C` is applied afterward to compute posterior goal
probability and forbidden-contact/disturbance risk for each action:

```text
u_t = argmax_u E_{x ~ b_t}[
        goal_progress(x,u,C)
        - lambda * forbidden_risk(x,u,C)
        + beta * information_gain(x,u)
      ].
```

With neural ratio estimation, a classifier distinguishes joint samples
`(H,x,U,G) ~ p(H,x,U,G)` from independently paired samples from
`p(H|U,G)p(x|U,G)`.  Hard-negative proposals require recorded proposal density
and importance correction.  The particle update is

```text
w_i' proportional_to w_i * r_theta(H_t, x_i, U_t, G_O, G_E).
```

`Phi_geometry` exposes candidate-conditioned signed distances, closest
surface normals, potential contact pairs, and clearance margins.  Contact
identity is essential: an insertion rim and a neighbouring Pringles-like can
cannot be collapsed into one generic `CONTACT` state.  Symmetry is handled
only for geometries that actually have equivalent poses.

### Candidate implementation

1. Compile a posed local scene and `C` from CAD/scans and task specification.
   In the controlled benchmark, `C` is provided directly.  A VLA-generated
   `C` is evaluated separately so semantic parsing cannot hide metric errors.
2. Encode object/environment SDF or surface samples under each candidate
   object pose with an SE(3)-equivariant or invariant scene encoder.
3. Encode a fixed-length history of `q`, `dq`, commanded joint/Cartesian
   motion, and tracking error.  Keep external wrench/tactile data out of the
   primary contract; estimated joint torque is a named ablation if used.
4. Train one neural likelihood-to-evidence ratio estimator across both task
   families.  Supervise active contact identity/mode as auxiliary outputs,
   not as replacements for calibrated pose evidence.
5. Maintain a factored posterior over object pose, active contact identity,
   and task-relevant neighbour motion.  Report rank, NLL, credible-set
   coverage, contact identity, and disturbance prediction before control.
6. Use frozen, family-appropriate primitive libraries first: approach and
   small alignment probes for mating; lift/translate/backoff primitives for
   narrow passage.  The inference model is shared even when feasible actions
   differ.
7. Compare point/MAP control with posterior-aware constrained receding-horizon
   selection.  Add learned probing only if fixed primitives leave repeatable
   ambiguity.

The first paper version should prefer a shared constraint-aware controller or
the same action-selection objective with family-specific primitive sets.  If
each task requires a separate encoder, contact state, and policy, the claimed
general method has failed and the result is two engineered systems.

## 9. Training program

### Phase A: simulation identifiability

- Generate many exact-clearance mating pairs and dense narrow-passage scenes.
- Randomize target/environment pose, layout, mount error, compliance,
  friction, and visual corruption independently.
- Collect standardized probing/action histories with privileged relative pose
  and contact-identity labels.  Include hard-negative states that share the
  same first contact but diverge under later motion.
- Test whether one estimator can recover pose/contact identity on held-out
  geometry and layout families in both task families before training a full
  controller.

**Go/no-go gate:** on held-out geometry families, the true pose must be ranked
above local hard negatives and posterior coverage must improve over a
geometry-agnostic history encoder.  If not, do not proceed to a control paper.

### Phase B: belief-conditioned control

- Start with one constraint-aware objective and frozen primitive libraries;
  train an end-to-end controller only if this is demonstrably insufficient.
- Freeze geometry encoder, filter, and controller before held-out evaluation.
- Compare mean-belief control with distribution-aware control and compare a
  shared inference model against task-specific estimator heads.

### Phase C: real transfer

- Manufacture a subset never used for training.
- Use CAD/scan only through the declared geometry front end.
- Perform no object-specific policy update.
- Report simulation-to-real degradation separately by geometry and layout
  family and by contact semantics.
- Evaluate both a controlled initial prior and an actual multi-view prior.

### Phase D: optional full system

- Restore a gripper, in-hand transform uncertainty, and tabletop perception
  only after the fixed-mount result is established.
- Treat this as a robustness demonstration, not part of the primary causal
  experiment.

## 10. Required baselines

1. nominal open-loop task trajectory;
2. MV-GoTrack/multi-object visual feedback only;
3. geometry-only signed-clearance controller with no contact update;
4. Blind-style proprioception-only history policy;
5. fixed visual alignment followed by a task-specific contact policy;
6. implicit RGB/pose + proprioception recurrent fusion, parameter matched;
7. InsertionWM-style latent world-model controller on mating, reproduced in
   simulation or compared under its published contract;
8. geometry-specific contact-manifold matching where applicable;
9. task-specific point-state estimator/controllers;
10. shared geometry-conditioned point estimator plus constraint controller;
11. proposed distributional contact-constraint belief;
12. direct VLA trained from the same physical task distribution and with its
   data budget reported;
13. task/geometry-specific specialist upper bounds where feasible.

Comparisons to BayesContact/Force Policy are conceptual unless the required
sensor suite can be reproduced fairly.  Do not claim an empirical win from a
different sensor contract.  InsertionWM's primary published evidence is
simulation; report that domain difference explicitly rather than presenting a
real-only comparison as an architectural win.

## 11. Core ablations

- remove geometry conditioning;
- train/test only within one geometry family;
- remove visual updates after initialization;
- remove proprioceptive history;
- point posterior versus calibrated distribution;
- fixed handoff versus evidence-dependent update;
- task constraint graph versus a task-ID token only;
- shared observation model versus task-specific estimator heads;
- shared controller objective versus task/geometry-specific specialists;
- symmetry canonicalization on/off only for the symmetric subset.

## 12. Metrics and success criteria

### Estimation

- relative-pose translation/rotation error;
- active-contact identity and mode accuracy, Brier score and negative log
  likelihood;
- neighbour-motion/disturbance prediction error;
- posterior coverage and calibration error;
- information gain from visual and contact observations;
- held-out-instance and held-out-family generalization gaps.

### Control

- task success;
- success conditioned on reaching contact;
- time and path/contact-action count;
- peak available interaction signal;
- jam/recovery rate;
- number of disturbed neighbours;
- maximum and total neighbour translation/rotation;
- forbidden-contact duration or proxy under the declared sensor contract;
- zero-shot real success by held-out geometry and layout family.

The paper must report both estimator and controller results.  Success alone
cannot prove that the belief is meaningful; pose accuracy alone cannot prove
that it is useful.

## 13. Falsification criteria

- If an implicit recurrent fusion policy matches belief calibration and task
  success, the explicit belief contribution is not supported.
- If geometry conditioning helps only seen instances, the generalization claim
  fails.
- If vision-only solves every held-out geometry at the target clearance, the
  contact-observability problem is too easy.
- If blind proprioception matches the proposed method from the same start
  distribution, visual evidence is unnecessary for that protocol.
- If gains require per-object fine-tuning, the method is not zero-shot across
  geometry.
- If results hold only for cylinders or one procedural family, do not submit a
  generalizable precision-manipulation claim.
- If mating and narrow passage require separate physical encoders or
  incompatible contact states, the cross-task representation claim fails.
- If the task graph can be replaced by a task-ID token without loss on
  held-out layouts and contact relations, its structured semantics are
  unsupported.
- If an InsertionWM-style latent policy matches success and data efficiency
  while the proposed posterior is not better calibrated, the explicit-belief
  claim is unsupported.
- If a geometry-specific contact manifold matches the proposed estimator and
  amortization provides no speed, data, or transfer advantage, the likelihood
  field is an implementation convenience rather than a contribution.
- If the learned likelihood ranks the true pose correctly only after a
  geometry-specific calibration sequence, the zero-shot claim fails.

## 14. Novelty claim that survives the audit

The paper should make exactly three linked claims and avoid broader wording:

1. **Representation:** a queryable, task-independent likelihood-ratio model
   maps scene geometry, candidate state, commanded motion, and ordinary joint
   history to pose and contact-identity evidence.
2. **Generalization:** one frozen observation model transfers across held-out
   geometry/layout families and two different contact semantics without
   fitting per-scene manifolds or task-specific inference heads.
3. **Decision value:** a machine-readable constraint graph turns the
   calibrated posterior into goal-seeking versus forbidden-contact decisions
   and improves real precision/disturbance over point, implicit, and
   specialist baselines.

The following are explicitly **not** novelty claims:

- combining vision and proprioception;
- using a world model or recurrent policy for insertion;
- applying a particle filter to contact;
- using active probing;
- zero-shot insertion on unseen geometry in simulation;
- safe target extraction from clutter;
- language-to-skill selection;
- symmetry-aware pose canonicalization;
- accumulated XY correction across episodes.

### Closest-work stress test

| Closest work | Stronger/equal aspect of that work | Required non-overlap for this paper |
| --- | --- | --- |
| InsertionWM (2026) | Unknown-geometry zero-shot control from wrist depth and proprioception; up to 90 training assemblies | Explicit calibrated relative-pose likelihood/posterior, real transfer, and estimator-level evidence |
| Blind Precise Assembly (2026) | Real zero-shot diverse assembly from joint history alone | Show where visual prior and explicit belief improve global initialization, speed, or success under matched contact starts |
| BayesContact (2026) | Explicit particle belief, visual/contact fusion, active information-gain probing, real robot | Amortized cross-geometry likelihood from joint history, no F/T/tactile, and lower per-test computation/preprocessing |
| Contact-manifold sampling (2025) | Explicit full-SE(3) contact pose estimation at 0.1--1.0 mm clearance with 96.7% insertion success | No per-geometry reference manifold; family-held-out transfer and calibrated ambiguity |
| Simulation-based Bayesian grasp inference (2021) | Amortized differentiable likelihood-to-evidence ratio for contact-rich robotic inference | Assembly-specific joint-history evidence and transfer of one observation model across held-out paired geometries |
| BILBA / partial-observability control | Belief-aware contact planning is established | Learned geometry-conditioned observation model and held-out-shape real evidence |
| Contact-Driven Framework for Manipulating in the Blind (2026) | Joint-torque contact particles, occupancy estimation and planning in unknown clutter | Known-geometry amortized pose/contact-identity likelihood, calibrated posterior, and transfer across opposing contact semantics |
| SafePicking / Gentle Object Retraction / selective rearrangement | Target extraction while limiting clutter disturbance is established | One explicit belief and constraint interface shared with mating, held-out geometry/layout transfer, and estimator-level evidence |
| Force Policy / RDP / FACTR | Global/local multimodal hierarchy is established | State-estimation contribution rather than another sensor gate or fusion architecture |

If these differences cannot be supported experimentally, the honest target is
a strong system/benchmark paper rather than an RSS method paper.

## 15. Role of VLA

The direct VLA remains a professor-requested short-term baseline.  It should
not be inserted into the core method merely to make the system appear more
general.  In the RSS system, a VLA may compile an instruction and scene into
`C`: target, goal region, allowed/required contacts, forbidden objects,
tolerances, and a primitive vocabulary.  The controlled primary experiment
supplies the same graph directly.  Metric state estimation and
contact-constrained correction remain the responsibility of the
belief/controller unless the direct VLA baseline demonstrates equal precision
under a matched data budget.

## 16. Evidence bar for an RSS submission

Minimum credible package:

1. large multi-geometry simulation training set with family-level held-out
   splits;
2. calibrated estimator results, not just control success;
3. all required modality and representation baselines;
4. at least three clearance levels and multiple fixture/scene layouts;
5. held-out physical mating pairs and narrow-passage payload/layouts spanning
   several families;
6. repeated real trials with confidence intervals;
7. no object-specific policy fine-tuning in the primary result;
8. videos and episode records that expose phase-specific failures;
9. a separate end-to-end gripper experiment only as a robustness extension.

Additionally, preregister the held-out geometry split and trial count before
the main real evaluation.  Use a pilot only to estimate effect size and choose
the number of repetitions; do not redesign the split after seeing test
results.

## 17. Closest-work boundary

- Blind Precise Assembly: <https://arxiv.org/abs/2610.23849>
- InsertionWM / Generalizable Robotic Insertion with World Models:
  <https://arxiv.org/abs/2609.28258>
- AutoMate: <https://roboticsproceedings.org/rss20/p064.html>
- Force Policy: <https://roboticsproceedings.org/rss22/p128.html>
- Reactive Diffusion Policy: <https://roboticsproceedings.org/rss21/p052.html>
- FACTR: <https://roboticsproceedings.org/rss21/p079.html>
- BayesContact: <https://arxiv.org/abs/2607.16123>
- BILBA: <https://arxiv.org/abs/2409.15774>
- Contact-rich control under partial observability:
  <https://roboticsproceedings.org/rss16/p023.html>
- CaPTURe: <https://proceedings.mlr.press/v329/marques26a.html>
- Contact-manifold sampling: <https://arxiv.org/abs/2507.03925>
- Simulation-based Bayesian inference for multi-fingered grasping:
  <https://arxiv.org/abs/2109.14275>
- A Contact-Driven Framework for Manipulating in the Blind:
  <https://arxiv.org/abs/2510.20177>
- SafePicking:
  <https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/dyson-robotics-lab/wada_etal_safepicking_ICRA2022.pdf>
- Gentle Object Retraction in Dense Clutter:
  <https://arxiv.org/abs/2508.19476>
- Selective Object Rearrangement in Clutter:
  <https://proceedings.mlr.press/v205/tang23a.html>

This document is a research design, not evidence that the proposed method has
been implemented or validated.
