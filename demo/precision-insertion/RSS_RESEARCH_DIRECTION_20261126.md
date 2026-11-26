# RSS direction: cross-geometry proprioceptive inference for precision assembly

Date: 2026-11-26  
Status: proposed clean-room research design  
Relation to undergraduate work: separate apparatus and scientific claim

## 1. Recommendation in one sentence

Learn an **amortized, geometry-conditioned proprioceptive observation model**
that scores relative-pose hypotheses from short robot joint/action histories,
fuse it with a visual pose prior into an explicit calibrated belief, and test
zero-shot on unseen part/fixture geometries with the part rigidly mounted to
the flange and the mating cavity recessed in a flat plate.

Provisional descriptive title:

> **Amortized Geometry-Conditioned Proprioceptive Inference for Zero-Shot
> Precision Assembly**

Do not lock a branded method name before the likelihood-ratio formulation and
closest-work boundary are experimentally validated.

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
The key is rigidly attached to a repeatable Franka flange mount.  The fixture
is a flat plate with an interchangeable recessed cavity.  Only the key insert
and fixture insert change between geometries.  This matches the clean
contact-start logic of *Blind Precise Assembly* while still allowing a
free-space visual phase before contact.

## 3. Critical assessment of candidate research directions

| Candidate | Scientific value | Main problem | Decision |
| --- | --- | --- | --- |
| Accumulated XY correction across trials | Useful engineering diagnostic | Cylinder/task specific; calibration learning is well known; does not address unseen geometry | Thesis auxiliary only |
| Learned modality gate between vision and proprioception | Tests phase dependence | Fixed/learned slow-fast modality division is already crowded by Force Policy, RDP, FACTR and related fusion work | Insufficient alone |
| Direct VLA for precise insertion | Important baseline and may generalize semantically | Data hungry, weak metric-state interpretability, hard to isolate why it works, crowded topic | Baseline, not initial core method |
| End-to-end RGB+joint-history recurrent policy | Strong baseline | Implicit state gives no calibrated uncertainty or contact interpretation; difficult to audit | Required baseline |
| Particle belief with per-hypothesis physics simulation | Principled and interpretable | Closely overlaps BayesContact/BILBA; expensive per geometry and sensor substitution alone is weak novelty | Comparison/inspiration |
| InsertionWM-style visual-proprioceptive world model | Already demonstrates zero-shot insertion on unseen unknown geometry in simulation | An end-to-end RGB/depth+proprioception controller is no longer a novelty claim | Required baseline/boundary |
| Geometry-specific contact manifold matching | Gives accurate explicit SE(3) estimates for complex tight-clearance parts | Requires an offline manifold per geometry and starts partially inserted | Required boundary |
| Geometry-conditioned amortized likelihood-ratio from joint/action history | Explicit uncertainty, queryable pose evidence, unseen-shape transfer, no extra tactile/F/T sensor | Likelihood amortization and geometry-conditioned contact likelihoods exist separately; the cross-geometry assembly transfer must be decisive | **Recommended core, high risk** |

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

The remaining defensible gap is narrower:

> Can a single amortized observation model evaluate how likely an ordinary
> robot joint/action history is under arbitrary relative-pose hypotheses and
> previously unseen part/fixture geometry, so that contact turns a coarse
> visual prior into a calibrated, actionable pose belief without a dedicated
> F/T or tactile sensor?

This is not merely sensor fusion.  The candidate technical contribution is a
**cross-geometry proprioceptive likelihood-ratio over relative assembly
pose**, amortized over many training geometries.  It must replace both
per-geometry contact manifolds and per-particle online physics simulation with
one learned, queryable observation model while retaining an explicit
posterior.  Neither "likelihood," "Bayesian," nor "amortized" is novel alone.

## 5. Central hypothesis and research questions

### Central hypothesis

Given a test-time geometric description of an unseen key/fixture pair, a
learned likelihood field can reuse contact signatures learned across many
simulated geometries to score SE(3) relative-pose hypotheses from joint/action
histories.  Bayesian fusion with a multi-view visual prior should produce a
better calibrated and more actionable posterior than point pose estimates,
implicit recurrent fusion, geometry-specific contact matching, and
end-to-end visual-proprioceptive policies.

### Research questions

1. **Likelihood generalization:** can one learned observation model rank the
   true residual pose above hard negative hypotheses for held-out geometry
   families?
2. **Belief calibration:** after contact observations, does the posterior put
   the declared probability mass on the true pose and contact mode?
3. **Actionability:** at matched observation and control budgets, does using
   the full posterior improve correction, probing, and insertion over its
   mean, an implicit recurrent state, and a geometry-specific manifold?
4. **Real zero-shot transfer:** can the frozen likelihood model and controller
   assemble manufactured held-out keys/cavities without object-specific
   learning?

Across-trial memory is not part of the core hypothesis.  It may be added only
after repeated-trial data show a stable calibration or mounting bias.

## 6. Generalization contract

The primary claim is:

```text
known_geometry_zero_shot
```

The test shape is held out from likelihood-model and controller training.  Its
CAD mesh or calibrated scan is available at deployment.  Deterministic
geometry preprocessing is allowed, but no learned parameters, reference
contact manifold, or policy may be fitted on the test object.

This contract is intentionally different from:

- a new pose of a training cylinder;
- a new object ID with the same cross-section;
- fine-tuning on the test object;
- `unknown_geometry_zero_shot`, where geometry must be inferred only from live
  RGB.  InsertionWM already makes this stronger geometry-availability claim in
  simulation, so it is a stretch comparison rather than the proposed novelty.

## 7. Clean physical apparatus

### 7.1 Robot and part mount

- Franka arm with no gripper in the primary experiment.
- Kinematically surveyed, stiff flange adapter.
- Interchangeable key cartridges with a common mechanical datum.
- Key-to-flange transform recorded for evaluation; the policy receives only
  the observation contract declared for its baseline.

### 7.2 Fixture plate

- Flat plate with interchangeable recessed fixture cartridges.
- Common dowel/screw datum makes fixture exchange repeatable.
- No protruding socket wall in the primary experiment.
- Mount the plate on a metrology-grade XY--yaw stage, with calibrated tilt
  shims or a 6-DoF stage for designated tests.  Stage settings provide
  evaluation-only ground truth for imposed residuals and do not enter the
  estimator.
- Plate pose varies across a declared workspace distribution.

### 7.3 Geometry families

At minimum include:

- circular and elliptical profiles;
- regular polygonal profiles;
- keyed/asymmetric profiles;
- spline/star/gear-like profiles;
- freeform profiles generated under printability and clearance constraints.

Train/validation/test splitting occurs by **generator family and parameters**.
Holding out only random seeds from the same narrow generator is insufficient.

### 7.4 Observation and evaluation metrology

- Policy input: calibrated multi-view RGB, robot `q`, `dq`, commanded action,
  and forward-kinematic history; no wrist F/T or fingertip tactile.
- Test-time geometry input: CAD/scan descriptor for the primary contract.
- Primary estimator experiment: initialize a controlled visual prior around
  stage-measured ground truth.  This cleanly measures how much contact
  evidence changes pose uncertainty without confounding the result with one
  perception front end.
- Full-system experiment: replace that controlled prior with calibrated
  multi-view pose/tracking evidence and keep all downstream code fixed.
- Evaluation-only ground truth: stage setting plus surveyed fixture and
  flange datums, with independent optical metrology on a subset.  It must not
  enter the policy.
- Contact labels in simulation: exact geometry/contact state.
- Real success: physically interpretable seating/insertion event, with
  passive stability reported separately.

## 8. State and observation model

Let `G_K` and `G_S` be key and fixture geometry, and let

```text
h_G = Enc_geometry(G_K, G_S, task).
```

The hidden state contains:

```text
x_t = (T_SK,t, m_t, eta_t),
```

where `T_SK,t` is relative pose, `m_t` is a geometry-agnostic contact mode,
and `eta_t` contains nuisance physical variables needed by the estimator, such
as effective compliance or friction.  A shape-specific fixed vector of
cylinder landmarks is not allowed.

The observations are:

```text
z^v_t = multi-view pose/tracking evidence
z^p_t = history(q, dq, FK, commanded_action, tracking_error)
```

and the filter maintains

```text
b_t = p(x_t | z^v_0:t, z^p_0:t, u_0:t-1, h_G).
```

The initial vision update produces particles `{x_i, w_i}` approximating a
global pose prior.  For a candidate state `x`, a short commanded probe `U` and
observed proprioceptive sequence `H`, the core model estimates a
likelihood-to-evidence ratio

```text
r_theta(H, x, U, G)
  approximately p(H | x, U, G) / p(H | U, G),
where G = (G_K, G_S).
```

`Phi_geometry` exposes candidate-conditioned geometric quantities such as
signed distances, closest surface normals, and potential contact regions; it
must not encode a fixed cylinder landmark layout.  With neural ratio
estimation, a classifier distinguishes joint samples
`(H,x,U,G) ~ p(H,x,U,G)` from independently paired samples from
`p(H|U,G)p(x|U,G)`.  Hard-negative proposals may be added only with their
proposal density recorded and importance correction applied; otherwise the
learned score is not the stated ratio.  The calibrated classifier logit
estimates `log r_theta` without requiring a normalized generative model of the
joint history.  The particle update is

```text
w_i' proportional_to w_i * r_theta(H_t, x_i, U_t, G_K, G_S).
```

Thus the model answers a falsifiable question: *if this relative pose were
true for these geometries, how compatible would the measured joint response
be with the commanded motion?*  Contact-stage evidence reshapes the belief
when the object is occluded or visual precision is insufficient.  Symmetry is
handled only when a particular geometry has equivalent poses; the method must
also work on fully asymmetric shapes.

### Candidate implementation

1. Encode paired key/cavity SDF or surface samples in the candidate relative
   pose with an SE(3)-equivariant or invariant geometry encoder.
2. Encode a fixed-length history of `q`, `dq`, commanded joint/Cartesian
   motion, and tracking error.  Keep external wrench and tactile data out of
   the primary sensor contract.
3. Train a neural likelihood-to-evidence ratio estimator.  Joint
   pose--history samples are positive and independently paired samples are
   negatives.  Mine nearby contact-compatible aliases through an explicitly
   importance-corrected proposal.  Supervise contact mode as an auxiliary
   output, not as the sole objective.
4. Represent the posterior with particles over relative SE(3) pose plus a
   categorical contact mode.  Normalize weights over the declared proposal,
   calibrate on validation geometries only, and report true-pose rank,
   posterior coverage, NLL, and calibration before any control result.
5. Use a short, standardized low-force probe library first.  This makes
   estimator comparisons identifiable and prevents a learned probing policy
   from hiding estimation errors.
6. Correct using posterior-aware receding-horizon control.  Compare against
   posterior-mean control; add information-gain-based probe selection only if
   fixed probes leave reproducible multimodality.

The first paper version should prefer a shared analytical or fixed controller
after inference, because it makes improvements attributable to the likelihood
model.  If diverse geometry requires a learned controller, freeze the
estimator and evaluate estimator and controller ablations separately.

## 9. Training program

### Phase A: simulation identifiability

- Generate many exact-clearance key/fixture pairs.
- Randomize fixture pose, mount error, compliance, friction, and visual
  corruption independently.
- Collect standardized probing/action histories with privileged relative pose
  and contact labels.  Include hard-negative poses that share the same first
  contact but diverge under later probes.
- Test whether the estimator can recover pose/contact on held-out families
  before training a full controller.

**Go/no-go gate:** on held-out geometry families, the true pose must be ranked
above local hard negatives and posterior coverage must improve over a
geometry-agnostic history encoder.  If not, do not proceed to a control paper.

### Phase B: belief-conditioned control

- Start with one fixed correction/probing controller across geometries; train
  a controller only if this is demonstrably insufficient.
- Freeze geometry encoder, filter, and controller before held-out evaluation.
- Compare mean-belief control with distribution-aware control.

### Phase C: real transfer

- Manufacture a subset never used for training.
- Use CAD/scan only through the declared geometry front end.
- Perform no object-specific policy update.
- Report simulation-to-real degradation separately by geometry family.
- Evaluate both a controlled initial prior and an actual multi-view prior.

### Phase D: optional full system

- Restore a gripper, in-hand transform uncertainty, and tabletop perception
  only after the fixed-mount result is established.
- Treat this as a robustness demonstration, not part of the primary causal
  experiment.

## 10. Required baselines

1. nominal open-loop insertion;
2. MV-GoTrack/visual pose feedback only;
3. Blind-style proprioception-only history policy;
4. fixed visual alignment followed by the blind contact policy;
5. implicit RGB/pose + proprioception recurrent fusion, parameter matched;
6. InsertionWM-style latent world-model controller, reproduced in simulation
   or compared under its published contract;
7. geometry-specific contact-manifold matching on geometries for which the
   offline manifold can be generated;
8. geometry-conditioned point-state estimator/controller;
9. proposed distributional contact-belief field;
10. direct VLA trained from the same physical task distribution and with its
   data budget reported;
11. geometry-specific specialist upper bound where feasible.

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
- shared controller versus geometry-specific specialists;
- symmetry canonicalization on/off only for the symmetric subset;
- optional across-trial memory on/off, only if persistent bias is observed.

## 12. Metrics and success criteria

### Estimation

- relative-pose translation/rotation error;
- contact-mode accuracy, Brier score and negative log likelihood;
- posterior coverage and calibration error;
- information gain from visual and contact observations;
- held-out-instance and held-out-family generalization gaps.

### Control

- task success;
- success conditioned on reaching contact;
- time and path/contact-action count;
- peak available interaction signal;
- jam/recovery rate;
- zero-shot real success by held-out geometry family.

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

1. **Representation:** a queryable likelihood-ratio model maps paired geometry,
   candidate relative pose, commanded probe, and ordinary joint history to
   contact evidence.
2. **Generalization:** one frozen likelihood model transfers to held-out
   geometry families without fitting a per-part manifold or policy.
3. **Decision value:** the calibrated posterior improves a shared controller's
   real-robot precision or efficiency over point, implicit, and
   geometry-specific alternatives.

The following are explicitly **not** novelty claims:

- combining vision and proprioception;
- using a world model or recurrent policy for insertion;
- applying a particle filter to contact;
- using active probing;
- zero-shot insertion on unseen geometry in simulation;
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
| Force Policy / RDP / FACTR | Global/local multimodal hierarchy is established | State-estimation contribution rather than another sensor gate or fusion architecture |

If these differences cannot be supported experimentally, the honest target is
a strong system/benchmark paper rather than an RSS method paper.

## 15. Role of VLA

The direct VLA remains a professor-requested short-term baseline.  It should
not be inserted into the core method merely to make the system appear more
general.  In the RSS system, a VLA may select the task, geometry pair, target,
or recovery primitive.  Metric state estimation and contact correction remain
the responsibility of the contact-belief module unless the direct VLA
baseline demonstrates equal precision under a matched data budget.

## 16. Evidence bar for an RSS submission

Minimum credible package:

1. large multi-geometry simulation training set with family-level held-out
   splits;
2. calibrated estimator results, not just control success;
3. all required modality and representation baselines;
4. at least three clearance levels and multiple fixture poses;
5. at least ten physically manufactured held-out part/fixture pairs spanning
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

This document is a research design, not evidence that the proposed method has
been implemented or validated.
