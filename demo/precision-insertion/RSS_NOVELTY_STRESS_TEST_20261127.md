# Novelty stress test: general precision execution beyond task families

Date: 2026-11-27
Status: prospective positioning; no empirical superiority is claimed

This document asks whether the proposed RSS direction is scientifically
distinct from nearby work. It treats insertion, dense extraction, narrow-gap
placement and guide following as manifestations, not as method categories.

## 1. Claim that survives the literature check

The defensible claim is:

> A frozen, policy-agnostic execution layer maintains a calibrated
> variable-size belief over the signed margins and activation/contact modes of
> a supplied geometric/contact constraint program, then applies local residual
> corrections to nominal actions on held-out geometry, layout, constraint
> composition and upstream policy class without test-task learning.

The method is general only within a frozen vocabulary of measurable rigid-body
constraints. It is not universal manipulation, open-world physics or zero-shot
learning of an undefined new physical relation.

## 2. Closest work and the exact overlap

| Direction | What is already covered | What must remain distinct here |
| --- | --- | --- |
| [GeoManip](https://arxiv.org/abs/2501.09783) | LLM/VLM-derived geometric constraints as a general manipulation interface and training-free trajectory optimization | online calibrated belief over physical constraint margins/contact identity; sequential vision/proprioceptive updates; policy-agnostic residual execution rather than replacing the nominal policy |
| [Adaptive Diffusion Constrained Sampling](https://arxiv.org/abs/2505.13667) | flexible equality/inequality constraint embeddings and constraint-weighted trajectory sampling | partially observed physical execution after a nominal trajectory exists; calibrated posterior and cross-policy residual decision value |
| [Generalizable Robotic Insertion with World Models](https://arxiv.org/abs/2609.28258) | one visual-proprioceptive world model trained across many insertion geometries; zero-shot unseen-object insertion | transfer across constraint motifs and graph compositions rather than only object geometry inside insertion; explicit auditable constraint belief; heterogeneous frozen base policies |
| [BEACON](https://arxiv.org/abs/2609.22730) | Bayesian belief conditioning outperforms raw observation history under manipulation state aliasing | a task-program-indexed, variable-size belief shared across manifestations; calibrated signed margins and contact entities; analytical policy-agnostic residual layer rather than a belief-conditioned task policy |
| [BayesContact](https://arxiv.org/abs/2607.16123) | multimodal visuo-contact pose belief and information-gain probing across insertion geometries | belief over active task constraints rather than only object pose; non-insertion manifestations; residual execution across base-policy classes |
| [CaPTURe](https://proceedings.mlr.press/v329/marques26a.html) | contact-aware conformal calibration of multimodal future configuration sets | program-conditioned constraint violation belief and its value for correcting arbitrary nominal manipulation policies |
| [ResiP](https://arxiv.org/abs/2407.16677) | closed-loop residual RL improves frozen chunked BC on precise assembly | no task-specific residual training at deployment; held-out constraint composition and base-policy class; explicit calibrated belief versus implicit residual state |
| [OmniTacTune](https://arxiv.org/abs/2607.03723) | policy-agnostic tactile residual adaptation improves several visual policies and contact-rich tasks | zero-test-learning primary claim; constraint-program transfer; calibrated belief and point-state/raw-history comparisons instead of per-task online RL |
| [ARC](https://arxiv.org/abs/2610.37547) | sparse human correction localizes when and where residual exploration is needed | no test-task human corrections; constraint-derived intervention and residual subspace; compositional zero-shot evaluation |

No row alone leaves a paper-sized gap. The proposed contribution is the joint
interface among compositional constraint specification, calibrated sequential
physical inference and policy-agnostic residual decision. Every one of those
parts must be experimentally necessary.

## 3. Claims that should be rejected

Do not submit the following as novelty:

1. **VLM chooses a correction direction.** This is a weak heuristic and does
   not establish millimetre-level geometry or uncertainty.
2. **Vision plus proprioception improves insertion.** Multimodal precision
   insertion and contact belief are already occupied.
3. **A residual policy improves a VLA.** Residual precision learning and
   policy-agnostic tactile adaptation already exist.
4. **A constraint language generalizes tasks.** Geometric-constraint
   interfaces and task-conditioned contracts already exist.
5. **One model handles many peg shapes.** The 90-task insertion world-model
   result is a stronger direct comparison.
6. **The same network runs on several demos.** Shared weights are not evidence
   of geometry, composition or policy-class zero-shot transfer.

## 4. Central scientific hypothesis

The hypothesis is not that a belief is generally useful. It is:

> The posterior distribution of task-specified normalized constraint margins,
> activation states and contact entities is a smaller and more transferable
> decision state for precision correction than either a visual point pose or
> an unconstrained observation-history latent.

This produces three linked tests:

1. **Identifiability:** the shared belief predicts signed margins and contact
   identity on held-out geometry/layout/composition.
2. **Calibration:** its violation probabilities and credible sets retain
   coverage under those holdouts.
3. **Decision value:** posterior-aware control beats posterior-mean/point-state
   MPC and a matched recurrent residual under the same action and time budget.

If any test fails, the paper claim must narrow. A larger end-to-end controller
must not be used to hide a failed representation hypothesis.

## 5. Clean experimental separation

### Primary causal apparatus

- Franka with a surveyed rigid payload mount;
- modular datum plate and interchangeable fixed/movable geometry;
- multi-view RGB and ordinary Franka proprioception as method inputs;
- evaluation-only metrology for physical margins and neighbour disturbance;
- supplied CAD/reconstruction and oracle constraint program initially.

The rigid mount removes grasp/slip as a confound. Grasping is a secondary
robustness condition, not the primary claim.

### Manifestation basis

Use at least six real manifestations spanning at least four motifs:

- terminal equality plus contact;
- narrow free-space passage;
- disturbance-sensitive extraction or placement among movable neighbours;
- required-contact tangent motion;
- path/manifold following.

Names such as insertion or Pringles extraction are only video labels. They
must not select a model head, reward, state vector or controller.

### Independent holdouts

- Z1: geometry generator family;
- Z2: layout generator family and tolerance;
- Z3: base-policy class;
- Z4: predicate combination and graph topology.

Report each axis and the joint holdout. A random seed or unseen mesh from the
same generator is not the headline zero-shot result.

## 6. Required empirical comparisons

### Estimation

1. visual point pose;
2. geometry-conditioned point state;
3. raw-history recurrent encoder with matched capacity;
4. task-specific belief upper bound;
5. proposed active-constraint belief;
6. privileged simulator/metrology oracle.

### Control

1. frozen base policy;
2. visual pose servo;
3. point-state constrained MPC;
4. matched recurrent residual;
5. posterior-mean constrained MPC;
6. posterior-aware constrained MPC;
7. task-specific residual adaptation as a non-zero-shot upper bound.

### Base policies

At least a model-based planner, an imitation/diffusion policy and a
language-conditioned/VLA policy are required. Hold out one entire policy class
while training the precision layer.

## 7. Headline figures

1. success and violation magnitude versus physical tolerance, not one success
   threshold;
2. false-safe rate and coverage versus Z1--Z4 holdout;
3. point-state versus raw-history versus explicit-belief decision quality;
4. gain/degradation for each frozen base policy;
5. intervention count, magnitude, latency and path/time overhead;
6. neighbour displacement for disturbance-sensitive programs.

## 8. Go/no-go rule for an RSS submission

Proceed with the full claim only if all are true:

- the explicit belief outperforms matched point-state and raw-history models
  in held-out calibration/contact diagnosis;
- posterior-aware control outperforms point-state and posterior-mean MPC;
- gains remain on a held-out constraint composition and base-policy class;
- at least four real constraint motifs show a shifted tolerance curve without
  test-task updates;
- the VLA compiler is evaluated separately and is not needed to obtain the
  oracle-program core result.

Otherwise publish a narrower result: multimodal cylinder thesis, contact
belief for one family, or a benchmark/tool paper. Do not label a task-specific
success improvement as general precision manipulation.

## 9. Separation from the undergraduate thesis

The undergraduate thesis remains a controlled cylinder study of when vision,
proprioception and a fixed handoff help. It may validate sensing, latency and
failure taxonomy, but it cannot support Z1--Z4 or policy-agnostic claims. RSS
training/test data and the clean rigid-mount apparatus must be rebuilt and
frozen independently.
