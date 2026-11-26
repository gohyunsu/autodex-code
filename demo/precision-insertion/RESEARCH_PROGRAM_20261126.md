# Staged research program: generalizable precision manipulation

Date: 2026-11-26  
Status: active research specification  
Supersedes as the top-level agenda: the same-trial visual-correction-only
scope in `RESEARCH_EVALUATION_PROTOCOL.md`

The authoritative RSS scope is now
`RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`.  Insertion and constrained
passage below are diagnostic examples, not a restriction of the method to two
task families.

The corresponding task-agnostic evaluation contract is
`RSS_ZERO_SHOT_PRECISION_BENCHMARK_20261127.md`.  It replaces named task
families with compositional constraint programs and independent hold-outs for
geometry, layout, graph composition and upstream policy.

The stage dependencies, current evidence boundary and exact transition gates
for both the undergraduate and RSS tracks are maintained in
`RESEARCH_PROGRAM_EVIDENCE_MAP_20261127.md`.

## 1. Scope and central question

The project is not three independent attempts to solve one cylinder-insertion
demo with (i) vision, (ii) proprioception, and (iii) a VLA.  The scientific
question is:

> How should a robot represent and reduce precision-task uncertainty as the
> task moves from visible free-space motion into partially observed physical
> contact?

The meeting identified two sources of difficulty that must initially be
measured separately:

1. **Geometric uncertainty before contact.**  Object, fixture, calibration,
   and object-in-gripper errors leave a residual relative pose.  Continuous
   multi-view 6D feedback can reduce this residual while the object remains
   observable.
2. **Contact uncertainty after impact.**  Occlusion and unmodeled contact make
   visual pose alone insufficient.  Joint-history/proprioceptive evidence can
   support local search, insertion, and recovery.

The short-term experiments locate the boundary between these regimes.  The
long-term method should adapt across that boundary instead of assuming in
advance that one sensing modality is uniformly superior.

Precision is not defined only by insertion clearance.  It also includes
moving through a narrow feasible set without disturbing neighbouring objects.
The long-term common abstraction is therefore **precision manipulation under
geometric and contact constraints**: the robot must infer which object or
surface is active, while the task specification declares whether that contact
is required, allowed, or forbidden.

## 2. What MV-GoTrack means in this program

The method referred to in the meeting as the Hyunwoo/Geonhee framework is
**MV-GoTrack**.  It is the required short-term modular baseline and the visual
observation component of a possible later hybrid method.  Connecting it to
the robot is necessary engineering, but is not by itself the paper novelty.

For the modular baseline, MV-GoTrack must:

1. receive the once-localized object pose as its initial pose;
2. track continuously before pickup and through pickup, lift, transfer, and
   the pre-contact hold;
3. produce a symmetry-reduced key pose relative to the target task frame;
4. drive explicit metric pose correction while visual evidence is available;
5. expose confidence/consistency evidence for analysis, without silently
   adding task-success gates.

The existing implementation and deployment contract are documented in
`MVGOTRACK_CAPTURE4.md`.

## 3. Stages that must not be conflated

### 3.1 Short-term goal: diagnostic baselines

The immediate goal is to implement and evaluate two deliberately vanilla
approaches under one frozen protocol:

#### A. Explicit modular baseline

- initial object and fixture localization;
- continuous MV-GoTrack 6D feedback;
- explicit relative-pose target;
- ordinary motion planning/control;
- no learned contact recovery in the first comparison;
- no trial-adaptive bias update in the first comparison.

#### B. Direct VLA baseline

- teleoperated demonstrations of the same physical tasks;
- the same observations available at evaluation;
- action chunks or delta end-effector actions from images, instruction, and
  proprioception;
- no privileged CAD pose supplied only to the VLA;
- no modular precision correction hidden behind the VLA label.

The objective is not to declare a winner.  It is to produce a phase-resolved
failure map that identifies what remains unsolved.

### 3.2 Intermediate goal: undergraduate thesis

After the diagnostic baselines, the undergraduate thesis should address the
phase boundary in the existing cylinder system rather than claim geometry or
task generalization.  The preferred working topic is:

> **Phase-Structured Visual and Proprioceptive Correction for Cylindrical
> Precision Insertion**

The thesis compares nominal open loop, MV-GoTrack pre-contact correction, a
frozen proprioceptive contact policy, and their fixed handoff under one
physical outcome protocol.  The modules remain interpretable and the contact
policy receives no visual pose.  A persistent XY bias is only an auxiliary
ablation if repeated-trial evidence shows a stable component.

The cylinder is an acceptable thesis testbed only if the claim concerns
**phase-dependent observability and correction**, not category-level shape
generalization.  The minimum thesis questions are:

- **RQ-T1 (visual value):** does continuous multi-view 6D correction improve
  physical outcome over nominal open loop as offset and clearance become
  harder?
- **RQ-T2 (contact value):** from the same registered contact-start
  distribution, does a frozen proprioceptive policy improve outcome over a
  fixed guarded stroke?
- **RQ-T3 (handoff value):** does visual correction followed by the same
  proprioceptive policy outperform either single-modality condition?
- **RQ-T4 (observability boundary):** which failures are resolved before
  contact and which remain after contact under occlusion and mechanical
  ambiguity?

The preferred thesis method is therefore a **fixed-handoff controller**, not
a full learned multimodal router.  MV-GoTrack supplies free-space feedback;
after a shared guarded contact approach, a frozen recurrent proprioceptive
policy owns local correction.  Simulator-only privileged pose/contact labels
are never actor inputs at deployment.  If the hybrid does not beat the two
single-modality conditions, the thesis reports that empirical boundary rather
than an unsupported fusion claim.

The thesis must vary at least clearance, target pose, object-in-gripper
transform, and contact-stage visibility.  A result from one nominal upright
cylinder pose is a system demonstration, not a thesis result.

The authoritative four-condition protocol, outcome definitions, evidence
audit and completion gates are in
`UNDERGRADUATE_THESIS_PROTOCOL_20261127.md`.

### 3.3 Long-term goal: RSS-targeted method

The RSS-scale research topic is:

> **A geometry-conditioned physical belief updated by visual tracking and
> proprioceptive evidence, paired with a task constraint graph that transfers
> across unseen object/environment geometries and different contact
> semantics.**

Working paper title:

> *From Sight to Contact: Geometry-Conditioned Belief Adaptation for
> Generalizable Precision Manipulation*

The VLA belongs at the task/subgoal/skill-selection level unless the direct
VLA baseline demonstrates that it can reliably own metric contact control.

This long-term method is **not** the cylinder thesis with more experiments.
The cylinder and its `D_inf` symmetry are a fast, clean diagnostic setting for
the undergraduate thesis.  The RSS method must train across diverse part and
fixture geometries and evaluate on held-out shapes, in the spirit of *Blind
Precise Assembly*.  Symmetry handling is an implementation requirement for
shapes that have symmetry, not the source of generalization or the headline
novelty.

The current clean-room hypothesis, generalization hierarchy and RSS evidence
bar are specified in `RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`.  The earlier
`RSS_RESEARCH_DIRECTION_20261126.md` and
`RSS_BENCHMARK_PROTOCOL_20261126.md` remain useful detailed precursors, but
their mating/narrow-passage split is an evaluation basis rather than the
method definition.

The primary RSS generalization contract is
`known_geometry_zero_shot`: test object/environment geometry and layout are
absent from policy training, but their CAD meshes or calibrated scans are
available to the geometry encoder and pose front end; no scene-specific
policy fine-tuning is allowed.
`unknown_geometry_zero_shot`, where geometry must be inferred only from live
images, is a separate stretch setting and must not be conflated with the
primary result.

### 3.4 Clean-room RSS apparatus

The RSS experiment should be rebuilt to isolate the scientific question.  It
does not need to inherit the undergraduate system's protruding socket,
tabletop grasp, Inspire/Robotiq fingers, or cylinder-only assets.

Recommended primary apparatus:

1. rigidly attach the interchangeable key/part to the Franka flange through a
   repeatable, surveyed mount;
2. use a flat fixture plate with the mating cavity recessed into the plate,
   matching the contact structure of *Blind Precise Assembly* more closely;
3. exchange only a small printed or machined key insert and fixture insert so
   many geometries share one mechanical datum;
4. add a separate interchangeable dense-row base whose neighbour geometry,
   gap and layout can vary under the same surveyed-datum principle;
5. place the fixture or dense scene in several poses within a calibrated workspace;
6. use multi-view RGB for free-space relative pose and ordinary Franka joint
   history for contact evidence;
7. begin every episode from a declared free-space or contact-start state and
   report these protocols separately.

This removes grasp success, in-hand slip, finger collision, and protruding
socket-wall collision from the primary experiment.  Those are valuable later
system tests, but they obscure the initial claim about global visual alignment
and local contact correction.  A secondary end-to-end experiment may restore
a gripper only after the fixed-mount claim is established.

### 3.5 RSS research questions

- **RQ-R1:** can one geometry-conditioned filter infer object pose, active
  contact identity and contact mode for held-out object/environment geometries
  and layouts from multi-view RGB and robot proprioception?
- **RQ-R2:** does controlling from a calibrated distribution improve success
  and recovery over a point estimate or implicit feature concatenation?
- **RQ-R3:** does evidence-dependent visual/proprioceptive updating outperform
  vision-only, proprioception-only, fixed handoff, and monolithic fusion on
  the same held-out geometry distribution?
- **RQ-R4:** can one physical observation model transfer across insertion and
  constrained passage while a task graph changes whether target/environment
  contact is useful or forbidden, without task-specific estimator heads?

Across-trial persistent bias is an optional extension.  It enters the RSS
claim only if repeated trials show a stable component after conditioning on
geometry, fixture pose, and the rigid flange-to-part mount.

## 4. Short-term task set

The meeting proposed two task families.  Their exact physical variants must
be frozen before collecting comparison data.

### Task A: key--socket insertion

- Development geometry: symmetric cylindrical key/socket.
- Required diagnostic extension: at least one keyed or non-circular pair.
- Initial study: fixed key-to-gripper relation or one repeatable grasp, so the
  study isolates precision rather than grasp selection.
- Difficulty axes: clearance, target pose, object-in-gripper perturbation,
  and contact-stage visibility.

### Task B: constrained passage/placement

- Pringles-like walls or a reproducible printed substitute constrain the
  object path.
- Contact with a forbidden wall constitutes failure when that is the chosen
  task definition.
- The task must be specified as either constrained placement or constrained
  extraction before data collection; the two must not be pooled.
- This task tests precision without relying on matching peg/socket geometry.

Cylinder insertion is a bring-up task, not sufficient final evidence of
generalization.

## 5. Frozen short-term evaluation protocol

### 5.1 Controlled simplification

The first comparison removes unnecessary sources of variance:

- fixed end effector and controller;
- fixed or mechanically repeatable object attachment;
- one calibrated camera configuration;
- one success detector shared by both methods;
- identical target-pose and obstacle distributions;
- no method-specific human intervention after an episode begins.

Grasping is reintroduced only after the precision controller has been
characterized.

### 5.2 Experimental axes

| Axis | Minimum short-term conditions | RSS extension |
| --- | --- | --- |
| Task | insertion, one constrained task | mating plus selective narrow passage |
| Geometry | cylinder, one keyed shape | unseen shape families |
| Clearance | bring-up and one tighter condition | multiple held-out clearances |
| Visibility | full, contact-stage partial occlusion | controlled corruption/occlusion sweep |
| Target pose | several in-workspace poses | held-out workspace regions |
| Object attachment | fixed/repeatable | randomized grasp transform |
| Trials | at least 30 per primary cell | confidence-powered count, multiple seeds |

### 5.3 Primary metrics

1. task success under one physically interpretable definition;
2. success conditioned on reaching pre-contact;
3. time and path length to success;
4. number of retries or contact-search actions;
5. peak interaction signal available from the robot;
6. terminal relative-pose residual when observable;
7. generalization gap from seen to held-out conditions.

For insertion, the primary success event is that the part reaches the defined
seated/insertion state at least once.  A zero-tolerance hold at every
simulation step is not the primary task metric.  Passive stability can be
reported separately.

### 5.4 Phase-resolved failure taxonomy

Every episode must terminate in exactly one primary category:

1. initial localization failure;
2. tracking loss or drift;
3. free-space planning/control failure;
4. pre-contact alignment failure;
5. premature/forbidden contact;
6. contact search or jam failure;
7. partial insertion/placement;
8. task success;
9. success followed by instability, recorded as a secondary diagnostic.

This taxonomy is the key short-term output.  A single end-to-end success rate
cannot distinguish a representation failure from a contact-control failure.

## 6. Short-term hypotheses and decision gates

### H0-M: modular geometric sufficiency

With reliable MV-GoTrack feedback, explicit relative-pose control is
sufficient for the selected task and clearance.

### H0-V: direct VLA sufficiency

A direct VLA trained on the chosen demonstration distribution reaches the
same task-level precision without an explicit pose interface.

### Diagnostic hypotheses

- **H1:** MV-GoTrack performs better before contact than after task-relevant
  occlusion/contact.
- **H2:** a direct VLA handles scene/task variation better than the vanilla
  modular baseline, but loses precision as clearance tightens.
- **H3:** proprioceptive history contains actionable information after contact
  that is absent or unreliable in the visual observation.

### Decision table

| Observed result | Research consequence |
| --- | --- |
| Modular succeeds across all meaningful conditions | Current task is too easy; tighten clearance, visibility, or task diversity before proposing fusion. |
| Modular succeeds pre-contact but fails after contact | Proceed to contact-conditioned residual control. |
| VLA succeeds coarsely but fails at tight contact | Test hierarchical VLA + explicit precision skill. |
| VLA matches modular at tight precision | Focus on data efficiency, held-out task generalization, or recovery; do not force a modular hybrid claim. |
| Both fail before contact | Improve task representation/perception; contact fusion is not yet the bottleneck. |
| Both reach contact and fail differently | This is the strongest evidence for an adaptive multimodal method. |

No learned modality gate should be introduced before this table is populated
with physical evidence.

## 7. Narrowed long-term research gap

A simple visual-to-contact phase switch is not a sufficient novelty.  Recent
work already separates global visual behavior from local high-frequency
force/tactile behavior, and recent VLA work specializes experts by contact
stage.  Proprioception-only policies have also demonstrated assembly from a
contact-start distribution.

The 2026 InsertionWM result closes the broadest version of this gap: combining
raw depth and proprioception in a learned world model already enables
simulation-only zero-shot insertion of unseen, unknown geometries.  The
remaining candidate gap is therefore narrower:

1. a queryable, task-independent geometry-conditioned **contact likelihood**,
   rather than an implicit multimodal policy state;
2. an explicit calibrated posterior over object pose, active contact identity,
   contact mode, and task-relevant neighbour motion;
3. amortization across held-out geometry and layout families, rather than a
   separately constructed contact manifold or online physics simulation for
   each scene;
4. a machine-readable constraint graph that distinguishes required/allowed
   target contact from forbidden neighbour contact;
5. real-robot decision value using ordinary robot joint/action history,
   without fingertip tactile or a wrist force/torque sensor.

The strongest candidate claim is therefore:

> One likelihood model conditioned on object/environment geometry can score
> pose and active-contact hypotheses from ordinary joint/action histories.
> A declarative task constraint graph then turns the calibrated posterior into
> useful-contact versus forbidden-contact decisions across held-out mating and
> narrow-passage scenes, outperforming point, implicit, and specialist
> alternatives.

Across-trial persistent bias remains an undergraduate-system extension.  It
is not part of the primary RSS hypothesis unless later evidence motivates a
separate study.

### 7.1 Why an explicit belief is not sufficient novelty

Several existing lines already cover large parts of a generic "belief for
contact" claim:

| Existing direction | What it already establishes | Boundary for this project |
| --- | --- | --- |
| InsertionWM | An implicit visual-proprioceptive world model can control unseen, unknown-geometry insertion in simulation. | We need explicit calibrated state inference, real transfer, and estimator-level evidence. |
| Particle/GMM belief control under partial observability | Contact-rich manipulation can be controlled by switching policies from a compressed particle belief. | We need a geometry-conditioned belief that transfers across unseen parts using ordinary RGB and robot proprioception. |
| BILBA compliant belief-space planning | Planned contacts can reduce pose uncertainty during low-clearance part mating. | Our method must learn/update from real multimodal observations rather than only construct a model-based contact schedule. |
| BayesContact | A particle pose belief can fuse depth and force/torque contact likelihoods and select informative probes. | Our setting omits a dedicated F/T or tactile sensor, uses continuous multi-view tracking plus joint history, and studies repeated-trial bias. |
| Blind Precise Assembly | Proprioceptive history alone can solve diverse assemblies from a contact-start distribution. | Vision must reduce global uncertainty before contact, while the contact policy must expose when proprioception adds information beyond the last visual residual. |
| RDP/FACTR and related visuotactile policies | Slow visual and fast contact feedback, or learned modality fusion, improves contact-rich behavior. | Concatenation or a fixed slow/fast split is not enough; the contribution must be an explicit calibrated task belief and its update/ablation. |
| CaPTURe | Contact-dependent, possibly multimodal uncertainty sets can be calibrated across free/contact motion. | We must estimate the current actionable relative pose from real observations and demonstrate control value on held-out geometry. |
| Contact-manifold sampling | Full-SE(3) pose can be estimated for complex low-clearance parts from contact primitives. | We need one amortized model that transfers to held-out geometry without an offline reference manifold per test part. |

Consequently, the RSS claim cannot be "we combine vision and
proprioception."  It must show that the proposed likelihood/posterior is better
calibrated and more actionable than an implicit recurrent state, transfers to
held-out geometry families, and avoids per-test-object contact fitting.

### 7.2 Cylinder instantiation for the undergraduate thesis

For the cylindrical task, axial yaw is unobservable and task-irrelevant.  A
practical continuous state is therefore

```text
xi_t = [radial_x, radial_y, axial_depth, tilt_x, tilt_y],
```

with a discrete contact mode such as

```text
c_t in {free, rim_contact, single_side_contact, wedged, inserted}.
```

The implementation should maintain a distribution over `xi_t` and `c_t`, not
only a point feature.  A minimal version may use a Gaussian mixture for the
continuous residual and categorical probabilities for contact mode; a
particle representation is required when contact produces clearly
multimodal hypotheses.  Its outputs must be evaluated by:

1. residual prediction error against simulator privilege or observable
   real-world geometry;
2. contact-mode calibration and negative log likelihood/Brier score;
3. uncertainty coverage before and after contact;
4. task success when the controller receives the mean only versus the full
   belief;
5. recovery choice and information gain, if active probing is added.

The across-trial state `theta_n` must have its own process model and update
rate.  A hierarchical prior such as

```text
theta_(n+1) = theta_n + epsilon_persistent
xi_(n,t+1)  = f(xi_(n,t), c_(n,t), u_(n,t)) + epsilon_transient
```

prevents a one-off jam or slip from being accumulated as calibration bias.
An ablation that replaces this decomposition with one cumulative XY offset is
mandatory.

This five-dimensional residual is deliberately **not** the RSS
representation.  It is a thesis-scale instrument for establishing whether a
contact-conditioned belief is useful at all.

### 7.3 Geometry- and constraint-general belief required for RSS

Let the manipulated-object and local-environment geometry be represented by
`G_O` and `G_E`, obtained from CAD, point clouds, SDF samples, or an equivalent
geometric observation.  A shared encoder produces a physical geometry context

```text
h_G = Enc_geometry(G_O, G_E).
```

A separate task constraint graph `C` names the target and goal region,
required/allowed/forbidden contact relations, disturbance tolerances, and
terminal predicates.  Keeping `C` separate matters: task semantics should not
change the physical likelihood of the same contact observation.

The transient state cannot be a fixed cylinder-specific `[x, y, depth,
tilt_x, tilt_y]` vector.  It must contain a relative rigid transform together
with a geometry-conditioned contact representation:

```text
xi_t = (T_SK,t, d_contact,t, m_t),
```

where `d_contact,t` may be a learned contact field, a sparse set of candidate
contact features, or another representation whose dimensionality does not
depend on one particular mesh.  `m_t` is a geometry-agnostic contact-mode
distribution such as free, one-sided contact, multi-contact/jam, and task
completion.  For a symmetric part, equivalent poses may be marginalized or
canonicalized; for an asymmetric part, the full task-relevant pose remains.

The estimator and controller are then conditioned on `h_G`:

```text
b_t = p(xi_t | z^vision_0:t, z^prop_0:t, u_0:t-1, h_G)
u_t ~ pi(b_t, h_G, C).
```

To justify generalization, the training generator must vary at least:

- cross-section and mating-surface geometry;
- clearance and insertion/contact depth;
- part and fixture scale within the robot's usable range;
- grasp transform and target pose;
- visual appearance/occlusion independently of physical geometry.

Evaluation must reserve entire geometry and layout families, not only new
poses of a seen cylinder or a new random Pringles spacing.  At minimum, report
seen-shape, held-out instance, held-out family, and held-out layout results.
Cylinder-only training followed by a keyed cylinder test, or a single
successful extraction row, is not evidence of the intended RSS claim.

## 8. Proposed physical belief and task constraints

Let `W`, `S`, `K`, and `E` denote world, socket/constraint, key/object, and end
effector.  The undergraduate cylinder instantiation may define a
symmetry-reduced pose residual as

```text
xi_t = argmin_{g in G_K} || Log( inverse(T_WS) T_WK g ) ||_Q.
```

but the RSS physical belief is scene-geometry conditioned:

```text
b_t = p(T_WO,t, a_contact,t, m_t, delta_E,t
        | z^vision_0:t, z^prop_0:t, u_0:t-1, h_G),
```

where:

- `T_WO,t` is the manipulated-object pose;
- `a_contact,t` identifies the active environment object/surface;
- `m_t` is the contact-mode distribution;
- `delta_E,t` represents task-relevant neighbouring-object motion;
- `h_G` describes current object/environment geometry;
- `z^vision` is the MV-GoTrack/multi-view observation;
- `z^prop` is joint position, velocity, end-effector kinematics, and action
  history available to the proprioceptive policy.

The controller separately evaluates this posterior under `C`.  Thus rim
contact may be informative and useful for mating while the same evidence of
contact with a neighbouring container incurs forbidden-contact risk in dense
extraction.  Physics inference is shared; task semantics are not baked into
the likelihood.

The method must work when the object has no useful symmetry.  When symmetry
is present, it must not average incompatible SE(3) modes or invent an
unobservable axial yaw; equivalent hypotheses are marginalized or reduced as
a special case.  Training, evaluation, and control must use the same geometry
and equivalence contracts.

### Within-trial update

- vision-dominant free-space state estimation;
- mixed evidence near first contact;
- proprioception-dominant local correction when visual observability falls;
- vision reacquisition after backoff when the task exposes the object again.

### Across-trial update

Across-trial memory is outside the primary RSS model.  It remains a separate
system extension for the undergraduate pipeline if repeated evidence reveals
a stable mounting, fixture, or controller bias.  Random slip and one-off
contact disturbances must never be accumulated as calibration state.

## 9. Role of the VLA

### Short-term

The direct VLA is an honest independent baseline.  It receives no hidden
modular correction unavailable to the declared model.

### RSS extension

The preferred hierarchical role is:

```text
instruction + scene
    -> VLA compiles target, goal and allowed/forbidden contacts into C
    -> task-frame goal, constraints and primitive vocabulary
    -> metric visual alignment
    -> local proprioceptive/contact skill
    -> verifier and retry/recovery choice.
```

The scientific question is not whether a VLA can call a handcrafted skill.
It is whether a task-conditioned high-level policy can generalize semantic
task choice while an explicit adaptive belief preserves metric precision.

## 10. Required baselines and ablations

### Minimum baselines

1. open-loop nominal motion;
2. MV-GoTrack visual feedback only;
3. proprioception-only policy from the same contact-start distribution;
4. fixed visual-to-proprioceptive handoff;
5. direct VLA;
6. proposed task-frame belief/adaptation method.

### Core ablations

- without geometry conditioning;
- point estimate versus calibrated distribution;
- generic full SE(3) versus the correct task equivalence for symmetric test
  shapes;
- without visual updates after initialization;
- without proprioceptive history;
- fixed handoff instead of evidence-dependent update;
- geometry-specific contact manifold versus amortized likelihood;
- implicit recurrent state versus explicit likelihood/posterior;
- direct VLA versus hierarchical VLA plus local precision skill.

Every learned policy must be compared on matched training data or explicitly
reported data budgets.  A VLA trained on substantially more demonstrations
cannot be used as an unqualified architecture comparison.

## 11. Undergraduate-thesis completion criteria

The undergraduate thesis is complete when the following are supported by
physical evidence:

1. both short-term vanilla baselines run on the same frozen protocol;
2. the phase-resolved failure map identifies a reproducible post-contact or
   partial-observability gap;
3. one contact-conditioned residual method addresses that gap;
4. the result includes cylinder bring-up and one non-axisymmetric or
   constrained condition;
5. at least the five minimum comparisons relevant to the selected thesis
   claim are reported with confidence intervals;
6. videos, raw episode records, labels, and configuration identities make
   every counted trial auditable.

If no post-contact gap is observed, the thesis must change its claim rather
than add a contact module without evidence.

## 12. RSS expansion criteria

An RSS submission requires more than a successful integrated demo:

1. an amortized geometry-conditioned contact likelihood and calibrated belief;
2. estimator-level evidence on held-out geometry **families**;
3. generalization across clearance, pose, and visibility;
4. one mating/engagement benchmark and one selective narrow-passage benchmark
   with opposite contact semantics;
5. physical comparisons to direct VLA, fixed modular handoff, and
   proprioception-only contact control;
6. sufficient trials and independent seeds for confidence intervals;
7. analysis of when each modality helps or hurts;
8. real-robot evidence that the belief changes actions and success, rather
   than only explaining past logs.

## 13. Falsification and pivot conditions

The following outcomes would falsify or narrow the proposed method:

- If repeated residuals have no stable component, keep accumulated trial bias
  out of both the method and the claim.
- If proprioceptive history cannot predict actionable contact modes above a
  current-state baseline, active probing or an additional contact sensor may
  be required.
- If MV-GoTrack remains accurate through contact and vision-only control
  solves tight held-out tasks, the primary gap is not modality handoff.
- If direct VLA solves tight unseen tasks with the same data budget, study why
  explicit geometry is unnecessary or move the contribution to data/online
  adaptation.
- If performance gains occur only on the cylinder, do not claim generalizable
  precision manipulation.
- If insertion and narrow passage need separate observation models, describe
  them as task-specific systems rather than a general contact-constraint
  belief.

## 14. Immediate authoritative deliverables

Before adding a fusion architecture, produce:

1. a frozen specification for the two short-term task variants;
2. a shared episode schema and physical success definition;
3. MV-GoTrack modular baseline results;
4. direct VLA baseline results under the same evaluation distribution;
5. a phase-resolved comparison table and representative failure videos;
6. a measured test of whether pre-contact residuals contain a persistent
   across-trial component;
7. a go/no-go decision for contact residual control and trial memory using the
   decision table in Section 6.

## 15. Closest-work boundary

The final related-work audit must cover at least:

- Blind Precise Assembly: proprioception-only contact-start assembly;
- Force Policy: global visual and local force control;
- Reactive Diffusion Policy: slow visual and fast tactile hierarchy;
- DEFT-VLA/ForceVLA: contact-stage force specialization in VLAs;
- AutoMate: diverse-geometry specialist/generalist assembly;
- visual verification and inference-time steering;
- task-frame/contact-state estimation and trial-adaptive assembly.

Primary references used to establish the present boundary:

- Blind Precise Assembly: <https://arxiv.org/abs/2610.23849>
- Force Policy, RSS 2026:
  <https://roboticsproceedings.org/rss22/p128.html>
- Reactive Diffusion Policy, RSS 2025:
  <https://roboticsproceedings.org/rss21/p052.html>
- DEFT-VLA: <https://arxiv.org/abs/2610.14854>
- AutoMate, RSS 2024:
  <https://roboticsproceedings.org/rss20/p064.html>
- Visual Verification, RSS 2026:
  <https://roboticsproceedings.org/rss22/p079.html>
- pi-0.5, CoRL 2025:
  <https://proceedings.mlr.press/v305/black25a.html>

This document states a research hypothesis and evaluation contract.  It does
not claim that the hybrid method has already been implemented or validated.
