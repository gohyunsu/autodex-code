# RSS implementation gap and reuse map

Date: 2026-11-26  
Scope: current `autodex-code` and `/home/hyunsu/blind-assembly-reproduction`

This audit distinguishes reusable infrastructure from cylinder/demo-specific
logic.  It does not claim that the RSS method is implemented.

## 1. Current authoritative boundary

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Multi-view visual prior | FoundPose/MV-GoTrack integration and capture4 transport exist in the precision pipeline. | Reusable front end; physical clean-room prior not yet recorded |
| Franka joint-history stream | Precision runtime and blind-assembly real adapter expose robot state/action histories. | Interface exists; synchronized likelihood dataset absent |
| Proprioception-only policy | Cylinder contact-start PPO environment, training and evaluation exist in `blind-assembly-reproduction`. | Cylinder simulation baseline only |
| Diverse geometry simulation | Current environment builds one circular key/socket contract and uses circular tip-edge labels. | Missing |
| Fixed flange part mount | Existing models retain a gripper/held-key body and grasp randomization. | Missing clean primary apparatus/model |
| Recessed interchangeable fixture | Existing precision assets center on protruding socket fixtures. | Missing |
| Geometry-conditioned likelihood-ratio model | No paired-geometry encoder, ratio estimator or particle update was found. | Missing core method |
| Calibrated posterior metrics | Existing scripts report task/contact policy outcomes, not pose-posterior NLL/coverage. | Missing |
| Direct VLA baseline | Existing Qwen/Gemini modules are VLMs, not action policies. | Missing |
| Real held-out geometry evidence | No clean-room cartridges, stage manifest or physical trials exist. | Missing |

## 2. Reuse without changing scientific meaning

### From the precision-insertion repository

- camera calibration and synchronized multi-view capture;
- FoundPose initialization and MV-GoTrack transport;
- transform, SE(3), and per-geometry symmetry utilities;
- Franka control adapters and phase timing;
- episode video/sensor persistence;
- method-neutral episode schema;
- failure/outcome evidence tooling where the physical success definition is
  replaced by the clean fixture contract.

### From `blind-assembly-reproduction`

- FR3 MuJoCo model and position-control interface;
- fixed-history observation and policy plumbing;
- vectorized simulation, PPO and deterministic evaluation machinery;
- simulator domain-randomization and evidence manifests;
- real rollout transport and validation structure;
- stage-gate, seed and curriculum evidence tools.

These components are infrastructure.  Their current cylinder task, rewards,
success criteria and actor observations are not the RSS method.

## 3. Logic that must not be promoted to RSS

### Cylinder-specific simulation logic

`blind_assembly/contact.py` currently contains:

- `circular_tip_points`;
- radial bore tests;
- one circular edge insertion metric;
- three labels: `FREE`, `CONTACT`, `INSIDE`.

`blind_assembly/env.py` currently instantiates `CylinderSocketEnv`, circular
tip samples, socket wall names, a held-key gripper body and cylinder-specific
reset parameters.  Generalizing class names while preserving these mechanics
would not create a geometry-general benchmark.

### Current system-specific logic

- accumulated XY correction;
- cylinder `D_inf` symmetry reduction;
- tabletop grasp candidates and squeeze stability;
- protruding socket collision world;
- VLM final success label;
- key bottom-face heuristic.

These may remain for the undergraduate system but are excluded from the clean
RSS primary experiment.

## 4. New software modules required

### 4.1 Geometry and dataset layer

1. `GeometryPairManifest`
   - part/fixture mesh digests;
   - common datums and seated transform;
   - clearance and insertion depth;
   - generator family and parameters;
   - symmetry/equivalence contract;
   - train/validation/test split.
2. Geometry-family generators for prismatic and genuinely spatial mating
   pairs.
3. Exact or validated collision representation that preserves clearance.
4. Fixed-flange part and recessed-fixture model builder.
5. Privileged simulator recorder for relative pose, contacts, material
   nuisance variables and task outcome.

### 4.2 Observation collection

1. Frozen probe-library executor in fixture coordinates.
2. Synchronized history recorder for `q`, `dq`, command, tracking error and
   actions.
3. Controlled visual-prior particle generator.
4. Actual multi-view-prior adapter using the same particle contract.
5. Candidate-particle and contact-alias sampler with symmetry-equivalence
   handling.

### 4.3 Inference

1. paired part/fixture geometry encoder;
2. temporal joint/action history encoder;
3. neural likelihood-to-evidence ratio estimator;
4. proposal-aware hard-negative sampler and importance correction;
5. particle weight update/resampling;
6. contact-mode auxiliary head;
7. held-out validation calibration;
8. posterior evaluator for rank, NLL, coverage, ECE and entropy reduction.

### 4.4 Control and evidence

1. shared analytical probe/correction controller;
2. posterior-mean and full-posterior action adapters;
3. geometry-independent seating detector contract;
4. research-episode schema adapters for modular, blind, implicit and proposed
   methods;
5. synchronized posterior/action/outcome replay.

## 5. Recommended package boundary

Do not implement the core RSS method inside the current cylinder demo package.
Use a new project or top-level package with dependencies flowing one way:

```text
precision camera/robot adapters ----+
                                    |
blind-assembly simulator utilities -+--> geometry_belief_assembly
                                          |-- geometry
                                          |-- simulation
                                          |-- probes
                                          |-- inference
                                          |-- control
                                          |-- evaluation
```

The new project may import stable adapters, but the urgent cylinder runtime
must not import the experimental RSS method.

## 6. Implementation order and evidence gates

### Milestone 0: benchmark generator

- two prismatic training families;
- one held-out prismatic family;
- one spatial diagnostic family;
- geometry manifest and exact seated/collision validation.

Evidence: mesh/clearance validator, deterministic reset tests and split-leak
audit.

### Milestone 1: identifiability dataset

- fixed-flange simulator;
- standardized probes;
- privileged pose/contact histories;
- candidate particle sets with aliases.

Evidence: the same history/candidate record can be replayed deterministically
and contains no privileged actor feature.

### Milestone 2: point and ratio baselines

- geometry-agnostic point regressor;
- geometry-conditioned point regressor;
- likelihood-ratio posterior.

Evidence: frozen `validation_instance`, `test_instance` and `test_family`
tables with rank, NLL and coverage.

### Milestone 3: control value

- no-update, point and distributional controllers under one probe/action
  budget.

Evidence: matched-seed paired outcomes and failure taxonomy.

### Milestone 4: real apparatus

- surveyed flange cartridge;
- recessed fixture plate and calibrated stage;
- controlled-prior real experiment;
- multi-view-prior complete-system experiment.

Evidence: preregistered held-out physical manifest, synchronized logs and
confidence intervals.

### Milestone 5: full baseline paper package

- Blind-style policy;
- implicit recurrent/world-model policy;
- geometry-specific contact manifold;
- direct VLA system baseline;
- optional gripper robustness experiment.

## 7. Current critical path

The next technical work is **not** more cylinder tracking or VLM prompting.
It is:

1. select or implement a diverse exact-clearance geometry generator;
2. replace the cylinder-only MuJoCo task with a fixed-flange geometry-pair
   interface;
3. generate a deterministic probe/history dataset;
4. establish that relative pose is identifiable on a held-out family before
   implementing belief-conditioned control.

Until Milestone 2 passes, RSS controller and VLA integration are premature.

