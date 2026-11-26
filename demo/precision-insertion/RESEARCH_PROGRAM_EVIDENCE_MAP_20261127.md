# Evidence map: staged generalizable precision-manipulation program

Date: 2026-11-27  
Status: authoritative progress and transition contract

This document connects the undergraduate cylinder study to the separate RSS
program without treating cylinder insertion as evidence of task
generalization.  The scientific unit of the RSS work is a normalized
constraint margin and its calibrated posterior, not a named task.

Authoritative specifications:

- undergraduate protocol: `UNDERGRADUATE_THESIS_PROTOCOL_20261127.md`;
- RSS method hypothesis: `RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`;
- RSS method implementation contract:
  `RSS_ACTIVE_CONSTRAINT_BELIEF_METHOD_20261127.md`;
- RSS benchmark: `RSS_ZERO_SHOT_PRECISION_BENCHMARK_20261127.md`.

## 1. Claim tree

### Undergraduate claim

For one cylindrical key/socket system, determine whether continuous visual
pose correction, proprioceptive contact adaptation and their fixed handoff
resolve different phases of physical uncertainty.

This is a phase-observability claim.  It is not zero-shot geometry, task or
policy generalization.

### RSS claim

For rigid-body manipulation specified by compositional geometric/contact
constraints, determine whether one frozen, policy-agnostic execution layer can
improve precision on held-out geometry, layout, constraint composition and
upstream policy by maintaining a calibrated belief over active constraint
margins.

The RSS claim is not established by adding more cylinder clearances.  It
requires a task-ID-free representation, leakage-free held-out splits and
cross-manifestation real evidence.

## 2. Stage and evidence matrix

| Gate | Required artifact or observation | Evidence that passes the gate | Current status |
| --- | --- | --- | --- |
| U0: instrumentation contract | frozen end effector, fixture, physical clearance, `T_tool_key`, calibration and synchronized logger | digested configuration plus one replayable synchronized episode | incomplete: static camera/robot assets exist; the final end-effector/contact-policy match is not frozen |
| U1: visual commissioning | same-session localization initializes MV-GoTrack before pickup | continuous tracker log and video through pickup, lift, transfer and stopped pre-contact hold | software preflight passed; no physical end-to-end tracker episode yet |
| U2: contact commissioning | frozen proprioceptive policy and matching physical tool/key support | shadow, zero-action and real contact-start rollouts with robot state/action logs | gap-5 simulation passed; no policy-controlled real rollout and current hardware model mismatch remains |
| U3: autonomous handoff | shared guarded approach from both nominal and visually corrected states | no human repositioning; identical handoff event contract in `O`, `V`, `P`, `VP` | missing |
| U4: commissioning pilot | blinded outcome labels and invalid-trial accounting | pilot estimates effect size, runtime and invalid rate without entering the final analysis | missing |
| U5: undergraduate final study | frozen randomized `O/V/P/VP` matrix | preregistered trial count, paired evidence, Wilson intervals and declared contrasts | missing |
| R0: task-agnostic program contract | constraint-program schema and split audit | structural tests plus rejection of Z1--Z4 leakage | implemented and unit-tested; no generated benchmark corpus yet |
| R1: procedural constraint world | generator for geometry, layouts, graph compositions and corrupted nominal trajectories | train/validation/test index whose joint hold-out passes the leakage audit | structural generator and backend-independent margin evaluator exist; physics geometry, trajectories and privileged evidence backend are missing |
| R2: active-constraint identifiability | point-state, raw-history and explicit-belief estimators | calibrated margin/contact predictions on held-out graph compositions | missing |
| R3: decision value of belief | matched controllers and action/time budgets | posterior-aware controller beats point-state MPC and implicit recurrent residual | missing |
| R4: policy-agnostic zero-shot transfer | at least three upstream interfaces with one policy class held out | the same frozen layer improves planner, imitation/diffusion and held-out VLA/policy class | missing |
| R5: real modular benchmark | reproducible manifestations spanning at least four constraint motifs | tolerance sweeps, violation metrology and complete episode evidence without test-task updates | missing |
| R6: semantic compiler | language/scene to constraint-program model | graph-generation error separated from execution error; frozen executor unchanged | optional final stage; missing |

## 3. Dependency order

```text
U0 -> U1 -> U3 -> U4 -> U5
  \-> U2 -/

R0 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6
```

The two tracks may run in parallel after their contracts are frozen, but
evidence cannot be substituted across them.  In particular:

- a successful MV-GoTrack cylinder episode does not pass R2;
- a simulation proprioceptive policy does not pass U2 without matched real
  hardware evidence;
- a VLA that selects a named skill does not pass R6 unless its generated
  constraint program is evaluated independently;
- average task success does not pass R3 without a matched point-state control
  baseline and belief calibration.

## 4. Undergraduate experiment outputs that legitimately inform RSS

The thesis contributes measurements, not generality claims:

1. phase-resolved failure and latency distributions;
2. visual residual versus physical outcome under visibility changes;
3. proprioceptive recovery versus fixed stroke from a matched contact-start
   distribution;
4. cases where visual and contact evidence agree or conflict;
5. synchronized raw records suitable for designing the RSS observation model.

These results determine sensor history length, required update rate, useful
contact labels and the handoff boundary for R1--R2.  They do not determine the
RSS architecture or held-out split.

## 5. RSS hypothesis ladder

Test the cheapest falsifiable hypothesis first.

| Step | Hypothesis | Required comparison | Stop or narrow when |
| --- | --- | --- | --- |
| H1 | constraint margin is a transferable state interface | geometry-conditioned point state versus task-specific state | the shared evaluator needs task IDs or task-specific heads |
| H2 | distributional belief is identifiable | explicit belief versus raw history and posterior mean | held-out calibration/contact identity is no better |
| H3 | uncertainty has decision value | posterior-aware MPC versus point-state MPC | matched point-state control is equivalent |
| H4 | one residual interface transfers across base policies | seen and held-out policy classes | gains occur for only one policy/backbone |
| H5 | composition transfers zero-shot | held-out constraint graph topology and geometry/layout | test-specific training or controllers are required |
| H6 | the simulation result survives physics | preregistered real tolerance sweeps | gains vanish without test-fixture tuning |

Only after H1--H3 pass should a learned residual controller be introduced.
Otherwise a larger policy can conceal an invalid belief claim.

## 6. Shared measurement vocabulary

Every RSS manifestation reports the same quantities, even when its visual
appearance and task semantics differ:

- signed residual `rho_k` and normalized margin `rho_k / tau_k`;
- posterior violation probability `P(rho_k < 0)`;
- active constraint/contact identity;
- nominal and residual Cartesian actions in declared frames;
- measured constraint violation magnitude and duration;
- environment disturbance where applicable;
- success across a physical tolerance sweep;
- residual intervention count, norm, latency and path/time overhead.

This vocabulary prevents the benchmark from becoming a collection of
unrelated task-specific success criteria.

## 7. Immediate next evidence

### Undergraduate track

1. freeze the physical end effector and `T_tool_key` contract;
2. record one same-session FoundPose-to-MV-GoTrack physical sequence through
   the pre-contact hold;
3. commission the matched proprioceptive contact policy;
4. execute a small balanced `O/V/P/VP` pilot and freeze the final trial plan.

### RSS track

1. connect the structural constraint-program generator to physics geometry,
   trajectories and privileged predicate labels;
2. generate a joint Z1--Z4 split and run `python3 -m
   precision_layer.audit INDEX.json`;
3. train no controller yet: first compare point state, raw history and an
   explicit constraint belief on held-out margin/contact inference;
4. proceed to analytical posterior-aware constrained control only after the
   identifiability and calibration gate passes.

## 8. Completion rule

The undergraduate thesis is complete only at U5.  The RSS research program is
not supported by real evidence until R5.  R6 strengthens semantic usability
but is not necessary for the core precision-execution claim.  Status prose,
software readiness and demonstration videos are supporting evidence, not
substitutes for the gate-specific artifacts above.
