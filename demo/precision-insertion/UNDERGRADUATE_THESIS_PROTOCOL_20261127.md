# Undergraduate thesis protocol: phase-structured cylinder insertion

Date: 2026-11-27

Status: authoritative prospective protocol; pilot evidence is not a result

Relation to long-term work: the cylinder, its axial symmetry and the current
hardware are an intentionally narrow thesis setting.  The RSS claim is
separate and governed by `RSS_ZERO_SHOT_PRECISION_LAYER_20261127.md`.

## 1. Thesis question and contribution boundary

Working title:

> **Phase-Structured Visual and Proprioceptive Correction for Cylindrical
> Precision Insertion**

The thesis asks whether the two observable phases should be handled by
different, explicitly connected modules:

1. continuous multi-view 6D tracking reduces free-space key/socket residual
   before contact;
2. a proprioceptive policy corrects the remaining uncertainty after contact;
3. a fixed, method-independent handoff combines the two without training a
   monolithic policy.

The contribution is not a general manipulation policy, a new VLA, or shape
generalization.  It is a controlled cylinder study of the vision-to-contact
handoff and its task-level value.  VLA task reasoning remains a separate
diagnostic baseline and long-term module.

## 2. Research questions

- **RQ-T1 -- visual phase:** does continuous MV-GoTrack feedback improve
  released-inside-socket success over nominal open loop as injected
  pre-contact offset and clearance become harder?
- **RQ-T2 -- contact phase:** from the same registered contact-start
  distribution, does the frozen proprioceptive policy improve success over a
  fixed guarded stroke?
- **RQ-T3 -- handoff:** does MV-GoTrack followed by the same proprioceptive
  policy outperform either vision-only or proprioception-only execution at a
  matched trial and action budget?
- **RQ-T4 -- observability boundary:** which failures are explained by visible
  pre-contact pose residual, and which remain after contact where vision is
  occluded or mechanically ambiguous?

Corresponding hypotheses must be frozen before final trials.  Internal
tracker residual is not independent ground truth, so RQ-T1 is decided by the
physical outcome and evaluation-only evidence, not by the tracker claiming
that its own residual decreased.

## 3. Four primary conditions

All conditions reuse one physical key/socket pair, frozen socket session,
end effector, key grasp/attachment contract, planner, guarded approach and
outcome definition.

| ID | Pre-contact phase | Contact phase | Purpose |
| --- | --- | --- | --- |
| `O` | nominal trajectory, no key feedback | fixed guarded 20 mm stroke | open-loop baseline |
| `V` | continuous MV-GoTrack plus symmetry-reduced XY/tilt correction | fixed guarded 20 mm stroke | visual correction value |
| `P` | nominal trajectory, no key feedback | frozen proprioceptive policy | contact correction value |
| `VP` | continuous MV-GoTrack correction | the same frozen proprioceptive policy | fixed-handoff method |

The legacy SAM half-ellipse endpoint route may be reported as an engineering
baseline but is not substituted for `V`.  A direct VLA may be evaluated with
the same outcome protocol, but it is not required to establish RQ-T1--T4 and
must not secretly call the modular correction stack.

## 4. Handoff contract

The comparison is invalid if a human establishes contact only for `P` or
`VP`.  Every condition uses the same deterministic sequence:

1. stop at a registered pre-contact plane in the measured socket frame;
2. save the last pre-contact image/tracker and robot state;
3. execute one shared low-speed guarded approach to the declared contact
   event;
4. for `O` and `V`, continue the fixed guarded stroke;
5. for `P` and `VP`, initialize the frozen history buffer from the measured
   robot state and give control to the proprioceptive policy;
6. hold at the task endpoint for the frozen verification duration;
7. release, move to clear view, and collect the same final outcome evidence.

The policy never receives the socket pose, tracker output, method ID or
privileged contact label.  Vision only changes the physical state from which
the shared contact approach begins.

## 5. End-effector and object-in-hand contract

Use one end effector for all four conditions.  The current ZeroDex runner
defaults to the Franka gripper, while the blind-policy simulator and prepared
grasp utility describe a Robotiq 2F-85.  These are not yet a matched system.

Before comparison, record the physical `T_tool_key` and verify that it lies
inside the contact policy's declared key-to-tool randomization support.  Also
verify that the actual gripper geometry remains clear of the fixture over the
contact workspace.  If those checks fail, rebuild/retrain the simulation with
the selected physical end effector; do not call a shape-compatible checkpoint
equivalent.  The policy observation does not contain finger state, but its
key transform and collision model still determine the physical problem.

For the primary experiment, use one repeatable grasp or a rigid key fixture.
Natural tabletop pickup can be added as an end-to-end secondary condition,
but must not be mixed into the controlled handoff comparison.

## 6. Difficulty and trial randomization

Difficulty is defined by measured radial clearance, injected pre-contact
translation/tilt and contact-stage visibility.  Start with the existing 5 mm
condition for commissioning, then include at least one tighter measured
fixture in final evidence.  Never pool clearances.

The final offset distribution is chosen from a pilot and then frozen.  It
must contain offsets on both sides of the open-loop feasibility boundary,
random directions, and declared tilt levels.  Use blocked randomization over
condition, clearance and offset bin so time, calibration drift and hardware
temperature cannot become method labels.  Store the randomization seed and
execute the preregistered order without optional stopping.

## 7. Outcomes and labels

### Primary outcome

`released_inside_socket`: after the commanded contact phase, release the key,
move the robot to clear view, and have a method-blind human label paired
initial/final multi-view images.  Save the independent VLM label but do not
use it as ground truth until its confusion matrix is supported.

### Secondary outcomes

- `tip_entry_success`: the key tip entered during controlled descent before
  release, judged from synchronized video;
- endpoint reached and maintained for the frozen hold duration;
- force/contact stop or jam;
- time, path length and number/magnitude of visual corrections;
- proprioceptive action count and policy latency;
- grasp/lift/transfer funnel outcomes for the secondary end-to-end study.

Robot travel or a policy `success` bit alone is not proof of physical
insertion.  Conversely, release-assisted entry is a valid final task success
but must be separated from smooth controlled entry.

## 8. Independent evaluation evidence

At minimum, preserve all raw camera streams, robot state/action history,
tracker messages, guarded/contact-controller events and final clear-view
frames.  Use an evaluation-only view or manual geometric annotation for
pre-contact pose analysis where possible.  If no independent pose reference
is installed, report tracker residual only as a process diagnostic and make
task success the primary endpoint.

Annotators receive randomized episode IDs and no method name.  Report label
coverage, disagreements and exclusions.  A missing label is neither success
nor a measured physical failure.

## 9. Statistical plan

Run a commissioning pilot only to estimate base success rates, variance,
runtime and invalid-trial frequency.  Use those estimates and a declared
minimum meaningful success-rate difference to calculate the final sample
size before unblinding the final split.  Freeze a maximum hardware budget and
do not stop early when a favorable rate appears.

For the final analysis:

- report counts and Wilson confidence intervals for every funnel denominator;
- fit a preregistered binomial model with method, clearance, injected XY
  magnitude and tilt as covariates;
- report `V-O`, `P-O`, `VP-V` and `VP-P` contrasts with confidence intervals;
- plot success against normalized offset and physical clearance;
- bootstrap continuous latency/action/path metrics by episode;
- show every attempted valid episode and every invalidation reason.

The strongest thesis conclusion requires `VP` to improve both single-modality
conditions.  If it does not, report the measured observability boundary rather
than retrofitting a new hypothesis.

## 10. Current evidence audit

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Complete legacy visual episode | One 2026-10-30 endpoint-based pickup-to-release success and a second human-confirmed final insertion exist. | Pilot evidence only |
| Controlled visual ablation | Existing 16-start corpus mixes sessions, policies and candidates; only two final positive labels exist. | Missing |
| Independent pre-contact key pose | Largest repeated-session group has four wrist records but no valid multi-view metric key tip. | Missing |
| Continuous MV-GoTrack path | Capture4 service, transport and runner integration exist. It has not initialized from a physical same-session key pose and run pickup-to-insertion. | Software ready; physical evidence missing |
| Static ZeroDex preflight | Board-free support plane, four-camera calibration, gap-5 Franka-gripper catalog and capture4 MV-GoTrack endpoint pass the 2026-11-27 static preflight. Capture4 CUDA and the GoTrack checkpoint are readable. | Passed; not a physical tracking result |
| Proprioceptive simulation | Gap-5 stable-seated-v2 development gate passed; gap-3 training remains active and gap-1 is locked. | Gap-5 simulation complete |
| Proprioceptive real rollout | Shadow/transport and operator-grasp software exist; no policy-controlled real insertion has run. | Missing |
| End-effector match | Vision runtime uses Franka gripper by default; proprioceptive assets declare Robotiq 2F-85. | Bridge contract missing |
| Fixed vision-to-contact handoff | No shared autonomous contact-start transition has been physically commissioned. | Missing |
| Four-condition final collection | No preregistered `O/V/P/VP` physical dataset exists. | Missing |
| Direct VLA baseline | Current Qwen/Gemini code labels images; it is not a robot action policy. | Optional baseline missing |

### 10.1 Machine-readable U1 audit

The visual-commissioning claim is now checked by the read-only auditor below.
It does not run in the robot action loop and introduces no motion stop or
failure condition.

```bash
python3 scripts/precision_layer/audit_undergraduate_evidence.py \
  /mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4 \
  --output /tmp/undergraduate_evidence_audit.json
```

For one run to pass U1, the *same episode* must preserve the accepted
FoundPose initializer, first native-valid MV-GoTrack pose, continuous tracker
pose stream, episode video and an offline continuity audit covering pickup,
lift, transfer and the stopped pre-contact hold.  The continuity-audit format
is frozen in `config/mvgotrack_continuity_audit.schema.json`.  File presence
alone is only an artifact bundle, not a tracking-success claim.

The 2026-11-27 NAS audit found 20 launch/recording receipts.  Five receipts
reported `release_and_clear_completed`, while only one corresponding video
was present in the canonical `2026/MM/DD/RUN_ID` tree.  None contained the
initializer, first pose, pose stream or continuity audit, so the U1 completion
count is zero.  This result does not invalidate the legacy demonstration
videos; it says they cannot support the newer continuous-tracking claim.

Future MV-GoTrack runs now write
`mvgotrack/mvgotrack_pose_stream.jsonl` asynchronously from the subscriber.
The I/O thread is evidence-only: it neither delays pose consumption nor turns
a journal error into a robot failure.  After a run, join that stream to the
main-PC timing trace without inventing a score or pose-jump threshold:

```bash
python3 scripts/precision_layer/build_mvgotrack_continuity_audit.py \
  RUN_ID \
  RUN_DIR/mvgotrack/mvgotrack_pose_stream.jsonl \
  RUN_DIR.timing/events.jsonl \
  --output RUN_DIR/mvgotrack/mvgotrack_continuity_audit.json
```

The compiler uses `received_unix_ns` from the main PC and the trace's UTC
motion spans.  A pass means at least one native-valid pose was observed in
each of pickup, lift, transfer and stopped pre-contact alignment, with one
calibration identity and strictly increasing tracker sequence numbers.  It
does not use tracker score, workspace, pose-jump or inter-frame-gap limits.

## 11. Completion gates

1. **Instrumentation freeze:** one end effector, measured fixture/clearance,
   `T_tool_key`, outcome labels and synchronized logging are fixed.
2. **Visual commissioning:** same-session FoundPose initializes MV-GoTrack and
   continuous physical tracking survives pickup, lift, transfer and the
   pre-contact stop.
3. **Contact commissioning:** a gate-eligible frozen policy completes shadow,
   zero-action and real contact-start rollouts with the chosen hardware.
4. **Handoff commissioning:** the shared guarded approach transfers from both
   nominal and visually corrected pre-contact states without human placement.
5. **Pilot and freeze:** estimate effect/runtime, freeze perturbation range,
   hypotheses, sample size, order and analysis.
6. **Final collection:** execute the complete randomized `O/V/P/VP` matrix and
   label it blind to method.
7. **Claim audit:** regenerate tables/figures from episode records and ensure
   every thesis statement is supported by the frozen final split.

The undergraduate thesis is not yet complete because gates 2--7 lack physical
evidence.  Existing success videos remain valid system demonstrations.

The exact gate-2 software audit and remaining physical evidence contract are
recorded in `MVGOTRACK_COMMISSIONING_STATUS_20261127.md`.

## 12. Separation from the RSS program

The thesis deliberately fixes cylinder geometry, axial symmetry, hardware and
the constraint sequence.  It tests phase-structured sensing/control, not
zero-shot geometry or task composition.  The RSS work replaces this fixed
state with a variable-size active-constraint belief, holds out geometry,
layout, constraint composition and upstream policy, and uses the generic
benchmark contract in `RSS_ZERO_SHOT_PRECISION_BENCHMARK_20261127.md`.
