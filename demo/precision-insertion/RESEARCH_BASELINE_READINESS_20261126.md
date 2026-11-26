# Research baseline readiness audit

Date: 2026-11-26  
Scope: evidence available on the current workstation  
Research agenda: `RESEARCH_PROGRAM_20261126.md`

This document separates implemented software, simulated evidence, and
physical evidence.  It is not a claim that any baseline comparison has
already been completed.

## 1. Executive status

| Baseline or requirement | Code/assets | Current evidence | Readiness |
| --- | --- | --- | --- |
| MV-GoTrack modular baseline | Precision-insertion state machine, FoundPose initializer, capture4 deployment, continuous tracker transport, symmetry reduction, XY/tilt correction and insertion handoff exist. | capture4 environment and four-camera streamer were validated.  The tracker has not yet been initialized from a same-session physical key pose and run through a complete physical episode in this setup. | **software integrated; physical baseline pending** |
| Proprioception-only contact baseline | Separate `blind-assembly-reproduction` project contains FR3/Robotiq MuJoCo task, PPO, deterministic evaluator, real-client contract and evidence tools. | The 5 mm simulation stage passed the amended three-seed development gate at 100/100, 97/100 and 100/100.  The 3 mm stage is active, 1 mm is locked, and no real robot rollout is claimed. | **5 mm simulation baseline ready; curriculum and hardware pending** |
| Direct VLA baseline | No OpenVLA/pi0/Octo/direct action-model source, checkpoint, demonstration dataset or execution adapter was found in the relevant local projects.  Existing Qwen/Gemini code is a read-only VLM checker/grounder, not a VLA policy. | none | **not implemented** |
| Task A: cylindrical insertion | Key/socket geometry, calibration/runtime assets, modular execution and proprioceptive simulation task exist. | Existing results were produced under different start-state and success protocols. | **hardware available; common protocol not yet run** |
| Task B: constrained passage/placement | No frozen physical geometry, direction (placement versus extraction), target distribution or success detector was found. | none | **not specified** |
| Shared episode record | Existing precision runtime writes detailed implementation-specific attempt records; the proprioception project writes its own evaluator reports. | No method-neutral record existed.  `config/research_episode.schema.json` now defines one without changing either runtime. | **schema specified; adapters pending** |

## 2. Evidence boundary by method

### 2.1 MV-GoTrack modular

Authoritative implementation evidence:

- `MVGOTRACK_CAPTURE4.md` defines initialization, continuous tracking,
  symmetry reduction and correction behavior.
- `precision_insertion/mvgotrack_runtime.py` connects tracker messages to the
  held-key correction path.
- `capture4/mvgotrack_stream.py` and `capture4/manage_mvgotrack.py` isolate the
  capture-PC process from shared camera services.
- `config/experiments/mvgotrack_axis_alignment_capture4.env.sh` is the
  experiment profile.

The baseline is not complete until one physical episode preserves all of:

1. accepted initial key pose;
2. tracker initialization from that pose;
3. uninterrupted pickup-to-pre-contact pose stream;
4. symmetry-reduced residual at the pre-contact hold;
5. commanded metric correction and insertion outcome;
6. phase timing and video evidence.

Tracker confidence is an analysis variable, not a newly invented task-success
gate.  The current integration intentionally adds no custom score, pose-jump,
workspace or age threshold beyond native validity and finite SE(3).

### 2.2 Proprioception-only contact policy

Authoritative evidence lives outside this repository at
`/home/hyunsu/blind-assembly-reproduction`.

The actor uses 30 samples of joint position, joint velocity, gripper
pose/twist and previous action.  It excludes vision, force/torque, tactile,
object pose and fixture pose.  Its current result is a **contact-start
simulation baseline**, not an end-to-end pick/transfer baseline.

The 5 mm policy can enter the common comparison only if the modular and VLA
methods are evaluated from the same declared contact-start distribution, or
if pre-contact success and post-contact success are reported separately.  It
must not receive credit or blame for localization, grasping or transfer it was
not designed to perform.

### 2.3 Direct VLA

The current Qwen/Gemini modules answer visual questions or provide grounding.
They do not produce robot action chunks and therefore cannot be relabeled as a
VLA baseline.

A minimum direct VLA baseline still requires:

1. a selected policy family and frozen pretrained checkpoint;
2. teleoperation demonstrations for both frozen tasks;
3. declared image/proprioception/instruction inputs;
4. delta end-effector or joint action semantics and control rate;
5. one execution adapter using the same robot controller as the modular
   comparison where possible;
6. a training-data manifest and exact demonstration count;
7. evaluation with no hidden CAD pose or modular correction.

Model selection should be made after measuring available demonstrations and
inference latency.  Naming a large pretrained model before the data and action
interface are fixed would not make the comparison runnable.

## 3. Protocol decisions still required

These are experimental definitions, not implementation failures:

1. Freeze Task B as either constrained placement or constrained extraction.
2. Decide whether the first comparison begins from a fixed attachment,
   completed grasp, or tabletop object.  The meeting favors removing grasp
   variance for the first precision comparison.
3. Freeze the physical success event shared by all methods.
4. Choose primary clearance values and held-out target poses.
5. Define the matched observation set for the direct VLA.
6. Declare equal demonstration/interaction budgets or report them explicitly.

## 4. Recommended execution order

1. Run Task A with fixed/repeatable attachment and the vanilla MV-GoTrack
   modular controller.
2. Replay the same contact-start distribution with the 5 mm proprioception
   policy to isolate contact-stage value.
3. Freeze Task B geometry and success detector before collecting VLA data.
4. Collect direct-VLA demonstrations for both tasks and train one unmodified
   baseline.
5. Populate the phase-resolved failure table using the common schema.
6. Only then decide whether the thesis method should add a proprioceptive
   residual estimator, trial memory, or both.

## 5. Go/no-go gates for method development

- **Go to contact-conditioned thesis method:** modular tracking reliably
  reaches pre-contact, but post-contact success remains materially lower and
  proprioceptive history predicts actionable correction/contact mode.
- **Stay with perception/control:** both methods fail before contact or the
  tracked residual is not repeatable.
- **Do not add trial memory:** episode-to-episode residual has no stable
  component after controlling target pose and grasp transform.
- **Reframe VLA contribution:** direct VLA matches tight precision under a
  matched data budget; study data efficiency, recovery or semantic task
  generalization instead of asserting that explicit geometry is necessary.

## 6. Separation between the urgent thesis and the clean RSS study

| Item | Undergraduate thesis | RSS-targeted program |
| --- | --- | --- |
| Purpose | Establish whether contact-conditioned proprioception adds value after tracked visual alignment. | Learn a geometry-conditioned belief and controller that transfer to unseen precision-task geometries. |
| Geometry | Cylindrical key/socket is acceptable; one asymmetric diagnostic is desirable. | Diverse procedural or collected part/fixture families; entire families held out at evaluation. |
| Representation | Cylinder task residual `[radial_x, radial_y, depth, tilt_x, tilt_y]`; axial symmetry handled correctly. | Relative SE(3), geometry encoding, contact field/features and contact-mode distribution; no dependence on cylinder coordinates. |
| Method complexity | Fixed visual-to-contact handoff and one residual/contact estimator. | Amortized geometry-conditioned proprioceptive likelihood-ratio, explicit posterior, and a shared controller. |
| Novelty target | Clear empirical boundary and a focused residual correction method. | Estimator-level calibration and real zero-shot transfer of one contact observation model across held-out geometry families. |
| Evidence | Real cylinder trials across clearance, pose, grasp error and visibility. | Multiple task families and held-out geometry/clearance/visibility splits with physical evidence. |

Code written for the thesis may be reused as infrastructure, but its
cylinder-specific state, success detector, accumulated XY update, and
symmetry assumptions must not become the RSS method definition.  The current
RSS proposal is specified in `RSS_RESEARCH_DIRECTION_20261126.md`; neither
generic multimodal fusion nor across-trial memory is its primary claim.

## 7. What this audit deliberately does not do

- It does not modify runtime stop conditions or robot safeguards.
- It does not promote simulation success to physical evidence.
- It does not treat a VLM success label as a VLA action policy.
- It does not claim Task B or the direct VLA is ready.
- It does not add a fusion architecture before the vanilla comparison.
