# Research and evaluation protocol

## 1. Recommended thesis scope

**Korean title (working):** 다중 시점 시각 피드백을 이용한 물체-손 자세
불확실성 보정 기반 정밀 삽입

**English title (working):** Multi-View Visual Correction of Object-in-Hand
Pose Uncertainty for Precision Insertion

The defensible research claim is narrower than “a general insertion system”:

> A grasp that passes a lift test can still leave an uncertain object-in-hand
> transform.  Measure the key end and axis at the pre-insertion hold, correct
> the held wrist within the same trial, and test whether the correction
> reduces alignment residual and improves insertion success.

In the thesis, describe the inherited flow generically as offline grasp
proposal, geometric/simulation screening, and automated physical validation.
The contribution is the task-specific observation and same-trial correction
after a nominally successful grasp, not the inherited grasp generator.

## 2. Questions and hypotheses

- **RQ1 — repeatability:** with grasp candidate, socket pose and commanded
  pre-insertion target fixed, how much does the actual key pose vary?
- **RQ2 — correction:** does same-trial visual correction reduce key-tip XY
  residual and key-axis tilt before insertion?
- **RQ3 — task outcome:** does the reduced residual increase conditional and
  end-to-end insertion success, and at what latency cost?

Corresponding hypotheses should be preregistered as H1/H2/H3. Do not infer H1
from image differences alone: wrist endpoint, camera and socket calibration
changes can produce the same image difference.

## 3. Transform-level repeatability definition

Use robot frame `R`, socket frame `S`, key frame `K`, and wrist/hand frame `H`.
The fixed candidate stores the nominal `T_KH`. The nominal pre-insertion target
implemented in `precision_insertion/targets.py` is

```
T_RK* = T_RS T_SK(pre)
T_RH* = T_RK* T_KH
```

Therefore, reusing the **same frozen** `T_RS`, the same task geometry and the
same candidate gives exactly the same nominal wrist SE(3) target. It does not
guarantee the same joint path or measured endpoint because the initial robot
state, planner seed and controller error can differ.

At each first pre-insertion hold, save measured wrist FK `T_RH(meas)` and the
multi-view key observation `T_RK(obs)`. Compute

```
T_HK(obs) = inverse(T_RH(meas)) T_RK(obs)
```

and compare it with the fixed candidate relation. This separates actual
object-in-hand variation from wrist endpoint variation. For the cylinder,
rotation around the key axis is unobservable and irrelevant to insertion;
factor this yaw out and report tip translation and axis tilt instead of a raw
6-DoF rotation norm.

**Important control:** do not remeasure or move the socket between repeatability
trials. Save and reuse one session calibration record and record its digest in
every episode. Existing successful logs with different socket-session digests
cannot by themselves establish grasp inconsistency.

The saved-evidence audit is reproducible with:

```bash
python3 demo/precision-insertion/summarize_repeatability.py \
  --attempt-root "$PRECISION_RUN_OUTPUT_ROOT/precision-insertion/inspire/precision_key_cylinder_r15_h80" \
  --candidate-id table/0/20195
```

The tool groups attempts by candidate, session-calibration digest, task family
and gap. It verifies whether the nominal pre-insertion wrist target was
identical inside each group, reports measured wrist endpoint error separately,
and admits a key-tip value to metric statistics only when at least two cameras
support a result explicitly recorded as metric. Single-view fixed-height
hypotheses remain visible as diagnostics but are excluded from key-pose
statistics. It does not subtract the wrist norm from the tip residual and call
the result object-in-hand error: that requires an independently observed key
pose (or at least a consistent tip and axis), not two scalar error magnitudes.

As of 2026-11-02, candidate `table/0/20195` has ten saved first-preinsert
reports split across four socket-session calibration digests. The largest
single-session group has four attempts with an identical nominal wrist target;
its median measured wrist translation error is 2.22 mm and median rotation
error is 0.83 degrees. None of those four attempts has a valid multi-view
metric key tip. This is useful controller-repeatability evidence, but it is not
yet evidence for the magnitude of object-in-hand variation. New controlled
trials must supply the latter.

## 4. Implemented same-trial correction

The base ZeroDex capture4 profile keeps the endpoint/temporal controller as
its default. The controlled
`fixed_grasp_partial_shape_alignment_5mm.env.sh` profile selects the joint
partial-cylinder fit described below and falls back to the same endpoint path
when visible shape is insufficient. Both profiles use the following shared
pipeline:

1. Undistort each camera frame with its calibrated map.
2. Project the nominal held-key axis to obtain a per-view crop. The crop is
   recomputed from the session socket pose and the cumulative held motion; it
   is not a fixed pixel rectangle.
3. Ask Qwen for a tight key box inside that crop.
4. Run SAM3 on the padded Qwen ROI and fit the distal half ellipse of the
   cylindrical key mask. Its center is the observed bottom-face proxy and its
   local line is an axis cue.
5. Rank views using the current session's socket-pose inliers. The default
   endpoint profile stops at the first two-view metric quorum. The partial-
   shape profile adds ranked views until the symmetry-reduced `(x, y,
   tilt_x, tilt_y)` fit is observable or the views are exhausted; unavailable
   shape falls back without ending the trial. Measured-motion temporal
   feedback links consecutive endpoint observations.
6. Compute socket-frame translation
   `delta_xy = - observed_tip_xy`. The configured convergence target is
   2.5 mm and one commanded increment is capped at 10 mm; there is no
   cumulative XY cap.
7. Compute the minimum symmetry-reduced key-axis rotation toward socket `-Z`.
   Axial yaw is omitted. The configured tilt target is 2 degrees and one tilt
   increment is capped at 3 degrees.
8. Hold the finger command fixed. Rotate the wrist about the observed key tip
   and translate it in socket XY with the local damped-Jacobian planner.
9. Capture again and repeat. Replan the axial insertion stroke from the
   measured corrected hold; do not replay the nominal trajectory start.

The 2.5 mm and 2 degree values are experimental convergence parameters, not
evidence that the estimator is accurate to those values. Report sensitivity
or use a tighter socket for a meaningful precision claim. The current 5 mm
radial-gap fixture is a bring-up condition; include at least one tighter
condition if the thesis claims precision rather than pipeline completion.

## 5. Experimental matrix

### 5.1 Repeatability experiment

- Fixed: candidate (currently `table/0/20195`), key/socket pair, socket pose,
  frozen calibration, pre-insertion target and camera calibration.
- Repeat: at least 30 independent pickup–lift–transfer trials if feasible.
- Vary only the tabletop pickup repetition; either use a placement jig or
  record the initial key pose and stratify it.
- Record commanded and measured wrist poses, key tip/axis, `T_HK(obs)`, grasp
  and transfer labels, and first-preinsert images before any correction.
- Report within-condition median/IQR and 95% bootstrap intervals for tip XY,
  axis tilt and wrist endpoint error.

### 5.2 Correction ablation

Use comparable initial conditions and report both residual and task outcome:

1. open-loop nominal insertion;
2. same-trial XY correction only;
3. same-trial XY + symmetry-reduced axis correction.

An additional estimator ablation (single snapshot versus measured-motion
temporal feedback) is useful only after the three primary conditions have
enough trials. Do not change grasp, socket calibration and correction mode in
the same comparison.

### 5.3 Difficulty

Start with the 5 mm radial-gap fixture to collect reliable traces. Then add a
tighter fixture (for example 1 mm, subject to the printed geometry actually
measured). Report results by physical fixture, never pool gaps into one success
rate.

## 6. Outcome and failure accounting

Use an **execution funnel** rather than only one success rate. Each row needs
an explicit denominator:

1. physical pickup started;
2. pickup/squeeze execution record completed;
3. key still held after the 2 s lift-settle snapshot (`grasp_success`);
4. held transfer execution record completed;
5. key still held at pre-insertion (`preinsert_reached`);
6. guarded insertion execution record completed;
7. key tip entered the socket during guarded descent, before opening the hand
   (`tip_entry_success`, manual process-quality label);
8. key released inside the socket (`insertion_success`, final task outcome).

Do **not** put endpoint observability or closed-loop convergence between stages
5 and 6 in this funnel. The configured pipeline may continue with the nominal
target when a metric endpoint is unavailable, so those rows are not mandatory
predecessors of insertion. Report a separate alignment branch among attempts
that reached pre-insertion:

- first-snapshot metric endpoint available;
- any metric endpoint available during the loop;
- physical XY/tilt correction attempted;
- configured alignment target reached;
- nominal/default insertion after unavailable endpoint evidence;
- loop ended before an insertion execution record.

Candidate/planner availability is another pre-physical table, not a physical
success stage. This separation prevents a planning rejection, missing visual
measurement and physical insertion failure from sharing one denominator.

Report:

- end-to-end success: stage 8 / all started physical trials;
- conditional grasp success: stage 3 / stage 2;
- conditional transfer success: stage 5 / stage 3;
- conditional smooth-entry success: stage 7 / stage 6;
- conditional final task success: stage 8 / stage 6;
- release-assisted entry: `tip_entry_success=false` and
  `insertion_success=true`, reported separately rather than counted as smooth
  insertion;
- planning availability separately from physical failure;
- Wilson confidence intervals for proportions.

For every manually judged stage, report label coverage separately. A missing
label is not a positive outcome, but it is also not evidence of a physical
negative. Therefore give both (a) observed successes divided by all eligible
executions, explicitly described as an end-to-end completion/lower-bound
measure, and (b) success among labeled executions. Do not present an unlabeled
episode as a measured miss.

The saved execution records can be summarized without camera, GPU or robot
access:

```bash
python3 demo/precision-insertion/summarize_outcomes.py \
  --attempt-root "$PRECISION_RUN_OUTPUT_ROOT/precision-insertion/inspire/precision_key_cylinder_r15_h80" \
  --gap-mm 5
```

As of 2026-11-02, the saved 5 mm corpus has 16 physical pickup starts, 13
human-positive grasps among 15 labeled lifts, 12 positive pre-insertion holds,
9 guarded insertion execution records and 2 final inside-socket labels. Only
1/9 insertion executions has a `tip_entry_success` annotation, so no smooth-
entry rate is yet supported. Among the 12 pre-insertion holds, one first
snapshot and three attempts at any iteration have a metric multi-view endpoint;
eight attempted a physical correction, five ended with an `aligned` choice,
and seven proceeded to insertion without any metric endpoint. These are pilot
accounting figures, not an ablation: sessions, candidates and historical
policies are mixed unless the command is filtered further.

The current final label is visual `inside_socket` after release. It is not an
independent proof of exactly 20 mm depth. Use that wording in the thesis, or
add an independent depth/fixture measurement before claiming depth success.
Manual labels should be the primary evaluation reference while Qwen is being
validated. Save the independent VLM output, then report agreement, confusion
matrix and failure examples instead of silently replacing human labels.

## 7. Required figures

1. **System and frames:** one uncluttered view of robot, cameras, key, socket,
   and `R/S/K/H` frames.
2. **Controlled repeatability:** same-camera first-preinsert images aligned on
   the measured wrist projection. Overlay key masks with low opacity and show
   the 2D tip scatter/confidence ellipse. Pair this with a 3D socket-frame tip
   scatter and axis cones.
3. **Estimator anatomy:** a 2×N multi-view panel showing raw undistorted frame,
   dynamically projected crop, Qwen box, SAM mask, half-ellipse center and
   inlier ray. Show excluded views in gray, not as missing data.
4. **Closed-loop sequence:** iteration 0…K with tip center, socket center,
   correction arrow and measured residual. Add a small plot of XY residual and
   tilt versus iteration under the images.
5. **Outcome funnel:** counts and conditional percentages at the eight
   execution stages, plus a separate alignment-observability branch.
6. **Latency:** median/IQR stacked bars for perception, planning, motion,
   VLM/SAM, evidence I/O and human wait. Plot human wait separately.
7. **Failure taxonomy:** representative miss/slip, transfer loss, endpoint
   confusion, alignment non-convergence, path unavailable and released-outside
   cases. Do not mix planner failures with physical insertion failures.

The nominal CAD overlay is useful as an analysis visualization, but it must be
visually distinguished from measured key evidence and is not supplied to the
VLM endpoint input.

## 8. Timing evidence

Every live run writes a local append-only `events.jsonl`, then copies its
sidecar to the run directory after orderly completion. `timing_summary.json`
contains:

- every nested span with inclusive, self and child-overlap time;
- per-operation groups;
- phase and resource-kind aggregates with both cumulative work and occupied
  wall time;
- per-attempt aggregates and ordered action boundaries;
- explicit closed-loop total (`trial.same_trial_alignment`), each observation
  and motion with an `iteration` field, and final axial replanning.

Use occupied wall time for latency and cumulative time for compute/I/O work.
Do not add inclusive parent and child durations. For experiments, report
median/IQR/p90 across completed runs, plus counts of incomplete traces.

## 9. Suggested report structure

The department guide specifies 10–30 A4 pages for the final thesis and, for a
Korean body, cover, title page, Korean abstract, table of contents, body,
references, optional appendix and English abstract. Keep at most six abstract
keywords. A defensible interim report can mirror the final structure:

1. Introduction: precision manipulation gap, RQ1–RQ3 and contributions.
2. Task and system formulation: frames, cylinder symmetry, controlled target.
3. Method: candidate validation, lift/transfer labels, multi-view endpoint and
   axis estimation, same-trial wrist correction, insertion and final label.
4. Experimental protocol: controls, ablations, metrics and label policy.
5. Current evidence: implementation, saved traces and only supported pilot
   observations; do not present pilot episodes as a powered success study.
6. Limitations and planned experiments.
7. References; put file schemas and extra overlays in an appendix.

