# Meeting deck: AutoDex for precision insertion

Format: 16:9 widescreen, light neutral background, minimal text. Use gray for
existing AutoDex components, blue for new insertion logic, purple for the
ZeroDex-style VLM evidence layer, green for validated offline evidence, and red
only for blockers or failure modes.

The deck must consistently call the task **keyed plug-and-socket precision
insertion** or **precision insertion**, not “peg-in-hold.” The primary task
endpoint is 20 mm below the socket entry plane.

## Slide 1 — Toward AutoDex for Precision Insertion

**Subtitle**

> From autonomous grasp validation to repeated, evidence-driven insertion

**Footer**

> Franka FR3 + Inspire Hand · AutoDex multi-camera setup

**Visual layout**

- Full-bleed crop of candidate 104 at the 20 mm endpoint.
- Small pipeline strip at the bottom: `Perceive → Grasp → Reorient → Insert → Diagnose → Reset`.

**Use**

- `07_unconstrained_ablation/pose_004/grasp_104/final_frame_key_socket.png`

## Slide 2 — Recap: Research Direction Requested in the Last Meeting

**On-slide copy**

> **Professor Han's request**
>
> - Move beyond grasping large, easy objects.
> - Modify AutoDex for a task that requires precision.
> - Use safe target objects that tolerate repeated failures.
> - Let the robot improve through multiple trials, following the AutoDex philosophy.
> - Add ZeroDex-style VLM reasoning and test whether visual reasoning helps.
> - Progress toward real targets such as electrical plugs or USB insertion.

**Bottom-line callout**

> Selected first task: repeated insertion of a 3D-printed keyed plug into a fixed socket.

**Visual layout**

- Left 55%: the request, grouped into `Task`, `Autonomy`, and `Reasoning`.
- Right 45%: key/socket compatibility loop video.

**Use**

- `01_compatibility/key_socket_compatibility.mp4`

## Slide 3 — Recap: Agreed Experimental Ladder

**On-slide copy**

> **Primary success:** reach 20 mm insertion depth without losing the key or exceeding safety limits.

| Clearance | Purpose | Planned control regime |
|---|---|---|
| 1.5 mm | System bring-up | Safe open-loop baseline |
| 1.0 mm | Accuracy audit | Perception + calibration + grasp repeatability |
| 0.5 mm | Contact search | Force/contact-based XY–yaw correction |
| 0.3 mm | Precision condition | Final closed-loop evaluation |

**Small note**

> The 0.1 mm print exists but is excluded from the initial study because manufacturing error makes it unreliable.

**Visual layout**

- Use the clearance-family render across the top half.
- Put the four-stage ladder below it; highlight 1.5 mm as `Current`.

**Use**

- `00_geometry/key_socket_clearance_family.png`

## Slide 4 — Baseline AutoDex Pipeline, Briefly

**On-slide flow**

> Object assets  
> → BODex grasp proposals  
> → cuRobo scene clearance  
> → MuJoCo grasp stability  
> → FoundPose object pose  
> → approach + grasp + 10 cm lift preflight  
> → physical grasp trial  
> → coverage update / reorientation

**Three key properties**

- Candidate generation is offline; physical trials validate candidates online.
- Current runtime success is fundamentally a grasp-and-lift outcome.
- Coverage and reorientation support repeated autonomous trials.

**Visual layout**

- One horizontal gray pipeline with three grouped regions: `Offline`, `Runtime preflight`, `Physical loop`.
- Use only icons and short labels on the slide; keep details in speaker notes.

**Speaker note**

> AutoDex already provides the repeated-trial structure we need. The main change is extending the unit of validation from “can I lift?” to “can I complete the task after lifting?”

## Slide 5 — From AutoDex Grasping to Precision Insertion

**On-slide comparison**

| | AutoDex | Precision-insertion extension |
|---|---|---|
| Objective | Grasp and lift | Grasp, reorient, insert 20 mm |
| Session reference | Table/ChArUco | Table + measured fixed socket pose |
| Candidate preflight | Approach + lift | Approach + lift + transfer + entry + insertion |
| Collision world | Table and perceived objects | Table + frozen socket fixture + attached key |
| Success labels | Grasp/lift success | Grasp success and task success separated |
| Recovery | Coverage/reorient | Diagnose, extract/drop reset, then re-perceive/reorient |

**Headline takeaway**

> A grasp is useful only if its rigid key-to-hand transform remains compatible with the entire insertion task.

**Visual layout**

- Two-column comparison with a blue right-hand column.
- Add a thin rigid-transform illustration under the table: `T_hand_key fixed after closure`.

## Slide 6 — Proposed Episode and Session Logic

**On-slide flow**

> **Once per session**  
> AutoDex camera audit → estimate socket pose repeatedly → robust SE(3) selection → freeze `T_robot_socket` → build collision world
>
> **Every episode**  
> perceive key pose → identify exact tabletop pose → retrieve task-compatible grasps → full-task preflight → pick/lift/transfer → guarded insertion → outcome verdict → reset or reorient

**Decision branches**

- Success: record insertion evidence → extract → move above reset region → drop → re-perceive.
- Failed insertion: classify failure → guarded retreat → preserve grasp and task labels separately.
- No valid candidate: reorient the key, then re-perceive; never replay stale world-frame joints.

**Visual layout**

- Top row: session-start box.
- Bottom row: closed episode loop.
- Use a lock icon on the frozen socket pose and collision world.

## Slide 7 — Task Assets and Geometry

**On-slide copy**

> **Known geometry is an advantage, not a reason to skip perception.**
>
> - Exact watertight meshes are used for the key and keyed socket.
> - The socket pose is measured once per session and then treated as fixed.
> - The key pose is re-estimated every episode.
> - Exact task symmetry is identity: the keyed profile fixes insertion yaw.

**Visual layout**

- Left: clearance-family render.
- Center: compatibility video.
- Right: all five tabletop poses, with pose 004 highlighted.

**Use**

- `00_geometry/key_socket_clearance_family.png`
- `01_compatibility/key_socket_compatibility.mp4`
- `02_tabletop_poses/key/all_poses.png`
- `02_tabletop_poses/socket/fixed_pose/pose_000.png`

## Slide 8 — Current Status: 10k Contact-Policy Ablation

**On-slide funnel**

> 10,000 full-key BODex proposals  
> → 1,873 pass project numerical thresholds  
> → 527 pass high-density sampled 20 mm geometry  
> → 51-candidate simulation pilot  
> → 35 make squeeze contact  
> → 7 pass MuJoCo gravity stability

**Right-side result**

> Only 4/1,873 satisfy the original six-surface + 2 mm declared-contact rule.

**Interpretation**

- The exact rule is highly selective.
- Removing it finds stable grasps, but does not make all contact locations insertion-safe.
- This pilot is quality-ranked, not an unbiased success-rate estimate.

**Visual layout**

- Large horizontal funnel or Sankey-style count flow.
- Small 5×5 grasp-policy grid in the lower right.

**Use**

- `03_contact_policy/grasp_policy_grid_5x5.png`
- Counts from `assets/precision_insertion/unconstrained_contact_ablation_20mm_10k_results.json`.

## Slide 9 — Current Status: Seven Offline-Validated Candidate Previews

**Headline**

> Seven candidates survive grasp stability and sampled full-sequence geometry to 20 mm.

**On-slide validation badges**

> BODex quality ✓ · tabletop cuRobo clearance ✓ · MuJoCo grasp stability ✓ · rigid attachment ✓ · 12k-sample frame-wise geometry ✓

**Red scope label**

> Not yet continuous cuRobo planning or physical insertion success.

**Visual layout**

- Main 70%: autoplay candidate 104 key–socket video.
- Right 30%: candidate IDs `35, 70, 79, 95, 99, 104, 5102` and the validation badges.
- Use candidate 5102 as an appendix/control video, not seven simultaneous videos.

**Use**

- Main: `07_unconstrained_ablation/pose_004/grasp_104/grasp_104_pose_004_20mm_key_socket.mp4`
- Backup: the corresponding `grasp_<id>_pose_004_20mm_key_socket.mp4` for all seven IDs.

## Slide 10 — What the Ablation Actually Tells Us

**On-slide comparison**

> **Five of seven stable candidates** touch the shaft/front region.  
> They are useful negative examples but remain risky for rigid insertion.
>
> **Candidates 104 and 5102** have no declared contact above the handle front.  
> They are rejected mainly by the original edge-margin rule and are the best candidates for the next planning audit.

**Policy proposal**

- Hard reject: contact that occupies the socket swept volume or causes continuous hand–socket collision.
- Soft preference: handle side/rear contact and larger edge clearance.
- Do not equate “outside the old policy” with either failure or success.

**Visual layout**

- Left: candidate 95 contact still, caption `shaft/front contact`.
- Right: candidate 104 contact still, caption `edge-margin violation; no contact above handle front`.
- Add no annotations inside the images; put the captions below.

**Use**

- `07_unconstrained_ablation/pose_004/grasp_95/grasp_contact_key_socket.png`
- `07_unconstrained_ablation/pose_004/grasp_104/grasp_contact_key_socket.png`

## Slide 11 — Expected Failures and Required Diagnostics

**On-slide taxonomy**

| Stage | Expected failure | Evidence needed |
|---|---|---|
| Perception | key/socket pose error | multi-view pose residual, reprojection |
| Pickup | miss or slip | images, joint state, object motion |
| Transfer | key loss | synchronized multi-view observation |
| Entry | rim contact / yaw error | force transient + image + depth |
| Insertion | jam / depth shortfall | axial depth, force/torque, motion history |
| Reset | extraction/regrasp/drop failure | state-machine phase evidence |

**Takeaway**

> Failure cause, not only a binary verdict, should update the next trial.

**Visual layout**

- Six compact failure cards around a central key/socket diagram.
- Use a terminal frame from a shaft-contact candidate as an expected-risk example; label it as a preview, not an observed failure.

## Slide 12 — ZeroDex-Style VLM Reasoning: Read-Only Evidence Layer

**On-slide architecture**

> AutoDex synchronized images + phase label + robot state summary  
> → VLM evidence reviewer  
> → `{verdict, confidence, observed cues, likely failure mode, evidence quality}`

**Responsibilities**

- Replace the brittle single ChArUco-corner heuristic with phase-aware visual evidence.
- Distinguish pickup success, transfer retention, entry alignment, insertion progress, and reset outcome.
- Explain likely failure causes and flag unjudgeable evidence.

**Safety boundary**

> The VLM does not command motion, estimate metric pose, override force limits, or certify safety. Robot state and force/depth gates remain authoritative.

**Visual layout**

- Purple VLM box beside—not inside—the blue robot control path.
- Dashed arrow from VLM output to the trial database/candidate selector.
- No arrow from the VLM directly to the controller.

## Slide 13 — Reset and Reorientation Remain Part of the Trial Loop

**On-slide copy**

> **Reset after success**  
> Observe → regrasp → guarded extraction → transfer above reset region → open/drop → re-perceive
>
> **Reorientation when coverage is exhausted**  
> Select a validated source-to-target pose action → execute → release → re-perceive → verify target pose

**Status labels**

- Reset: composed presentation preview available; continuous planning and hardware validation remain.
- Reorientation: candidate 104 passes MuJoCo gravity retention and the AutoDex
  FR3 full-chain cuRobo motion preflight for pose 004→000. The 12 cm drop and
  post-drop pose verification remain composed/unvalidated.

**Visual layout**

- Left 60%: reset video.
- Right 40%: five-pose diagram with a dashed transition and a red `candidate generation required` badge.

**Use**

- `05_reset/pose_004/grasp_40/full_trial_20mm_drop_reset.mp4`
- `06_reorientation/reorientation_pose_004_to_000_stable_grasp_104.gif`
- `02_tabletop_poses/key/all_poses.png`
- `06_reorientation/status.json` for the exact evidence boundary.

## Slide 14 — Short-Term Todo List

**On-slide copy**

> **1. Promote an honest 1.5 mm baseline**  
> Run continuous attached-object cuRobo planning for candidates 104 and 5102; reject on any hand–socket swept-volume collision.
>
> **2. Commission the AutoDex robot setup**  
> Generate FoundPose assets, measure the socket at session start, verify hand–eye calibration, and record open-loop error statistics.
>
> **3. Execute guarded real-robot trials**  
> Start with low speed/force limits and 1.5 mm clearance; log grasp, task, and reset outcomes separately.
>
> **4. Add phase-aware outcome reasoning**  
> Build a synchronized evidence packet and benchmark heuristic-only vs. VLM-assisted verdicts against human labels.
>
> **5. Increase precision progressively**  
> 1.0 mm accuracy audit → 0.5 mm XY–yaw contact search → 0.3 mm final condition.

**Decision requested from the meeting**

> Should the contact rule become a soft ranking prior while swept-volume collision remains the hard gate?

**Visual layout**

- Five-step left-to-right roadmap, with step 1 highlighted as `Next`.
- Put the decision question in a single blue box at the bottom.

## Appendix A — Claims that must remain explicit

- “Offline-validated preview” is acceptable for the seven new animations.
- Do not call the numerical waypoint IK preview a continuous cuRobo plan.
- MuJoCo currently tests grasp retention under gravity, not socket insertion.
- A sampled zero-collision result is necessary evidence, not a continuous proof.
- No asset in this deck is a physical success until a robot episode record says so.

## Asset checklist

All visual assets should be placed on 16:9 slides without stretching. Images
and videos should retain their native aspect ratio and be cropped only when the
key and socket remain visible.

| Asset | Status | Recommended slide |
|---|---|---|
| Clearance-family key/socket render | Ready | 3, 7 |
| Key/socket compatibility animation | Ready | 2, 7 |
| Five tabletop-pose overview | Ready | 7, 13 |
| Existing 5×5 contact-policy grid | Ready | 8 |
| 10k ablation funnel graphic | Build directly in slides from recorded counts | 8 |
| Candidate 104 20 mm key–socket animation | Ready; primary | 1, 9 |
| Candidate 5102 20 mm key–socket animation | Ready; control | 9 appendix |
| Remaining five candidate animations | Ready; appendix | 9 appendix |
| Candidate 95 vs. 104 contact stills | Ready | 10 |
| Expected-failure taxonomy diagram | Build as slide-native vector graphic | 11 |
| VLM evidence-layer architecture | Build as slide-native vector graphic | 12 |
| Success/reset composed animation | Ready; label as preview | 13 |
| Key-specific reorientation animation | Ready; cuRobo motion + composed drop, not physical success | 13 |
| Optional finish geometry animation | Ready; exact meshes only, no robot/controller claim | Appendix |
| GIF derivatives for every active MP4 | Ready; see `gif_manifest.json` | All video slides |
| Real robot photo with camera labels | Capture after runtime audit | 14 or appendix |

Root asset directory:

`~/shared_data/AutoDex/precision_insertion/presentation_assets/`
