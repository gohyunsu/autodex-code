# AutoDex grasp pipeline vs. precision-insertion extension

This comparison describes the current code, not an idealized paper pipeline.
The precision-insertion column distinguishes implemented assets and sampled
previews from runtime work that is still required.

| concern | current AutoDex grasp/lift pipeline | precision-insertion extension |
|---|---|---|
| Primary objective | Discover and physically validate grasps that acquire and lift an object. | Acquire the key, rigidly reorient it, insert it 20 mm, optionally finish seating with a separate press, and reset by extraction and drop. |
| Unit of success | A grasp candidate and one lift trial. | One episode with independent `grasp_success`, `task_success`, optional `finish_success`, and `reset_success`. |
| Session startup | Audit cameras/calibration and measure an empty ChArUco board/table reference. | Retains the AutoDex startup and additionally measures the socket repeatedly, selects an SE(3) medoid, freezes `T_robot_socket`, and builds one immutable fixture collision world for the session. |
| Camera topology | AutoDex FLIR capture PCs, RCC, hardware trigger, timestamp camera. | Identical AutoDex camera topology; no ZeroDex camera port. |
| Per-episode perception | FoundPose estimates the object pose and stable tabletop pose. | FoundPose estimates the key every episode; socket FoundPose runs once at session startup and is frozen until invalidation. |
| Task symmetry | Object-dependent symmetry can expand/fold equivalent grasp orientations. | Exact key and socket task symmetry are identity because the keyed shaft fixes yaw; all five tabletop poses remain distinct. |
| BODex proposal surface | Normally the complete object mesh. | A handle-only proxy may propose grasps, but it never enters runtime collision/planning as the key geometry. |
| Declared contact policy | General BODex contact/quality thresholds. | All declared contacts must lie at least 2 mm from the boundary on one of five pentagonal-handle lateral faces or the rear face. Shaft, tip, shaft-tip bevels, and socket-facing shoulder are forbidden. |
| Complete-hand contact gate | Not a general AutoDex promotion requirement. Candidate 78 demonstrates that declared contacts can pass while another hand link penetrates a forbidden region. | Samples every Inspire visual link against the exact full key and requires thumb/finger opposition with no forbidden-region penetration. This is still a sampled necessary condition, not a continuous proof. |
| Offline cuRobo in `run_sim_filter.py` | Checks pregrasp hand/world and self collision in the tabletop scene; checks that the squeeze pose reaches the object. | Retained, but must run on the exact key and exact pose. It does not validate insertion or dynamics. |
| Offline MuJoCo in `run_sim_filter.py` | Closes/squeezes the hand, applies a gravity-direction external force, and rejects a grasp if the object moves over 5 cm or 15 degrees. | Retained as a grasp-stability gate. A dedicated key/socket contact simulation would be additional work; the current MuJoCo filter does not simulate socket insertion. |
| Validation provenance enforcement | `load_candidate()` loads any directory containing the expected NPY files; it does not itself require `coll_valid.npy`, `sim_eval.json`, or `simulation_validation.json`. Correct pool promotion is therefore the current trust boundary. | Promotion and the task-aware loader must fail closed on missing/hash-mismatched contact, MuJoCo, cuRobo, and asset evidence instead of relying only on directory placement. |
| Runtime cuRobo candidate filter | Rejects backward grasps, world collision, and self collision; solves batch arm IK. | Retained, with the socket fixture present as an obstacle from the beginning. |
| Runtime approach preflight | MotionGen plans from the initial joint state to the approach endpoint. | Retained unchanged. |
| Runtime lift preflight | Substitutes the selected closed-hand state at the exact approach endpoint and validates a 10 cm world-Z stroke with the full object rigidly attached. It densely checks collision, limits, monotonic Z, lateral/orientation drift, velocity, and acceleration. | Retained as the first part of a larger immutable task bundle. |
| Planning horizon before a physical pick | Approach plus 10 cm held-object lift. Downstream placement is not part of candidate acceptance. | Must include approach, lift, collision-free transfer, true entry pose, guarded 20 mm insertion geometry, extraction, and reset-drop reachability before picking. Optional finish mode additionally needs a release/retreat/press bundle. This full hook is not yet integrated into `run_pipeline.py`. |
| Attached-object invariant | Used during lift; later execution paths depend on the operation. | `T_hand_key` is immutable from closure until release. Every held key pose is derived from hand FK; no visual animation or plan may slide the key in the hand. |
| Collision world | Table and perceived target/obstacles; target is removed where intended hand-object contact is required. | Adds the frozen socket. Socket collision is disabled only for the declared bore corridor during guarded axial contact; it remains an obstacle everywhere else. |
| Intended contact motion | Not required for the grasp/lift objective. | Insertion, extraction, and press are guarded contact primitives. A nominal cuRobo free-space path cannot establish physical insertion success. |
| Primary insertion endpoint | None. | Tip-at-entry is `T_socket_key_entry`; primary success is 20 mm deeper at `T_socket_key_verification`. Full seating is another 20.5 mm nominal travel. |
| Full seating | Not applicable. | Same-grasp seating is explicitly not required: all ten current policy-pass candidates cross the table at the seated pose. Finish mode releases and retreats, then uses a separately preflighted guarded top-down press. |
| Physical grasp verdict | Current auto mode uses lift-time ChArUco evidence/heuristics; manual labeling remains available. | Replace the brittle single heuristic with a phase-aware multimodal verdict. Robot state and force/depth gates are authoritative; a read-only VLM reviews synchronized images and provides a verdict/explanation but cannot command motion. |
| Failure taxonomy | Primarily planning failure, grasp/lift failure, and recovery/reorientation paths. | Separates perception/calibration error, pickup slip, transfer loss, rim contact, XY/yaw misalignment, jam, depth shortfall, press failure, extraction failure, reset-grasp failure, and unjudgeable evidence. |
| Candidate result label | Candidate `result.json: success` means grasp/lift success and drives grasp coverage. | Preserve that meaning. Insertion/finish/reset failures must not relabel a mechanically good grasp as a grasp failure. Task evidence belongs in the episode record. |
| Coverage | Tracks grasp coverage and can trigger pose adjustment/reorientation when useful candidates are exhausted. | Retains pose-conditioned grasp coverage, but candidate selection is conditioned on a full task-plan pass. After drop reset, the key is re-perceived and its resulting exact tabletop pose determines the next pool. |
| Reset | Existing placement/reorientation logic aims to continue grasp exploration; it is not an insertion-specific extraction operation. | After success, observe, regrasp only permitted handle surfaces, guarded-extract, transfer above a reset zone, open/drop, then re-perceive. It neither replays the whole forward path nor restores the exact original pose. |
| Reorientation | Existing AutoDex assets/policies are used when coverage or reachability requires a new tabletop pose. | Still used when the observed pose has no full-task-valid grasp. Reset drop may itself produce a useful new pose; otherwise use the normal validated AutoDex reorientation path. |
| VLM/ZeroDex contribution | None in the current grasp verdict. | ZeroDex-style reasoning is a read-only evidence layer for phase verdicts and failure explanation on AutoDex images. It is not a camera replacement, pose estimator, force controller, collision checker, or safety authority. |
| Fail-closed conditions | Planning/CUDA/hardware failures stop or recover according to current runner policy. | Adds socket-pose invalidation, sensor/VLM disagreement, missing depth evidence, failed reset, and unjudgeable outcome as stop conditions. |
| Current evidence | Candidate 78 has recorded offline cuRobo/MuJoCo and hardware-free FR3 approach/10 cm lift-plan evidence, but its later whole-hand audit rejects it for precision insertion. | Five pose-004 candidates pass the corrected six-surface, 20 mm sampled geometric preview. None yet has the required MuJoCo rerun, continuous cuRobo end-to-end plan, or physical insertion label. |

## What “preflight passed” means

The phrase must always name its scope:

1. **BODex/contact screen:** numerical grasp quality and declared object-side
   contact locations only.
2. **Offline scene/MuJoCo screen:** pregrasp scene clearance, squeeze contact,
   and resistance to one gravity-direction force test.
3. **Current AutoDex runtime preflight:** approach plus the exact closed-hand,
   attached-object 10 cm lift trajectory.
4. **Sampled precision preview:** numerical waypoint IK and sampled visual-mesh
   checks to 20 mm; useful for rejecting obvious geometry, but not cuRobo.
5. **Required precision runtime preflight:** one immutable continuous plan
   bundle through transfer and 20 mm insertion, plus reset reachability and,
   when requested, the independent press bundle.
6. **Physical success:** measured execution outcome. No simulation or planner
   artifact can substitute for it.

## cuRobo and MuJoCo are complementary

cuRobo answers whether the modeled Franka/Inspire system can reach and move
through a collision world while respecting kinematic and trajectory limits.
It does not prove that friction will retain the key or that contact-rich
insertion will converge.

MuJoCo answers a narrower dynamics question for the current AutoDex filter:
after closing and squeezing, does the simulated object remain sufficiently
stable under the tabletop-pose gravity direction? It does not plan the Franka
arm, certify camera/calibration accuracy, or currently test the key/socket
insertion. Both gates are needed, and neither is physical validation.
