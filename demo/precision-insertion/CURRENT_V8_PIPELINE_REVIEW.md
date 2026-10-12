# Current AutoDex v8 versus precision insertion: implementation review

This review compares the **local** stock AutoDex checkout at `448732d` with
`feat/precision-insertion` and its isolated robot-PC staging root. Remote
GitHub freshness must be checked separately. It describes code paths and
offline execution evidence, not a physical insertion result. The stock
`src/execution/{run_pipeline.py,run_auto.py,scene_cfg.py}` files remain
unchanged. The only shared source change in this review is
`src/grasp_generation/adaptive_orchestrator.py`, described below.

## Data path and stage comparison

| Stage | Stock AutoDex v8 input → implementation → output | Precision cylinder `gap_mm=20` input → implementation → output | Material difference and current evidence |
| --- | --- | --- | --- |
| Object root | `run_auto.py` requires `grasp_version=v8`; `autodex.utils.path.get_obj_root("v8")` selects `shared_data/object_processing`. `scene_cfg.py` selects the processed planning mesh and turns a measured world pose into a planner scene. | `AssetPaths(shared_root, mode)` uses explicit recipient `object_processing/precision_key_cylinder_r15_h80` and `precision_socket_cylinder_gap_20mm`. Raw key OBJ is used by FoundPose; processed key OBJ/URDF and exact socket collision OBJ are separate inputs. | No runtime dependency on `object/paradex` or its pose index. Cylinder candidate scenes now require target mesh, URDF, and tabletop pose to match the explicit v8 root. |
| Grasp proposals | `src/grasp_generation/adaptive_orchestrator.py` reads v8 tabletop pose/OBB, generates table/wall/shelf/box scene JSON, calls BODex, then MuJoCo sim filter and writes `AutoDex/candidates/inspire/v8/...`. The common adaptive script defaults to wall/shelf/box. | Full-length cylinder proposal handoff supplies 112 **current v8** table candidates, 109 for pose `000` and 3 for `001`; the recipient installer verifies source files and rewrites absolute scene references. | The 112-ID pool is sufficient; the earlier 12-ID pool is not required. A stock generator bug used `generate_scene.obj_path` from a different root for the target mesh; `write_scene_file` now binds and restores the passed `obj_root`, and the CLI defaults it through `get_obj_root(version)`. |
| Offline filtering | Stock candidate `sim_eval.json` reflects MuJoCo hand/key grasp survival and stock files such as `wrist_se3.npy`, `pregrasp_pose.npy`, `grasp_pose.npy`. | `candidates.build_endpoint_catalog` rechecks complete candidate files, simulation validation, v8 scene source, input SHA-256, then calls `endpoint.screen_grasp_endpoint` on exact key/socket/hand geometry at the nominal 20 mm endpoint. | The recipient must rebuild a complete `gap=20` catalog. A producer `audit_only` JSON is not a runtime catalog. Endpoint clearance and stable simulation grasp do not establish physical grasp or insertion. |
| Cameras and calibration | Stock AutoDex camera/snapshot/ChArUco infrastructure and `InitOrchestrator` consume configured serials, intrinsics/extrinsics, masks, FoundPose data, and `c2r`; `run_auto` can auto-label grasp/place using a board snapshot proxy. | `session_launch` validates explicit calibration, clock/error limits and frame metadata. `live_capture` uses stock snapshot and `InitOrchestrator.collect_payloads`; `session_bootstrap` first binds the board, then repeated socket-only captures, admitted multiview masks/poses, and a frozen socket scene. | Stage-1 file audit has run. No live camera capture on this robot PC has been run; saved NAS frames are separate evidence. `start-session --capture` is camera-only and cannot claim key-plus-socket accuracy. |
| Segmentation and pose | Stock `run_auto` passes `--prompt` to `InitOrchestrator.trigger_init`, default `"object on the checkerboard"`. SAM3 masks and FoundPose use the v8 raw mesh/repre; cross-view IoU/silhouette selects/refines a pose before scene construction. | Socket SAM prompt example is `"fixed white cylindrical socket on the ChArUco board"`; a key capture must supply a specific white peg prompt, e.g. `"loose white cylindrical peg on the tabletop"`. New `prompt_contract` rejects nonwhite color words before active capture. `key_perception` checks same-request frames, admission and multi-view agreement. | Both physical objects are white. Shape and location distinguish them. Exact mesh/repre availability and saved-image replay do not prove live mask quality or metric pose accuracy. |
| Candidate selection and pickup | `GraspPlanner` plans against the measured v8 scene, with candidate evaluation/selection and Franka/Inspire pickup. Stock execution squeezes, lifts and later places using its standard trajectory flow. | `live_planning_handlers` binds a fresh key pose to the frozen socket session; `candidates` selects only source-bound catalog rows for the observed tabletop stem and passes one stock candidate via `planner_candidate_override`. `trial_preflight` reuses the stock planner for pickup; demo motion handlers then gate lift and transfer on measured feedback. | Runtime recomputes catalog hashes. The offline `wrist_se3.npy` key-to-hand transform remains nominal until measured after physical grasp. Current staging has no commissioned actuator adapters. |
| Preinsert geometry | Stock placing descends relative to the board/world; it has no fixed-socket preinsert or 20 mm key-axis task. Stock `jacobian_stroke.py` is for world-Z strokes. | `T_robot_key = T_robot_socket @ T_socket_key`, `T_robot_hand = T_robot_key @ T_key_hand`. `preflight` creates preinsert, entry and depth target on `R_robot_socket @ insertion_direction_socket`; `axis_jacobian.plan_jacobian_axis_stroke` uses damped numerical Jacobian continuation, fixed wrist orientation, joint limits, collision checking and retiming. `path_audit` samples key/hand/socket clearance and axial/lateral error. | It is a kinematic nominal path, not force-controlled contact. It handles a tilted socket axis without passing that axis into stock world-Z stroke code. |
| Contact, release and retry | Stock executor follows planned trajectories and places/retracts; it has no socket contact contract. | `follow_guarded_insertion` must be supplied by the robot-PC integration with watchdog, force/torque trace and stop acknowledgement; `release_and_clear` must open the hand only after a non-aborted held-key stroke, then reach a measured clear view. Visual failure may produce next-attempt XY advice; a released key is re-observed and restored before another attempt. | The ordinary stock joint follower is not accepted as a guarded insertion controller. No contact callback or physical reset path was run here. |
| Directed reset | Stock v8 reset scene/seed loader covers pose transitions, but the stock runner does not include this demo's fixed socket fixture and has no verified map for the new key. | Curated source-bound v8 `000→001/191` and current addendum `001→000/1687` are rehydrated to recipient paths, hash-checked and installed as directed MuJoCo seeds. | Both cells have one stable seed in the recipient audit. A seed is not a socket-aware Franka path or qualified landing. The older reverse `631` stays in noncanonical source staging and is not installed. |

## Exact active prompts and outputs

The common VLM wrapper in `observer._infer_closed_set` prepends the exact
ordered image labels and valid camera IDs. It instructs the model to return
**one compact JSON object** with exactly the requested class field,
`evidence_views`, and a ≤12-word `evidence` phrase. A local Qwen backend
passes ordered PIL images and the prompt to `infer_images_prompt`; it does
not acquire cameras. The robot config should set `vlm.model_id` to the
verified staged Qwen directory to avoid relying on a public model name.

| Active observation | Ordered raw input and prompt rule | Parsed output and decision |
| --- | --- | --- |
| Lift | For each camera, `before_grasp`, then `after_lift`. Prompt says both key and socket are white; track the loose key by shape and tabletop/hand location; `held` requires distinct key pixels moving with the hand. Hidden key or commanded closure is insufficient. | `class ∈ {held,miss,slip,unobservable}`, camera IDs and short evidence. At least two agreeing decisive cameras, no contradictory decisive view. This is a visual held-key label. |
| Preinsert | For each **same** camera, raw `after_lift`, then raw `preinsert`. Prompt asks whether the same white peg remains visibly held after transfer and is **roughly** above the white socket, using shape/location. It explicitly forbids metric XY, contact, success or depth inference. | `class ∈ {coarse_match,gross_misalignment,slip_or_miss,unobservable}` and provenance. The CAD overlay is retained for human geometry review but is excluded from this VLM input. |
| Final released key | For each camera, raw `initial_tabletop`, then raw `final_clear_view`, acquired after hand opening and robot exit. Prompt requires positive visible evidence for the same white key inside the white socket; disappearance, occlusion or empty hand is insufficient. | `visual_class ∈ {inside_socket,outside_socket,unobservable}`. `inside_socket` is a task image label, not an independent 20 mm depth measurement. |
| Failed-attempt XY grounding | Saved preinsert raw images only. The cylinder prompt says key and socket are both white; mark the visible key tip and two separated shaft-centerline pixels, or null if occluded/ambiguous. It forbids CAD/robot command extrapolation. | Per-view JSON `tip_px` or null, `axis_line_px` or null, short evidence. Geometry code triangulates eligible views and checks projection/parallax; a single view is nominal-height diagnostic, and no tip gives explicitly exploratory advice. Only `outside_socket` can use it; the update is `next_offset = current_offset + 1.0 × measured_error`, capped at 10 mm **per step**, with no cumulative cap. |

All prompt evidence records retain prompt text, raw VLM answer, image order,
camera ID, exposure timestamp, frame ID and image digest. Manual labeling
still stores an independent VLM result. The older single-frame/overlay
preinsert utilities and stock generic object prompt are not the active live
cylinder label path.

## What was actually executed

The isolated staging tree is under
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392`.
The source-bound base socket and key scenes were rebound there; 112 current
full-length grasp IDs were installed without the historical 12-ID pool. The
two directed reset seeds above were source-verified and accepted by the v8
loader. The recipient's complete `gap=20` endpoint scan screened all 112
candidate directories: 98 eligible (95 for `000`, 3 for `001`), zero errors, `complete_scan=true`,
`robot_ready=false`. The file audit found all eight required input classes,
2 tabletop poses and 2 directed height-12 reset seeds, with
`file_inputs_present=true` and `robot_ready=false`. The staged Qwen copy
matched all 10 source files by SHA-256; the staged key and 20 mm socket
FoundPose PTH hashes matched their recorded handoff digests. Unit/integration
tests use fixtures or mock adapters; they are not camera, controller, force,
or physical insertion tests.
The nonmoving verification record is saved at
`AutoDex/precision_insertion/cylindrical/recipient_staging_verification_20261013.json`
inside the staging root.
