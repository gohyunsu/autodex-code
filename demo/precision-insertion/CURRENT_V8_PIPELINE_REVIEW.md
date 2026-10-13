# Current AutoDex v8 versus precision insertion: implementation review

This review compares the stock AutoDex `main` at `448732d` with
`feat/precision-insertion` and its isolated robot-PC staging root. On
2026-10-13, `git ls-remote` confirmed the fork's `main` at `448732d`.
Use `git rev-parse HEAD` for the current feature-branch SHA. It describes code paths and
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
| Cameras and calibration | Stock AutoDex camera/snapshot/ChArUco infrastructure and `InitOrchestrator` consume configured serials, intrinsics/extrinsics, masks, FoundPose data, and `c2r`; `run_auto` can auto-label grasp/place using a board snapshot proxy. | `session_launch` validates explicit calibration, clock/error limits and frame metadata. `live_capture` uses stock snapshot and `InitOrchestrator.collect_payloads`; `session_bootstrap` first binds the board, then repeated socket-only captures, admitted multiview masks/poses, and a frozen socket scene. | A camera-only transport probe received 20/20 live JPEGs after temporarily arming the idle streams, then returned them to `READY`. All 20 frames had zero known ChArUco markers; the scene was not the key/socket task. Neither socket pose nor calibrated image timing has been verified. |
| Segmentation and pose | Stock `run_auto` passes `--prompt` to `InitOrchestrator.trigger_init`, default `"object on the checkerboard"`. SAM3 masks and FoundPose use the v8 raw mesh/repre; cross-view IoU/silhouette selects/refines a pose before scene construction. | Socket SAM prompt example is `"fixed white cylindrical socket on the ChArUco board"`; a key capture must supply a specific white peg prompt, e.g. `"loose white cylindrical peg on the tabletop"`. New `prompt_contract` rejects nonwhite color words before active capture. `key_perception` checks same-request frames, admission and multi-view agreement. | Both physical objects are white. Shape and location distinguish them. Exact mesh/repre availability and saved-image replay do not prove live mask quality or metric pose accuracy. |
| Candidate selection and pickup | `GraspPlanner` plans against the measured v8 scene, with candidate evaluation/selection and Franka/Inspire pickup. Stock execution squeezes, lifts and later places using its standard trajectory flow. | `live_planning_handlers` binds a fresh key pose to the frozen socket session; `candidates` selects only source-bound catalog rows for the observed tabletop stem and passes one stock candidate via `planner_candidate_override`. `trial_preflight` reuses the stock planner for pickup; demo motion handlers then gate lift and transfer on measured feedback. | Runtime recomputes catalog hashes. The offline `wrist_se3.npy` key-to-hand transform remains nominal until measured after physical grasp. Current staging has no commissioned actuator adapters. |
| Robot feedback boundary | Stock `FrankaExecutor` creates command-capable FR3/Inspire controllers. Its constructor calls `stop_streaming`, possible `error_recovery`, and `set_collision_behavior`; the current Inspire controller constructor writes speed/force/angle registers, including `angleSet=1000`. | Motion-enabled `open_franka_core` still uses that executor after explicit motion opt-in. New `probe_robot_feedback_only.py` instead subscribes only to FR3 state PUB port 5556 and reads Inspire `angleAct`/`angleSet` with Modbus function 03. | The previous Stage-3 guide incorrectly described `open_franka_core` as connection-only. It is corrected. The read-only probe received advancing FR3 state and six actual/commanded Inspire angles, without constructing a motion controller. This is not exposure-aligned 13-joint feedback or contact commissioning. |
| Preinsert geometry | Stock placing descends relative to the board/world; it has no fixed-socket preinsert or 20 mm key-axis task. Stock `jacobian_stroke.py` is for world-Z strokes. | `T_robot_key = T_robot_socket @ T_socket_key`, `T_robot_hand = T_robot_key @ T_key_hand`. `preflight` creates preinsert, entry and depth target on `R_robot_socket @ insertion_direction_socket`; `axis_jacobian.plan_jacobian_axis_stroke` uses damped numerical Jacobian continuation, fixed wrist orientation, joint limits, collision checking and retiming. `path_audit` samples key/hand/socket clearance and axial/lateral error. | It is a kinematic nominal path, not force-controlled contact. It handles a tilted socket axis without passing that axis into stock world-Z stroke code. |
| Contact, release and retry | Stock executor follows planned trajectories and places/retracts; it has no socket contact contract. | `follow_guarded_insertion` must be supplied by the robot-PC integration with watchdog, force/torque trace and stop acknowledgement; `release_and_clear` must open the hand only after a non-aborted held-key stroke, then reach a measured clear view. Visual failure may produce next-attempt XY advice; a released key is re-observed and restored before another attempt. | The ordinary stock joint follower is not accepted as a guarded insertion controller. No contact callback or physical reset path was run here. |
| Directed reset | Stock v8 reset scene/seed loader covers pose transitions, but the stock runner does not include this demo's fixed socket fixture and has no verified map for the new key. | Curated source-bound v8 `000→001/191` and current addendum `001→000/1687` are rehydrated to recipient paths, hash-checked and installed as directed MuJoCo seeds. The v5 audit reads only the v8 object-processing tree and installed candidate cells. | Both cells have one stable seed in the recipient audit. A seed is not a socket-aware Franka path or qualified landing. The older reverse `631` stays in noncanonical source staging and is not installed. |

## Exact active prompts and outputs

The common VLM wrapper in `observer._infer_closed_set` prepends the exact
ordered image labels and valid camera IDs. It instructs the model to return
**one compact JSON object** with exactly the requested class field,
`evidence_views`, and a ≤12-word `evidence` phrase. The first-party local
Qwen backend passes ordered PIL images to Transformers Qwen3-VL, requests
offline-only weights and decodes only newly generated tokens; it does not
acquire cameras or import ZeroDex. The built-in FR3 integration defaults to
the verified staged Qwen directory under the launch `shared_root`.

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
The refreshed v5 reset audit made no query to `AutoDex/object/paradex` and
still found one stable installed seed in each directed cell; its report is
`AutoDex/precision_insertion/cylindrical/v8_reorient_audit_no_legacy_20261013.json`
inside staging.
The recipient-mode profile with an explicit, source-verified catalog path and
canonical installed-reset audit reports 98 eligible `gap=20` candidates
(`000`: 95, `001`: 3) and one seed in each direction. Its report is
`AutoDex/precision_insertion/cylindrical/offline_asset_profile_installed_v8_20261013.json`.
The other nine modes are incomplete in this deliberately partial recipient
staging root; a zero all-mode completion count from the producer-name-only
profile would misdescribe the installed `gap=20` catalog.
After these audit changes, the full demo suite passed in the robot PC's
NumPy-1.26 `flir_env`: **907 passed, 4 skipped**. The separate NumPy-2
`flir_env_2` run failed mesh tests because its installed trimesh calls removed
`ndarray.ptp()`; that environment failure did not recur in `flir_env`.
The first-party Qwen adapter loaded the staged 2B model on CPU and returned
text for two synthetic white images. That answer used a Markdown JSON fence
and gave `evidence` as an array, so the strict observer would abstain; this
is a model-loading smoke test, not a passed task-label benchmark. CUDA was
not visible to PyTorch in this sandbox, despite host `nvidia-smi` output.
A second saved-image probe applied the **active lift prompt** to two pairs of
synthetic white shapes with no robot hand. Both camera calls returned `held`
and the consensus accepted `held`; the report is
`AutoDex/precision_insertion/cylindrical/qwen_synthetic_smoke_20261013/report.json`
inside staging. This is a concrete semantic false positive on deliberately
unrealistic images, not evidence of a physical grasp. Real paired camera
benchmarking and independent held-key checks remain required before motion.
The nonmoving verification record is saved at
`AutoDex/precision_insertion/cylindrical/recipient_staging_verification_20261013.json`
inside the staging root.

Actual recipient transport checks on 2026-10-13 used the stock
`SnapshotOrchestrator` with the current capture PCs. The first request returned
0/20 JPEGs because the cameras were `READY` and not streaming; every remote
snapshot daemon logged empty-frame JPEG encoding. A second request, after
briefly arming the idle camera streams, received 20/20 JPEGs and all 20 frame
IDs advanced. The streams were stopped, and all PCs returned to `READY` with
no controller. One viewed image showed soda cans, and detection against the
current rig ChArUco dictionaries found zero markers in every saved view.
The saved reports are under
`AutoDex/precision_insertion/cylindrical/live_camera_snapshot_probe_20261013/`
and `live_camera_stream_probe_20261013/` in staging. No socket-only or
key-with-socket SAM/FoundPose accuracy was measured.

A separate read-only FR3/Inspire probe used only the FR3 state subscription
and Inspire register reads. It received two advancing FR3 packets with seven
joint positions, seven velocities and six wrench values, and six actual plus
six commanded Inspire angles. The one diagnostic sample reported maximum
absolute FR3 velocity 0.00198 rad/s and hand actual/command gap 13 raw units;
the report is
`AutoDex/precision_insertion/cylindrical/feedback_only_probe_20261013.json`.
Neither robot nor hand motion was commanded. This does not establish the
production feedback thresholds, cross-device timing, or camera exposure
alignment.

The supplied handoff contains a `20260921_213318` camera snapshot and an
older `20260821_215517` C2R snapshot. A newer **separate** shared-data tree
contains paired `cam_param/20261012_060559` and
`handeye_calibration/20261012_060559/0/C2R.npy`; their camera JSON bytes
match. The live snapshot's 20 serials are all present in that pair, and all
20 saved JPEGs match its 1536×2048 calibration dimensions; the candidate
file-match report is
`AutoDex/precision_insertion/cylindrical/live_serial_calibration_file_match_20261013.json`
in staging. Its calibration report records a 4.15 mm mean and 8.10 mm maximum
leave-one-pose-out robot-predicted corner error, plus a 4.54 mm mean / 22.45
mm maximum AX=XB translation residual. These are measured report values,
not a current socket-pose bound or proof that this is the commissioned rig.
No current camera exposure-time calibration, fixed scene, actuator feedback
config, or ten-stage integration factory was supplied with the handoff.
