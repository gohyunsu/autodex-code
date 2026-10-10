# Precision insertion: square and cylinder modes

This is an implementation/status document, not a claim that the robot can
perform insertion today. The present `run_pipeline.py` still performs the
original grasp/lift/place action. `insertion_session.py` is an offline,
read-only decision layer and **does not command Franka or Inspire**.

## Ordered execution comparison

| Order | AutoDex `run_pipeline.py` today | Intended precision-insertion session | Present status |
|---|---|---|---|
| 1. Startup | Initialize robot and AutoDex cameras; optionally measure the fixed socket and ChArUco tabletop. | Require session socket pose, hand–eye/camera calibration, and immutable fixture record; construct table + socket collision scene. | Socket preflight supports both modes. Cylinder sockets use a C∞-aware center/axis repeatability metric that ignores unobservable axial yaw. Cylinder FoundPose representations remain missing. |
| 2. Perception | Distributed FoundPose estimates object pose; tabletop pose is classified. | Estimate key pose using the same AutoDex cameras and mesh; use socket pose frozen at startup. Symmetry quotient differs by mode: square yaw matters; cylinder axial yaw does not. | Cylinder geometry declares D∞ key/C∞ socket symmetry; scene snapping and fixture repeatability consume those declarations. Learned FoundPose weights still require onboarding and robot-camera validation. |
| 3. Trial choice | Use the observed pose, candidate coverage and attempted-candidate exclusions. | Filter the pre-simulated catalog by mode, gap **and observed tabletop pose**, then choose one scenario. Keep that scenario/grasp fixed across bounded retries to identify the effect of pose correction. | Offline catalog and pose-conditional selection implemented. No full-task-simulation pass exists; robot execution is blocked. |
| 4. Planning | Select a v8 grasp using coverage plus IK/collision; preflight approach and attached 10 cm lift. | For the selected scenario, require collision-checked approach, grasp, lift, *held-key transfer*, pre-insertion hold, and a 20 mm insertion path with fixture/hand clearance. | AutoDex preflight stops at lift. Square pilot has sampled insertion endpoint geometry, not a continuous Franka path; cylinder has nominal CAD fit only. |
| 5. Grasp/lift | Execute arm + Inspire grasp/lift. | Execute the same pickup, then observe a synchronized post-lift checkpoint; stop on slip or uncertain grasp. | Existing AutoDex pickup path available. New read-only VLM observer implemented, not wired into robot capture/execution. |
| 6. Transfer/hold | Transfer to a tabletop placement and plan descent. | Hold hand joints fixed; transfer key rigidly to calibrated socket pre-insertion pose; compare measured key pose with target and retain socket-frame XY residual. | Not implemented in live task pipeline. Offline bounded correction policy implemented. |
| 7. Insertion | No insertion action. | Force-limited guarded axial motion to measured 20 mm depth; square may need yaw alignment, cylinder ignores yaw. Optional finish press is separate. | Controller, F/T limits and contact dynamics not commissioned. No physical execution. |
| 8. Outcome | ChArUco lift label or manual label; update grasp candidate result. | Fuse measured depth, force/abort, grip state and read-only VLM visual class; record grasp success and task success separately. Unknown/occluded is not success. | Separate outcome schema exists; offline evidence gate exists. Live insertion outcome integration pending. |
| 9. Failure | Failed grasp/candidate can lead to another candidate, rotate, reset or reorient. | On misalignment with reliable metric pose, propose a bounded XY offset for the *same* grasp; on slip/safety abort/unknown, stop or inspect. Replan full path and enforce attempt cap. Reorient/regrasp only if fixed-grasp route has no feasible path. | Offline recommendation and attempt cap implemented. No robot retry loop. |
| 10. Success/reset | Return/release to tabletop and update grasp coverage. | Confirm task success, then pick from socket, move to safe reset region and release; record insertion success separately from grasp coverage. | Not implemented. |

## Mode and evidence contract

- `square`: current keyed socket plus keys with nominal clearances 0.1, 0.3,
  0.5, 1.0 and 1.5 mm. Yaw is task-relevant. The historical 10k full-key
  ablation produced seven candidates that passed *grasp* stability in MuJoCo
  after sampled 20 mm endpoint geometry. Only `104` and `5102` avoid declared
  contact above the handle front; even these lack whole-hand continuous
  insertion checks. The catalog defaults to `104` for **offline inspection**.
- `cylinder`: one 15 mm-radius, 80 mm-tall key and six sockets with *radial*
  gaps 1, 3, 5, 10, 15 and 20 mm. Axial yaw is irrelevant. The 20 mm radial
  gap is a bring-up condition, not a precision result. The 7 bundled STLs
  generate AutoDex mesh/scene metadata; FoundPose representation and BODex
  grasp candidates are explicitly missing. Nominal centered CAD fit is only
  `geometry_only`, not a simulated robot scenario.

Evidence levels are ordered `geometry_only` → `sampled_insertion_geometry` →
`grasp_sim_pass` → `full_task_sim_pass` → `hardware_validated`. A catalog item
must not be promoted without saved evidence for every prior gate. The live
execution gate currently requires hardware validation **and** a commissioned
controller; it is intentionally closed.

The cylinder contact proposal mask exposes the `z=0..25 mm` lateral grip zone
and the `z=0` rear cap, with the representative insertion end at `z=80 mm`.
The remaining body is forbidden. This is an initial
proposal rule, not a proof: every Inspire link must clear the socket for the
whole 20 mm descent. The square mode retains its existing key-specific
contact policy. Neither uses a VLM to choose a grasp contact point.

The square fixture is at `~/shared_data/AutoDex/precision_insertion/fixtures/unified_socket`;
all six cylinder fixtures are its siblings under `fixtures/precision_socket_cylinder_gap_XXmm`.
The cylinder family manifest is under `precision_insertion/cylindrical/` only
because it describes the family as a whole. All runtime object assets keep
the standard AutoDex `object_processing/<object_id>` layout. The legacy
root-level `cylindrical_assets.json` was archived and is not a catalog source.
The cylinder `gap_XXmm` suffix denotes one-sided *radial* nominal clearance.

## Offline usage

Use the existing AutoDex Conda environment. The commands below create local
assets and a scenario catalog but move no robot:

```bash
cd ~/autodex-code
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/build_cylindrical_assets.py \
  --shared-root ~/shared_data
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/validate_cylindrical_assets.py \
  --shared-root ~/shared_data
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/insertion_session.py catalog \
  --shared-root ~/shared_data \
  --out ~/shared_data/AutoDex/precision_insertion/scenario_catalog.json
```

Start one square pilot inspection, optionally binding the calibration JSON
created by session socket preflight:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/insertion_session.py start \
  --catalog ~/shared_data/AutoDex/precision_insertion/scenario_catalog.json \
  --mode square --gap-mm 1.5 --tabletop-pose 4 \
  --session ~/shared_data/AutoDex/precision_insertion/sessions/square_001.json \
  --fixture-pose /path/to/fixture_pose.session.json
```

For cylinder CAD-only inspection, use `--mode cylinder --gap-mm 20
--tabletop-pose 0 --minimum-level geometry_only`. This is **not** a valid robot scenario. Omitting
`--minimum-level geometry_only` correctly rejects it; `--purpose robot`
rejects both modes today.

After an external simulator or supervised trial has saved *calibrated*
measurements, provide an observation JSON. Example:

```json
{
  "grasp_held": true,
  "measured_depth_mm": 0,
  "force_within_limits": true,
  "abort_reason": null,
  "pose_error_xy_m": [0.001, -0.0006],
  "pose_uncertainty_mm": 0.2,
  "camera_labels": ["front", "top"]
}
```

The pose error means observed key center minus target center, expressed in the
**frozen socket frame**. It must come from calibrated geometry, never from a
VLM's guessed millimeters. With saved checkpoint frames and an explicit API
call, the read-only observer classifies `misaligned`, `rim_jam`, `slip`,
`partial_insertion`, `seated`, `occluded`, or `unknown`:

```bash
GEMINI_API_KEY=... ~/miniconda3/envs/autodex_bodex/bin/python \
  scripts/precision_insertion/insertion_session.py update \
  --session ~/shared_data/AutoDex/precision_insertion/sessions/square_001.json \
  --observation /path/to/observation.json --checkpoint pre_insert \
  --image front /path/to/front.png --image top /path/to/top.png \
  --gemini-model YOUR_COMMISSIONED_MODEL
```

For no-network replay, use `--vlm-response /path/to/vlm.json` instead of
`--gemini-model`. VLM JSON must include `checkpoint`, `class`, `confidence`,
`visible_evidence`, and `cameras_used`. Bad JSON, an unsupported label, or
unreferenced cameras becomes `unknown`. The decision policy stops on force
abort/slip, asks for inspection when evidence is insufficient, and changes XY
by at most 0.5 mm per step and 2 mm total (default, **not yet robot-approved**).
Task success requires measured 20 mm depth, confirmed grasp, force within
limits, no abort, and a sufficiently confident consistent visual class. The
fixture calibration file is hashed at session start and cannot silently change.

## Next gates before live use

1. Generate and test cylinder FoundPose representations and key-grasp pools.
   For both modes, verify *all* Inspire link contacts and rigid grasp across
   the continuous transfer/insertion, not only endpoint meshes.
2. Measure fixture pose in the actual AutoDex camera/hand–eye frame. Square
   preflight uses an SO(3) repeatability gate; the new cylinder code uses a
   symmetry-aware axis/center comparison but still needs actual camera and
   fixture validation. Validate calibration uncertainty against each gap.
3. Save reproducible cuRobo whole-path and MuJoCo contact simulations for
   chosen scenarios, including force/contact abort behavior and reset.
4. Commission a low-speed guarded insertion controller with measured depth,
   F/T limits, emergency stop, rollback and operator supervision. Integrate
   checkpoint capture/VLM, frozen scenario, and retry policy into a task
   action hook in `run_auto.py`; preserve original AutoDex as default.
