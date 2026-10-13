# AutoDex PC: staged precision-insertion bring-up

This is the recipient-side handoff for the **current cylinder live path**.
It is intentionally separate from stock `src/execution/run_pipeline.py` and
`run_auto.py`: use `demo/precision-insertion/run_pipeline.py` for offline and
camera-only probes, and `demo/precision-insertion/run_live.py` for the opt-in
robot loop. The square family has offline assets but the current live CLI
rejects square mode. The checked-out code, NAS payload, robot-PC calibration,
and robot-PC controller are four different prerequisites; success at one
level must not be reported as success at the next.

## Current behavior and decisions to preserve

1. The default live label mode is `vlm_only`. Lift receives each camera's
   initial-tabletop/after-lift **raw** pair. Preinsert receives that same
   camera's raw `after_lift` and raw `preinsert` frames, in time order, and
   asks whether the same key remains visibly held and roughly above the
   socket. The report binds both manifests, frame IDs, exposure times, PNG
   hashes, prompts and raw model answers. It is a coarse visual continuity
   label, not a metric XY or insertion result. CAD overlays remain a separate
   geometry audit and are never VLM inputs in this active checkpoint.
2. The pre-insertion target and Jacobian continuation follow the
   **session-frozen measured socket axis**,
   `R_robot_socket @ insertion_direction_socket`, with fixed wrist orientation.
   Demo-local `axis_jacobian.py` reuses stock AutoDex numerical Jacobian,
   collision rollout, joint limits and retiming while preserving the stock
   world-Z planner. `path_audit.py` checks sampled key, hand and socket
   geometry, progress and lateral error on the dense result. This is a
   kinematic reference for the stock FR3 force-stop follower, not a measured
   physical 20 mm stroke.
3. Following one completed nominal 20 mm stroke, the built-in release callback
   **opens Inspire and moves Franka to clear view before final capture**. It
   plans the reverse-axis exit and stock joint retract before insertion. Each
   camera then supplies its initial-tabletop/final-clear-view raw pair to
   the VLM. `inside_socket` is the default experimental task-success label
   and ends the session; it is **not** an independently measured claim that
   the physical key penetrated 20 mm. Preserve the commanded-stroke/force
   record separately. Do not rename the visual label "verified 20 mm".
4. On an eligible visual `outside_socket` failure, use the *already saved*
   pre-insertion raw multiview images for a cylinder XY diagnostic; do not
   lift the released key for another image or correct within the failed
   attempt. The next commanded wrist bias is `p_next = p_previous + w*d`,
   with default `w=1.0`, a 10 mm bound on each weighted step and no separate
   cumulative cap. A no-visible-tip fallback is an **exploratory 2 mm step**,
   not a VLM-measured millimetre error. Metric, nominal-height planar and
   exploratory advice have distinct statuses in the saved report. Then clear
   view, perceive the released
   key, and restore its original tabletop pose/position in the same failed
   attempt. The next attempt starts with another fresh key observation.
   `unobservable`, a controller/safety abort, or lost grasp is not the same
   failure branch and may stop for review instead of producing XY advice.

The active VLM paths receive undistorted raw images, **not CAD overlays**.
The pre-insertion checkpoint still renders an overlay for a separate geometry
audit, so its renderer dependency remains even though the VLM does not see
the overlay. The dormant `observe_preinsert_hold_views` overlay utility and
the historical single-image `observe_preinsert_raw_views` utility are not the
active label path.

## What exists, and what the robot PC must supply

| Concern | Already in this checkout/NAS | Robot-PC task and acceptance evidence |
| --- | --- | --- |
| Key/socket and grasp assets | Object meshes, exact fixed-socket collision mesh, FoundPose representations, v8-filtered full-key candidates, per-gap 20 mm endpoint catalogs, Qwen bundle and some directed reset seeds | Rehydrate/rebuild catalogs under the recipient's absolute `shared_data` root. File presence is not pose accuracy or an executable Franka path. |
| Cameras and perception | AutoDex camera transport, undistortion, SAM/FoundPose orchestration, frozen socket world and raw checkpoint bundles | Supply real serials, intrinsics/extrinsics, camera-to-robot transform, exposure-time metadata and synchronized measured state. Verify socket-only then key-with-socket frames, masks and repeatability on the actual rig. |
| FR3/Inspire basic resources | `franka_core.open_franka_core` constructs the existing AutoDex planner, completion-checked FR3 executor, measured-state reader and clear-view gate | Use the separate read-only feedback probe for Stage 3. `open_franka_core` creates command-capable controllers and may send setup and hand-angle writes; reserve it for the motion-authorized stage. |
| Lift and non-contact transfer | Stage handlers, saved paths, FK/path checks, optional measured-motion adapter | Supply stage-specific execution adapters/interlocks and check measured trajectory completion and hand state. The generic trajectory follower is not a contact-insertion controller. |
| Cylinder insertion | 20 mm axial path/handoff verifier, stock FR3 12 N wrist-Z force stop at stock speed, reverse-path ascent on contact, and bound metric record | Supply the stage limits and live interlock. Contact is a failed attempt; the key stays gripped for the held-return action. No separate contact-controller review or contact-site validation is used by this route. |
| Release and final view | The built-in `StockSocketRelease` plans opening, reverse-axis exit and stock clear-view retract before insertion; after a non-aborted stroke it opens Inspire, executes those paths and saves a SHA-bound `precision_insertion_poststroke_release_clear_view_v1` record | Supply the existing `PathAuditLimits` as `held_return.limits` or `shared.limits`, measured feedback and a valid frozen world. An explicit `task.release_and_clear` can override the default. A force stop does not release at the socket. Plan and offline tests are not evidence of physical insertion depth. |
| Failed key restoration and reorient | The built-in stock FR3 held-return recorder saves measured waypoints after a force stop. The built-in `StockReposeExecutor` can execute a complete directed v8 pickup, held motion, release and clear-view plan, and saves seven measured checkpoints. | Supply the existing pickup motion interlock and stage parameters; an explicit `execute_restoration` or `execute_repose` may override the stock callback. A newly observed landing requires a fresh plan. Offline passing paths and measured robot waypoints are not evidence of where the key landed; recapture it before declaring restoration. |
| VLM outcome | Local Qwen default, optional manual label with independent VLM result retained, visual label reports | Test on *real paired AutoDex frames*. A parseable answer or `inside_socket` is not a metric depth test. Decide separately whether an independently verified 20 mm metric is part of the study endpoint. |

The precise callback signatures and record fields are in
[LIVE_RUNTIME.md](LIVE_RUNTIME.md),
[`guarded_insertion_execution.py`](precision_insertion/guarded_insertion_execution.py),
[`post_insertion_release.py`](precision_insertion/post_insertion_release.py),
and [REPOSE_EXECUTION_EVIDENCE.md](REPOSE_EXECUTION_EVIDENCE.md).
The optional robot-PC `stages_factory` supplies the ten stage mappings for
`precision_insertion.franka_integration:factory`; `franka_core` injects the
shared planner/reader, default Qwen and checkpoint camera callbacks where
specified. There is **no** complete commissioned factory/controller in the
NAS asset bundle. This is a robot-specific integration gap, not a missing
STL, `repre.pth`, or grasp candidate.

## Stage 0 — separate checkout and immutable baseline

On the producer, first push the intended commit. At handoff time the local
`feat/precision-insertion` branch may be ahead of `origin`; copying a command
from this file does not push it. On the AutoDex PC, use a **new directory**:

```bash
git -C /home/robot/autodex-precision-insertion status --short --branch
git -C /home/robot/autodex-precision-insertion rev-parse HEAD
git -C /home/robot/autodex-precision-insertion remote get-url origin
git -C /home/robot/autodex-precision-insertion config user.name
gh auth status  # verify the active account is gohyunsu before pushing
git -C /home/robot/autodex-precision-insertion push origin feat/precision-insertion
```

If the push is blocked by DNS/authentication, do not claim that the AutoDex
PC's clone has the latest branch. Resolve that first or hand off the exact
commit by an explicitly verified archive. Then clone on the recipient:

```bash
git clone --single-branch --branch feat/precision-insertion \
  https://github.com/gohyunsu/autodex-code.git \
  /home/ROBOT_USER/autodex-precision-insertion
cd /home/ROBOT_USER/autodex-precision-insertion
git rev-parse HEAD
git status --short --branch
git remote get-url origin  # must be gohyunsu/autodex-code.git
git config user.name gohyunsu
git config user.email gohyunsu@users.noreply.github.com
```

Compare the SHA to the producer's `git rev-parse HEAD`. Do not `git pull`
inside the stock AutoDex checkout or copy demo files into stock
`src/execution`. Use the existing AutoDex Python/CUDA environment (or a
separate environment with equivalent dependencies); keep the Qwen dependency
installation isolated if necessary. Check `python --version` and imports
before proceeding.

## Stage 1 — no camera and no robot connection

Set `DATA_DIR` to the recipient's actual installed/staging `shared_data`
root, not the producer's `/home/hyunsu/shared_data`. The NAS root below is
the mounted handoff location, **not** a replacement for a source-rebound
runtime catalog.

```bash
cd /home/ROBOT_USER/autodex-precision-insertion
DATA_DIR=/absolute/new/staging/shared_data
NAS_DIR=/home/robot/shared_data/hyunsu  # use the actual mount on the recipient PC
CHECK_DIR="$(mktemp -d /tmp/precision-pc-check.XXXXXX)"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" \
  python -m pytest -q demo/precision-insertion/tests
```

The later examples assume the same shell retains `DATA_DIR`, `NAS_DIR` and
`CHECK_DIR`; set them again if starting a new shell. `CHECK_DIR` is an
exclusive disposable report directory, not the runtime asset root.
On this robot PC, the complete demo test suite passed with
`/home/robot/anaconda3/envs/flir_env/bin/python` (NumPy 1.26): 911 passed,
4 skipped on 2026-10-13. `flir_env_2` has NumPy 2 with the installed older
trimesh, whose `ndarray.ptp()` call raises during mesh tests; use a compatible
environment when repeating the suite.

Read back the relevant NAS manifests **before installing**. These commands
do not connect to a camera or robot:

```bash
cd "$NAS_DIR/autodex_precision_insertion_handoff_20261010_652ac909"
sha256sum -c SHA256SUMS
cd "$NAS_DIR/autodex_precision_insertion_runtime_addendum_20261010_652ac909"
sha256sum -c SHA256SUMS
cd "$NAS_DIR/autodex_precision_insertion_perception_eval_20261010_652ac909"
sha256sum -c SHA256SUMS
cd /home/ROBOT_USER/autodex-precision-insertion
python demo/precision-insertion/verify_handoff_manifest.py \
  --bundle-root "$NAS_DIR/autodex_precision_insertion_tabletop_v8_handoff_20261011_955e133"
python demo/precision-insertion/verify_handoff_manifest.py \
  --bundle-root "$NAS_DIR/autodex_precision_insertion_foundpose_generated_20261011"
printf '%s  %s\n' \
  3198df432a5beb7d3a3470b6021512045171d27d3db85fec50a444f330b67c96 \
  "$NAS_DIR/autodex_precision_insertion_foundpose_20261011/autodex_precision_foundpose_4objects_20261011.tar" \
  | sha256sum -c -
printf '%s  %s\n' \
  c11a637bf167e48585baadd13734cfef5fd50072c1256d771518ba9c88fcf8a8 \
  "$NAS_DIR/autodex_precision_insertion_foundpose_20261011/precision_foundpose_socket_20mm_20261011.tar" \
  | sha256sum -c -
python demo/precision-insertion/export_full_length_cylinder_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993"
python demo/precision-insertion/export_current_cylinder_reset_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_cylinder_reset1687_addendum_20261012"
python demo/precision-insertion/export_local_qwen_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_qwen3_vl_2b_20261012_89644892"
```

Install into a **new, empty** staging `shared_data` tree. The current
full-length 112-ID cylinder pool is sufficient as the v8 source pool; the
earlier 12-ID pool is not a runtime dependency. The base bundle's tabletop
scenes and socket fixture contain producer absolute paths and must be rebound
before screening. For this verified base bundle, the embedded producer root
is `/home/robot/shared_data`:

```bash
test ! -e "$DATA_DIR" && mkdir -p "$DATA_DIR"
BASE="$NAS_DIR/autodex_precision_insertion_handoff_20261010_652ac909/payload/shared_data"
cp -a "$BASE/." "$DATA_DIR/"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/rehydrate_cylinder_socket_handoff.py \
  --source-shared-root "$BASE" \
  --source-origin-shared-root /home/robot/shared_data \
  --target-shared-root "$DATA_DIR" --install
python demo/precision-insertion/rehydrate_cylinder_key_scenes.py \
  --source-shared-root "$BASE" \
  --source-origin-shared-root /home/robot/shared_data \
  --target-shared-root "$DATA_DIR" --install
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/rehydrate_full_length_cylinder_handoff.py \
  --bundle-root "$NAS_DIR/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993" \
  --target-shared-root "$DATA_DIR" --install
```

The installers verify source files and refuse conflicting recipient files.
Install the exact key/socket FoundPose representations and the selected
directed reset seeds using their handoff manifests. Never use a source-bound
`audit_only` catalog as a runtime catalog. Rebuild the complete selected-gap
catalog on this recipient root. See
[ROBOT_PC_HANDOFF_20261012.md](ROBOT_PC_HANDOFF_20261012.md) for the remaining
asset inventories. The key and every socket are **white**: use specific white
key/socket SAM prompts, not color cues from old examples.
The following commands install the current cylinder v8 runtime inputs into
the new staging root. The curated `000→001/191` archive is only source
evidence for the v8 seed; runtime uses the rebound canonical copy, never its
producer paths. The reverse cell uses the current `001→000/1687` addendum.

```bash
tar --no-same-owner -xf \
  "$NAS_DIR/autodex_precision_insertion_foundpose_20261011/autodex_precision_foundpose_4objects_20261011.tar" \
  -C "$DATA_DIR" --wildcards \
  'AutoDex/foundpose_assets/precision_key_cylinder_r15_h80/*'
tar --no-same-owner -xf \
  "$NAS_DIR/autodex_precision_insertion_foundpose_20261011/precision_foundpose_socket_20mm_20261011.tar" \
  -C "$DATA_DIR" --wildcards \
  'AutoDex/foundpose_assets/precision_socket_cylinder_gap_20mm/*'
cp -a /home/robot/shared_data/AutoDex/content/assets/robot/fr3_inspire_description \
  "$DATA_DIR/AutoDex/content/assets/robot/"
RESET_BUNDLE="$NAS_DIR/autodex_precision_insertion_cylinder_reset1687_addendum_20261012"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/rehydrate_reset_handoff.py \
  --source-shared-root "$RESET_BUNDLE/payload/shared_data" \
  --source-origin-shared-root /home/hyunsu/shared_data \
  --target-shared-root "$DATA_DIR" \
  --source-candidate-root "$RESET_BUNDLE/payload/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12" \
  --target-candidate-root "$DATA_DIR/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12" \
  --source-audit "$RESET_BUNDLE/audit_only/SOURCE_AUDIT.json"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/install_cylinder_reset_seeds.py \
  --shared-root "$DATA_DIR" \
  --source-reset-root "$DATA_DIR/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12" \
  --gap-mm 20 --from-pose-stem 1 --to-pose-stem 0 --seed-id 1687 \
  --output-manifest "$CHECK_DIR/reset1687_install.json"
CURATED="$NAS_DIR/autodex_precision_insertion_reorient_curated_20261011/cylinder"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/rehydrate_curated_reorient.py \
  --mode cylinder --gap-mm 20 \
  --source-shared-root "$CURATED/source_shared_data" \
  --source-handoff-root "$CURATED" --target-shared-root "$DATA_DIR" \
  --target-handoff-root "$DATA_DIR/AutoDex/precision_insertion/cylindrical/reorient_curated_v8_rehydrated"
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/install_cylinder_reset_seeds.py \
  --shared-root "$DATA_DIR" \
  --source-reset-root "$DATA_DIR/AutoDex/precision_insertion/cylindrical/reorient_curated_v8_rehydrated/reset_12" \
  --gap-mm 20 --from-pose-stem 0 --to-pose-stem 1 --seed-id 191 \
  --output-manifest "$CHECK_DIR/reset191_install.json"
MODEL_DIR="$DATA_DIR/AutoDex/models/Qwen3-VL-2B-Instruct"
test ! -e "$MODEL_DIR" && mkdir -p "$MODEL_DIR"
cp -a "$NAS_DIR/autodex_precision_insertion_qwen3_vl_2b_20261012_89644892/payload/model/." "$MODEL_DIR/"
```

With `precision_insertion.franka_integration:factory`, an omitted
`vlm.model_id` resolves to this directory under the launch `shared_root`.
An explicit absolute `vlm.model_id` may select another verified local copy.
The first-party Qwen loader uses Transformers with `local_files_only=True`;
it never requires ZeroDex's `main.vlm_base` or downloads a model. The FR3
copy command assumes the robot PC has its existing
`/home/robot/shared_data/AutoDex` asset tree; if it is mounted elsewhere,
substitute that verified path before copying. A successful file copy is not a
camera, GPU inference, or robot connection test.

Select the actual source-rebound `gap_mm=20` catalog path on this recipient.
The staged recipient used in the 2026-10-13 checks has a verified catalog
named `recipient_gap20_endpoint_catalog_20261013.json`:

```bash
CATALOG20="$DATA_DIR/AutoDex/precision_insertion/cylindrical/recipient_gap20_endpoint_catalog_20261013.json"
```

If that path does **not** already exist on a different recipient, select a new
output path and build a catalog there:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20 \
  --min-hand-clearance-mm 0 \
  --output "$CATALOG20"
```

Do not overwrite a pre-existing catalog: audit it first and use a new path
for regeneration if the source pool changed. Set `CATALOG20` to that selected
new absolute path before the remaining commands. Then run:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_pipeline.py audit \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20 \
  --output "$CHECK_DIR/reorient_audit.json"
```

For a source-bound catalog verification on the **selected mode**, run this
read-only check; the full all-mode profile below is optional:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python - "$DATA_DIR" "$CATALOG20" <<'PY'
import sys
from pathlib import Path
from precision_insertion.config import select_mode
from precision_insertion.live_session_runner import preflight_endpoint_catalog_file
root = Path(sys.argv[1]).resolve()
catalog = Path(sys.argv[2]).resolve()
result = preflight_endpoint_catalog_file(
    mode=select_mode("cylinder", 20), shared_root=root,
    catalog_path=catalog)
print("candidate rows:", len(result["candidates"]))
print("eligible:", sum(row.get("eligible") is True for row in result["candidates"]))
PY
```

Inspect the selected `audit` result for `file_inputs_present=true`, the
source-bound catalog for nonzero eligible candidates in the intended
tabletop poses, and both directed reset cells in the reset audit.
`audit-reorient` exits
successfully even when a directed cell has zero seeds: inspect its JSON
counts, not just the process exit code. None of these checks plans a physical
Franka motion or tests a camera. A reset seed does not imply a complete
pickup-to-release trajectory at a new observed location.

If **all ten** square/cylinder modes and their staged reset roots were
installed, also produce the full inventory:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/audit_offline_asset_profile.py \
  --shared-root "$DATA_DIR" --output "$CHECK_DIR/asset_profile.json" \
  --catalog-override "cylinder:20=$CATALOG20" \
  --audit-installed-resets
```

This same command is useful on a partial recipient staging root: other
uninstalled modes will be reported missing. The explicit catalog override
is rechecked against v8 source hashes and the selected `shared_root`; the
reset option counts installed canonical v8 cells rather than handoff files.

## Stage 2 — cameras, board and socket; robot not commanded

Make a copy of
[`configs/live_session_start.example.json`](configs/live_session_start.example.json)
**outside the repository**, then replace every placeholder with the AutoDex
PC's measured serials, capture-daemon address, camera calibration, exposure
time mapping, camera-to-robot transform, fixed scene and new evidence path.
The example tolerances are a schema illustration, not rig-approved values.
With the arm already physically out of the view, use the demo-local CLI:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/recipient/session_launch.json
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/recipient/session_launch.json --capture
```

The first invocation validates the launch files only. Before `--capture`,
start the AutoDex camera streams and keep the key absent. With the default
`empty_before_socket` phase, start with an empty board: the CLI validates
its full 54-corner measurement, prompts the operator to place the fixed
white socket, then captures the socket. For the current board-9 scene with
the socket already present, set `board_id: "9"`,
`board_capture_phase: "socket_present_partial"`, and an explicit
`minimum_board_corners` from 36 to 53. This mode checks board-wide corner
coverage and the same reprojection, plane, and rigid-fit bounds. `--capture`
connects to the **AutoDex cameras only** and freezes the
collision world. Inspect raw frames, SAM masks, CAD fit/rim orientation,
per-view agreement, repeated-pose residuals, camera IDs and exposure times.
The output path must be new; the command does not drive Franka to clear view.
The operator must arrange a clear view without using this command to move
the robot. A PTH loading or silhouette-loss pass alone is not socket-pose
accuracy. See [NAS_PERCEPTION_QA.md](NAS_PERCEPTION_QA.md) for the known
small-object/wrong-socket mask failure modes.

On 2026-10-13, a camera-only transport probe briefly armed the five idle
capture PCs, received **20/20 live JPEGs**, observed advancing frame IDs on
all 20 cameras, and returned every camera to `READY` with no controller.
The saved report is at
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical/live_camera_stream_probe_20261013/report.json`.
One inspected view (`24122734.jpg`) showed soda cans in a box, not the
ChArUco/key/socket setup. A saved-JPEG check found **zero known ChArUco
markers in all 20 views** (`charuco_presence_probe.json` beside the report).
This proves camera transport only; the board,
socket-only SAM/FoundPose and key-with-socket stages still need real captures.
The separate `20261012_060559` camera/C2R pair contains all 20 live camera
serials, and its 1536×2048 dimensions match every saved JPEG (see
`live_serial_calibration_file_match_20261013.json` in the same cylindrical
staging directory). This file match does not commission its robot-frame
accuracy or camera exposure timing.

After the operator changed the scene, a second camera-only probe saved 20/20
JPEGs showing the ChArUco board and white fixed socket. Its report, images,
and per-view marker detections are in
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical/live_socket_only_camera_probe_stream_20261013/`.
The streams were turned off and all five capture PCs returned to `READY`.
The installed AutoDex SAM3 source, local 2026-10-10 weight, and exact prompt
`white plastic cylindrical socket` yielded nonempty masks in 19/20 views;
one view had no mask. On inspected overlays, the mask covered the socket wall
but omitted part of its base flange. Saved-mask FoundPose replay with the
recipient v8 gap-20 mesh/repre and candidate `20261012_060559` camera pair
returned a pose in 19/20 views, but their CAD-center spread reached
**425.8 mm**. These are raw per-camera candidates, before stock selection.
A second offline replay used the unchanged current AutoDex IoU/silhouette
refinement on the 19 pose-bearing views: it selected camera `25322643`,
kept 14/19 views inside the silhouette optimizer, and returned a pose with
loss **0.00154** below its **0.003** limit. Mean post-refinement IoU was
**0.792**; 8/19 views exceeded the example **0.8** per-view threshold.
The report is `foundpose_stock_refinement_19_views.json` beside the JPEGs.
The demo records every view score and falls back to this stock result if its
extra all-view IoU condition would block a capture. The saved replay did not
verify acquisition times, repeated captures or camera-to-robot accuracy; do
not freeze a socket pose or plan motion from this diagnostic alone.

The visible board's marker IDs identify **board 9**. The stock tabletop
preflight defaults to board 11 and all 54 corners; the demo now selects board
9 explicitly while preserving that default. With the socket present, 42/54
board-9 corners were triangulated from the saved 20-camera capture. The
partial-board diagnostic had 0.41 px median / 0.65 px maximum median-corner
reprojection error, 0.18 mm plane RMS, and 0.30 mm rigid-fit RMS. Its report
is `board9_partial_tabletop_probe.json` beside the JPEGs. These are internal
fit residuals for the candidate calibration, not independent robot-frame
accuracy. No key-with-socket image was captured in this probe.

The camera-only key/socket diagnostic now captures the board and fixed socket
with the current AutoDex session, pauses while the operator places the white
key without moving the socket, then collects the same request's 20 raw key
frames, SAM masks and FoundPose outputs. It uses the staged board-9/gap-20
launch with a fresh evidence directory and **never opens FR3/Inspire**. The
CLI arms the idle capture PCs at the stock 10 FPS before the first image and
stops/releases that camera stream on exit:

```bash
cd /home/robot/autodex-precision-insertion
QA_BASE=/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/probe_live_key_socket.py \
  --launch-config "$QA_BASE/static_preflight_board9_gap20_20261013/launch.json" \
  --evidence-dir "$QA_BASE/key_socket_camera_qa_$(date +%Y%m%d_%H%M%S)"
```

The exclusive `key_socket_qa/report.json` binds each frame's acquisition
metadata, raw PNG digest, saved mask, prompt and FoundPose result. An empty
mask or failed pose stays visible in the report. This is **camera-only QA**;
it does not admit a key pose for robot motion, fabricate measured robot
state, or prove socket/key pose accuracy. Inspect each view before pickup.
At 12:35 on 2026-10-13, a fresh 20/20 camera snapshot succeeded after the
capture PCs were armed. Its contact sheet and frame manifest are in
`current_scene_snapshot_20261013_123549/` in the cylindrical staging root.
The workspace was covered by a black cloth, so board 9, the socket and key
were not visible in the downsampled preview. That capture is a scene check,
not the required socket/key pose QA. The operator must restore the board and
white socket before the camera-only diagnostic above can run.
For a reproducible **historical NAS-frame** QA exercise (not a substitute
for new recipient-camera frames), run:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/audit_perception_handoff.py \
  --capture-index "$NAS_DIR/autodex_precision_insertion_runtime_addendum_20261010_652ac909/latest_capture_index.json" \
  --candidate-index "$NAS_DIR/autodex_precision_insertion_runtime_addendum_20261010_652ac909/existing_foundpose_candidates.json" \
  --evaluation-root "$NAS_DIR/autodex_precision_insertion_perception_eval_20261010_652ac909" \
  --output "$CHECK_DIR/perception_handoff.json"
```

Its masks and pose results include known failures; this checks readable
evidence and points to views for review, not the current live camera path.
Similarly, a saved-image Qwen smoke test proves model loading/JSON parsing,
not grasp or insertion-label accuracy. The existing `probe_vlm.py`
`--task insertion_visual` exercises a **historical** preinsert/final prompt,
not the active released-key initial/final success checkpoint.

## Stage 3 — robot connection and measured feedback, no motion

Only on the authorized AutoDex PC, use the current robot-network configuration
to select the FR3 state PUB endpoint and Inspire Modbus endpoint. The probe
subscribes to **FR3 state port 5556** and uses only Modbus function 03 reads
for Inspire actual/command angle registers. It never opens FR3 command port
5555, constructs a robot controller, or writes hand registers. Use a new
output path and the rig operator's normal connection procedure:

```bash
python demo/precision-insertion/probe_robot_feedback_only.py \
  --franka-host 127.0.0.1 --franka-state-port 5556 \
  --inspire-host 192.168.11.210 --inspire-port 6000 \
  --output /absolute/new/feedback_only_probe.json
```

The addresses above match this PC's current ParaDex network configuration;
verify them before running elsewhere. Confirm two advancing 7-joint FR3
packets and six measured/commanded Inspire registers. This read-only probe
does not prove the two devices are exposure-aligned or stationary, nor does
it construct the production 13-joint `LiveRobotState`. Commission the
production feedback limits and `state_for_capture` against actual camera
exposure intervals separately. Do not use `open_franka_core` for Stage 3:
constructing its stock executor can write hand-angle and FR3 setup commands.

The 2026-10-13 run on this PC received two advancing FR3 state packets and
six Inspire actual and command registers, using only the read paths above.
The saved report is at
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical/feedback_only_probe_20261013.json`.
It observed a maximum absolute FR3 velocity of 0.00198 rad/s and maximum
Inspire actual/command difference of 13 raw units in that one sample. These
are diagnostic observations, not commissioned hold or exposure-time limits.
On a later pre-clear-view check the same state subscriber received no two
advancing FR3 packets within five seconds, although ports 5555/5556 were
listening. The existing `franka` tmux daemon repeatedly logged `Net
Exception`. The command-capable controller was not opened and no clear-view
motion was sent. Recheck FR3 power, Desk/FCI and the robot network, then
repeat the read-only feedback probe before planning a first free-space move.
The operator then activated FCI. Restarting the existing daemon restored
advancing FR3 state. After one more interrupted connection during concurrent
CUDA compilation, that compilation was stopped, the daemon reconnected, and
the read-only FR3/Inspire probe passed again. With on-site supervision and
the operator's explicit `home()` authorization, stock
`FrankaExecutor.home(clear_view=True)` was called once with an **agent-chosen
0.2 arm speed scale**. The operator objected to that slower-than-stock
setting; future commanded motions should use the stock default unless the
operator specifies otherwise. The move completed in 16.47 s. A separate
read-only probe measured maximum FR3 clear-view joint error **0.00989 rad**,
maximum velocity **0.00090 rad/s**, and Inspire command registers all at
1000 with actual readings `[999, 999, 998, 998, 1000, 986]`. Evidence is
`stock_clearview_home_20261013.json` and
`feedback_only_probe_post_home_20261013.json` in the cylindrical staging
directory. This verifies the endpoint after one free-space move; it does not
commission pickup, contact control, socket accuracy or insertion.

## Stage 4 — full robot loop, only after stage callbacks exist

Supply an importable robot-PC `stages_factory` and its commissioned settings
as in [LIVE_RUNTIME.md](LIVE_RUNTIME.md). Check the factory's ten stage
mappings before motion. Cylinder insertion uses the built-in stock FR3
12 N force stop and reverse-axis ascent, without a separate contact
commissioning record. The FR3 integration injects stock `release_and_clear`
with the existing path-audit limits and stock `execute_restoration`/
`execute_repose` with the pickup motion interlock. Supply all ten stage
mappings, the site limits, pickup/lift/transfer adapters, interlocks and
exposure-aligned feedback. The stock
`FrankaExecutor.follow_joint_trajectory` is not the force-stop entry point;
the cylinder adapter calls `_follow` with `stop_wrench_z=12.0`. Validate
callback records against the verifier schemas with mocked actuation first.
Do not use `--max-steps` as a physical
single-step controller: stopping the Python loop is not a safe robot hold.

The launch and 20 mm catalog can first be checked without cameras,
CUDA or robot construction:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" python \
  demo/precision-insertion/run_live.py --preflight \
  --launch-config /absolute/recipient/new_live_session.json \
  --endpoint-catalog "$CATALOG20" \
  --robot-config /absolute/recipient/franka_runtime.json
```

This checks robot-config syntax without importing the external stage factory
or claiming physical readiness. A file with
`"execution_scope": "static_preflight_only"` cannot start camera capture or
robot motion.

On this robot PC, the exact **non-moving** 20 mm board-9 check is already
available with the staged recipient files:

```bash
STAGE=/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" \
  /home/robot/anaconda3/envs/flir_env/bin/python \
  demo/precision-insertion/run_live.py --preflight \
  --launch-config "$STAGE/AutoDex/precision_insertion/cylindrical/static_preflight_board9_gap20_20261013/launch.json" \
  --endpoint-catalog "$STAGE/AutoDex/precision_insertion/cylindrical/recipient_gap20_endpoint_catalog_20261013.json"
```

The saved `static_preflight_board9_gap20_20261013/report.json` records 20
serials present in the camera calibration files and 98 eligible candidates
out of 112. Its launch uses
bounded request timing because same-frame exposure calibration is not yet
commissioned; the `static_preflight_only` scope also prevents that example
scene and illustrative tolerances from being used for a live run. The report
has `robot_ready: false` and no camera session was created.

After stage commissioning, the motion command is:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src:/absolute/robot_pc_binding" \
  python demo/precision-insertion/run_live.py \
  --launch-config /absolute/recipient/new_live_session.json \
  --endpoint-catalog "$CATALOG20" \
  --integration precision_insertion.franka_integration:factory \
  --robot-config /absolute/recipient/franka_runtime.json \
  --label-mode manual --annotator OPERATOR_NAME \
  --xy-update-weight 1.0 --enable-robot-motion
```

For autonomous experimental labels change only `--label-mode` to
`vlm_only`; manual mode still saves the independent VLM label. The launch
uses a **new** session evidence path. `--exp-name` defaults to
`precision-insertion`; without `--output-dir` the new run goes under stock
`~/shared_data/AutoDex/experiment/precision-insertion/inspire/<cylinder_key>/_pipeline_runs/`.
Each physical episode appears beside `_pipeline_runs` with its own
`result.json`, append-only states and evidence; `summary.json` indexes them.
The loop
itself begins with a clear-view move, so omitting `--enable-robot-motion`
does **not** turn it into a full dry run. Inspect `action_loop_result.json`,
the three checkpoint reports, raw camera manifests, guarded metric/release
records, and, after failure, the saved preinsert XY report and restoration
landing. Exit code 0 means the selected **task label** was positive; it is
not proof that the physical key reached exactly 20 mm. Exit code 2 is a
normal non-success terminal result. A thrown exception is not converted to
success or automatic retry. Preserve all attempt evidence for diagnosis.

### Before calling the handoff complete

- Demonstrate one camera-only socket session and one actual key-with-socket
  mask review using the recipient's calibrations and exposure metadata.
- Demonstrate that the robot-only connection/feedback probe works without
  motion, and that the callback factory imports with all ten stages.
- Demonstrate controller-side stop, contact/force record, ordered hand-open
  then clear-view record, and fresh post-release image timestamps.
- Demonstrate both `inside_socket` and visible `outside_socket` branches;
  the latter must use the **saved** pre-insertion images, restore the key in
  the same failed attempt, and begin the next attempt with new perception.
- Record `unobservable`, abort and lost-grasp branches separately. They
  are not silently converted into measured XY corrections.
- Replan reset/reorient for the actually observed start/target poses. The
  installed directed seeds are not universal Franka trajectories; one
  synthetic `000→001` descent was unreachable, while one `001→000` scene
  passed only hypothetical full-chain preflight.
