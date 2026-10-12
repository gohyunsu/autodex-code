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
   initial-tabletop/after-lift **raw** pair. The current transfer label sees
   **one raw pre-insertion image per camera**, not an after-lift/pre-insertion
   pair. That is sufficient to ask whether the *current visible state* looks
   ready to insert, but does not directly compare the state across transfer.
   If the research label is specifically "the key remained held through
   transfer," a temporal pair is preferable; adding it is a **future code
   change**, not a feature the recipient should assume is present.
2. The current CAD pre-insertion target and segmented descent use the
   **measured socket axis**, `R_robot_socket @ insertion_direction_socket`.
   Lift uses AutoDex's Jacobian vertical-stroke API; descent currently uses
   successive `plan_cartesian_pose` calls, **not** a Jacobian stroke.
   Neither a world-Z descent nor a socket-axis Jacobian descent should be
   claimed for the present code. For the upright, easy cylinder experiment,
   retain the current socket-axis target and segmented descent. Switching to
   world-Z is optional only after measuring that the fixture is sufficiently
   upright for the selected clearance; changing only the descent direction
   while leaving its target on the measured socket axis creates a mixed-frame
   path. A passing cuRobo plan is not a contact-control policy.
3. Following one guarded nominal 20 mm stroke, the robot-PC callback must
   **open Inspire and move Franka to clear view before final capture**. Each
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
   not a VLM-measured millimetre error. Then clear view, perceive the released
   key, and restore its original tabletop pose/position in the same failed
   attempt. The next attempt starts with another fresh key observation.
   `unobservable`, a controller/safety abort, or lost grasp is not the same
   failure branch and may stop for review instead of producing XY advice.

The active VLM paths receive undistorted raw images, **not CAD overlays**.
The pre-insertion checkpoint still renders an overlay for a separate geometry
audit, so its renderer dependency remains even though the VLM does not see
the overlay. The dormant `observe_preinsert_hold_views` utility is not the
active label path and does accept an overlay; do not wire it by accident.

## What exists, and what the robot PC must supply

| Concern | Already in this checkout/NAS | Robot-PC task and acceptance evidence |
| --- | --- | --- |
| Key/socket and grasp assets | Object meshes, exact fixed-socket collision mesh, FoundPose representations, v8-filtered full-key candidates, per-gap 20 mm endpoint catalogs, Qwen bundle and some directed reset seeds | Rehydrate/rebuild catalogs under the recipient's absolute `shared_data` root. File presence is not pose accuracy or an executable Franka path. |
| Cameras and perception | AutoDex camera transport, undistortion, SAM/FoundPose orchestration, frozen socket world and raw checkpoint bundles | Supply real serials, intrinsics/extrinsics, camera-to-robot transform, exposure-time metadata and synchronized measured state. Verify socket-only then key-with-socket frames, masks and repeatability on the actual rig. |
| FR3/Inspire basic resources | `franka_core.open_franka_core` constructs the existing AutoDex planner, completion-checked FR3 executor, measured-state reader and clear-view gate | Check the actual daemon/hand connection and feedback on the robot PC. Construction connects but does not call `home()`. Commission any motion separately. |
| Lift and non-contact transfer | Stage handlers, saved paths, FK/path checks, optional measured-motion adapter | Supply stage-specific execution adapters/interlocks and check measured trajectory completion and hand state. The generic trajectory follower is not a contact-insertion controller. |
| Guarded insertion | 20 mm path/handoff verifier, contact decision/trace schema and execution-evidence verifier | Implement `insertion_adapter.follow_guarded_insertion(...)` and `stop_and_acknowledge()` on the robot side, with a watchdog/contact stop, force/torque acquisition and an authentic bound metric record. Provide reviewed watchdog/contact commissioning records; do not bypass this by replaying joint points through an ordinary follower. |
| Release and final view | Ordering and SHA-bound `verify_post_insertion_release` record schema | Implement `task.release_and_clear(live, runner, decision, execution)`: open the hand, perform the checked exit to clear view, and return a new `precision_insertion_poststroke_release_clear_view_v1` JSON record. The code checks its ordering and later measured clear-view state; it cannot manufacture the physical motion. |
| Failed key restoration and reorient | Fresh released-key perception, source-pose/directed-v8 candidate selection and socket-aware reset preflight | Implement `held_return.execute_restoration` and `repose_execution.execute_repose` with authentic execution logs. Plans must be recomputed for each newly observed landing; a seed or one hypothetical passing plan is not a universal path. Observe the landing before declaring the attempt restored. |
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
git -C /home/hyunsu/autodex-code push origin feat/precision-insertion
git -C /home/hyunsu/autodex-code rev-parse HEAD
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
DATA_DIR=/absolute/recipient/shared_data
NAS_DIR=/mnt/paradex2/hyunsu
CHECK_DIR="$(mktemp -d /tmp/precision-pc-check.XXXXXX)"
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src" \
  python -m pytest -q demo/precision-insertion/tests
```

The later examples assume the same shell retains `DATA_DIR`, `NAS_DIR` and
`CHECK_DIR`; set them again if starting a new shell. `CHECK_DIR` is an
exclusive disposable report directory, not the runtime asset root.

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
python demo/precision-insertion/export_full_length_cylinder_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993"
python demo/precision-insertion/export_current_cylinder_reset_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_cylinder_reset1687_addendum_20261012"
python demo/precision-insertion/export_local_qwen_handoff.py \
  --verify-root "$NAS_DIR/autodex_precision_insertion_qwen3_vl_2b_20261012_89644892"
```

Install into a staging `shared_data` tree first, with the bundle-specific
rehydration procedures in [ROBOT_PC_HANDOFF_20261012.md](ROBOT_PC_HANDOFF_20261012.md),
[CYLINDER_SOCKET_HANDOFF_RELOCATION.md](CYLINDER_SOCKET_HANDOFF_RELOCATION.md),
and [FULL_LENGTH_CYLINDER_PROPOSALS.md](FULL_LENGTH_CYLINDER_PROPOSALS.md).
Never copy source-bound `audit_only` endpoint catalogs as runnable files:
their absolute paths and hashes belong to the producer. Install the
historical cylinder 12-ID pool before appending the new 112-ID pool if the
full 124-row catalog is desired. Install the exact key/socket FoundPose
representations and only the selected directed reset seeds; do not promote
`pending_foundpose`. Rebuild each selected per-gap catalog on the recipient.
For the current `gap_mm=20` mode, if its canonical catalog path does **not**
already exist:

```bash
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20 \
  --min-hand-clearance-mm 0 \
  --output "$DATA_DIR/AutoDex/precision_insertion/cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json"
```

Do not overwrite a pre-existing catalog: audit it first and use a new path
for regeneration if the source pool changed. Then run:

```bash
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/run_pipeline.py audit \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root "$DATA_DIR" --mode cylinder --gap-mm 20 \
  --output "$CHECK_DIR/reorient_audit.json"
```

For a source-bound catalog verification on the **selected mode**, run this
read-only check; the full all-mode profile below is optional:

```bash
PYTHONPATH=demo/precision-insertion:src python - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path
from precision_insertion.config import select_mode
from precision_insertion.live_session_runner import preflight_endpoint_catalog_file
root = Path(sys.argv[1]).resolve()
catalog = root / "AutoDex/precision_insertion/cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json"
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
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/audit_offline_asset_profile.py \
  --shared-root "$DATA_DIR" --output "$CHECK_DIR/asset_profile.json"
```

## Stage 2 — cameras, board and socket; robot not commanded

Make a copy of
[`configs/live_session_start.example.json`](configs/live_session_start.example.json)
**outside the repository**, then replace every placeholder with the AutoDex
PC's measured serials, capture-daemon address, camera calibration, exposure
time mapping, camera-to-robot transform, fixed scene and new evidence path.
The example tolerances are a schema illustration, not rig-approved values.
With the arm already physically out of the view, use the demo-local CLI:

```bash
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/recipient/session_launch.json
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/run_pipeline.py start-session \
  --config /absolute/recipient/session_launch.json --capture
```

The first invocation validates the launch files only. `--capture` connects
to the **AutoDex cameras only**: first measure the ChArUco board, then place
only the fixed socket, acquire repeated socket observations and freeze the
collision world. Inspect raw frames, SAM masks, CAD fit/rim orientation,
per-view agreement, repeated-pose residuals, camera IDs and exposure times.
The output path must be new; the command does not drive Franka to clear view.
The operator must arrange a clear view without using this command to move
the robot. A PTH loading or silhouette-loss pass alone is not socket-pose
accuracy. See [NAS_PERCEPTION_QA.md](NAS_PERCEPTION_QA.md) for the known
small-object/wrong-socket mask failure modes.

There is presently **no one-command, camera-only live key-with-socket trial**
in this CLI: key capture is normally coupled to an exposure-aligned robot
state and the live trial runner. Before physical pickup, review key-plus-
socket masks in real recipient-camera images, using a saved-image perception
probe or a robot-PC camera-only adapter that reuses the existing key
admission code without fabricating a robot state. Do not report the
socket-only `start-session --capture` as a successful key-perception test.
For a reproducible **historical NAS-frame** QA exercise (not a substitute
for new recipient-camera frames), run:

```bash
PYTHONPATH=demo/precision-insertion:src python \
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

Only on the authorized AutoDex PC, after validating the robot-specific
feedback JSON, the following *connection-only* probe constructs the existing
Franka/Inspire resources, reads one measured state and closes them. It does
not call `home()`, execute a grasp, or insert. It **does connect** to the
Franka daemon and initializes the CUDA planner, so run it under the rig
operator's normal connection procedure:

```bash
PYTHONPATH=demo/precision-insertion:src python - \
  /absolute/recipient/franka_runtime.json <<'PY'
import json
import sys
from pathlib import Path
from precision_insertion.franka_core import (
    FrankaFeedbackSettings, open_franka_core,
)
settings = json.loads(Path(sys.argv[1]).read_text())["feedback"]
core = open_franka_core(FrankaFeedbackSettings(**settings))
try:
    state = core.read_state()
    print("state_time_s:", state.sample_timestamp_s)
    print("full_q_shape:", state.full_q.shape)
finally:
    core.close()
PY
```

Confirm that the measured array has 13 FR3/Inspire joints, timestamps are
current, arm/hand samples agree, and closing releases the daemon. Then
commission `state_for_capture` against actual camera exposure intervals.
Do not infer an image-time state from a read taken after a slow VLM call.

## Stage 4 — full robot loop, only after stage adapters exist

Supply an importable robot-PC `stages_factory` and its commissioned settings
as in [LIVE_RUNTIME.md](LIVE_RUNTIME.md). Check the factory's ten stage
mappings before motion. In particular, provide a guarded-contact adapter
with real stop acknowledgement and metric trace, `release_and_clear`,
`execute_restoration`, directed `execute_repose`, pickup/lift/transfer
adapters, motion interlocks and exposure-aligned feedback. The stock
`FrankaExecutor.follow_joint_trajectory` must **not** be presented as a
guarded contact controller. Validate callback records against the verifier
schemas with mocked actuation first; then commission each physical stage
under the rig operator's procedure. Do not use `--max-steps` as a physical
single-step controller: stopping the Python loop is not a safe robot hold.

Only after these are implemented and reviewed, the current complete loop is:

```bash
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src:/absolute/robot_pc_binding" \
  python demo/precision-insertion/run_live.py \
  --launch-config /absolute/recipient/new_live_session.json \
  --endpoint-catalog "$DATA_DIR/AutoDex/precision_insertion/cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json" \
  --output-dir /absolute/new/precision_run \
  --integration precision_insertion.franka_integration:factory \
  --robot-config /absolute/recipient/franka_runtime.json \
  --label-mode manual --annotator OPERATOR_NAME \
  --xy-update-weight 1.0 --enable-robot-motion
```

For autonomous experimental labels change only `--label-mode` to
`vlm_only`; manual mode still saves the independent VLM label. The launch
uses a **new** session evidence path and a **new** run directory. The loop
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
