# ZeroDex capture4 pipeline review — 2026-10-27

Scope: kds-Z790 main PC, capture4's four FLIR cameras, straight Franka–Inspire
mount, cylindrical key/socket. Franka and Inspire were powered off; no robot
command, physical grasp, insertion, or recovery was tested. The older AutoDex
right-angle mount profile remains separate.

| Runtime boundary | Current implementation | Evidence and present limit |
| --- | --- | --- |
| Startup | `run_zerodex_capture4.sh` selects local profile, three VLM views and the straight-mount URDF/catalog. `run_live.py --preflight` reads static files without opening hardware. | 20 mm static check passed with 110/124 eligible candidates. An explicitly selected old right-angle catalog is rejected by its URDF hash. |
| Camera ownership | `session_launch.py` / `live_orchestration.py` arm the `stream` mode declared by the capture4 profile; capture4's separate daemons provide snapshot, SAM3 and FoundPose. The ZeroDex profile now selects `inline_png`, while AutoDex retains `shared_png`. | A 4/4 JPEG snapshot returned in 0.46 s. Same-frame source PNGs now ride with mask messages and are persisted locally. The camera-only CLI arms the stream and accepts the separate init ports. |
| Fixed socket session | Board-9 ChArUco measures the board. One SAM3 socket capture and `board_socket_fit.py` estimate board XY and board-normal axis; `session_bootstrap.py` freezes the socket collision scene. | Lit socket-only masks: 3/4 views, mean body IoU 0.618. Body IoU is now logged rather than used as an arbitrary 0.800 veto in this profile. With a key present, the refined socket prompt found 3/4 standing and 2/4 lying views. Socket identity is not guaranteed by zero IoU threshold; measure it with the key absent. |
| Per-trial key observation | The clear-view gate precedes `collect_key_capture`; SAM3 masks and FoundPose are associated with the same camera frame. `key_perception.py` selects the key pose and tabletop class. | `white plastic cylindrical key` segmented the lit lying key in 4/4 views. No lying-key FoundPose pose was obtained in the latest scene: the new images were nearly black and board corners were 0/54. |
| Grasp selection | `candidates.py` filters v8 MuJoCo-validated grasps by tabletop pose and 20 mm exact CAD hand/socket endpoint. `trial_preflight.py` then invokes the stock cuRobo pickup planner plus transfer/insertion audit. | Straight URDF catalog for gap 20: 106 eligible upright-pose grasps and 4 eligible lying-pose grasps. These are endpoint-eligible, **not** current-scene arm plans. Local RTX 3090 loaded the straight-URDF cuRobo planner. |
| Execution and labels | Demo handlers call stock pickup/lift, pre-insertion transfer, guarded axial insertion, release/clear, paired-image VLM outcome and, after failure, next-trial accumulated XY correction and repose/reorient. `action_loop.py` ends on `session_success`. | Code path and unit tests pass. Hardware callbacks, force trace and actual robot motion remain untested on this PC. `--plan-only` still moves to clear view/observes the key before stopping ahead of pickup; it is not an offline/no-motion command. |
| Repose/reorient | The demo uses directed v8 reset seeds and plans a live path from the observed key pose. It does not reuse the stock reset runner, which lacks the new key pose-index map and the fixed socket fixture. | `audit-reorient` finds one MuJoCo-stable seed at 12 cm for each direction (upright→side, side→upright), none at 0/4/8 cm. No complete current-scene Franka reset path has been tested. |

The GPU split is capture4 SAM3/FoundPose, local RTX 3090 Qwen/cuRobo. This
avoids shipping intermediate model activations. Before the latest GPU batch
change, local Qwen processed the three selected VLM views sequentially.
Four same-frame full-resolution PNGs
crossing NAS took about 20–40 s in earlier lit probes. The new inline-PNG
transport piggybacks the lossless source frame on the existing mask message,
checks its request/frame ID and decoded-pixel SHA256 against FoundPose, and
writes the evidence to this PC's local SSD. Camera-only 4-view key and socket
transport probes measured **1.75 s** and **1.72 s**, respectively, including
SAM3 mask-only processing and local PNG persistence. This is not a measured
full episode speedup, and these dark-scene probes do not show segmentation or
FoundPose success. No new port or stock capture4 file was needed.

**Runtime scope correction:** the normal per-trial key path calls
`collect_key_capture(pose_only=True)` when an initial VLM snapshot is scheduled.
It already skipped the slow NAS source-PNG wait; that snapshot runs in parallel
with candidate preflight. Thus inline PNG principally removes the one-time
socket-session PNG wait and accelerates full-image camera QA/recovery paths,
not an already-absent per-trial key PNG wait. Do not add the 20–40 s historical
probe delay to every expected episode. The old 20-camera Oct 22 trace records
`key.model_init` at 0.005 s, key FoundPose request at 3.00 s, and initial
candidate planning at 17.62 s followed by a 25.89 s same-capture continuation.
Those planning timings are not measurements of the new four-camera/straight-
mount configuration. The existing planner stops at the first full-chain pass;
parallelizing cuRobo candidates on one GPU is not assumed safe or faster.

The latest board JPEGs are in
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/lying_key_foundpose_20261027_5/`.
Their 99th-percentile intensities were 6–19/255 across four views; restore
illumination before interpreting any new SAM/FoundPose result. The current
Franka daemon binary is also absent: local CMake configure cannot find
`FrankaConfig.cmake` for libfranka >= 0.15. This is a future robot-integration
dependency, not a reason to claim the camera or planner code failed.

The Oct 24 NAS restic backup does contain the robot PC's 206,616-byte
`franka_daemon` executable, but `ldd` on a selectively restored copy reports
`libfranka.so.0.18 => not found` on kds-Z790. The backup also holds a Docker
image with libfranka 0.18, but the current account cannot access the Docker
daemon and the local filesystem has about 16 GB free versus a 26 GB image.
Do not point the live runtime at the restored executable until its matching
native library and the ZeroDex robot endpoint are commissioned. No robot
connection was attempted during this dependency check.

An Oct 28 00:00 camera-only retry confirmed that the dark scene had not
changed: four same-frame key PNGs arrived in 2.02 s, but SAM returned zero
mask pixels in all four views. Their grayscale 99th-percentile intensities
were 8, 6, 9 and 19/255. The report is
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/inline_key_20261027_2359/report.json`.
The namespaced capture4 services were stopped after the probe. The service
manager now waits for all eight camera/snapshot/perception sockets before
returning from `start`; the previous immediate return could race model startup.
This was confirmed by starting and stopping only the namespaced services.

Robot control is a separate unresolved integration decision. This profile
currently reuses AutoDex's ParaDex ZMQ `FrankaExecutor`, whose shared
`network_info["franka"]` is `172.16.0.2`. On this PC, the interface has
`172.16.1.10/24`; `ip route get 172.16.0.2` goes via the campus gateway,
not a direct robot subnet. ZeroDex's `realtime_vlm` code instead references a
ROS2 `FrankaController`. Neither observation proves the correct endpoint for
the presently powered-off robot. The intended ZeroDex control backend and
address must be identified before connecting; the shared ParaDex
`system/current/network.json` was not edited.

### Lying-key synthetic planning check (not the camera scene)

With the straight URDF, current 20 mm-gap catalogue and native locked-hand
planner mode, a hypothetical tabletop at robot Z=0.04 m, key XY=(0.4, 0) m,
and socket XY=(0.6, 0) m produced pickup plans for all four eligible lying-key
grasps. None passed the full synthetic chain: two transfer-unreachable, one
axial-waypoint-unreachable, and one held-key/table collision during transfer.
This does **not** classify the actual lying key: its measured pose, current
robot joints and socket pose were unavailable in the dark camera scene. The
diagnostic outputs are under `/tmp/precision_synthetic_lie_20261027_all`.

The synthetic diagnostic now writes the normal nested timing trace beside its
report. For the collision candidate `table/1/20777`, one run measured 39.44 s
in stock v8 pickup, 12.20 s in held transfer, and 5.13 s in geometry sampling.
Candidate selection now stops the geometry loop at its first definite
collision; a repeat measured 41.12 s, 9.65 s, and 3.59 s, respectively. These
are separate stochastic planning runs, not a controlled end-to-end speedup
estimate. Rejected reports mark the partial geometry scan, while passing
trajectories still receive the full sampled audit. The synthetic diagnostic
uses `use_cuda_graph=False`, unlike the normal live planner default, so its
cold pickup duration must not be presented as a live latency estimate.

Verification performed: the precision demo test suite passed **1252 tests**
with 5 skipped; the straight-URDF local cuRobo planner instantiated; 20 mm
static preflight passed; the old mount catalog was rejected; camera-only
services were stopped after probes. No current-scene full trajectory or
physical success is claimed.

Inline transport reports are in the local profile's
`runs/inline_transport_only_20261027_2/report.json` (key) and
`runs/inline_transport_socket_20261027_1/report.json` (socket).

All six cylindrical socket-gap catalogs were subsequently rebuilt from the
current 124-candidate v8 pool with the straight URDF and zero extra hand
clearance. Each full scan was complete with no screening errors and passed
`run_zerodex_capture4.sh --check` for its gap:

| Radial gap (mm) | Endpoint-eligible candidates | Upright / lying |
| ---: | ---: | ---: |
| 1 | 109 | 105 / 4 |
| 3 | 109 | 105 / 4 |
| 5 | 112 | 108 / 4 |
| 10 | 112 | 108 / 4 |
| 15 | 112 | 108 / 4 |
| 20 | 110 | 106 / 4 |

This table counts MuJoCo-evidenced, nominal 20 mm CAD endpoint-eligible
grasps, not completed Franka motions. The lying pose's four-candidate pool
is a coverage limitation worth addressing after a lit FoundPose and real
joint state make current-scene planning possible.

### 2026-10-28 saved lying-key replay and pipeline review

The earlier lit capture, request `2107810824`, has four same-request
undistorted PNGs and SAM3 key masks. The read-only
`probe_zerodex_saved_foundpose.py` replay uses the existing key representation
and capture4's FoundPose environment without opening a camera or robot.
Capture4's xFormers build has no float32 attention kernel for its GPU, so
FoundPose initially threw `NotImplementedError` at DINOv2 attention. The
ZeroDex precision service now sets DINOv2's supported `XFORMERS_DISABLED=1`
environment variable before import, selecting PyTorch SDPA. This is scoped
to the namespaced capture4 precision daemon and saved replay; the shared
capture4 checkout and AutoDex robot-PC launch are unchanged. A direct GPU
attention test and the four-view saved replay succeeded.

All four views returned initial poses, but symmetry-aware physical key-centre
spread was **92.3 mm** maximum and unoriented-axis spread **19.1°** maximum.
Therefore per-view FoundPose success alone cannot authorize a grasp. The
live cylinder tabletop path already treats raw disagreement as evidence and
uses AutoDex IoU selection plus silhouette refinement. Capture4's older
`InitOrchestrator` lacks the local `refine_from_payloads` wrapper; an optional
saved diagnostic instead calls its identical-hash `pose_select.py` IoU method
and capture4's silhouette optimizer directly. It selected view `25305462`
at mean IoU **0.481**, then fit the four masks to mean IoU **0.866**
(per-view **0.812–0.950**) and loss **0.000281**, below the existing
`0.003` stock loss threshold. This is mask agreement, not independently
measured 3D ground truth, and the capture4 optimizer version is not identical
to the live local optimizer. The saved result is
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/lying_key_probe_20261027_1/foundpose_core_refined_20261028.json`.

Applying the current `C2R.npy` and demo symmetry classifier to that fitted
pose selects lying tabletop stem `001` with **2.82°** axis residual. The
20 mm-gap straight-mount catalog contains four endpoint-eligible grasps for
that stem, but no actual-scene arm plan was run: this saved key probe contains
no measured robot joints or same-session frozen socket/board calibration.
Earlier synthetic planning failures must not be applied to this observed
key. Current live imagery is still almost black (the later probe had 0/4
key masks), and robot/hand remained powered off.

The principal remaining end-to-end integration gap is robot ownership, not
these four candidate files. The current precision FR3 core instantiates the
AutoDex ParaDex ZMQ executor, whereas ZeroDex documentation identifies a
ROS2 controller and robot IP `172.16.1.11`; the configured local daemon
binary is unavailable with its required libfranka version. Choose and
commission the intended ZeroDex robot backend before any motion. Static
`run_zerodex_capture4.sh --check` passes with 124 catalog rows and 110
endpoint-eligible rows, but deliberately does not establish live readiness.

The main-PC runtime environment had a separate missing dependency: its
`precision-vlm` Python did not contain `nvdiffrast` or the FoundationPose
renderer imports, although the capture4 environment did. The exact
`nvdiffrast v0.3.3` source already on the NAS was installed into that local
environment, with the required Python imports. The ZeroDex launcher now
exposes the existing local CUDA 12.1 toolkit/header directories to
`nvdiffrast`'s first-use JIT; it does not alter the AutoDex launcher. A real
RTX 3090 initialization of the current solid-MTL cylinder-key renderer
passed with 514 vertices and 1,024 faces.

`probe_zerodex_saved_current_refine.py` then fed the saved same-frame
FoundPose candidates and masks into the **current local**
`InitOrchestrator.refine_from_payloads`, not capture4's older implementation.
The same source view `25305462` was selected. Mean four-view IoU improved
from **0.481** to **0.871**, and current silhouette loss was **0.000273**;
the existing `0.003` stock gate accepted it. IoU selection took **0.064 s**
and the 100-iteration silhouette step **0.858 s** after renderer startup.
With the same `C2R.npy`, the current fitted pose classifies as lying stem
`001` at **2.38°** axis residual (the older capture4 optimizer gave 2.82°).
These times exclude capture, SAM, FoundPose, JIT/cold model load and arm
planning; they are not an episode latency claim. The exact local result is
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/lying_key_probe_20261027_1/current_stock_refined_20261028.json`.

### 2026-10-28 pipeline improvement: batched classification views

The active local Qwen path now batches the three independent same-camera
temporal pairs for lift, pre-insertion transfer and final released-key labels.
Each view retains its own prompt, raw answer, parse result and vote; only the
single RTX 3090 model invocation is shared. Metric key-point grounding is
unchanged. The run ledger records `vlm.view_batch` and `vlm.infer` with
`gpu_batch_size=3`; the aggregate observation records wall-clock batch latency
instead of adding three simultaneous view durations.

On the same three saved camera images, production-sized 384×288 inputs and
128-token maximum, a warm comparison gave **1.506 s** sequential inference
versus **0.706 s** batched inference. Both returned `miss` from all three
views; this is a deliberately identical before/after-image smoke test, not a
grasp-success evaluation. Model load was **4.934 s** separately on the same
PC. The full suite passed **1,261 tests, 5 skipped** after the capture4 lift
crop change; static `--check` reports
124 candidates, 110 endpoint-eligible. No current-scene arm plan or live
robot/camera integration was exercised in this benchmark.

The initial ZeroDex local Qwen lift route used the generic full-frame temporal
pair. A saved AutoDex held-key example was previously misclassified by that
generic observer, whereas the old system's measured-hand crop replayed
correctly on its own cameras. The ZeroDex launch now reuses that crop method
with its straight URDF, measured Franka/Inspire joints and three configured
capture4 views. Unlike the old distributed 2-view route, its three close-ups
are batched on the local RTX 3090. The old 2-view schema and saved reports
remain readable. Unit tests cover straight-mount projection, batch wiring and
report verification, and a three-image 640×640 saved-frame GPU smoke took
**0.711 s**, peaking at **4.23 GiB**. That image was not an actual held-key
crop; real capture4 hand visibility and classification accuracy remain
unmeasured. No additional success/failure threshold was introduced.

### 2026-10-28 00:50 current-scene camera check

The isolated capture4 camera/SAM3/FoundPose services were started only for a
four-view key transport probe and then stopped. All four source frames arrived
locally in **2.138 s** after init (**0.010 s**) with matching mask/pose
messages. SAM3 produced **0/4 nonempty key masks**. Grayscale 99th-percentile
intensities were **5, 8, 9, 19/255** across the cameras; the current scene
remains effectively unlit. The report is
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/inline_key_20261028_continuation_1/report.json`.
This is a fresh transport/illumination result, not a key-pose or robot-plan
result. No camera settings or shared capture4 source were changed.
