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

### 2026-10-28 execution-path audit and measured optimizations

This is the path that the current **cylindrical ZeroDex profile** would take;
it is not a claim of a completed physical episode on kds-Z790.

| Boundary | Implementation and current evidence | Latency or correctness conclusion |
| --- | --- | --- |
| Launch | `run_zerodex_capture4.sh` selects four capture4 cameras, three local-Qwen checkpoint views, the straight Franka–Inspire URDF and gap-specific endpoint catalog. `--check` gives 124 rows and 110 endpoint-eligible rows for 20 mm gap. | File preflight is not robot readiness. The straight-mount catalogue has only four eligible lying-pose rows. |
| Session socket | One board-9 measurement plus socket SAM/body fit freezes the session collision scene. | One-time, not repeated per trial. Socket-with-key masks were less consistent than socket-only masks; capture the socket before placing the key. |
| Trial key | Clear view, one same-request SAM3/FoundPose acquisition, AutoDex IoU selection and multiview silhouette fit, then tabletop pose classification. | Latest live frames were nearly black and yielded 0/4 key masks, so no fresh lying-key pose or current-scene plan is established. The saved lit lying-key replay selected stem `001`; that is not a live observation. |
| Candidate planning | Select matching v8/MuJoCo evidence and exact 20 mm hand/socket endpoint rows; run stock cuRobo pickup and the demo's held transfer/insertion audit, stopping at the first full-chain pass. | The four lying rows are not four executable plans. In an earlier synthetic scene all four failed the full path. That scene cannot substitute for measured joints and socket/key poses. |
| Motion and labels | Measured-state pickup/lift, three-view cropped lift label, held transfer, paired after-lift/preinsert label, guarded 20 mm insertion, release and clear view, paired initial/final three-view Qwen label. | VLM checkpoint views are batched locally. The final `inside_socket` label does not independently measure 20 mm depth; the motion/force record is separate evidence. No robot command was sent here. |
| Failed trial | The pre-insertion frames are retained before insertion. After a released-key failure, the XY estimate from those frames is accumulated for the **next** trial; repose/reorient restores the initial tabletop state. | This path is implemented but not physically exercised with the ZeroDex mount. The robot backend/address is still unresolved. |

The saved lit four-view key replay was repeated with the *current local*
silhouette optimizer, changing only its iteration count. The diagnostic CLI
now accepts `--sil-iters`; each run writes a separate JSON and cannot
overwrite the 100-iteration reference.

| Iterations | Mean mask IoU | Silhouette loss | Optimizer time | Symmetry-aware key-axis difference from 100 iterations |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0.787 | 0.000583 | 0.448 s | 10.55° |
| 50 | 0.807 | 0.000493 | 0.579 s | 7.25° |
| 75 | 0.837 | 0.000395 | 0.744 s | 3.62° |
| 100 | 0.871 | 0.000273 | 0.858 s | reference |

All four runs pass the existing loss threshold, but shorter runs change the
axis enough to affect grasp selection and insertion planning. Therefore the
live default remains **100 iterations**; a sub-second saved-frame gain is not
worth asserting equivalent pose quality from this one scene. These are
silhouette-fit metrics, not independent 3D ground truth. The JSON records are
under `shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/lying_key_probe_20261027_1/`.

Repeated VLM checkpoints rebuilt OpenCV's undistortion map for each camera
even though this session's calibration and resolution are fixed. The map is
now cached by dimensions and exact intrinsics/distortion (four entries, one
per capture4 camera). A changed calibration gets a new map; remapping and
image provenance are unchanged. Timing has separate
`camera.undistort_map` and `camera.undistort_remap` spans. On this PC with the
four real 2048×1536 calibration records, a standalone map-build measurement
was 0.034 s for four cameras, versus about 0.001 s for a cache hit; four
remaps took 0.059 s. This is a small repeated-checkpoint saving, **not** a
solution to the dominant FoundPose/arm-planning or unlit-scene issues.

The full precision test suite passed **1,261 tests with 5 skipped** after the
map-cache change; the directly affected camera/probe tests also passed.
The capture4 namespaced services are currently stopped. On kds-Z790 the
AutoDex ZMQ Franka executor still lacks a commissioned ZeroDex robot endpoint
and its matching native libfranka runtime; do not interpret `--check` as a
motion-ready verdict.

### 2026-10-28 01:05 camera and control follow-up

The namespaced capture4 services were started for one new camera-only key
probe and stopped immediately afterward. All four same-request frames and
mask messages arrived in **2.010 s** after a 0.005 s init. All four key
masks were empty. The saved undistorted images had grayscale 99th percentiles
of **8, 6, 9 and 19 / 255** (serials `25305462`, `25322639`, `25322642`,
`26053248`). The result is in
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/inline_key_20261028_followup_1/report.json`.
This is direct evidence that the current lying-key scene remains too dark for
SAM3; changing the prompt or lowering an acceptance threshold would not
recover information absent from the images. The probe now records grayscale
mean and p99 in future reports as **diagnostics only**, without a new gate.

The robot integration mismatch is architectural. The selected demo runtime
creates `src.execution.franka_executor.FrankaExecutor`, which obtains the
ParaDex **ZMQ** `FrankaController` from `paradex.system.current` and streams
joint velocities to a `franka_daemon`. Its held-motion adapter also checks
the local daemon binary identity. The current `paradex.system.current`
network file points at AutoDex's `172.16.0.2`, and the configured local daemon
binary is absent. `realtime_vlm/docs/ros_setup.md` instead documents a
**ROS2/franka_ros2** controller on a different machine with robot IP
`172.16.1.11`; kds-Z790 currently has no `ros2` executable or `/opt/ros`.
The ZeroDex launcher therefore declines `--run` and `--plan-only` rather
than silently sending the inherited AutoDex ZMQ commands to `172.16.0.2`.
The AutoDex launcher and shared executor were not changed.

### 2026-10-28 01:20 repeat camera probe

After starting only the namespaced capture4 services, one new same-request
four-view key probe again returned **zero SAM masks**. Grayscale p99 values
were 8, 6, 9 and 19/255 for `25305462`, `25322639`, `25322642` and
`26053248` respectively; capture and local PNG persistence took 2.172 s.
The report and raw frames are in
`shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/inline_key_20261028_continuation_2/`.
The isolated services were stopped after the probe. This confirms the live
scene is still too dark for a meaningful key-pose or full-session timing run;
it does not measure a robot or planner bottleneck. The new
`run_zerodex_capture4.sh --camera-only` path is covered by a call-order test,
but has not completed on current live frames: the key must be removed for
initial socket measurement and the board illuminated before that test.
That document alone does not prove which control backend owns this physical
ZeroDex FR3. Choosing one is required before implementing a command adapter;
merely substituting the IP in the ZMQ client would not make ROS2's joint
trajectory controller provide the ZMQ velocity/feedback API. No robot
connection or motion was attempted.

### 2026-10-28 renderer scheduling correction

The live session previously prewarmed a socket silhouette renderer before
camera streaming even when `socket_perception_mode=board_constrained_body`.
That socket path uses SAM masks and board-constrained body geometry; it never
uses the local socket silhouette optimizer. The session now prewarms the
**key** renderer in this mode, before the camera stream's heartbeat starts.
The unchanged `start_precision_session()` mask-only socket initialization
does not replace the already loaded key renderer, so the first key trial can
reuse it. FoundPose's source frames, masks and acceptance rules are unchanged.

An offline RTX 3090 constructor timing measured 12.663 s for the first
renderer (socket in that order) and 0.003 s for the second renderer (key).
Most of the 12.663 s is cold CUDA/JIT warmup and will still be paid when the
key is constructed first. This change eliminates an unused object-specific
load and shifts the key initialization out of the first trial; **it does not
establish a 12-second total-session speedup**. The board-constrained routing
is covered by a focused unit test. No new gating threshold was added.

### 2026-10-28 control-backend and launch-parameter audit

Read-only inspection of this kds-Z790 host found `enp4s0` at
`172.16.1.10/24` with a direct route to `172.16.1.11`. That establishes a
network path, **not** a running controller: the robot and hand were off and
no connection or command was attempted. The local ParaDex
`system/current/network.json` still selects `172.16.0.2`; its
`FrankaController` is a ZMQ client. In contrast, ZeroDex's
`docs/ros_setup.md` documents ROS2 Humble and a controller on another PC for
`172.16.1.11`, and its `run/components/robot_exec.py` uses ROS2
`FollowJointTrajectory`/servo primitives. This host has no `ros2` executable
or `/opt/ros`. Changing only ParaDex's robot IP cannot make the existing
AutoDex velocity-stream executor speak the ROS2 API. The ZeroDex robot owner
and intended backend remain an integration decision, not a camera or grasp
failure.

The live CLI already accepts `--xy-update-weight`, but `run_robot_pc.sh`
previously passed a literal `1.0`, making the parameter impossible to tune
through either profile launcher. It now forwards
`PRECISION_XY_UPDATE_WEIGHT` with the same `1.0` default. This is an opt-in
parameter fix; the retry update equation, per-attempt cap and outcome rules
are unchanged.

### 2026-10-28 camera latency follow-up

The capture4 precision log for the dark-scene request `1338853889` shows a
0.408 s four-camera snapshot, then four serial SAM3 attempts ending in empty
masks; the last attempt completed about 1.24 s after the snapshot. The local
inline capture report measured 2.172 s including source-frame delivery and
local PNG persistence. These figures cannot predict lit-scene FoundPose time.
The stock `Sam3ImageSegmentor.segment_batch()` currently calls sequential
`segment()` for every image after its batch setup, so invoking that method
would **not** establish a four-view batch speedup. Rehashing one decoded
1536×2048 BGR frame took about 8.6 ms in a local 20-call microbenchmark;
removing a few hashes would not address the observed second-scale path and
would complicate same-frame evidence. The normal key trial remains
`pose_only=True` and does not request source PNGs.

The same capture4 log showed two Hugging Face HEAD requests before the local
SAM3 checkpoint loaded. Both SAM3 files were present in the capture4 cache;
`HF_HUB_OFFLINE=1` resolved them and a direct SAM3 GPU load passed without
opening cameras. The namespaced precision service now sets that variable,
leaving shared capture4 programs unchanged. A controlled cold-start latency
comparison has not been run.

### 2026-10-28 candidate geometry-screen scheduling

The recorded AutoDex episode `20261022_103429_752726` spent 54.1 s across
two `audit.geometry_samples` spans (489 and 220 planned FK samples), versus
6.8 s of cumulative VLM inference. That older robot-PC trace does not measure
the current ZeroDex host or prove a current speedup. Inspection found that a
candidate rejected by geometry still paid for exact Coal distance queries on
far-separated key/hand–world pairs before reaching its first collision.

`path_audit.py` now does a collision-first pass for candidate screening:
the enclosing-AABB distance is a mathematical lower bound on mesh distance,
so a pair farther than its existing clearance requirement cannot cause a
rejection and needs no exact Coal query in that pass. A rejected candidate
stops at its first definite collision; a passing candidate then runs the
original exact minimum-distance scan. No collision threshold, clearance
requirement, accepted path, or contact policy changed. The two passes have
separate `audit.collision_screen` and `audit.minimum_distance_scan` timing
spans. Unit tests compare first-failure identity, passing exact minima and
Coal call counts. A wall-time reduction on realistic ZeroDex trajectories
remains to be measured once a valid scene and robot state are available.
