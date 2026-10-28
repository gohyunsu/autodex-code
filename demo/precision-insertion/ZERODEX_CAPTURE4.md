# ZeroDex capture4 profile (kds-Z790)

This profile lives on branch `feat/zerodex-precision-insertion`. The AutoDex
camera/robot profile and `src/execution` sources are not changed. The local
kds-Z790 PC runs session orchestration, Qwen VLM and planning. Capture4
(`147.46.219.136`, SSH port `77`) provides four cameras and a persistent
SAM3/FoundPose process. Its existing ParaDex checkout, global `system/current`
configuration, calibration and Python environments are **not edited**.
For the robot-free replay of previously lit four-camera socket/key photographs,
see [SAVED_CAPTURE4_SCENE_20261028.md](SAVED_CAPTURE4_SCENE_20261028.md).

The capture-side adapter is copied only to
`/home/capture4/precision_insertion_capture4_20261027`. It starts three
processes (camera, snapshot, precision perception) on the established ports.
`start` returns only after their eight required sockets are listening, so the
following capture command does not race the perception model's startup.
The namespaced precision process uses its existing cached SAM3 checkpoint
offline, avoiding Hugging Face HEAD requests on each cold start; a direct
offline model-load test passed on capture4. This does not change the shared
camera or snapshot processes.
Use `manage_zerodex_capture4_services.py stop` when a camera-only session
finishes; it checks its PID files before signaling those processes. Do not
start the adapter if another experiment owns the cameras.

## Local profile and assets

- Profile: `/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4`
- Launch: `launch.json` (board 9, one fixed socket capture, four cameras)
- Runtime: `franka_runtime.json` (local Qwen and planning)
- Measured ZeroDex clear view: `clear_view_arm_q.json` beside the runtime file;
  loaded only by the ZeroDex integration factory, never by stock AutoDex
- Straight-mount cylinder catalogs: `recipient_gap{1,3,5,10,15,20}_endpoint_catalog_straight.json`
- Franka/Inspire mount: `autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf`
- Local Qwen weights: `/home/hyunsu/shared_data/AutoDex/models/Qwen3-VL-2B-Instruct`
- Local silhouette renderer: `nvdiffrast v0.3.3` and FoundationPose imports
  in `/home/hyunsu/.venvs/precision-vlm`. The ZeroDex launcher exports the
  existing CUDA 12.1 toolkit and PyTorch-bundled header paths for first-use
  RTX 3090 JIT compilation. `--check` is file-only; the saved-view GPU replay
  below tests this renderer without a robot.
- Local Qwen batches the three selected classification-camera pairs on one
  GPU. Prompts, replies and votes remain per camera. Metric key-point
  grounding still uses its separate native-pixel path.
- The ZeroDex launcher sets `PRECISION_PRELOAD_VLM=1`: after the cuRobo
  planner core is constructed, local Qwen starts loading in a background
  thread while the arm moves to clear view and the fixed socket is captured.
  Handler assembly joins that one model-load future; it never loads a second
  copy. The already parsed launch supplies its camera image size, so this
  overlap does not re-read calibration or change the original pixel geometry.
  The AutoDex launcher has no preload default. Set
  `PRECISION_PRELOAD_VLM=0` to compare the original serial startup. The
  2026-10-22 AutoDex episode recorded 3.924 s for Qwen load and 5.687 s for
  clear-view motion; this is an overlap opportunity, **not** a measured
  ZeroDex live-session saving.
- The ZeroDex launcher sets `PRECISION_PREWARM_FIXED_WORLD=1` when the key
  will be placed after the fixed-socket measurement. Once a live
  session has measured the fixed socket and loaded Qwen, the planner initializes
  its table/socket cuRobo world in a background thread while the operator
  places the key. Fresh key capture can also overlap the warmup; the key
  handler joins it immediately before candidate planning. Other planner
  actions join before entry. Later trials reuse the initialized planner.
  Set the variable to
  `0` to use the original first-candidate initialization. When
  `PRECISION_KEY_ALREADY_PLACED=1`, the default is `0`: key perception runs
  first because there is no placement wait to hide the GPU work. An explicit
  `PRECISION_PREWARM_FIXED_WORLD=1` still overrides that choice. This does not
  change any grasp, collision or task-success criterion. It is not exercised
  by `--check` or `--camera-only`; live use requires the ZeroDex FR3 daemon
  and robot/hand commissioning described below.
  The capture4 and main GPUs are separate, although local silhouette fitting
  can contend with cuRobo on the main RTX 3090. A live speedup has not yet
  been measured.
- The ZeroDex launcher sets `PRECISION_VISIBLE_GRIP_CAMERA_IDS` to the same
  three selected views. Its lift checkpoint uses measured Franka/Inspire FK,
  the straight-mount URDF and calibrated projections to crop the hand in each
  raw after-lift frame; three single-image YES/NO prompts share one GPU batch.
  The original AutoDex fixed two-camera crop and its saved report schema are
  unchanged. A real capture4 held-key example is still needed to audit crop
  placement and classification accuracy.
- Same-frame full-resolution capture exchange: the existing SAM mask PUB message
  now also carries one lossless PNG of its source frame. The main PC writes it
  under local `zerodex_capture4/<capture_id>_request_<id>/images/`.

Fixed-socket board/body localization now compares each admitted SAM mask with
the CAD wall, base flange, and whole-body silhouettes. It backprojects at
the selected region's height rather than assuming every mask depicts the
wall. Per-view region choice, IoUs and board-XY disagreement are saved as
diagnostics; no additional acceptance threshold was introduced. The
current-scene pose still needs a new lit socket-only capture.

The launch uses all four serials `26053248`, `25322639`, `25305462`,
`25322642`. VLM inference defaults to the first three; this is separate from
socket/key SAM3 perception. Board 9 is registered from this profile inside
the local Python process, not in ParaDex's global configuration. Capture4's
older camera server requires `camera_acquisition_mode: stream` in the launch
profile; the AutoDex profile still defaults to `acquire`.
`init_frame_transport: inline_png` is specific to this ZeroDex launch. The
unchanged AutoDex profile still uses `shared_png`. No extra capture4 port or
stock daemon edit is required. The PNG shares the mask's request ID, serial,
sensor frame ID and decoded-pixel SHA256; the main PC compares that identity
with the FoundPose payload before using the image. Raw capture files are
written on this PC, not through NAS, so the old 20–40 s NAS wait is removed
from this camera path. Camera-only measurements must still establish the new
end-to-end latency.
Normal per-trial key planning already uses a pose-only FoundPose request and
an overlapping initial VLM snapshot; it did not wait for the NAS source PNG.
This change mainly helps the session's socket capture and full-image QA or
recovery requests. It is not a measured 20–40 s saving on every trial.

`requirements-local-vlm.txt` does not install the renderer. The working
`nvdiffrast v0.3.3` source is already at
`/home/hyunsu/shared_data/nvdiffrast`; install a copy into `precision-vlm`
without editing that NAS source. The same environment needs the FoundationPose
imports, including Open3D, ruamel.yaml, transformations and scikit-learn.
The ZeroDex launcher sets the existing local CUDA toolkit/header paths for
the renderer's first-use JIT. The saved-view replay in the review document
checks the full local refinement path without opening cameras or the robot.

```bash
renderer_build=$(mktemp -d /tmp/precision-nvdiffrast.XXXXXX)
cp -a /home/hyunsu/shared_data/nvdiffrast "$renderer_build/source"
/home/hyunsu/.venvs/precision-vlm/bin/python -m pip install \
  --no-build-isolation --no-deps "$renderer_build/source"
```

The current local `precision-vlm` environment already has this renderer and
the required Python imports; these commands are for rebuilding it.

The profile can be rebuilt with `prepare_zerodex_capture4.py --source-stage
<commissioned-stage> --calibration-dir <capture4-calibration> --c2r-npy
<valid-C2R.npy>`. This copies/rebinds camera/runtime assets, but deliberately
does not copy the AutoDex right-angle mount catalogs. Screen each socket gap
with the ZeroDex straight-mount URDF; for example:

```bash
cd /home/hyunsu/autodex-code
export PRECISION_FR3_INSPIRE_URDF="$PWD/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf"
export PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src"
/home/hyunsu/.venvs/precision-vlm/bin/python demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root /home/hyunsu/shared_data --mode cylinder --gap-mm 20 \
  --min-hand-clearance-mm 0 \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/recipient_gap20_endpoint_catalog_straight.json
```

As of 2026-10-28, all six default `_straight.json` catalogs have been
rescanned against the expanded 126-candidate v8 pool; the per-gap upright
and side-down counts are in `SIDE_DOWN_CYLINDER_20261028.md`. The ZeroDex
launcher selects `_straight.json` by default. Static preflight
compares its screened mount-URDF hash to the actual local URDF once; the old
right-angle catalog is rejected without scanning every candidate. The six
previously copied right-angle catalogs are retained, not deleted, under
`zerodex_capture4/archive/autodex_right_angle_catalogs_20261027/`.

## Franka/Inspire execution route

`run_zerodex_capture4.sh --run` and `--plan-only` now select
`precision_insertion.zerodex_franka_integration:factory`. This thin wrapper
uses the unchanged AutoDex planner, Franka executor, measured-state reader,
contact-insertion and trial stages. It first reads the local process table
and requires exactly one ParaDex `franka_daemon` whose command line targets
the ZeroDex FR3 FCI address `172.16.1.11` on command/state ports 5555/5556.
It also checks that the daemon binary named in `franka_runtime.json` exists.
Only then does it set the **current Python process's** ParaDex Franka client
address to `127.0.0.1`; the previous value is restored on shutdown or
startup failure. Shared ParaDex `network.json`, capture4 programs and the
AutoDex launcher defaults are unchanged. No daemon is started by this script.

The FR3 and Inspire hand were connected on 2026-10-28. The AutoDex-PC
`franka_daemon` binary was restored into the local `zerodex_capture4/robot_deps`
tree without editing the shared ParaDex checkout, and its matching libfranka
0.18 runtime dependencies resolve locally. The configured daemon path is in
`franka_runtime.json`. The command route still needs a running daemon with
exclusive FCI ownership before either live mode. A
`--plan-only` session **still moves** the arm to clear view and observes the
key before planning; it is not an offline test. `--check` and `--camera-only`
remain robot-free. This route check prevents an AutoDex-PC daemon targeting
another robot from receiving ZeroDex commands; it adds no new grasp, pose,
insertion or VLM success threshold.

The read-only FCI route is `172.16.1.11 dev enp4s0 src 172.16.1.10`.
Before the kernel change, starting the daemon as an ordinary user failed
with `unable to set realtime
scheduling`, and starting it with `sudo` failed with `Running kernel does not
have realtime capabilities`. The operator installed Ubuntu 22.04's
`6.8.1-1061-realtime` HWE kernel; `/sys/kernel/realtime` now reads `1`.
NVIDIA 535.309.01 was built for that kernel with DKMS, and the host's
`nvidia-smi` reports an RTX 3090. Do not silently change the daemon to
libfranka's `RealtimeConfig::kIgnore`; the RT kernel is now available.
Franka's [FCI real-time kernel instructions](https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html)
recommend Ubuntu Pro's real-time kernel for Ubuntu 22.04. Canonical's
[enablement guide](https://documentation.ubuntu.com/pro-client/en/latest/howtoguides/enable_realtime_kernel/)
documents the 22.04 HWE RT kernel. The required host kernel and GPU driver
are now present. A post-reboot read-only FCI state sample succeeded, with all
seven joints still within 0.00004 rad of the recorded clear-view target.
The unchanged daemon subsequently reached ready state with RT permission,
and the measured clear-view command completed as recorded below. NVIDIA
support on PREEMPT_RT is not officially guaranteed by Franka even though
this build loaded and reported the GPU.

Operator-side RT setup used on this Ubuntu 22.04 host (Ubuntu Pro subscription is
needed for the HWE package; attach interactively and do not put its token in
the repository):

```bash
sudo apt update
sudo apt install ubuntu-pro-client
sudo pro attach
sudo pro enable realtime-kernel --access-only
sudo apt install linux-realtime-hwe-22.04
# Reboot at an operator-selected time, then:
uname -a
cat /sys/kernel/realtime  # expected: 1
nvidia-smi
```

The unchanged daemon can then be started from a dedicated terminal; leave it
open until the pipeline session finishes:

```bash
sudo env LD_LIBRARY_PATH=/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/robot_deps/prefix/usr/lib/x86_64-linux-gnu:/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/robot_deps/prefix/usr/lib \
  /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/robot_deps/restored/home/robot/paradex/cpp/franka_daemon/build/franka_daemon \
  172.16.1.11 --command_port 5555 --state_port 5556
```

Wait for `[READY] Daemon ready, waiting for commands...`. The socket session
already captured with camera-only mode must be recaptured if the fixture has
moved during the reboot. Without a key on the board, do not start a pick or
insertion trial; the first motion check is only clear-view with the measured
joint target.

The existing
ZeroDex `realtime_vlm/run/components/robot_exec.py` uses a ROS2
`FrankaController`/JTC plus Inspire commands; it is not a drop-in replacement
for this demo's stock 50 Hz ParaDex follower and measured wrench/hand trace.

## Camera-only checks (robot and hand may remain off)

```bash
cd /home/hyunsu/autodex-code
python3 demo/precision-insertion/manage_zerodex_capture4_services.py start
bash demo/precision-insertion/run_zerodex_capture4.sh --check
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" \
/home/hyunsu/.venvs/precision-vlm/bin/python \
  demo/precision-insertion/probe_zerodex_socket.py \
  --launch-config /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/launch.json \
  --evidence-dir /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/socket_probe_NEW
python3 demo/precision-insertion/manage_zerodex_capture4_services.py stop
```

If the board is unlit, measure only the new transport without relaxing board
pose criteria:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" \
/home/hyunsu/.venvs/precision-vlm/bin/python \
  demo/precision-insertion/probe_zerodex_inline_frames.py \
  --launch-config /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/launch.json \
  --evidence-dir /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/inline_probe_NEW \
  --target key
```

Start the isolated services first, and stop them after the probe. This captures
four raw source frames but does not measure a board/object pose or authorize
robot motion.

The capture4 precision daemon uses `PRECISION_SAM3_BATCH=auto` by default:
source-PNG requests stay serial because that path was faster end to end;
normal pose-only key requests use a four-view SAM3 batch. To time the latter,
repeat the inline-frame probe with `--pose-only` and a new evidence directory.
It records four mask/pose messages but intentionally saves no source PNGs or
brightness measurements. To force serial SAM3 for a comparison, stop our
namespaced services and restart them with
`PRECISION_SAM3_BATCH=0 python3 demo/precision-insertion/manage_zerodex_capture4_services.py start`.
The flag affects only this namespace and does not change AutoDex's stock
camera daemon. Measured A/B timings and limits are in
`ZERODEX_CAPTURE4_REVIEW_20261027.md`.

If the original raw frames are underexposed, the capture4-specific camera
server can use an **explicit, temporary** exposure override without editing
capture4's shared ParaDex source or `system/current/camera.json`. Run only
when the four cameras are free and the shared-rig operator agrees to the
temporary camera setting:

```bash
python3 demo/precision-insertion/manage_zerodex_capture4_services.py start --exposure-us 20000
# Run the inline-frame probe above with a new evidence directory.
python3 demo/precision-insertion/manage_zerodex_capture4_services.py stop
```

Without `--exposure-us`, each camera retains its configured baseline. The
override applies only in our namespaced daemon process; it does not relax a
SAM, FoundPose, board or planning threshold. A running service must be
stopped before selecting a different exposure. A 20 ms test of the current
scene brightened the images but still returned zero key masks; see the review
document before choosing a longer-exposure default.

A fresh default-exposure four-view request on 2026-10-28 again returned zero
key-mask pixels in every view; grayscale p99 was 6, 8, 9 and 19/255. Its
evidence is in `zerodex_capture4/runs/inline_key_20261028_current_pipeline_audit_1/`.
Capture4 services were stopped after this probe. This image-input issue must
be resolved before interpreting any live lying-key planning result.

The probe saves original-frame SAM mask overlays, a four-view grid and
`report.json`; it saves board JPEGs even if ChArUco detection fails. It never
connects to Franka or Inspire. Use a new evidence
directory for each run. `run_zerodex_capture4.sh --check` is a static asset
preflight, not a camera or robot acceptance test. Its
`candidates_by_tabletop_pose` counts distinguish the 106 eligible upright
`000` grasps from the five eligible lying `001` grasps in the current gap-20
catalog. These are endpoint-screen results, not executable arm plans. The
ZeroDex profile tries pose-matched grasps for both upright `000` and lying
`001` first. A lying key enters reorientation only when its direct
current-scene plans are exhausted. Do **not** use `--run` while
the robot/hand are off or before the local robot adapter is commissioned.
The ZeroDex launcher accepts `--run` and `--plan-only` through its separate
integration factory, but the factory refuses to construct an executor unless
exactly one local ParaDex daemon targets the ZeroDex FR3 on the expected
ports. On this PC the daemon binary/runtime and exclusive FCI ownership are
not yet commissioned, so **neither live mode can currently run**. This is
target-routing correctness, not a new grasp, perception or insertion
acceptance threshold. `--check` and `--camera-only` remain robot-free.
Once the daemon route is commissioned, `PRECISION_XY_UPDATE_WEIGHT` can set the
next-trial residual gain (default `1.0`); the launcher forwards it to
`run_live.py` instead of fixing the value in its command line.

To test a complete camera-only fixed-socket session with a lit board and the
key absent, start the isolated services, then run:

```bash
PYTHONPATH="$PWD:$PWD/demo/precision-insertion:$PWD/src" \
/home/hyunsu/.venvs/precision-vlm/bin/python demo/precision-insertion/run_pipeline.py start-session \
  --config /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/launch.json \
  --capture --init-port-mask 15006 --init-port-pose 15007 --init-port-cmd 16893
```

This CLI now starts/stops the capture stream and connects to the isolated
perception ports. It never opens the robot. Give each capture a new
`evidence_dir` in the launch file.

To check the **connected camera-only pipeline** rather than just independent
socket/key masks, leave the fixed socket on the lit board with the key absent,
start the isolated capture4 services, and run:

```bash
cd /home/hyunsu/autodex-code
python3 demo/precision-insertion/manage_zerodex_capture4_services.py start
bash demo/precision-insertion/run_zerodex_capture4.sh --camera-only
python3 demo/precision-insertion/manage_zerodex_capture4_services.py stop
```

The command prompts once before socket measurement and once after it to place
the key. It reuses the live board/socket session, key SAM3/FoundPose admission,
tabletop classification and matching screened catalog, and writes a unique
`zerodex_capture4/runs/camera_only_*/report.json`, `socket_evidence/`,
`key_capture/`, `ledger/` and a sibling `.timing/` directory. This is the
current 20 mm launch/catalog profile. The report includes
`direct_candidate_preflight_available` and `live_pose_route`. A lying key
with matching endpoint-screened grasps now enters current-scene candidate
planning first; if those plans fail, the existing tabletop reorientation
route remains available. These fields do not claim a passing arm plan.
Key admission is pose-only, as in the
live planning path: it does not wait for a second set of same-request PNGs.
Use `probe_zerodex_inline_frames.py` above if raw key frames and masks are
needed for visual QA; do not compare its PNG-inclusive duration to the
pose-only trial capture. The camera-only command does **not** create a Franka
executor, plan a trajectory, move the robot, or validate a physical grasp or
insertion. If camera images are still nearly black, restore illumination
before using this command; an empty SAM mask cannot establish a key pose.
For repeated key-pose checks with the *same unmoved socket*, pass
`--reuse-session-evidence /absolute/previous/socket_evidence` after
`--camera-only`. The camera-only path applies the same saved-image,
calibration and board checks as live reuse, then skips board/socket capture;
the key can already be on the board. The default remains a fresh socket
measurement. A new output directory is generated even when this option is
present.

```bash
bash demo/precision-insertion/run_zerodex_capture4.sh --camera-only \
  --reuse-session-evidence /absolute/previous/socket_evidence \
  --key-already-placed
```

`--key-already-placed` skips only the operator placement prompt. It requires
verified fixed-socket evidence and does not skip key capture, pose admission,
or any planning filter. Without it, the prompt remains available for a key
that will be placed after camera startup.

For a chosen output directory, pass `--output-dir /absolute/new/path` after
`--camera-only`. Keep the capture4 services isolated; this launcher does not
alter the shared capture4 checkout.
`PRECISION_GAP_MM` selects both the matching straight-mount endpoint catalog
and the per-run launch mode for `--camera-only`, just as it does for `--check`
and the live modes. The original stage `launch.json` remains unchanged; the
camera-only run saves the effective `launch.json` beside its report.
Calling the service manager's `start` again while all namespaced ports are
already ready now reuses the running services with one readiness probe,
without repeating deployment. This reduces warm-start SSH/rsync overhead;
the effect on a complete episode has not been measured.

## Current board/socket capture and FR3 pose, 2026-10-28

The physical socket in the 2026-10-28 live setup is the **10 mm radial-gap**
variant. Use `PRECISION_GAP_MM=10`; this selects its own endpoint catalog and
socket mesh. The ZeroDex `launch.json` now sets
`board_fit_camera_ids = [26053248, 25322639, 25322642]`. All four capture4
images are still acquired and available for socket/key perception. The subset
applies only to ChArUco triangulation: on the current board, camera
`25305462` made corner 8 reprojection about 4.0 px, whereas the other three
cameras fitted 41 corners with median reprojection 0.275 px and board rigid-fit
RMS 0.675 mm. No reprojection threshold was changed. The diagnostic source
frames are under `/tmp/precision_gap10_board_diag_20261028/`.

The first integrated 10 mm `--plan-only` session froze the socket in
`zerodex_capture4/runs/live_charuco_board_gap10/session_evidence_20261028_135908_048797173_21840/`.
After placing the upright key, four-view SAM3/FoundPose and local silhouette
refinement completed. Its current-scene preflight selected `table/0/20232`:
pickup approach, measured-start lift, transfer and a sampled 20 mm axial
path passed. The saved report and trajectories are under
`AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261028_135909_098315/trial_preflights/0000/`.
The plan-only result is `planning_ready`, not a physical insertion result.

The first live `--run` reused that frozen socket. A stale FR3/Inspire pair
immediately after the blocking clear-view command stopped the first start;
the feedback reader now reacquires within its existing sample-age interval.
The next run executed approach, pregrasp, grasp, squeeze and part of lift.
Two empty Inspire Modbus replies during lift caused the measured-motion
adapter to abort at 156/221 references. The controller trace is at
`.../_pipeline_runs/20261028_140538_313569/controller_traces/lift_295a405722414788b63c991e3f033c6f/controller_trace.json`.
The key remained upright on the board in the post-abort four-camera snapshot
`/tmp/precision_gap10_after_lift_abort_20261028/`. No transfer, insertion,
VLM outcome or next-episode XY update occurred. The measured-motion adapter
now tolerates a missing Modbus reply within its existing 0.5 s state-gap
limit, while continuing to read FR3 state; this fix is not yet confirmed in a
subsequent physical trial.

With the socket on board 9, the camera-only `start-session --capture` path
completed without moving the FR3 or Inspire. Evidence is under
`/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/session_evidence/`.
The four cameras triangulated 41/54 board corners (plane RMS 0.288 mm,
median reprojection 0.926 px). Socket SAM3 masks were nonempty in
`26053248` and `25322639`; their board-constrained body fits had IoU
0.941 and 0.948. The other two camera masks were empty and did not
contribute to socket localization. The two fitted socket centers differed
by 4.73 mm. These are diagnostics from one capture, not a repeatability test
or independently verified socket-pose accuracy. This earlier camera-only
session used the default 20 mm asset and must not be reused for the current
10 mm socket.

The operator-designated FR3 pose was recorded after the operator added a
robot-source-only UDP rule on `enp4s0`. A read-only libfranka probe is in
`read_fr3_state_once.cpp`; its local built binary is at
`zerodex_capture4/robot_deps/read_fr3_state_once`. Before that rule, TCP to
`172.16.1.11:1337` worked but `readOnce()` timed out on UDP. Afterward two
stationary reads agreed to within 0.000007 rad. The second seven-joint sample
is saved at `zerodex_capture4/clear_view_arm_q.json`. The ZeroDex factory
loads it before constructing AutoDex's executor/planner, updates the shared
in-process ndarray referenced by the clear-view/release/recovery code, and
restores the stock value when the session closes. It does not edit the stock
AutoDex clear-view constant or ParaDex calibration on disk. A
read-only Modbus check of Inspire `angleAct` returned
`[800, 802, 800, 800, 1000, 787]` in controller order. The stock pipeline's
`INSPIRE_INIT` means all raw angles near 1000 (fully open); this first sample
predates the actual clear-view/open-hand command documented below.

`--run` and `--plan-only` need the local ParaDex daemon and an explicit
motion opt-in. The daemon reached ready state after the RT kernel boot.
Initial
`home(clear_view=True)` will open Inspire and command the *measured* FR3
target, even if the arm is already near it. This is a command-capable action,
not what the earlier read-only pose probe or socket capture did. Keep the
board/socket fixed if reusing this one-capture calibration.

## Observed fixed socket, 2026-10-27

The socket is visible in all four raw camera views. One camera-only probe
segmented its cylindrical wall in three views; `25322642` returned an empty
SAM mask. The fitted body IoUs for `26053248`, `25322639`, `25305462` were
0.642, 0.584, 0.627 (mean 0.618). The original 0.800 body-IoU veto blocked
the session even though the socket was visibly segmented in three views.
For this ZeroDex board-constrained profile only, `minimum_socket_refined_iou`
is now `0.0`: IoU is retained as a diagnostic but does not veto the estimated
pose. The AutoDex/FoundPose modes retain their original threshold semantics.
The overlays are at
`/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/socket_probe_20261027_1/overlays_4view.jpg`.
This is a segmentation/body-fit finding, not evidence that the socket is
absent. With a standing key present, the former short socket prompt selected
the key in all four views. The refined prompt, `large hollow white
cylindrical socket with a wide flat circular base`, segmented the socket in
3/4 standing-key and 2/4 lying-key views. Zero IoU veto is **not** an object
identity guarantee. The fixed session should measure the socket with the key
absent, then add the key. The production key prompt is now `white plastic
cylindrical key`; it segmented the lying key in 4/4 lit views, but this was
SAM only at the time of that probe. The later read-only replay did obtain
FoundPose candidates in 4/4 views and passed the current local AutoDex
IoU/silhouette refinement on those saved masks; see
`ZERODEX_CAPTURE4_REVIEW_20261027.md`. This does not establish metric pose
truth or live robot readiness.

In the latest retry, all four board frames arrived in 0.46 s but ChArUco
found 0/54 corners. The saved views were effectively unlit: their 99th
percentile pixel intensities were 6–19/255. Raw frames are under
`runs/lying_key_foundpose_20261027_5/*_board.jpg`. No current lying-key
FoundPose or physical path is claimed.

The four-camera JPEG snapshot returned in about 0.5–0.8 s; remote SAM masks
were published in about 1.3–1.6 s after the request. Waiting for four
full-resolution same-frame PNGs through NAS took substantially longer and is
the recorded camera-only latency bottleneck. A later dark-scene inline
transport probe delivered four same-frame key images and masks locally in
2.172 s. It measured transport, not a successful key pose or full episode.

## 2026-10-28 RT robot bring-up and clear-view socket probe

The local daemon reached `[READY]` on the RT kernel, targeting
`172.16.1.11` with ZMQ command/state ports 5555/5556. The missing Franka
gripper is expected: this rig uses Inspire on Modbus TCP, not the stock
Franka gripper. The first stock `FrankaExecutor` startup exposed a local
dependency mismatch: ParaDex imports `pymodbus.client.ModbusTcpClient` while
the installed 2.5.3 package exposes it from `pymodbus.client.sync`. The
ZeroDex integration now installs that class alias **in its own Python
process** before stock executor construction; neither shared ParaDex code
nor capture4 files were edited.

With the operator-designated arm target, the stock `home(clear_view=True)`
completed. The arm's measured final max joint error was 0.000785 rad and the
Inspire measured raw angles were `[1000, 998, 1000, 998, 1000, 988]` versus
all-1000 open command. The result record is in
`zerodex_capture4/runs/clear_view_probe_20261028.json`. This proves the
clear-view arm and open-hand command path, not pickup or insertion.

After clear view, a fresh four-view socket SAM3 probe saved
`zerodex_capture4/runs/socket_probe_20261028_rt_clear_view/overlays_4view.jpg`.
Masks were nonempty in two views (`26053248`, `25322639`) with body IoU
0.940 and 0.946; the other two returned empty masks. Mean IoU among the two
contributing views was 0.943. This is a camera-only mask/body-fit probe, not
a new frozen socket pose or key trial. The camera-only full session still
needs the physical key after its socket phase. No physical insertion or task
success is claimed here.

## 2026-10-28 live pickup attempt and current hardware finding

The 10 mm-gap socket and upright key were measured in a frozen capture4
session. A live selected v8 grasp (`table/0/20232`) reached squeeze and then
completed the planned 10 cm lift. The key remained on the board in the saved
after-lift views. An offline replay of three before/after camera pairs with
the configured Qwen3-VL-2B backend returned `miss` in all three views. The
feedback sampler exited after a transient missing Inspire Modbus reply, so
the live attempt stopped before its VLM checkpoint. No pre-insertion,
insertion, XY correction, or next-episode correction was executed. The stock
planned missed-grasp reset returned the arm to clear view.

The commanded raw Inspire angles after squeeze were
`[899, 267, 512, 604, 32, 0]`; measured thumb-bending channel 4 was about
`995` rather than its commanded `32`, while the other bending channels
tracked their commands. A separate, unloaded thumb-bend command of `600`
also left its measured angle at `1000`; the command was then restored to
`1000`. This localizes the observed nonresponse to the thumb-bend channel. The
available measurements do **not** distinguish its connector/wiring, motor,
or driver. The external Modbus link communicates with the other channels.

A later read-only register check decoded `ERROR` and `STATUS` as six packed
bytes from **three** Modbus registers each, matching ParaDex's Inspire
controller. At the restored all-open target, every error byte was `0` and
every status byte was `2` (target reached). The earlier raw `514` values
were packed status bytes and must not be interpreted as an error code.
All six `FORCE_SET` values were `400`; thumb-bending `FORCE_ACT` was about
`386`, close to that limit. In a separate unloaded `1000 -> 900` thumb-only
probe, `ANGLE_ACT(4)` briefly reached `988` and then returned to `1000`
while `ANGLE_SET(4)` remained `900`; its status was `1` (grasping), not `3`
(force-stop). The target was restored to `1000`. This shows a transient
response and leaves the cause unresolved: the force threshold is a plausible
factor, but the expected force-stop status was not observed. It does not
prove a disconnected cable or a failed actuator.

With only thumb `FORCE_SET(4)` temporarily raised to `600`, the small
unloaded `1000 -> 900` command reached `ANGLE_ACT(4)=900` and the actuator
position changed from `240` to `357`; this proves the thumb actuator can
move and the command reaches it. The actual grasp target `32` still did not
track: it briefly reached `843` before returning to about `956` while the
target remained `32`. The angle and force settings were restored to `1000`
and `400`. Thus the `400` force limit contributes to small-command failure,
but raising it to `600` did not restore the full grasp motion. The remaining
possibilities include channel wiring/contact, motor/driver, force sensing or
mechanical loading; cable replacement is a useful next comparison. No
robot motion or insertion was attempted during these unloaded hand probes.

The ZeroDex FR3 robot config accepts an optional six-value
`hand_force_set_raw` array in little/ring/middle/index/thumb-bend/thumb-rotate
order. When absent, the stock ParaDex `400` per channel remains unchanged.
When present, the demo applies the specified values after stock Inspire
construction and before the trial. For example,
`[400, 400, 400, 400, 600, 400]` reproduces the small-command diagnostic;
it is **not** validated for the full grasp target. The current local
`zerodex_capture4/franka_runtime.json` intentionally has no override until
the cable comparison and no-load full-range behavior are resolved.

The integration now tolerates a missing hand-feedback sample and exposes a
sampler failure if one occurs. It reuses the candidate's offline 20 mm
hand/socket endpoint screen after squeeze; measured finger joints are used
for the lift/transfer planner, not treated as a measured key-in-hand pose.
These code changes do not make the missed physical grasp a success.

## 2026-10-28 live latency baseline

The two failed physical attempts retained structured timing records under
`_pipeline_runs/20261028_141454_188235.timing` and
`_pipeline_runs/20261028_142201_394786.timing`. Their full session spans
were 52.59 s and 93.22 s, respectively; these end at different failure
points and are **not** complete-episode latency measurements. Fresh key
FoundPose requests were 2.19 s and 2.37 s. The first cuRobo pickup plan was
27.55 s and 26.08 s. In the second run, preferred candidate `table/0/20620`
did not pass the transfer preflight, so planning tried 14 pickup candidates:
33.38 s total for pickup planning and 8.37 s for eight nominal insertion
plans. The physical pickup/squeeze segment was 8.39 s and 8.30 s; only the
second run reached the 5.61 s lift motion. Local Qwen load was 5.71 s and
5.19 s, partially overlapped with other startup work. A one-time 25.4 s
unattributed integration import/setup interval appeared in the second run;
its first-run counterpart was only about 0.04 s. It should be profiled
separately before treating it as a stable bottleneck. The first pickup plan
and repeated candidate planning dominate the observed critical path; the
four-camera key request is not the main delay in these two traces.
An isolated, robot-free import of `franka_integration` subsequently took
1.33 s, so the 25.4 s live interval is not yet reproducible as module import
time. New timing spans split integration module import from construction in
future runs. The saved second-run pickup report attributes 22.80 s of its
26.08 s first candidate plan to initial cuRobo world setup.

## Hand state after reconnecting the same cable

After the operator disconnected and reconnected the same Inspire cable,
Modbus TCP still answered at `192.168.11.210:6000`, but the read-only state
changed across **all six** channels: `ANGLE_SET=[65535]*6`,
`FORCE_SET=[2000]*6`, `STATUS=[7]*6`, and the voltage register returned
`65535`. The six error bytes were zero. The [manufacturer's RH56 register
table](https://static.generation-robots.com/media/rh56-series-user-manual.pdf)
assigns status `7` to actuator fault; the angle/force values are
outside the documented command ranges. The hand's indicator light was on,
which does not by itself establish actuator power or normal operation.
This is a new whole-hand state following reconnection, distinct from the
earlier thumb-only nonresponse. Do not infer that the cable, actuators, or
software caused it without a normal power/connector comparison.

One manufacturer force-calibration request (`register 1009 = 1`) was sent
while the hand was clear of objects. The pre-request read already showed
the abnormal all-channel state; the diagnostic script failed to check that
state before writing. No finger motion was observed during the following
ten seconds, and repeat read-only checks still showed the same abnormal
values. No further hand/arm motion commands were sent. This calibration
attempt must **not** be treated as completed or as evidence that the
original thumb-force offset was corrected.

The operator then verified the hand indicator was lit, checked the connector,
and performed a full hand power restart. A fresh **read-only** Modbus check
still returned `VLTAGE=65535`, `ANGLE_SET=[65535]*6`,
`FORCE_SET=[2000]*6`, and `STATUS=[7]*6`. The controller ID remained `1`
and TCP connected. This makes a transient restart state less likely, but
does not identify whether the connector, cable, supply or hand internals
are responsible. The demo must not resume finger calibration or pickup from
this current all-channel state.
