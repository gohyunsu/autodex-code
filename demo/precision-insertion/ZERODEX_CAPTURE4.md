# ZeroDex capture4 profile (kds-Z790)

This profile lives on branch `feat/zerodex-precision-insertion`. The AutoDex
camera/robot profile and `src/execution` sources are not changed. The local
kds-Z790 PC runs session orchestration, Qwen VLM and planning. Capture4
(`147.46.219.136`, SSH port `77`) provides four cameras and a persistent
SAM3/FoundPose process. Its existing ParaDex checkout, global `system/current`
configuration, calibration and Python environments are **not edited**.

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

The ZeroDex launcher selects `_straight.json` by default. Static preflight
compares its screened mount-URDF hash to the actual local URDF once; the old
right-angle catalog is rejected without scanning every candidate. The six
previously copied right-angle catalogs are retained, not deleted, under
`zerodex_capture4/archive/autodex_right_angle_catalogs_20261027/`.

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
stopped before selecting a different exposure. The override's effect on the
current lying-key scene has **not** yet been measured.

The probe saves original-frame SAM mask overlays, a four-view grid and
`report.json`; it saves board JPEGs even if ChArUco detection fails. It never
connects to Franka or Inspire. Use a new evidence
directory for each run. `run_zerodex_capture4.sh --check` is a static asset
preflight, not a camera or robot acceptance test. Its
`candidates_by_tabletop_pose` counts distinguish the 106 eligible upright
`000` grasps from the four eligible lying `001` grasps in the current gap-20
catalog. These are endpoint-screen results, not executable arm plans. The
ZeroDex profile tries pose-matched grasps for both upright `000` and lying
`001` first. A lying key enters reorientation only when its direct
current-scene plans are exhausted. Do **not** use `--run` while
the robot/hand are off or before the local robot adapter is commissioned.
The ZeroDex launcher now refuses `--run` and `--plan-only` outright: both
would instantiate the inherited AutoDex ZMQ executor, while this PC's ParaDex
`system/current/network.json` points Franka at `172.16.0.2`, not a verified
ZeroDex controller. This is target-routing correctness, not a new grasp,
perception or insertion acceptance threshold. `--check` and `--camera-only`
remain available. A ZeroDex-specific FR3/Inspire backend and measured-state
binding are required before motion or live planning can be enabled.
When that backend is commissioned, `PRECISION_XY_UPDATE_WEIGHT` can set the
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
Calling the service manager's `start` again while all namespaced ports are
already ready now reuses the running services with one readiness probe,
without repeating deployment. This reduces warm-start SSH/rsync overhead;
the effect on a complete episode has not been measured.

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

Robot/hand power and a local Franka daemon are still required for execution.
No robot movement, insertion or real-world task success is claimed here.
