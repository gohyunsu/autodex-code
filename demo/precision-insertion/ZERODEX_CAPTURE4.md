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
- Same-frame full-resolution capture exchange: `/mnt/paradex2/hyunsu/zerodex_capture4_runtime`

The launch uses all four serials `26053248`, `25322639`, `25305462`,
`25322642`. VLM inference defaults to the first three; this is separate from
socket/key SAM3 perception. Board 9 is registered from this profile inside
the local Python process, not in ParaDex's global configuration. Capture4's
older camera server requires `camera_acquisition_mode: stream` in the launch
profile; the AutoDex profile still defaults to `acquire`.

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

The probe saves original-frame SAM mask overlays, a four-view grid and
`report.json`; it saves board JPEGs even if ChArUco detection fails. It never
connects to Franka or Inspire. Use a new evidence
directory for each run. `run_zerodex_capture4.sh --check` is a static asset
preflight, not a camera or robot acceptance test. Do **not** use `--run` while
the robot/hand are off or before the local robot adapter is commissioned.

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
SAM only, not a confirmed FoundPose pose.

In the latest retry, all four board frames arrived in 0.46 s but ChArUco
found 0/54 corners. The saved views were effectively unlit: their 99th
percentile pixel intensities were 6–19/255. Raw frames are under
`runs/lying_key_foundpose_20261027_5/*_board.jpg`. No current lying-key
FoundPose or physical path is claimed.

The four-camera JPEG snapshot returned in about 0.5–0.8 s; remote SAM masks
were published in about 1.3–1.6 s after the request. Waiting for four
full-resolution same-frame PNGs through NAS took substantially longer and is
the current camera-only latency bottleneck. A faster lossless frame transfer
can be implemented in this namespaced adapter without changing capture4's
stock ParaDex files.

Robot/hand power and a local Franka daemon are still required for execution.
No robot movement, insertion or real-world task success is claimed here.
