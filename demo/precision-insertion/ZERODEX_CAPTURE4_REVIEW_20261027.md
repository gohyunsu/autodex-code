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
avoids shipping intermediate model activations. The existing local Qwen
backend processes the three selected VLM views sequentially; measured warm
three-pair inference was about 0.84 s. Four same-frame full-resolution PNGs
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

Verification performed: the precision demo test suite passed **1247 tests**
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
