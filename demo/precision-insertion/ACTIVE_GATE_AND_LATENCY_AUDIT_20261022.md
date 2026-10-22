# Active cylinder gate and latency audit (2026-10-22)

This audit compares the active board-9, 20 mm radial-gap route in the
separate checkout with stock `/home/robot/AutoDex` at `448732d`. It reads
three saved real socket captures from 2026-10-21. No camera or robot was
connected or moved. The stock checkout was not changed.

| Stage | Stock AutoDex | Active precision route and decision |
| --- | --- | --- |
| Board | ChArUco tabletop fit | Reuses `measure_tabletop_from_images`; requires acquisition-bound board frames. The board is needed to place this flat-flange socket and the table collision surface. |
| Initial object mask/pose | SAM mask plus FoundPose hypothesis per view, followed by view selection and optional whole-mesh silhouette optimization | Board/body socket fit uses SAM masks and board geometry, not FoundPose hypotheses. The old FoundPose quality/inlier/pose gate was a redundant admission condition and is removed **only** for this mode. The capture-PC adapter now skips the unused FoundPose estimate on a `skip_foundpose` request, while publishing the same frame-bound `ok=false` pose envelope expected by the existing transport. |
| Socket fit | No fixed socket | Mask-centroid rays determine board XY; CAD wall reprojection supplies per-view and mean IoU. The active 0.8 **mean** body IoU, mask size/border, minimum two views, and board support constraints remain. They measure an input or result actually used by the fit. |
| Key pose and grasp | FoundPose, v8 candidate and collision planning | Retains FoundPose pose/quality/inlier checks for the key, because its 6D pose is an input to the stock v8 grasp. The new socket admission mode is not used by `key_perception.py`. |
| Pickup, lift and transfer | v8 pickup planner, FR3 follower, measured state and collision checks | Reuses the same primitives, with task-specific socket collision world and preinsert endpoint. The current route does not use legacy held-relation or lateral-shift experiments. No physical limit was loosened without a trace showing a false stop. |
| Insertion and release | Stock place/reset has no socket stroke or inside-socket label | Axis continuation, force stop, release/clear and final visual label are task requirements. They remain separate from a visual claim of metric 20 mm depth. |
| Failed attempt | Stock reset/re-observe | Current route recaptures the key at its new tabletop pose. A definite grasp miss already skips unnecessary original-XY restoration; released-key XY advice uses the saved failed preinsert frames for the **next** attempt. Source/attempt identity and measured-state checks remain because otherwise a stale plan could be applied to a different physical state. |

`SocketObservation.pose_world` can now be null **only** when the accepted
socket capture has a board/body refinement. The frozen socket pose still comes
from that refinement. The session record and offline replay verifier preserve
the null explicitly; they do not fabricate agreeing FoundPose poses. Stock
socket modes and key perception still require valid FoundPose poses. Captured
image/frame IDs, exposure timestamps, mask hashes and request binding remain
checked; saved raw pose envelopes are retained for diagnosis.

The three saved October 21 captures admitted 18, 19 and 19 views before this
change. Read-only replay of their original masks and exact launch thresholds
admitted the same 18, 19 and 19 views after it. Their rejected views had too
little SAM mask as well as no FoundPose pose, so this change does not alter
those historical socket results. The board/body fit itself is unchanged.

The old captures recorded 5.761–6.821 seconds of aggregate per-view FoundPose
worker time over 20 cameras (0.288–0.341 seconds per view on average). Five
capture PCs overlap that work, so this sum is **not** a predicted wall-time
saving. There is no post-change hardware timing. The remaining socket capture
time is dominated by SAM and saving the full undistorted source images; the
new request avoids only unused FoundPose inference. It does not skip the
session's image provenance or its evidence write.

The deployed capture-PC adapters under
`/home/captureX/precision_init_adapter_20261013_983ba80/` are separate copies.
Pulling this checkout on the robot PC does not update those processes. Before
claiming the inference-time change in a live session, stage the updated
`demo/precision-insertion/precision_init_daemon.py` on each capture PC and
restart the precision init daemons. An old adapter ignores `skip_foundpose`;
the robot-side mask-only admission remains correct but receives no inference
speedup. The current service manager does not synchronize adapter files.

Validation: 60 focused tests passed; the full demo suite passed 1,101 tests
with four existing skips, excluding `test_franka_completion_checked.py` whose
CUDA hardware is unavailable in this offline environment. The saved-capture
replay was read-only. `run_robot_pc.sh --check` passed its static file check
for the current 20 mm launch and 98 eligible endpoint candidates. That check
reports `robot_ready: false` by design and does not connect to a camera,
CUDA worker or robot. No live robot outcome is implied.
