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
| Pickup, lift and transfer | v8 pickup planner, FR3 follower, measured state and collision checks | Reuses the same primitives, with task-specific socket collision world and preinsert endpoint. The current route does not use legacy held-relation or lateral-shift experiments. A stale exact-binary SHA pin was removed after it stopped a live post-squeeze lift; local process, port, stream-expiry and measured-state checks remain. |
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
session's image provenance or its evidence write. A live 10 mm capture on
2026-10-22 measured `camera.socket_mask_request` at 42.158 s,
`camera.saved_frame_wait` at 8.805 s, and `socket.body_fit` at 0.158 s.
These are elapsed durations; per-camera work overlaps.

The deployed capture-PC adapters under
`/home/captureX/precision_init_adapter_20261013_983ba80/` are separate copies.
Pulling this checkout on the robot PC does not update those processes. The
updated adapter was copied to all five capture PCs on 2026-10-22, with each
previous file backed up, and the precision init daemons were then started.
An old adapter ignores `skip_foundpose`; the robot-side mask-only admission
remains correct but receives no inference speedup. The current service manager
does not synchronize adapter files.

Validation: the full demo suite passed 1,106 tests with four existing skips,
excluding `test_franka_completion_checked.py` because CUDA was unavailable
inside the test sandbox. The saved-capture replay was read-only. The 10 mm
`run_robot_pc.sh --check` passed with 100 eligible endpoint candidates and
reports `robot_ready: false` by design. One live 10 mm capture accepted 19
views and fit the wall at mean IoU 0.921. Its first grasp reached squeeze,
then stopped at the now-removed exact-binary SHA pin before the lift; no
insertion success is implied by that capture.

## 10 mm preinsert stop and stock follower comparison

The saved 2026-10-22 run `20261022_101456_545928` completed key planning,
pickup, lift, transfer and a `coarse_match` preinsert visual label. It stopped
before any axial insertion command. The arm was measured stationary while
holding the key above the socket. Its joint residual from the saved axial
start was 0.02624 rad L2 (largest joint 0.02003 rad). The duplicate handoff
limit was 0.02 rad per joint. The stock FR3 `_follow` does not impose a new
start-pose or post-observation timestamp threshold between consecutive
segments; it tracks the planned path with a bounded lead and reports its own
endpoint. The demo's completion-checked follower had already accepted the
preceding transfer endpoint at 0.04 rad L2.

The active cylinder route now records that measured handoff state without a
second start-pose, sample-order or stationary check. The same extra checks
were removed from observed XY retry handoff. The first-party force-stop path
no longer repeats them immediately before the stroke or applies a second
post-stroke age/endpoint threshold after its checked follower returns. The
post-completion duration threshold was removed: the stock follower already
owns its execution timeout. Release no longer checks the same insertion
endpoint and timestamp a third time, and the final image does not repeat the
release callback's measured clear-view check. These changes remove incidental
failures after an otherwise completed segment; they do not change the
physical FR3 follower, force stop, collision-planned release path, attempt
binding or the requirement to read actual robot feedback.

The released-key outcome still uses its distinct final images and VLM label.
No insertion success is inferred from the stopped 10 mm run. Offline checks
after this change: 50 targeted tests passed; the demo suite passed 1,112
tests with four skips, excluding the CUDA-dependent follower test because
the test environment exposed no GPU. The 10 mm `run_robot_pc.sh --check`
passed as a static file check and did not connect cameras or motors.
