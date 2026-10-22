# Fixed cylindrical socket localization: source and saved-frame review

Scope: branch `feat/precision-insertion` on 2026-10-22, compared with
`/home/robot/AutoDex` and replayed on three real, saved 2026-10-21
socket-only captures. No camera or robot was connected, and no surveyed
socket pose exists. Rendered overlays are predictions on recorded undistorted
images, not new observations or physical accuracy measurements.

## Active sequence and quantities

`T_AB` maps coordinates in B into A. `R` is robot base, `W` camera world,
`B` ChArUco board, and `S` socket CAD. The frozen `C2R` is `T_WR`.

1. `run_robot_pc.sh` chooses cylinder `board_constrained_body`. The launch
   uses board 9, one socket capture, and the white cylindrical socket prompt.
   The socket must already be seated on the board for this partial-board mode.
2. `live_session_start.py` obtains one board snapshot from the stock AutoDex
   camera system. `measure_tabletop_from_images` uses calibrated, undistorted
   views of ChArUco corners to return `T_RB`. Its result is a measured board
   plane and orientation; no socket position comes from this step.
3. The capture-PC init daemons keep stock SAM3 and undistortion. The
   board/body request marks socket initialization and inference as mask-only;
   each PC emits an undistorted image, white-socket SAM mask, exact frame ID,
   and an `ok=false` placeholder on the expected pose channel. The key still
   initializes and runs stock FoundPose in later observations. This init
   optimization uses the precision adapter installed on each capture PC. On
   2026-10-22 the updated `precision_init_daemon.py` was copied to the five
   stopped hosts after preserving a `.backup-20261022` copy; all five remote
   SHA-256 values matched the local source. No daemon was started.
4. `live_capture.py` binds masks and saved images to frame IDs, pixel hashes,
   request IDs, and acquisition-time evidence. `perception_evidence.py` admits
   nonempty, non-border-clipped, calibrated masks. It does not require
   FoundPose quality in mask-only mode. At least two views are required; the
   current real captures admitted 18, 19, and 19 of 20 cameras.
5. `_barrel_dimensions` reads the socket raw OBJ and derives the cylindrical
   wall radius and local lower/upper Z. For this 20 mm gap asset these are
   40 mm, 5 mm, and 55 mm. The model deliberately excludes the wider flange
   from the compared silhouette, but the complete collision mesh remains in
   the robot planning scene.
6. `fit_board_socket_body` sets the open axis `u = -R_RB[:,2]` and board X
   axis `x = R_RB[:,0]`, then `y = u × x`. For each admitted mask, it takes the
   mean pixel location of the foreground, unprojects through `K_undist`,
   transforms the ray through the calibrated camera/world pose and `T_WR`,
   intersects it with the plane 30 mm above the measured board, and subtracts
   30 mm along `u`. This yields one board XY estimate per camera. The
   coordinate-wise median gives the socket base XY. The base Z lies on the
   board; round-socket yaw is chosen from board X and is not observed.
7. The code projects the two wall rings and their convex hull into each
   camera. It records per-view wall-mask IoU and gates on **mean** IoU ≥0.8,
   not on every camera. `socket_refinement.py` stores the pose and diagnostics;
   `calibration.py` converts W→R and freezes `T_RS` for the session. The
   launcher currently takes one socket capture, so old two-capture
   translation/angle repeatability thresholds are inactive in this path.

This is a board-constrained closed-form estimate, not a soft prior or an
optimizer that later frees the axis. A board-aligned seed followed by stock
full-mesh silhouette optimization was tested on these records and produced
larger held-out wall disagreement and a flange crossing the board. The stock
optimizer's objective compares the entire mesh with masks that often contain
only the wall; initializing it correctly does not repair that mismatch.

## Independent-camera replay

Run `evaluate_saved_socket_views.py` with repeated `--run-dir` and one
`--output` directory. For each camera, it recomputes the XY median from all
*other* admitted cameras and scores the omitted camera's original mask.
Pixel hashes are checked against saved acquisition evidence. The JSON keeps
camera ID, frame ID, original pixel hash, PNG hashes, bounded timestamp and
its maximum error. The plotted cyan contour is the saved SAM mask and green
rings are the held-out prediction. No view's mask contributes to its own
green pose. This is a camera-generalization check; all cameras still share
the same board, CAD, and camera calibration. All 20 cameras' stored
intrinsics/extrinsics and `C2R` in the three frozen sessions match the
current static launch values numerically (maximum difference 0).

| Saved run | Views | Full-fit mean IoU | Omitted-view mean / median IoU | Views ≥0.8 | Worst omitted view / IoU | Largest single-view influence on fitted XY |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `20261021_090945_053190` | 18 | 0.943 | 0.942 / 0.969 | 17 | `25322641` / 0.625 | 0.137 mm |
| `20261021_104732_950124` | 19 | 0.925 | 0.925 / 0.969 | 17 | `25322643` / 0.593 | 0.133 mm |
| `20261021_140856_019116` | 19 | 0.921 | 0.921 / 0.967 | 17 | `25322643` / 0.607 | 0.211 mm |

The 0.59–0.63 views look down onto the broad flange: SAM includes flange
pixels while the diagnostic renders only the narrower wall. Their individual
centroid rays can be 10–12 mm away from the multi-view median, but omitting
one shifts that median by at most 0.211 mm in these recordings. Camera
`25322649` scores 0.831–0.852 and also has a visible flank/flange mismatch.
The real image grids, with every admitted camera labeled, and source-bound
metrics are saved under
`/home/robot/shared_data/hyunsu/precision_socket_pose_review_20261022/`.
The non-admitted cameras had empty SAM masks, so they provide no measured
socket silhouette to score. The first run's bounded acquisition timestamp
has ±32.9 s uncertainty; the others have about ±1.2 s. These are static
socket scenes, but that time evidence cannot prove subsecond cross-camera
synchrony. The XY influence statistic is sensitivity to deleting a camera,
not absolute center error or a millimeter accuracy guarantee.

## Method judgment and remaining limits

The method is appropriate for a flat, seated, upright fixed cylinder with
this known CAD and board. Its strong held-out wall agreement supports its
geometric consistency. It would be invalid for a tilted, lifted, or shimmed
socket, an incorrect board plane/C2R, or a different socket CAD. The current
admission minimum of two cameras cannot resolve an erroneous two-view pair;
the real deployment has 20 calibrated cameras, and no new arbitrary gate was
added from this retrospective replay. Mean IoU detects gross disagreement
but can conceal a bad single view; per-view IoUs remain recorded. Without an
independent pose survey or a later contact/insertion result, absolute axis
and center accuracy remains unproven.

The stock multi-camera capture and ChArUco measurement remain shared.
Removed for this active socket mode: stock whole-mesh silhouette refinement,
per-view socket FoundPose inference, socket FoundPose model construction on
the updated capture adapter, and socket FoundPose file requirements in live
startup. The pose channel placeholder remains because the stock orchestrator
collects paired mask/pose messages. Frozen older sessions can still be opened
without a socket FoundPose file; the key file remains required for subsequent
key recognition. Static `run_robot_pc.sh --check` passed; the test suite had
1104 passes and four skips when its CUDA-dependent FR3 module was excluded.
That one module cannot collect without CUDA on this host. No live capture or
insertion was attempted in this review.
