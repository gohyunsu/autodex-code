# AutoDex camera-frame provenance handoff

The precision demo must not freeze a socket collision world from an image whose
capture time or identity is unknown. The demo-local `live_capture.py` rejects
the unchanged AutoDex snapshot/init outputs **by design**. A conservative
request/response *interval* can now bound exposure for bring-up; it is not a
millisecond-accurate camera timestamp or a precision-calibration result.

The current stock live init daemon makes one undistortion map per camera from
`original_intrinsics`, `dist_params`, and `intrinsics_undistort`. For each
request it remaps the raw SHM BGR frame once, saves that undistorted BGR array
as a PNG, and converts the same array to RGB for SAM3 and FoundPose. FoundPose
uses `intrinsics_undistort` for those pixels. The board snapshot follows the
stock tabletop path: ChArUco is detected in the original JPEG and only its
corner coordinates are undistorted before triangulation. The stock diagnostic
`probe_stock_camera_qa.py` does not remap either output again. These are two
different, consistent image paths; a board JPEG and a socket PNG must not be
compared as if they shared the same pixel coordinates.

## Demo-only transport and timing modes

`precision_insertion/camera_transport.py` supplies two narrow adapters while
reusing the stock camera capture and FoundPose inference code:

- `SnapshotMetadataTap` subscribes to the same stock snapshot PUB stream and
  records its `fid` plus a JPEG hash. `ProvenanceSnapshotAdapter` lets the
  unchanged `SnapshotOrchestrator` dispatch/collect, but returns a frame ID
  only if its decoded JPEG exactly matches the tap's same-request JPEG.
  A missing tap message rejects that capture; no later image is substituted.
- `precision_init_daemon.py` is a **demo-local capture-PC adapter** that may
  run on a separate set of ports alongside the unchanged stock daemon. It
  subclasses the existing
  SAM3/FoundPose pipeline, records the SHM `(frame ID, image)` used by the
  request, hashes the exact undistorted pixels and adds `fid`/pixel hash to
  both existing mask and pose PUB metadata. On the robot PC,
  `PrecisionInitOrchestrator(stock_init_orchestrator)` preserves these extra
  fields in the stock buffers. `collect_socket_capture` and
  `collect_key_capture` require both payload hashes and IDs to match their
  same-request saved PNG. The two daemons need distinct mask, pose and command
  ports; the staged adapter uses 15006/15007/16893 while stock continues on
  5006/5007/6893. The adapter can import the installed stock code via
  `AUTODEX_STOCK_ROOT` without changing that checkout.

Example robot-PC construction, before starting any capture:

```python
identities = FrameIdentityRegistry()
windows = RequestWindowRegistry()
stock_snap = SnapshotOrchestrator(pc_list, capture_ips)
tap = SnapshotMetadataTap(capture_ips)
board_snap = ProvenanceSnapshotAdapter(
    stock_snap, tap.buffer, identities, request_windows=windows)
stock_init = InitOrchestrator(pc_list, capture_ips)
init = PrecisionInitOrchestrator(
    stock_init, identities, request_windows=windows)
calibrations = {
    serial: CameraTimeCalibration.load(path)
    for serial, path in per_camera_calibration_paths.items()
}
acquisition_metadata_for_request = AcquisitionTimeProvider(
    identities, calibrations, request_windows=windows)
# Pass board_snap to collect_board_snapshot() and init to
# collect_socket_capture()/collect_key_capture(), using the provider as
# acquisition_metadata_for_request. Close tap and stock orchestrators when
# the session ends.
```

The example names are imports from the unchanged AutoDex orchestrators and
the demo-local `camera_transport` module. The metadata tap's PUB connection
is not a guaranteed barrier: if it misses an initial message, the capture
fails closed and must be repeated. On 2026-10-13 the separate-port init
adapter was staged on all five AutoDex capture PCs and exercised against all
20 live cameras: each returned mask, pose and complete undistorted PNG had
matching request/frame IDs and pixel hashes. This proves those per-request
bindings; the snapshot metadata tap, exposure-time alignment, strict session
admission and robot-frame accuracy have not been commissioned by that test.
The adapter accepts a positive frame ID per request; ParaDex may restart its
counter when a stopped camera stream is armed again.

`camera_time.py` supplies two modes for
`acquisition_metadata_for_request(request_id)`:

- `calibrated_frame_id` (default): the existing per-camera exposure-time
  calibration below. Use this for quantitative timing and precision claims.
- `bounded_request_interval`: on the robot PC, record Unix-clock times before
  the stock request and after its response. The unchanged **live** AutoDex
  daemons call `wait_for_new_frames()` after receiving that request, so the
  exact matched frame's exposure lies inside this interval. Store its midpoint
  only as a representation of `[start, end]`, with `max_error_s` equal to half
  the full interval. This requires no capture-PC clock synchronization and
  never uses publication `ts` as exposure. A slow SAM/FoundPose response can
  produce seconds of uncertainty. The stationary board, socket and tabletop
  stationary key admission gates retain exact same-request frame binding but do not
  compare the inference-inflated interval with a cross-camera exposure-skew
  limit. Motion-linked frame-age and robot-feedback gates still see the full
  bound and may refuse the capture.
  It is suitable for stationary bring-up, not measured millimetre-level
  temporal accuracy. It does not work with disk-mode replay, where the frame
  predates the request.

To try bounded timing, change the launch JSON to
`"camera_time_mode": "bounded_request_interval"` and **remove**
`camera_time_calibrations`. Keep the precision init daemon and snapshot tap
for exact frame IDs/pixels. The initial `start-session --config ...` command
is a file-only preflight; adding `--capture` actually opens the cameras.
Robot motion still needs the separate `run_live.py` integration and controllers.
If a bound is wider than a configured timing/feedback limit, use a faster
capture or a true camera calibration; do not shrink `max_error_s` by hand.

The calibrated mode is not itself a calibration measurement. For each imaging
camera, a
`precision_insertion_camera_time_calibration_v1` JSON needs:

- `calibration_id`, `camera_serial`, `source_method` equal to
  `same_imaging_camera_exposure_chunk_utc` or
  `independent_per_camera_trigger_metrology`, and hashed absolute
  `source_files` containing the external measurements;
- at least three `fit_samples` and three disjoint `validation_samples`,
  each with `frame_id`, `exposure_utc_s`, and its `max_error_s`;
- `clock_offset_error_s`, `drift_error_s_per_frame`,
  `max_total_error_s`, `max_extrapolation_frames`, and
  `valid_until_utc_s` measured for this rig.

The loader fits the frame-ID period, checks the held-out residuals, derives a
conservative error bound including source error, clock offset and bounded
future drift, rejects fits beyond the independently commissioned error limit,
and refuses frames beyond its validity window or a calibration past its UTC
expiration. The registry
joins that time to the *same* request's exact image hash and frame ID. Do not
create this JSON from snapshot/FoundPose publication `ts`, `pc_time` after
`GetNextImage`, or the **separate** timestamp-monitor camera's frame IDs.
The live calibrated provider also records the robot-host request/response
window and rejects a calibrated exposure interval that does not overlap it.
This catches stale frame-ID time mappings after a camera-stream restart when
the mapped time lies outside the current request. It is still not a substitute
for binding a physical calibration to the correct camera-stream epoch when
frame IDs can reset and recur within a narrow time window.
ParaDex enables image timestamp chunk data, but its current `get_image()`
returns only `pc_time` and `frameID`; the chunk exposure timestamp is not
presently carried into SHM. Reading/validating that timestamp per imaging
camera (or instrumenting an external trigger) and measuring its UTC clock
conversion is the next capture-PC commissioning task. No real calibration
record is included, so calibrated-frame-ID mode cannot make the current rig
robot-ready without that handoff.
An opt-in [same-image chunk journal](CAMERA_CHUNK_CAPTURE.md) now records raw
frame/tick pairs without changing ParaDex source. It remains an offline
commissioning input: raw ticks and host receipt time are **not** UTC exposure
measurements and are never fed into a session by themselves.

## What the existing AutoDex path actually provides

- `src/execution/daemon/snapshot_daemon.py` reads `(image, frame_id)` from
  ParaDex SHM and publishes a JPEG plus `fid`. Its `ts` is JPEG publication
  time. `autodex/perception/snapshot_orchestrator.py` discards the published
  `fid` and returns decoded pixels without it.
- `src/execution/daemon/init_daemon.py` reads `(image, frame_id)`, but drops
  `frame_id` before undistortion, SAM and FoundPose. Saved PNGs and the
  mask/pose payloads therefore cannot be proved to share a specific sensor
  frame. Their `ts` fields are inference-publication times.
- ParaDex `Camera.acquire` receives a `pc_time` from `get_image()`, but the
  current SHM ring stores only image and frame ID. That `pc_time` is recorded
  after `GetNextImage` and is **not** a hardware exposure timestamp. The
  separate timestamp-monitor camera is not a per-frame clock for each
  imaging camera without an independently validated correspondence.

Do not put publication, dispatch, or host-receipt time alone into
`timestamp_s` merely to pass the gate. The bounded mode uses *both* endpoints
of a verified new-frame request and reports the entire interval as error.

## Required demo-local camera producer/adapters

Keep the three original execution files and the existing AutoDex daemons
unchanged. Deploy demo-specific capture-side variants and robot-side adapters
that provide the following for *the exact frame* used in each request:

1. Preserve `(camera serial, request ID, sensor frame ID)` from SHM through
   board JPEG return, and through both socket SAM and FoundPose payloads.
   One camera's mask and pose must have the same ID. IDs must be positive;
   the stock reader waits for a new SHM frame within the current request.
   ParaDex may reset its counter when the camera stream restarts, so compare
   frame identity within a request rather than against a previous stream.
2. For precise exposure timing, provide capture-PC hardware time or validated
   per-camera frame-ID calibration in the common clock and retain its error
   bound. For stationary bring-up only, the robot-host bounded-request mode
   is available as described above. It never claims per-camera exposure time.
3. Bind this record to the decoded BGR pixels with
   `frame_provenance.image_sha256(image)`. For board capture, hash pixels
   decoded from the *same encoded JPEG* that is transmitted. For socket
   capture, hash the exact undistorted BGR pixels saved as the lossless PNG
   and fed to SAM/FoundPose. Do not hash a separate later snapshot.
4. Expose a request-ID lookup returning the schema below. A lookup may be
   built over a reliable camera-side metadata stream; the demo does not
   prescribe transport or pretend that a string `source` proves hardware
   provenance. Record and commission its implementation before robot use.

```json
{
  "request_id": 42,
  "source": "camera_acquisition",
  "frames": {
    "CAMERA_SERIAL": {
      "frame_id": 12345,
      "image_sha256": "64 lowercase hex digits of decoded BGR pixels plus shape/dtype",
      "timestamp_s": 1791572400.0,
      "max_error_s": 0.001,
      "timestamp_method": "hardware_exposure",
      "clock_domain": "unix_utc"
    }
  }
}
```

For `timestamp_method: "calibrated_frame_id"`, add a nonempty
`calibration_id`; its underlying measurements must be retained. The example
time and error above are **schema examples, not measured rig values**.

The board adapter must return `{"image": BGR, "frame_id": int}` for every
camera. The socket adapter must add `frame_id` to each matching SAM mask and
FoundPose pose payload, and save the same-request PNG. `collect_board_snapshot`
and `collect_socket_capture` reject missing/mismatched IDs, image hashes,
uncalibrated camera IDs and incomplete metadata. `bootstrap_session` checks
the stated worst-case cross-camera skew as
`max(timestamp + error) - min(timestamp - error)` against its configured
session limit. Evidence files record each frame's hash, ID, time and method.

The same contract applies after a failed insertion: `assess_xy_retry` takes
the full-frame `LabeledFrame` images, their separately retained sensor frame
IDs, request ID and camera-side metadata. It checks the exact BGR pixel
digest, timestamp equality and worst-case cross-camera skew/age **before**
projecting 1 mm choices or calling the VLM. Cropped overlays are derived only
after this gate. If the acquisition time is too uncertain or old, the result
is `visual_abstain`, not an XY motion command.

It also applies to each new tabletop key observation. After reinitializing
the existing AutoDex FoundPose orchestrator for the selected v8 **key**,
`collect_key_capture()` uses the strict same-request PNG/mask/pose capture;
`admit_key_capture()` checks two or more per-view masks/poses, multi-view
physical-pose agreement and unchanged AutoDex IoU/silhouette refinement.
It additionally requires the selected key-mask/refined-CAD IoU to exceed an
explicit commissioned threshold; the stock whole-image silhouette loss alone
can admit an incorrect mask for a small key. This is still not a pose-accuracy
certificate or a guarantee that every admitted camera mask is correct.
The resulting timestamp is the selected view's exposure time, but its
record also keeps the earliest/latest bounds across **all** admitted views.
`plan_admitted_key_trial()` requires the robot-state timestamp to be close to
both ends of that interval before invoking the existing candidate/preflight
logic. Save the raw per-trial bundle with `write_key_capture_artifacts()`.
The key gate compares current intrinsics/extrinsics to the session's saved
camera snapshot, then projects the frozen socket mesh into each view and
rejects a key mask with excessive socket-hull overlap. This is a conservative
veto, not a pixel-accurate segmentation proof: a key behind/adjacent to the
socket may be rejected, and a calibration error can shift the projected ROI.
Set its overlap/dilation limits from annotated AutoDex-rig images, retain
key-specific prompts, and review bring-up examples before robot use.

This contract catches accidental cross-frame mixing; it is not a signature
or an independent certification of the producer. A physical bring-up still
needs a measured timestamp-error budget, camera/robot calibration, socket
repeatability tests, and guarded insertion commissioning. Bounded-request
timing can exercise a stationary live session without the camera-clock files,
but it does not by itself authorize robot motion or certify insertion depth.

Test the software gate without hardware:

```bash
cd /home/hyunsu/autodex-code
/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests/test_live_capture.py \
  demo/precision-insertion/tests/test_session_bootstrap.py \
  demo/precision-insertion/tests/test_key_perception.py \
  demo/precision-insertion/tests/test_xy_retry.py
```
