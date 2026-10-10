# AutoDex camera-frame provenance handoff

The precision demo must not freeze a socket collision world from an image whose
capture time or identity is unknown. The demo-local `live_capture.py` now
rejects the unchanged AutoDex snapshot/init outputs **by design**. This is a
deployment requirement, not a claim that the camera rig is already timed.

## Demo-only transport now implemented (still not a timestamp solution)

`precision_insertion/camera_transport.py` supplies two narrow adapters while
reusing the stock camera capture and FoundPose inference code:

- `SnapshotMetadataTap` subscribes to the same stock snapshot PUB stream and
  records its `fid` plus a JPEG hash. `ProvenanceSnapshotAdapter` lets the
  unchanged `SnapshotOrchestrator` dispatch/collect, but returns a frame ID
  only if its decoded JPEG exactly matches the tap's same-request JPEG.
  A missing tap message rejects that capture; no later image is substituted.
- `precision_init_daemon.py` is a **capture-PC replacement for the stock
  init daemon**, not a modification of it. It subclasses the existing
  SAM3/FoundPose pipeline, records the SHM `(frame ID, image)` used by the
  request, hashes the exact undistorted pixels and adds `fid`/pixel hash to
  both existing mask and pose PUB metadata. On the robot PC,
  `PrecisionInitOrchestrator(stock_init_orchestrator)` preserves these extra
  fields in the stock buffers. `collect_socket_capture` and
  `collect_key_capture` require both payload hashes and IDs to match their
  same-request saved PNG. Only one daemon may bind ports 6893/5006/5007 on a
  capture PC; do **not** start stock and precision daemons together.

Example robot-PC construction, before starting any capture:

```python
identities = FrameIdentityRegistry()
stock_snap = SnapshotOrchestrator(pc_list, capture_ips)
tap = SnapshotMetadataTap(capture_ips)
board_snap = ProvenanceSnapshotAdapter(stock_snap, tap.buffer, identities)
stock_init = InitOrchestrator(pc_list, capture_ips)
init = PrecisionInitOrchestrator(stock_init, identities)
calibrations = {
    serial: CameraTimeCalibration.load(path)
    for serial, path in per_camera_calibration_paths.items()
}
acquisition_metadata_for_request = AcquisitionTimeProvider(
    identities, calibrations)
# Pass board_snap to collect_board_snapshot() and init to
# collect_socket_capture()/collect_key_capture(), using the provider as
# acquisition_metadata_for_request. Close tap and stock orchestrators when
# the session ends.
```

The example names are imports from the unchanged AutoDex orchestrators and
the demo-local `camera_transport` module. The metadata tap's PUB connection
is not a guaranteed barrier: if it misses an initial message, the capture
fails closed and must be repeated. These adapters have synthetic transport
tests, **not** an AutoDex-camera-PC deployment test.

`camera_time.py` supplies the **interface and validation** for
`acquisition_metadata_for_request(request_id)`. It is not a calibration
measurement. For each imaging camera, a
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
ParaDex enables image timestamp chunk data, but its current `get_image()`
returns only `pc_time` and `frameID`; the chunk exposure timestamp is not
presently carried into SHM. Reading/validating that timestamp per imaging
camera (or instrumenting an external trigger) and measuring its UTC clock
conversion is the next capture-PC commissioning task. No real calibration
record is included, so the adapter cannot make the current rig robot-ready.
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

Do not put publication, dispatch, or unbounded host-receipt time into
`timestamp_s` merely to pass the gate.

## Required demo-local camera producer/adapters

Keep the three original execution files and the existing AutoDex daemons
unchanged. Deploy demo-specific capture-side variants and robot-side adapters
that provide the following for *the exact frame* used in each request:

1. Preserve `(camera serial, request ID, sensor frame ID)` from SHM through
   board JPEG return, and through both socket SAM and FoundPose payloads.
   One camera's mask and pose must have the same ID. IDs must be positive and
   should be checked for monotonic advance at the producer.
2. On the capture PC, provide exposure time converted to the common Unix UTC
   clock using either a real hardware exposure timestamp or a validated
   per-camera frame-ID-to-exposure-time calibration. Record the method, a
   worst-case clock-conversion error in seconds, and the calibration ID when
   applicable. Measure the camera-PC↔robot-PC clock offset/error; do not
   assume NTP alone satisfies a millimetre-level task.
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
repeatability tests, and guarded insertion commissioning. Until then, use
saved offline evidence only and do not authorize robot motion from these
session helpers.

Test the software gate without hardware:

```bash
cd /home/hyunsu/autodex-code
/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python -m pytest -q \
  demo/precision-insertion/tests/test_live_capture.py \
  demo/precision-insertion/tests/test_session_bootstrap.py \
  demo/precision-insertion/tests/test_key_perception.py \
  demo/precision-insertion/tests/test_xy_retry.py
```
