# AutoDex camera-frame provenance handoff

The precision demo must not freeze a socket collision world from an image whose
capture time or identity is unknown. The demo-local `live_capture.py` now
rejects the unchanged AutoDex snapshot/init outputs **by design**. This is a
deployment requirement, not a claim that the camera rig is already timed.

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
  demo/precision-insertion/tests/test_session_bootstrap.py
```
