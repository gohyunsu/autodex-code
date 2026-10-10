# Opt-in ParaDex camera chunk-timestamp collection

The checked-out ParaDex `PyspinCamera` enables GenICam `ChunkSelector=Timestamp`,
but its `get_image()` drops that field and returns only `frameID` plus
`pc_time` sampled after `GetNextImage`. The demo-only
`precision_camera_daemon.py` wraps the **same unreleased PySpin image** at
`GetNextImage` and journals `(camera serial, frame ID, raw chunk ticks)`.
It delegates normal camera methods and does not modify ParaDex or AutoDex
source files. This is an **offline clock-commissioning recorder**, not an
exposure-time provider for live precision-insertion sessions.

On one capture PC in an isolated test window, first stop the existing camera
server and confirm it released its 5480/5481/5482 ports. Then launch this
replacement **instead of** the stock server, using a new per-run directory on
durable storage accessible from the robot PC:

```bash
cd /path/to/autodex-code
PYTHONPATH="$PWD/demo/precision-insertion:/path/to/paradex" \
  /path/to/paradex-python demo/precision-insertion/precision_camera_daemon.py \
  --journal-dir /mnt/paradex2/hyunsu/NEW_chunk_clock_run_001
```

Do not run both camera daemons concurrently. Verify the ParaDex camera
controller can start/stop normally and compare its image/frame stream with
an uninstrumented run before deploying beyond one capture PC. This checkout
has only fake-PySpin unit tests; it has **not** been run against the live
camera server, PySpin SDK, or robot. If the chunk node is unavailable, frame
IDs or ticks regress, or the bounded writer queue fills, the wrapper fails
the camera grab rather than fabricating a timestamp. The capture PC's normal
camera error/recovery procedures then apply.

Each camera acquisition gets an exclusive `SERIAL_TIME_RANDOM.jsonl` journal,
so a stopped/restarted camera may legitimately restart its frame ID or clock
without mixing two epochs. ParaDex's post-stop buffer-drain reads are not
journaled. The first
line identifies schema `precision_insertion_camera_chunk_journal_v1`; each
following line carries `frame_id`, `chunk_timestamp_ticks` and
`host_received_utc_s`. The last field is **after** image retrieval and is not
an exposure timestamp. The tick frequency/epoch are intentionally unknown;
the raw ticks must not be interpreted as nanoseconds or Unix time. Journals
are fully flushed when the camera daemon stops cleanly. A crash may leave a
partial final line, which the verifier rejects. Preserve original journals
and record their SHA-256 hashes for commissioning.

For an offline integrity check **after the daemon stops**:

```bash
/path/to/autodex-python \
  demo/precision-insertion/audit_camera_chunk_journal.py \
  --journal /abs/path/to/SERIAL_TIME_RANDOM.jsonl \
  --camera-serial SERIAL
```

The underlying `verify_chunk_journal` checks the header and monotonic
frame/tick sequence and returns a file hash. It always reports
`exposure_utc_admissible=false`.

Next, obtain independently timed per-camera hardware trigger or same-camera
exposure references on a common UTC clock. Match those references to the
*exact imaging-camera* frame IDs, measure clock conversion/offset/drift error,
and create held-out fit/validation data. The separate
`camera_chunk_clock.ChunkUTCClock` can then fit and replay the **closed**
journal against those independent UTC references. One reference JSON has this
schema (the numbers are structural examples, **not** measurements):

```json
{
  "schema": "precision_insertion_chunk_utc_calibration_v1",
  "calibration_id": "camA_session001",
  "camera_serial": "CAMERA_SERIAL",
  "chunk_journal": {
    "path": "/abs/path/to/closed_camera_journal.jsonl",
    "sha256": "<journal SHA-256>"
  },
  "source_method": "independent_per_camera_trigger_metrology",
  "source_files": [
    {"path": "/abs/path/to/trigger_metrology_log.csv",
     "sha256": "<metrology log SHA-256>"}
  ],
  "tick_frequency_hz": 1000000000,
  "max_clock_rate_error_ppm": 100,
  "fit_samples": [
    {"frame_id": 101, "exposure_utc_s": 1790000000.01,
     "max_error_s": 0.0001},
    {"frame_id": 103, "exposure_utc_s": 1790000000.03,
     "max_error_s": 0.0001},
    {"frame_id": 105, "exposure_utc_s": 1790000000.05,
     "max_error_s": 0.0001}
  ],
  "validation_samples": [
    {"frame_id": 102, "exposure_utc_s": 1790000000.02,
     "max_error_s": 0.0001},
    {"frame_id": 104, "exposure_utc_s": 1790000000.04,
     "max_error_s": 0.0001},
    {"frame_id": 106, "exposure_utc_s": 1790000000.06,
     "max_error_s": 0.0001}
  ],
  "clock_offset_error_s": 0.0002,
  "drift_error_s_per_s": 0.00001,
  "max_total_error_s": 0.001,
  "max_extrapolation_s": 0.05,
  "valid_until_utc_s": 1790000001.0
}
```

Use the camera's measured or documented tick frequency; this value cannot
be inferred from `pc_time`. The independent references must identify the
*same imaging-camera frame* and include a worst-case trigger-to-exposure
latency and clock error. Fit and held-out frame IDs must be disjoint. Run:

```bash
/path/to/autodex-python \
  demo/precision-insertion/audit_chunk_utc_calibration.py \
  --calibration /abs/path/to/new_chunk_utc_calibration.json
```

The fitter subtracts integer ticks before floating-point regression, checks
clock rate against the declared frequency, scores both fit and held-out
references, and refuses frames beyond bounded extrapolation/expiry. Neither
this fit nor the journal proves the external instrument is calibrated. It is
an **offline replay only**: no current live camera-PC → robot-PC tick stream
or real UTC reference dataset has been supplied. Only after camera-side
deployment, frame-ID/hash transport matching, rig-specific physical
calibration, held-out timing validation and a live startup replay should a
calibrated acquisition-time provider be connected to
`start_precision_session()`. The default path must continue to reject
missing/uncalibrated acquisition times.
