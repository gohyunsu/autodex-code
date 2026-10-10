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
and create held-out fit/validation data for `camera_time.py`. The journal
alone cannot supply UTC or an error bound. Only after camera-side deployment,
frame-ID/hash transport matching, rig-specific calibration, held-out timing
validation and a live startup replay should `AcquisitionTimeProvider` be
connected to `start_precision_session()`. The default path must continue to
reject missing/uncalibrated acquisition times.
