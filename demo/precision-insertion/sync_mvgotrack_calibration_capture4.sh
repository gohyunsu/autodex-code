#!/usr/bin/env bash
# Refresh only the experiment-owned MV-GoTrack camera world on capture4.
set -Eeuo pipefail

CAM_PARAM=${PRECISION_CALIB_DIR:-}
HOST=${PRECISION_MVGOTRACK_SSH_HOST:-capture4@147.46.219.136}
PORT=${PRECISION_MVGOTRACK_SSH_PORT:-77}
REMOTE=${PRECISION_MVGOTRACK_REMOTE_ROOT:-/home/capture4/precision_insertion_mvgotrack}

while (($#)); do
  case "$1" in
    --cam-param) CAM_PARAM=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z $CAM_PARAM ]]; then
  echo "source zerodex_capture4.env.sh or pass --cam-param" >&2
  exit 2
fi
for file in intrinsics.json extrinsics.json; do
  [[ -f $CAM_PARAM/$file ]] || {
    echo "missing calibration input: $CAM_PARAM/$file" >&2
    exit 2
  }
done

status=$(ssh -p "$PORT" "$HOST" "$REMOTE/manage_mvgotrack.py" status)
python3 - "$status" <<'PY'
import json
import sys

status = json.loads(sys.argv[1])
if status.get("stream_pid") is not None or status.get("tracker_pid") is not None:
    raise SystemExit("stop the experiment-owned MV-GoTrack process before calibration sync")
PY

ssh -p "$PORT" "$HOST" mkdir -p "$REMOTE/cam_param"
scp -P "$PORT" "$CAM_PARAM/intrinsics.json" \
  "$HOST:$REMOTE/cam_param/intrinsics.json.next"
scp -P "$PORT" "$CAM_PARAM/extrinsics.json" \
  "$HOST:$REMOTE/cam_param/extrinsics.json.next"
ssh -p "$PORT" "$HOST" \
  "mv '$REMOTE/cam_param/intrinsics.json.next' '$REMOTE/cam_param/intrinsics.json' && \
   mv '$REMOTE/cam_param/extrinsics.json.next' '$REMOTE/cam_param/extrinsics.json'"

local_intrinsics_sha=$(sha256sum "$CAM_PARAM/intrinsics.json" | awk '{print $1}')
local_extrinsics_sha=$(sha256sum "$CAM_PARAM/extrinsics.json" | awk '{print $1}')
remote_hashes=$(ssh -p "$PORT" "$HOST" \
  "sha256sum '$REMOTE/cam_param/intrinsics.json' '$REMOTE/cam_param/extrinsics.json'")
grep -q "$local_intrinsics_sha" <<<"$remote_hashes"
grep -q "$local_extrinsics_sha" <<<"$remote_hashes"
printf '%s\n' "$remote_hashes"
