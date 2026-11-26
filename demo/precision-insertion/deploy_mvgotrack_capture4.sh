#!/usr/bin/env bash
# Install only under /home/capture4/precision_insertion_mvgotrack.  Existing
# ParaDex camera services, environments and source trees are not modified.
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE=
CHECKPOINT=
CAM_PARAM=
HOST=${PRECISION_MVGOTRACK_SSH_HOST:-capture4@147.46.219.136}
PORT=${PRECISION_MVGOTRACK_SSH_PORT:-77}
REMOTE=${PRECISION_MVGOTRACK_REMOTE_ROOT:-/home/capture4/precision_insertion_mvgotrack}

while (($#)); do
  case "$1" in
    --source) SOURCE=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --cam-param) CAM_PARAM=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for path in "$SOURCE" "$CHECKPOINT" "$CAM_PARAM/intrinsics.json" "$CAM_PARAM/extrinsics.json"; do
  [[ -e "$path" ]] || { echo "missing deployment input: $path" >&2; exit 2; }
done
[[ -d "$SOURCE/.git" ]] || { echo "--source must be an MV-GoTrack checkout" >&2; exit 2; }

ssh -p "$PORT" "$HOST" mkdir -p \
  "$REMOTE/MV-GoTrack/checkpoints" "$REMOTE/state" \
  "$REMOTE/sessions" "$REMOTE/cam_param"
rsync -a --delete -e "ssh -p $PORT" \
  --exclude .git --exclude checkpoints/ \
  "$SOURCE/" "$HOST:$REMOTE/MV-GoTrack/"
ssh -p "$PORT" "$HOST" mkdir -p "$REMOTE/MV-GoTrack/checkpoints"
checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
remote_checkpoint_sha=$(ssh -p "$PORT" "$HOST" \
  "sha256sum '$REMOTE/MV-GoTrack/checkpoints/gotrack_checkpoint.pt' 2>/dev/null | cut -d ' ' -f1" || true)
if [[ "$remote_checkpoint_sha" != "$checkpoint_sha" ]]; then
  scp -P "$PORT" "$CHECKPOINT" \
    "$HOST:$REMOTE/MV-GoTrack/checkpoints/gotrack_checkpoint.pt"
fi
scp -P "$PORT" "$CAM_PARAM/intrinsics.json" "$CAM_PARAM/extrinsics.json" \
  "$HOST:$REMOTE/cam_param/"
scp -P "$PORT" \
  "$HERE/capture4/mvgotrack_stream.py" \
  "$HERE/capture4/manage_mvgotrack.py" \
  "$HERE/capture4/mvgotrack_capture4.patch" \
  "$HOST:$REMOTE/"
ssh -p "$PORT" "$HOST" \
  "grep -q MVGOTRACK_ANY_GPU '$REMOTE/MV-GoTrack/deploy/live_tracker.py' || \
   (cd '$REMOTE/MV-GoTrack' && patch -p1 < '$REMOTE/mvgotrack_capture4.patch')"
ssh -p "$PORT" "$HOST" \
  "chmod 755 '$REMOTE/manage_mvgotrack.py' '$REMOTE/mvgotrack_stream.py' && \
   printf '%s\n' '{\"capture4\":{\"ip\":\"127.0.0.1\",\"cam_list\":[\"26053248\",\"25322639\",\"25305462\",\"25322642\"]}}' > '$REMOTE/pc.json' && \
   MVGOTRACK_ANY_GPU=1 '$REMOTE/manage_mvgotrack.py' status"
