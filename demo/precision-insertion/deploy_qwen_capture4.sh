#!/usr/bin/env bash
# Prepare an isolated, persistent-Qwen deployment on capture4. This script
# writes only below /home/capture4/precision_insertion_qwen and the local
# precision_insertion_project runtime directory. It does not open cameras,
# start a robot process, or modify capture4's existing Conda environments.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/config/zerodex_capture4.env.sh"

ACTION=${1:---check}
HOST=${PRECISION_CAPTURE4_SSH_HOST:-capture4@147.46.219.136}
PORT=${PRECISION_CAPTURE4_SSH_PORT:-77}
REMOTE_ROOT=${PRECISION_CAPTURE4_QWEN_ROOT:-/home/capture4/precision_insertion_qwen}
REMOTE_BASE_PY=${PRECISION_CAPTURE4_QWEN_BASE_PYTHON:-/home/capture4/anaconda3/envs/gotrack_cu128/bin/python}
LOCAL_MODEL=${PRECISION_VLM_MODEL_ID:-$PRECISION_LOCAL_PROJECT_ROOT/cache/models/Qwen3-VL-2B-Instruct}
LOCAL_RUNTIME=${PRECISION_LOCAL_PROJECT_ROOT}/runtime
LOCAL_WORKER=${LOCAL_RUNTIME}/capture4_qwen_worker
OUTPUT_CONFIG=${PRECISION_CAPTURE4_QWEN_ROBOT_CONFIG:-$LOCAL_RUNTIME/franka_runtime_capture4_qwen.json}
SOURCE_CONFIG=${PRECISION_CAPTURE4_QWEN_SOURCE_CONFIG:-$PRECISION_STAGE_ROOT/franka_runtime.json}
WORKERS_JSON=${PRECISION_CAPTURE4_QWEN_WORKERS_JSON:-$SCRIPT_DIR/config/capture4_qwen_workers.json}
REMOTE_MODEL=$REMOTE_ROOT/model/Qwen3-VL-2B-Instruct
REMOTE_WORKER=$REMOTE_ROOT/worker
REMOTE_PY=$REMOTE_ROOT/venv/bin/python

ssh_command=(ssh -p "$PORT" -o BatchMode=yes "$HOST")
rsync_ssh="ssh -p $PORT -o BatchMode=yes"

prepare_local() {
  mkdir -p "$LOCAL_RUNTIME"
  "$PRECISION_CHECK_PY" "$SCRIPT_DIR/setup_capture_qwen.py" \
    --source-config "$SOURCE_CONFIG" \
    --output-config "$OUTPUT_CONFIG" \
    --worker-root "$LOCAL_WORKER" \
    --workers-json "$WORKERS_JSON"
}

check_remote() {
  local local_hash remote_hash
  local_hash=$(sha256sum "$LOCAL_MODEL/model.safetensors" | awk '{print $1}')
  remote_hash=$("${ssh_command[@]}" \
    "test -x '$REMOTE_PY' && test -f '$REMOTE_MODEL/model.safetensors' && test -f '$REMOTE_WORKER/precision_insertion/distributed_qwen_worker.py' && '$REMOTE_PY' -c 'import torch, transformers, accelerate, PIL; print(torch.__version__, transformers.__version__, accelerate.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))' >&2 && sha256sum '$REMOTE_MODEL/model.safetensors' | awk '{print \$1}'")
  if [[ $remote_hash != "$local_hash" ]]; then
    echo "capture4 Qwen model hash differs" >&2
    return 1
  fi
  printf 'capture4 Qwen ready: config=%s model_sha256=%s\n' \
    "$OUTPUT_CONFIG" "$remote_hash"
}

case "$ACTION" in
  --prepare)
    prepare_local
    ;;
  --deploy)
    prepare_local
    "${ssh_command[@]}" \
      "mkdir -p '$REMOTE_ROOT/model' '$REMOTE_WORKER'; if test ! -x '$REMOTE_PY'; then '$REMOTE_BASE_PY' -m venv --system-site-packages '$REMOTE_ROOT/venv'; fi"
    "${ssh_command[@]}" \
      "'$REMOTE_PY' -m pip install --disable-pip-version-check 'transformers==5.5.0' 'accelerate==1.13.0' 'pillow==12.3.0'"
    rsync -a --checksum -e "$rsync_ssh" \
      "$LOCAL_WORKER/" "$HOST:$REMOTE_WORKER/"
    rsync -a --checksum -e "$rsync_ssh" \
      "$LOCAL_MODEL/" "$HOST:$REMOTE_MODEL/"
    check_remote
    ;;
  --check)
    test -f "$OUTPUT_CONFIG"
    check_remote
    ;;
  *)
    echo "usage: $0 [--prepare|--deploy|--check]" >&2
    exit 2
    ;;
esac
