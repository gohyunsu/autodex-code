#!/usr/bin/env bash
# Copy the exact current AutoDex working tree and robot-PC cuRobo source to
# dedicated capture-PC worker directories.  Existing camera-daemon checkouts
# and conda environments are not touched.
#
# Usage:
#   bash scripts/sync_capture_planner_workers.sh
#   bash scripts/sync_capture_planner_workers.sh capture1 capture3
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUROBO_SOURCE=${CUROBO_SOURCE_ROBOT:-$HOME/RSS_2026/planner}
REMOTE_ROOT=${AUTODEX_PLANNER_WORKER_ROOT:-AutoDex-planner-worker}

if [ ! -f "$CUROBO_SOURCE/setup.py" ]; then
    echo "[sync] robot-PC cuRobo source missing: $CUROBO_SOURCE" >&2
    exit 2
fi

if (( $# > 0 )); then
    PCS=("$@")
else
    PCS=(capture1 capture2 capture3 capture5 capture6)
fi

for pc in "${PCS[@]}"; do
    echo "===== sync $pc ====="
    ssh -o BatchMode=yes -o ConnectTimeout=6 "$pc" \
        "mkdir -p \"\$HOME/$REMOTE_ROOT/.worker_vendor/curobo\""
    rsync -a \
        --exclude='outputs/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.so' \
        --exclude='build/' \
        --exclude='autodex/perception/thirdparty/' \
        "$REPO_ROOT/" "$pc:$REMOTE_ROOT/"
    rsync -a \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.so' \
        --exclude='build/' \
        "$CUROBO_SOURCE/" "$pc:$REMOTE_ROOT/.worker_vendor/curobo/"
done

echo "[sync] complete: ${#PCS[@]} worker(s)"
