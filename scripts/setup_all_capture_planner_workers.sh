#!/usr/bin/env bash
# Invoke the isolated planner-worker setup on capture PCs after syncing code.
# Installs are intentionally sequential to avoid five simultaneous large
# PyTorch downloads. Camera/perception daemons are never stopped by this tool.
#
# Usage:
#   bash scripts/setup_all_capture_planner_workers.sh
#   bash scripts/setup_all_capture_planner_workers.sh capture1 capture3
set -euo pipefail

REMOTE_ROOT=${AUTODEX_PLANNER_WORKER_ROOT:-AutoDex-planner-worker}
if (( $# > 0 )); then
    PCS=("$@")
else
    PCS=(capture1 capture2 capture3 capture5 capture6)
fi

for pc in "${PCS[@]}"; do
    echo "===== setup $pc ====="
    ssh -o BatchMode=yes -o ConnectTimeout=6 "$pc" \
        "bash -lc 'cd \"\$HOME/$REMOTE_ROOT\" && bash scripts/setup_capture_planner_worker.sh'"
done

echo "[setup-all] complete: ${#PCS[@]} worker(s)"
