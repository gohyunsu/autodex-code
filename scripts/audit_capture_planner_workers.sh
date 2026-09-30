#!/usr/bin/env bash
# Read-only preflight for using capture-PC GPUs as AutoDex planning workers.
#
# Usage:
#   bash scripts/audit_capture_planner_workers.sh
#   bash scripts/audit_capture_planner_workers.sh capture1 capture3
#
# The script deliberately does not stop daemons, install packages, update a
# checkout, or write remote files.  Its output is the input to the separate
# worker setup step.
set -uo pipefail

if (( $# > 0 )); then
    PCS=("$@")
else
    # capture4 is not part of the current operational camera-PC set.
    PCS=(capture1 capture2 capture3 capture5 capture6)
fi

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=6)

audit_one() {
    local pc="$1"
    ssh "${SSH_OPTIONS[@]}" "$pc" 'bash -s' <<'REMOTE'
set -u

echo "identity host=$(hostname) user=$USER"
echo "os $(uname -srmo)"

echo "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,compute_mode --format=csv,noheader
    echo "gpu_processes"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
else
    echo "nvidia-smi=missing"
fi

echo "cuda_toolkit"
if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    nvcc --version | tail -n 1
else
    echo "nvcc=missing"
fi
echo "transfer_tools"
command -v rsync || echo "rsync=missing"

echo "conda"
CONDA_ROOT=""
for candidate in "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -x "$candidate/bin/conda" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done
if [ -n "$CONDA_ROOT" ]; then
    echo "conda_root=$CONDA_ROOT"
    "$CONDA_ROOT/bin/conda" env list 2>/dev/null || true
    for env_name in planner autodex_planner_worker gotrack_cu128; do
        py="$CONDA_ROOT/envs/$env_name/bin/python"
        if [ ! -x "$py" ]; then
            echo "env=$env_name missing"
            continue
        fi
        "$py" -c '
import importlib.util
import platform
print("env_python", platform.python_version())
for name in ("torch", "numpy", "trimesh", "warp", "curobo"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(name, "missing")
        continue
    module = __import__(name)
    version = getattr(module, "__version__", "unknown")
    location = getattr(module, "__file__", "unknown")
    print(name, version, location)
if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch_cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch_gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
' 2>&1 | sed "s/^/env=$env_name /"
    done
else
    echo "conda=missing"
fi

echo "repositories"
for repo in "$HOME/AutoDex" "$HOME/RSS_2026" "$HOME/paradex"; do
    if [ -d "$repo/.git" ]; then
        branch=$(git -C "$repo" branch --show-current 2>/dev/null || true)
        revision=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || true)
        dirty=$(git -C "$repo" status --porcelain 2>/dev/null | wc -l)
        echo "$repo branch=$branch revision=$revision dirty_paths=$dirty"
    else
        echo "$repo missing"
    fi
done

echo "shared_assets"
for path in \
    "$HOME/shared_data/AutoDex/candidates/inspire/v8" \
    "$HOME/shared_data/AutoDex/content/configs/robot" \
    "$HOME/shared_data/object_processing"; do
    if [ -r "$path" ]; then
        echo "$path readable=yes"
    else
        echo "$path readable=no"
    fi
done
if [ -d "$HOME/shared_data" ]; then
    resolved=$(readlink -f "$HOME/shared_data" 2>/dev/null || true)
    echo "shared_data_resolved=$resolved"
    df -h "$HOME/shared_data" | tail -n 1
else
    echo "shared_data=missing"
fi

echo "active_autodex_daemons"
ps -eo pid,args | grep -E '[p]ython .*([i]nit_daemon|[g]otrack_daemon|[s]napshot_daemon|[p]erception_daemon)' || true
REMOTE
}

status=0
for pc in "${PCS[@]}"; do
    echo "===== $pc ====="
    if ! audit_one "$pc"; then
        echo "audit_failed pc=$pc"
        status=1
    fi
done

exit "$status"
