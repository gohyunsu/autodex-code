#!/usr/bin/env bash
# Build an isolated AutoDex planning environment on one capture PC.
#
# Run this script FROM the worker checkout that should be executed:
#   cd ~/AutoDex-planner-worker
#   bash scripts/setup_capture_planner_worker.sh
#
# Required beforehand:
#   - an NVIDIA driver compatible with CUDA 12.8 wheels;
#   - network access to the PyTorch/NVIDIA conda channels.  When a system
#     nvcc is absent, CUDA 12.8 nvcc is installed inside the worker env;
#   - ~/shared_data mounted with AutoDex candidates/configs and object_processing;
#   - an exact cuRobo source tree at .worker_vendor/curobo (preferred),
#     or set CUROBO_SOURCE explicitly.
#
# This script never modifies or removes the capture PC's perception envs.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${AUTODEX_PLANNER_ENV:-autodex_planner_worker}
CUROBO_SOURCE=${CUROBO_SOURCE:-$REPO_ROOT/.worker_vendor/curobo}
# sync_capture_planner_workers.sh exports cuRobo without its .git directory.
# cuRobo's setuptools-scm configuration consequently needs an explicit package
# version during the editable install.  Keep this override configurable for a
# deliberately different cuRobo release; 0.7.0 is the compatible worker
# baseline when metadata is unavailable.
CUROBO_PACKAGE_VERSION=${CUROBO_PACKAGE_VERSION:-0.7.0}

CONDA_ROOT=""
for candidate in "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -x "$candidate/bin/conda" ]; then
        CONDA_ROOT="$candidate"
        break
    fi
done
if [ -z "$CONDA_ROOT" ]; then
    echo "[setup] conda not found under ~/anaconda3 or ~/miniconda3" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] nvidia-smi is missing" >&2
    exit 2
fi
if [ ! -f "$CUROBO_SOURCE/setup.py" ]; then
    echo "[setup] exact cuRobo source is missing: $CUROBO_SOURCE" >&2
    echo "[setup] sync it from the robot PC or set CUROBO_SOURCE" >&2
    exit 2
fi

required_assets=(
    "$HOME/shared_data/AutoDex/candidates/inspire/v8"
    "$HOME/shared_data/AutoDex/content/configs/robot"
    "$HOME/shared_data/object_processing"
)
for path in "${required_assets[@]}"; do
    if [ ! -r "$path" ]; then
        echo "[setup] required shared asset is not readable: $path" >&2
        exit 2
    fi
done

if ! "$CONDA_ROOT/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating conda env $ENV_NAME"
    "$CONDA_ROOT/bin/conda" create -n "$ENV_NAME" python=3.10 -y
else
    echo "[setup] reusing conda env $ENV_NAME"
fi

PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
PIP="$CONDA_ROOT/envs/$ENV_NAME/bin/pip"

if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "$CONDA_ROOT/envs/$ENV_NAME/bin/nvcc" ]; then
    echo "[setup] installing CUDA 12.8 nvcc inside $ENV_NAME"
    "$CONDA_ROOT/bin/conda" install -n "$ENV_NAME" -y --override-channels \
        -c nvidia/label/cuda-12.8.1 -c defaults \
        cuda-nvcc=12.8.93
fi
export PATH="$CONDA_ROOT/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="$CONDA_ROOT/envs/$ENV_NAME"
# Conda's CUDA 12 packages keep the target headers outside CUDA_HOME/include.
# torch.utils.cpp_extension invokes the host C++ compiler without that target
# include directory, so expose it through CPATH for both C++ and nvcc builds.
CUDA_TARGET_ROOT="$CUDA_HOME/targets/x86_64-linux"
if [ ! -f "$CUDA_TARGET_ROOT/include/cuda_runtime_api.h" ]; then
    echo "[setup] CUDA runtime headers are missing: $CUDA_TARGET_ROOT/include" >&2
    exit 2
fi
export CPATH="$CUDA_TARGET_ROOT/include${CPATH:+:$CPATH}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[setup] nvcc is still unavailable after environment setup" >&2
    exit 2
fi
echo "[setup] nvcc=$(command -v nvcc)"
echo "[setup] CUDA headers=$CUDA_TARGET_ROOT/include"
nvcc --version | tail -n 1

echo "[setup] installing the robot-PC PyTorch/CUDA contract"
"$PIP" install \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128
"$PIP" install -r "$REPO_ROOT/scripts/requirements-planner-worker.txt"

CUDA_ARCH=$(
    "$PY" -c 'import torch; p=torch.cuda.get_device_capability(0); print(f"{p[0]}.{p[1]}")'
)
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS=${MAX_JOBS:-4}
echo "[setup] GPU=$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))') arch=$CUDA_ARCH"

# Make ordinary `conda activate autodex_planner_worker; python ...` commands
# retain the architecture selected for this specific host.
ACTIVATE_DIR="$CONDA_ROOT/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$CUDA_ARCH" \
    > "$ACTIVATE_DIR/autodex_planner_worker_arch.sh"
printf 'export CUDA_HOME=%q\n' "$CONDA_ROOT/envs/$ENV_NAME" \
    >> "$ACTIVATE_DIR/autodex_planner_worker_arch.sh"

echo "[setup] compiling/installing cuRobo from $CUROBO_SOURCE"
"$PIP" uninstall -y nvidia-curobo >/dev/null 2>&1 || true
(
    cd "$CUROBO_SOURCE"
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="$CUROBO_PACKAGE_VERSION" \
        "$PIP" install -e . --no-build-isolation --no-deps
)

echo "[setup] installing AutoDex from $REPO_ROOT"
"$PIP" install -e "$REPO_ROOT" --no-deps

echo "[setup] validating imports and shared assets"
(
    cd "$REPO_ROOT"
    TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" "$PY" - <<'PY'
from pathlib import Path
import numpy as np
import torch
import curobo
import autodex

from src.validation.planning.pipeline_lift_reachability.core import (
    available_objects,
    planner_robot_for,
    tabletop_files,
)

assert torch.cuda.is_available()
assert planner_robot_for("xarm", "inspire") == "inspire"
assert "pepsi" in available_objects("inspire", "v8")
pepsi_008 = next(
    path for path in tabletop_files("pepsi", "v8") if path.stem == "008"
)
print("[verify] python", __import__("sys").version.split()[0])
print("[verify] torch", torch.__version__, "cuda", torch.version.cuda)
print("[verify] gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("[verify] curobo", Path(curobo.__file__).resolve())
print("[verify] autodex", Path(autodex.__file__).resolve())
print("[verify] pepsi_008", np.load(pepsi_008).shape)
PY
)

cat <<EOF
[setup] complete
[setup] run commands with:
  conda activate $ENV_NAME
  cd $REPO_ROOT
  python src/validation/planning/pipeline_lift_reachability/run.py ...
EOF
