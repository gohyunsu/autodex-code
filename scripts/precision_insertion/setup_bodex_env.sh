#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_root="${AUTODEX_CONDA_ROOT:-${HOME}/miniconda3}"
env_name="${AUTODEX_BODEX_ENV:-autodex_bodex}"
env_prefix="${conda_root}/envs/${env_name}"
paradex_root="${PARADEX_ROOT:-${HOME}/paradex}"

if [[ ! -x "${conda_root}/bin/conda" ]]; then
  echo "Missing conda: ${conda_root}/bin/conda" >&2
  echo "Install Miniconda first or set AUTODEX_CONDA_ROOT." >&2
  exit 2
fi
if [[ ! -f "${paradex_root}/setup.py" ]]; then
  echo "Missing ParaDex checkout: ${paradex_root}" >&2
  echo "Clone https://github.com/snuvclab/paradex.git or set PARADEX_ROOT." >&2
  exit 2
fi

conda_bin="${conda_root}/bin/conda"
if [[ ! -x "${env_prefix}/bin/python" ]]; then
  "${conda_bin}" create --yes --override-channels -c conda-forge \
    -n "${env_name}" python=3.10 pip setuptools wheel
fi

"${conda_bin}" install --yes --override-channels -c conda-forge -n "${env_name}" \
  numpy=1.26.4 qhull=2020.2 eigen octomap=1.10 assimp=5.4 boost=1.84 coal=3.0
"${conda_bin}" install --yes --override-channels \
  -c nvidia/label/cuda-12.1.1 -c conda-forge -n "${env_name}" \
  cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-nvrtc-dev=12.1

pip_bin="${env_prefix}/bin/pip"
"${pip_bin}" install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1
"${pip_bin}" install -r "${repo_root}/scripts/requirements-planner-worker.txt"
"${pip_bin}" install mujoco==3.3.7
"${pip_bin}" install pytest==9.1.1
"${pip_bin}" install ultralytics==8.4.15 --no-deps
"${pip_bin}" install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
"${pip_bin}" install \
  chime==0.7.0 \
  msgpack==1.1.1 \
  opencv-contrib-python-headless==4.10.0.84 \
  pymodbus==2.5.3 \
  pyserial==3.5 \
  pyzmq==27.1.0

export CONDA_PREFIX="${env_prefix}"
export CUDA_HOME="${env_prefix}"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS="${MAX_JOBS:-4}"

"${pip_bin}" install --no-build-isolation -e \
  "${repo_root}/src/grasp_generation/BODex/src/curobo/geom/cpp"
"${pip_bin}" install --no-build-isolation -e \
  "${repo_root}/src/grasp_generation/BODex"
"${pip_bin}" install -e "${repo_root}"
"${pip_bin}" install -e "${paradex_root}"

"${env_prefix}/bin/python" \
  "${repo_root}/scripts/precision_insertion/verify_bodex_env.py"

echo "BODex environment ready: ${env_prefix}"
echo "Activate with: ${conda_root}/bin/conda activate ${env_name}"
