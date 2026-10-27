#!/usr/bin/env bash
# Run the separate ZeroDex capture4 profile with the existing cylinder pipeline.
# Capture4 owns camera acquisition and SAM3/FoundPose; this PC owns Qwen/cuRobo.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PRECISION_REPO_ROOT=${PRECISION_REPO_ROOT:-$ROOT}
export PRECISION_STAGE_ROOT=${PRECISION_STAGE_ROOT:-/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4}
export PRECISION_PLANNER_PY=${PRECISION_PLANNER_PY:-/home/hyunsu/.venvs/precision-vlm/bin/python}
export PRECISION_CHECK_PY=${PRECISION_CHECK_PY:-$PRECISION_PLANNER_PY}
export PRECISION_VLM_CAMERA_IDS=${PRECISION_VLM_CAMERA_IDS:-26053248,25322639,25305462}
export PRECISION_XY_ESTIMATOR=${PRECISION_XY_ESTIMATOR:-vlm}
# ZeroDex uses the original straight flange-to-Inspire mount, not the
# October 12 right-angle mount selected by run_robot_pc.sh for AutoDex.
export PRECISION_FR3_INSPIRE_URDF=${PRECISION_FR3_INSPIRE_URDF:-$ROOT/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf}
export PRECISION_ENDPOINT_CATALOG=${PRECISION_ENDPOINT_CATALOG:-$PRECISION_STAGE_ROOT/recipient_gap${PRECISION_GAP_MM:-20}_endpoint_catalog_straight.json}

# The main PC performs the current AutoDex silhouette refinement on its own
# RTX 3090. nvdiffrast compiles its CUDA extension on first use and needs the
# CUDA 12.1 headers bundled with this PC's existing PyTorch environment.
export CUDA_HOME=${CUDA_HOME:-/home/hyunsu/miniconda3/envs/autodex_bodex}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-4}
PRECISION_CUDA_HEADERS="$CUDA_HOME/lib/python3.10/site-packages/nvidia"
for precision_header_dir in cuda_runtime cusparse cublas cusolver; do
  precision_include="$PRECISION_CUDA_HEADERS/$precision_header_dir/include"
  if [[ -d "$precision_include" ]]; then
    export CPATH="$precision_include${CPATH:+:$CPATH}"
  fi
done

exec bash "$SCRIPT_DIR/run_robot_pc.sh" "${1:---check}"
