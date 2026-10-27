#!/usr/bin/env bash
# Run the separate ZeroDex capture4 profile with the existing cylinder pipeline.
# Capture4 owns camera acquisition and SAM3/FoundPose; this PC owns Qwen/cuRobo.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PRECISION_REPO_ROOT=${PRECISION_REPO_ROOT:-$ROOT}
export PRECISION_EXTRA_PYTHONPATH=${PRECISION_EXTRA_PYTHONPATH:-$ROOT/demo/precision-insertion/compat}
export PRECISION_STAGE_ROOT=${PRECISION_STAGE_ROOT:-/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4}
export PRECISION_PLANNER_PY=${PRECISION_PLANNER_PY:-/home/hyunsu/.venvs/precision-vlm/bin/python}
export PRECISION_CHECK_PY=${PRECISION_CHECK_PY:-$PRECISION_PLANNER_PY}
export PRECISION_VLM_CAMERA_IDS=${PRECISION_VLM_CAMERA_IDS:-26053248,25322639,25305462}
export PRECISION_VISIBLE_GRIP_CAMERA_IDS=${PRECISION_VISIBLE_GRIP_CAMERA_IDS:-$PRECISION_VLM_CAMERA_IDS}
export PRECISION_XY_ESTIMATOR=${PRECISION_XY_ESTIMATOR:-vlm}
export PRECISION_PRELOAD_VLM=${PRECISION_PRELOAD_VLM:-1}
if [[ -z ${PRECISION_PREWARM_FIXED_WORLD+x} ]]; then
  # With the key already in place there is no operator wait to hide warmup;
  # observe it first rather than spending GPU time on a potentially bad image.
  if [[ ${PRECISION_KEY_ALREADY_PLACED:-0} == 1 ]]; then
    export PRECISION_PREWARM_FIXED_WORLD=0
  else
    export PRECISION_PREWARM_FIXED_WORLD=1
  fi
fi
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

if [[ ${1:-} == --camera-only ]]; then
  shift
  export PYTHONPATH="$PRECISION_EXTRA_PYTHONPATH:$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
  export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-/tmp/precision-insertion-timing}
  camera_only_args=(
    --launch-config "${PRECISION_LAUNCH_TEMPLATE:-$PRECISION_STAGE_ROOT/launch.json}"
    --endpoint-catalog "$PRECISION_ENDPOINT_CATALOG"
  )
  camera_only_has_output=0
  for camera_only_arg in "$@"; do
    if [[ "$camera_only_arg" == --output-dir || "$camera_only_arg" == --output-dir=* ]]; then
      camera_only_has_output=1
    fi
  done
  if [[ $camera_only_has_output -eq 0 ]]; then
    run_tag="$(date +%Y%m%d_%H%M%S_%N)_$$"
    camera_only_args+=(--output-dir "$PRECISION_STAGE_ROOT/runs/camera_only_$run_tag")
  fi
  exec "$PRECISION_PLANNER_PY" "$SCRIPT_DIR/run_zerodex_camera_only.py" \
    "${camera_only_args[@]}" "$@"
fi

if [[ ${1:-} == --run || ${1:-} == --plan-only ]]; then
  # The profile-specific factory verifies that a local ParaDex daemon targets
  # the ZeroDex FR3 before the AutoDex executor can send any setup commands.
  # --plan-only still moves the robot to clear view and observes the key.
  export PRECISION_INTEGRATION_FACTORY=precision_insertion.zerodex_franka_integration:factory
fi

exec bash "$SCRIPT_DIR/run_robot_pc.sh" "${1:---check}"
