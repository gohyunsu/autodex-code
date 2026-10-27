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
export PRECISION_CAMERA_ACQUISITION_MODE=stream

exec bash "$SCRIPT_DIR/run_robot_pc.sh" "${1:---check}"
