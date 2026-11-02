#!/usr/bin/env bash
# Run the separate ZeroDex capture4 profile with the existing cylinder pipeline.
# Capture4 owns acquisition and object-perception SAM3/FoundPose. This PC owns
# Qwen, the pre-insertion SAM3 ROI worker, and cuRobo.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# Apply local overrides first; the versioned profile fills missing values and
# derives CUDA include paths from the selected toolkit.
if [[ -n ${PRECISION_SYSTEM_PROFILE:-} ]]; then
  source "$PRECISION_SYSTEM_PROFILE"
fi
source "$SCRIPT_DIR/config/zerodex_capture4.env.sh"
source "$SCRIPT_DIR/config/nas_namespace.sh"
if [[ -n ${PRECISION_EXPERIMENT_PROFILE:-} ]]; then
  precision_experiment_profile=$PRECISION_EXPERIMENT_PROFILE
  if [[ $precision_experiment_profile != /* ]]; then
    precision_experiment_profile="$SCRIPT_DIR/config/experiments/$precision_experiment_profile"
  fi
  source "$precision_experiment_profile"
  unset precision_experiment_profile
fi
for precision_path_var in \
  PRECISION_NAS_PROJECT_ROOT PRECISION_SHARED_ROOT PRECISION_STAGE_ROOT \
  PRECISION_RUN_OUTPUT_ROOT PRECISION_RECORDING_RECEIPT_ROOT \
  PRECISION_DEV_ROOT PRECISION_SAM_CHECKPOINT; do
  precision_require_allowed_nas_path \
    "$precision_path_var" "${!precision_path_var:-}"
done
unset precision_path_var
export PRECISION_GAP_MM=${PRECISION_GAP_MM:-5}
export PRECISION_XY_ESTIMATOR=${PRECISION_XY_ESTIMATOR:-sam_half_ellipse}
export PRECISION_PRELOAD_VLM=${PRECISION_PRELOAD_VLM:-1}
# The first live pickup previously spent 21.97 s initializing the fixed
# cuRobo world. Start that work while capture4 acquires/refines the next key.
# A caller can set 0 for a measured serial-latency comparison.
export PRECISION_PREWARM_FIXED_WORLD=${PRECISION_PREWARM_FIXED_WORLD:-1}
if [[ -z ${PRECISION_CANDIDATE_POLICY_PROFILE+x} && $PRECISION_GAP_MM == 5 ]]; then
  export PRECISION_CANDIDATE_POLICY_PROFILE="$SCRIPT_DIR/config/candidate_policies/cylinder_gap5.env.sh"
fi
if [[ -n ${PRECISION_CANDIDATE_POLICY_PROFILE:-} ]]; then
  precision_candidate_policy=$PRECISION_CANDIDATE_POLICY_PROFILE
  if [[ $precision_candidate_policy != /* ]]; then
    precision_candidate_policy="$SCRIPT_DIR/config/candidate_policies/$precision_candidate_policy"
  fi
  source "$precision_candidate_policy"
  unset precision_candidate_policy
fi
export PRECISION_ENDPOINT_CATALOG=${PRECISION_ENDPOINT_CATALOG:-$PRECISION_STAGE_ROOT/recipient_gap${PRECISION_GAP_MM}_endpoint_catalog_straight.json}

if [[ ${1:-} == --camera-only ]]; then
  shift
  export PYTHONPATH="$PRECISION_CUROBO_PYTHONPATH:$PRECISION_PARADEX_ROOT:$PRECISION_EXTRA_PYTHONPATH:$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
  mkdir -p "$PRECISION_DEV_ROOT/camera_only" "$PRECISION_TIMING_LOCAL_ROOT"
  camera_only_args=(
    --launch-config "${PRECISION_LAUNCH_TEMPLATE:-$PRECISION_STAGE_ROOT/launch.json}"
    --endpoint-catalog "$PRECISION_ENDPOINT_CATALOG"
    --gap-mm "$PRECISION_GAP_MM"
  )
  camera_only_has_output=0
  for camera_only_arg in "$@"; do
    if [[ "$camera_only_arg" == --output-dir || "$camera_only_arg" == --output-dir=* ]]; then
      camera_only_has_output=1
    fi
  done
  if [[ $camera_only_has_output -eq 0 ]]; then
    run_tag="$(date +%Y%m%d_%H%M%S_%N)_$$"
    camera_only_args+=(--output-dir "$PRECISION_DEV_ROOT/camera_only/$run_tag")
  fi
  exec "$PRECISION_PLANNER_PY" "$SCRIPT_DIR/run_zerodex_camera_only.py" \
    "${camera_only_args[@]}" "$@"
fi

if [[ ${1:-} == --run || ${1:-} == --plan-only ]]; then
  # The profile-specific factory verifies that a local ParaDex daemon targets
  # the ZeroDex FR3 before the AutoDex executor can send any setup commands.
  # --plan-only still moves the robot to clear view and observes the key.
  export PRECISION_INTEGRATION_FACTORY=precision_insertion.zerodex_franka_integration:factory
  # Keep the full-resolution selected view on capture4's local SSD while the
  # pipeline runs. NAS transfer is a separate post-run command.
  exec "$PRECISION_PLANNER_PY" "$SCRIPT_DIR/run_zerodex_recorded.py" "$1"
fi

exec bash "$SCRIPT_DIR/run_robot_pc.sh" "${1:---check}"
