#!/usr/bin/env bash
# Run the separate ZeroDex capture4 profile with the existing cylinder pipeline.
# Capture4 owns camera acquisition and SAM3/FoundPose; this PC owns Qwen/cuRobo.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# Apply local overrides first; the versioned profile fills missing values and
# derives CUDA include paths from the selected toolkit.
if [[ -n ${PRECISION_SYSTEM_PROFILE:-} ]]; then
  source "$PRECISION_SYSTEM_PROFILE"
fi
source "$SCRIPT_DIR/config/zerodex_capture4.env.sh"
export PRECISION_GAP_MM=${PRECISION_GAP_MM:-5}
export PRECISION_XY_ESTIMATOR=${PRECISION_XY_ESTIMATOR:-sam_half_ellipse}
export PRECISION_TABLETOP_POSE_ERROR_DEG=${PRECISION_TABLETOP_POSE_ERROR_DEG:-20}
export PRECISION_PRELOAD_VLM=${PRECISION_PRELOAD_VLM:-1}
# Observe whether the grasp remains held briefly after the lift completes.
export PRECISION_POST_LIFT_SETTLE_S=${PRECISION_POST_LIFT_SETTLE_S:-2.0}
# The first live pickup previously spent 21.97 s initializing the fixed
# cuRobo world. Start that work while capture4 acquires/refines the next key.
# A caller can set 0 for a measured serial-latency comparison.
export PRECISION_PREWARM_FIXED_WORLD=${PRECISION_PREWARM_FIXED_WORLD:-1}
# Previously selected upright grasps are only ordering hints. Every new
# tabletop observation still runs the normal v8 pickup and insertion checks;
# an infeasible preferred grasp falls through to the remaining candidates.
# An explicitly empty override disables the hint.
if [[ -z ${PRECISION_INITIAL_CANDIDATE+x} ]]; then
  case $PRECISION_GAP_MM in
    # This grasp reached the released, human-confirmed inside-socket state in
    # two recorded 5 mm episodes. The former hint 208 has only offline plans.
    5) export PRECISION_INITIAL_CANDIDATE=table/0/20195 ;;
    10) export PRECISION_INITIAL_CANDIDATE=table/0/20267 ;;
  esac
fi
if [[ $PRECISION_GAP_MM == 5 && -z ${PRECISION_EXCLUDED_CANDIDATES+x} ]]; then
  # In two corrected-hold replays 20096 intersected the socket at 20 mm;
  # three released trials with this grasp left the key outside the socket.
  export PRECISION_EXCLUDED_CANDIDATES=table/0/20096
fi
export PRECISION_ENDPOINT_CATALOG=${PRECISION_ENDPOINT_CATALOG:-$PRECISION_STAGE_ROOT/recipient_gap${PRECISION_GAP_MM}_endpoint_catalog_straight.json}

if [[ ${1:-} == --camera-only ]]; then
  shift
  export PYTHONPATH="$PRECISION_PARADEX_ROOT:$PRECISION_EXTRA_PYTHONPATH:$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
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
