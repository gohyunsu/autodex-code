#!/usr/bin/env bash
# Run only the board-9 / 20 mm pipeline. Manage daemons separately with
# manage_robot_pc_services.sh; this script never starts or stops a service.
set -Eeuo pipefail

ROOT=${PRECISION_REPO_ROOT:-/home/robot/autodex-precision-insertion}
STAGE=${PRECISION_STAGE_ROOT:-/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical}
PY=${PRECISION_PLANNER_PY:-/home/robot/anaconda3/envs/planner/bin/python}
CHECK_PY=${PRECISION_CHECK_PY:-/home/robot/anaconda3/envs/flir_env/bin/python}
MODE=${1:---run}

if [[ "$MODE" != --run && "$MODE" != --check ]]; then
  echo "usage: $0 [--run|--check]" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-/tmp/precision-insertion-timing}

LAUNCH="$STAGE/live_board9_gap20_20261013/launch.json"
CATALOG="$STAGE/recipient_gap20_endpoint_catalog_20261013.json"
ROBOT="$STAGE/franka_runtime_board9_gap20.json"

if [[ "$MODE" == --check ]]; then
  exec "$CHECK_PY" demo/precision-insertion/run_live.py --preflight \
    --launch-config "$LAUNCH" --endpoint-catalog "$CATALOG" \
    --robot-config "$ROBOT"
fi

echo "[pipeline] using existing camera, perception and Franka services"
exec "$PY" demo/precision-insertion/run_live.py \
  --launch-config "$LAUNCH" --endpoint-catalog "$CATALOG" \
  --integration precision_insertion.franka_integration:factory \
  --robot-config "$ROBOT" \
  --init-port-mask 15006 --init-port-pose 15007 --init-port-cmd 16893 \
  --xy-update-weight 1.0 --enable-robot-motion
