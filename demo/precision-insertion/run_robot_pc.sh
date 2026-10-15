#!/usr/bin/env bash
# Run only the board-9 / 20 mm pipeline. Manage daemons separately with
# manage_robot_pc_services.sh; this script never starts or stops a service.
set -Eeuo pipefail

ROOT=${PRECISION_REPO_ROOT:-/home/robot/autodex-precision-insertion}
STAGE=${PRECISION_STAGE_ROOT:-/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical}
PY=${PRECISION_PLANNER_PY:-/home/robot/anaconda3/envs/planner/bin/python}
CHECK_PY=${PRECISION_CHECK_PY:-/home/robot/anaconda3/envs/flir_env/bin/python}
MODE=${1:---run}

if [[ "$MODE" != --run && "$MODE" != --check && "$MODE" != --plan-only ]]; then
  echo "usage: $0 [--run|--check|--plan-only]" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-/tmp/precision-insertion-timing}
export PRECISION_FR3_INSPIRE_URDF=${PRECISION_FR3_INSPIRE_URDF:-$ROOT/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire_oct12_mount.urdf}

LAUNCH_TEMPLATE="$STAGE/live_board9_gap20_20261013/launch.json"
CATALOG="$STAGE/recipient_gap20_endpoint_catalog_20261013.json"
ROBOT="$STAGE/franka_runtime_board9_gap20.json"

if [[ "$MODE" == --check ]]; then
  exec "$CHECK_PY" demo/precision-insertion/run_live.py --preflight \
    --launch-config "$LAUNCH_TEMPLATE" --endpoint-catalog "$CATALOG" \
    --robot-config "$ROBOT"
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S_%N)_$$"
LAUNCH="$STAGE/live_board9_gap20_20261013/launch_$RUN_TAG.json"
"$CHECK_PY" - "$LAUNCH_TEMPLATE" "$LAUNCH" "$RUN_TAG" \
  "${PRECISION_REUSE_SESSION_EVIDENCE:-}" <<'PY'
import json
from pathlib import Path
import sys

template, output, run_tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
reused_evidence = sys.argv[4]
record = json.loads(template.read_text(encoding="utf-8"))
original_evidence = Path(record["evidence_dir"])
fresh_evidence = original_evidence.with_name(f"session_evidence_{run_tag}")
if fresh_evidence.exists():
    raise FileExistsError(f"session evidence already exists: {fresh_evidence}")
if not fresh_evidence.parent.is_dir():
    raise FileNotFoundError(f"session evidence parent is missing: {fresh_evidence.parent}")
record["evidence_dir"] = str(fresh_evidence)
with output.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(f"[pipeline] launch config: {output}")
if reused_evidence:
    print(f"[pipeline] saved socket evidence: {reused_evidence}")
    print(f"[pipeline] launch evidence path unused: {fresh_evidence}")
else:
    print(f"[pipeline] fresh evidence: {fresh_evidence}")
PY

echo "[pipeline] using existing camera, perception and Franka services"
EXTRA_ARGS=()
if [[ "$MODE" == --plan-only ]]; then
  EXTRA_ARGS+=(--stop-after-planning)
fi
if [[ -n ${PRECISION_REUSE_SESSION_EVIDENCE:-} ]]; then
  EXTRA_ARGS+=(--reuse-session-evidence "$PRECISION_REUSE_SESSION_EVIDENCE")
fi
if [[ ${PRECISION_KEY_ALREADY_PLACED:-0} == 1 ]]; then
  EXTRA_ARGS+=(--key-already-placed)
fi
exec "$PY" demo/precision-insertion/run_live.py \
  --launch-config "$LAUNCH" --endpoint-catalog "$CATALOG" \
  --integration precision_insertion.franka_integration:factory \
  --robot-config "$ROBOT" \
  --init-port-mask 15006 --init-port-pose 15007 --init-port-cmd 16893 \
  --xy-update-weight 1.0 --enable-robot-motion "${EXTRA_ARGS[@]}"
