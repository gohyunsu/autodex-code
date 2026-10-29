#!/usr/bin/env bash
# Run the board-9 cylinder pipeline for an explicit radial-gap size. Manage daemons separately with
# manage_robot_pc_services.sh; this script never starts or stops a service.
set -Eeuo pipefail

ROOT=${PRECISION_REPO_ROOT:-/home/robot/autodex-precision-insertion}
STAGE=${PRECISION_STAGE_ROOT:-}
PY=${PRECISION_PLANNER_PY:-/home/robot/anaconda3/envs/planner/bin/python}
CHECK_PY=${PRECISION_CHECK_PY:-/home/robot/anaconda3/envs/flir_env/bin/python}
MODE=${1:---run}
GAP_MM=${PRECISION_GAP_MM:-20}
SCENE_MODE=${PRECISION_SCENE_MODE:-charuco_board}
TABLE_SURFACE_Z_M=${PRECISION_TABLE_SURFACE_Z_M:-}

case "$GAP_MM" in
  1|3|5|10|15|20) ;;
  *) echo "PRECISION_GAP_MM must be one of 1, 3, 5, 10, 15, 20" >&2; exit 2 ;;
esac

if [[ "$MODE" != --run && "$MODE" != --check && "$MODE" != --plan-only ]]; then
  echo "usage: $0 [--run|--check|--plan-only]" >&2
  exit 2
fi
if [[ "$SCENE_MODE" != charuco_board && "$SCENE_MODE" != board_free ]]; then
  echo "PRECISION_SCENE_MODE must be charuco_board or board_free" >&2; exit 2
fi
if [[ "$SCENE_MODE" == board_free && -z "$TABLE_SURFACE_Z_M" ]]; then
  echo "board_free needs explicit PRECISION_TABLE_SURFACE_Z_M (metres in robot base frame)" >&2; exit 2
fi
if [[ -z "$STAGE" ]]; then
  echo "PRECISION_STAGE_ROOT must name the active cylindrical recipient stage" >&2
  exit 2
fi
if [[ ! -d "$STAGE" ]]; then
  echo "active cylindrical recipient stage is missing: $STAGE" >&2
  exit 2
fi
STAGE=$(realpath -e "$STAGE")

cd "$ROOT"
export PYTHONPATH="${PRECISION_PARADEX_ROOT:+$PRECISION_PARADEX_ROOT:}${PRECISION_EXTRA_PYTHONPATH:+$PRECISION_EXTRA_PYTHONPATH:}$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"
export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-/tmp/precision-insertion-timing}
export PRECISION_FR3_INSPIRE_URDF=${PRECISION_FR3_INSPIRE_URDF:-$ROOT/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire_oct12_mount.urdf}
export PRECISION_VLM_CAMERA_IDS=${PRECISION_VLM_CAMERA_IDS:-24122734,25305463,25322638,25322649,25305461}
PRECISION_XY_ESTIMATOR=${PRECISION_XY_ESTIMATOR:-vlm}
XY_UPDATE_WEIGHT=${PRECISION_XY_UPDATE_WEIGHT:-1.0}
LABEL_MODE=${PRECISION_LABEL_MODE:-vlm_only}

LAUNCH_TEMPLATE=${PRECISION_LAUNCH_TEMPLATE:-$STAGE/launch.json}
CATALOG=${PRECISION_ENDPOINT_CATALOG:-$STAGE/recipient_gap${GAP_MM}_endpoint_catalog.json}
ROBOT=${PRECISION_ROBOT_CONFIG:-$STAGE/franka_runtime.json}
for required in "$LAUNCH_TEMPLATE" "$CATALOG" "$ROBOT"; do
  if [[ ! -f "$required" ]]; then
    echo "active recipient-stage input is missing: $required" >&2
    exit 2
  fi
done

if [[ "$MODE" == --check ]]; then
  CHECK_LAUNCH=$(mktemp /tmp/precision-launch-check.XXXXXX.json)
  trap 'rm -f "$CHECK_LAUNCH"' EXIT
  "$CHECK_PY" - "$LAUNCH_TEMPLATE" "$CHECK_LAUNCH" "$GAP_MM" "$SCENE_MODE" "$TABLE_SURFACE_Z_M" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
record["mode"] = {"family": "cylinder", "gap_mm": int(sys.argv[3])}
scene_mode, table_z = sys.argv[4:6]
record["scene_mode"] = scene_mode
record["socket_perception_mode"] = ("ignore_sil_loss" if scene_mode == "board_free"
                                    else "board_constrained_body")
if scene_mode == "board_free":
    record["table_surface_z_m"] = float(table_z)
    for key in ("board_id", "minimum_board_corners", "board_capture_phase"):
        record.pop(key, None)
Path(sys.argv[2]).write_text(json.dumps(record), encoding="utf-8")
PY
  "$CHECK_PY" demo/precision-insertion/run_live.py --preflight \
    --launch-config "$CHECK_LAUNCH" --endpoint-catalog "$CATALOG" \
    --robot-config "$ROBOT" --xy-estimator "$PRECISION_XY_ESTIMATOR"
  exit
fi

RUN_TAG=${PRECISION_RUN_TAG:-"$(date +%Y%m%d_%H%M%S_%N)_$$"}
LAUNCH_DIR="$STAGE/runs/live_${SCENE_MODE}_gap${GAP_MM}"
mkdir -p "$LAUNCH_DIR"
LAUNCH="$LAUNCH_DIR/launch_$RUN_TAG.json"
"$CHECK_PY" - "$LAUNCH_TEMPLATE" "$LAUNCH" "$RUN_TAG" "$GAP_MM" \
  "${PRECISION_REUSE_SESSION_EVIDENCE:-}" "$LAUNCH_DIR" "$SCENE_MODE" "$TABLE_SURFACE_Z_M" <<'PY'
import json
from pathlib import Path
import sys

template, output, run_tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
gap_mm, reused_evidence, launch_dir = int(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
scene_mode, table_z = sys.argv[7:9]
record = json.loads(template.read_text(encoding="utf-8"))
record["mode"] = {"family": "cylinder", "gap_mm": gap_mm}
record["scene_mode"] = scene_mode
record["socket_perception_mode"] = ("ignore_sil_loss" if scene_mode == "board_free"
                                    else "board_constrained_body")
if scene_mode == "board_free":
    record["table_surface_z_m"] = float(table_z)
    for key in ("board_id", "minimum_board_corners", "board_capture_phase"):
        record.pop(key, None)
fresh_evidence = launch_dir / f"session_evidence_{run_tag}"
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
echo "[pipeline] VLM camera IDs: $PRECISION_VLM_CAMERA_IDS"
echo "[pipeline] failed-trial XY estimator: $PRECISION_XY_ESTIMATOR"
echo "[pipeline] next-trial XY update weight: $XY_UPDATE_WEIGHT"
if [[ "$MODE" == --run && "$PRECISION_XY_ESTIMATOR" == tracked_tip ]]; then
  bash demo/precision-insertion/manage_robot_pc_services.sh check-tracker
fi
EXTRA_ARGS=()
EXTRA_ARGS+=(--label-mode "$LABEL_MODE")
if [[ -n ${PRECISION_MAX_STEPS:-} ]]; then
  EXTRA_ARGS+=(--max-steps "$PRECISION_MAX_STEPS")
fi
if [[ -n ${PRECISION_SAME_TRIAL_XY_TOLERANCE_MM:-} ]]; then
  EXTRA_ARGS+=(--same-trial-xy-tolerance-mm "$PRECISION_SAME_TRIAL_XY_TOLERANCE_MM")
fi
if [[ "$LABEL_MODE" == manual ]]; then
  EXTRA_ARGS+=(--annotator "${PRECISION_ANNOTATOR:-Hyunsu Go}")
fi
if [[ "$MODE" == --plan-only ]]; then
  EXTRA_ARGS+=(--stop-after-planning)
fi
if [[ -n ${PRECISION_REUSE_SESSION_EVIDENCE:-} ]]; then
  EXTRA_ARGS+=(--reuse-session-evidence "$PRECISION_REUSE_SESSION_EVIDENCE")
fi
if [[ ${PRECISION_KEY_ALREADY_PLACED:-0} == 1 ]]; then
  EXTRA_ARGS+=(--key-already-placed)
fi
if [[ -n ${PRECISION_INITIAL_CANDIDATE:-} ]]; then
  EXTRA_ARGS+=(--initial-candidate "$PRECISION_INITIAL_CANDIDATE")
fi
if [[ -n ${PRECISION_EXCLUDED_CANDIDATES:-} ]]; then
  IFS=, read -ra excluded_candidates <<< "$PRECISION_EXCLUDED_CANDIDATES"
  for excluded_candidate in "${excluded_candidates[@]}"; do
    EXTRA_ARGS+=(--exclude-candidate "$excluded_candidate")
  done
fi
exec "$PY" demo/precision-insertion/run_live.py \
  --launch-config "$LAUNCH" --endpoint-catalog "$CATALOG" \
  --integration "${PRECISION_INTEGRATION_FACTORY:-precision_insertion.franka_integration:factory}" \
  --robot-config "$ROBOT" \
  --init-port-mask 15006 --init-port-pose 15007 --init-port-cmd 16893 \
  --xy-update-weight "$XY_UPDATE_WEIGHT" --xy-estimator "$PRECISION_XY_ESTIMATOR" \
  --enable-robot-motion "${EXTRA_ARGS[@]}"
