#!/usr/bin/env bash
# One-command board-9 / 20 mm FR3 run with session-owned daemon cleanup.
set -Eeuo pipefail

ROOT=/home/robot/autodex-precision-insertion
STAGE=/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical
PY=/home/robot/anaconda3/envs/planner/bin/python
CHECK_PY=/home/robot/anaconda3/envs/flir_env/bin/python
IMAGE=franka3/real:bimanual_control_libfranka0.18.0_v2
FRANKA_IP=172.17.1.11
PCS=(capture1 capture2 capture3 capture5 capture6)
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
RUN_START_NS="$(date +%s%N)"
SERVICE_TIMING_FILE="/tmp/precision-insertion-$RUN_TAG-service-timing.tsv"
FRANKA_NAME="precision-insertion-franka-$RUN_TAG"
FRANKA_OWNED=0
MODE="${1:---run}"

if [[ "$MODE" != --run && "$MODE" != --check && "$MODE" != --services-only ]]; then
  echo "usage: $0 [--run|--check|--services-only]" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/demo/precision-insertion:$ROOT/src"

# Shell-side service startup is outside run_live.py's Python timeline.
# Logging must never determine whether a robot service may run.
record_duration() {
  local label="$1" started_ns="$2" status="${3:-ok}" ended_ns
  ended_ns="$(date +%s%N)"
  printf '%s\t%s\t%s\n' "$label" "$status" \
    "$(((ended_ns - started_ns) / 1000000))" >> "$SERVICE_TIMING_FILE" ||
    echo "[timing] could not append $SERVICE_TIMING_FILE" >&2
}
printf 'stage\tstatus\tduration_ms\n' > "$SERVICE_TIMING_FILE" || true
echo "[timing] service profile: $SERVICE_TIMING_FILE"

ssh_pc() {
  local host="$1" command="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "$command"
}

# Each pid file is unique to this invocation. Borrowed services have no file
# and are never stopped by this run. Check the command line before each kill.
stop_remote_owned() {
  local host="$1" kind="$2" needle="$3"
  local pidfile="/tmp/precision-insertion-$RUN_TAG-$kind.pid"
  ssh_pc "$host" "
    if test -f '$pidfile'; then
      pid=\$(cat '$pidfile')
      case \"\$pid\" in *[!0-9]*|'') exit 1;; esac
      if test -r /proc/\$pid/cmdline; then
        cmd=\$(tr '\\000' ' ' < /proc/\$pid/cmdline)
        case \"\$cmd\" in *'$needle'*) kill -TERM \$pid;; esac
      fi
      rm -f '$pidfile'
    fi" >/dev/null 2>&1 || echo "[services] cleanup failed: $host $kind" >&2
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${app_started_ns:-}" && "${app_timing_recorded:-0}" == 0 ]]; then
    record_duration python_live_session "$app_started_ns" "$status"
  fi
  record_duration launcher_before_cleanup "$RUN_START_NS" "$status"
  if [[ "$MODE" == --check ]]; then
    exit "$status"
  fi
  if [[ "$FRANKA_OWNED" == 1 ]]; then
    docker stop --timeout 3 "$FRANKA_NAME" >/dev/null 2>&1 ||
      docker kill "$FRANKA_NAME" >/dev/null 2>&1 || true
  fi
  local host
  for host in "${PCS[@]}"; do
    stop_remote_owned "$host" precision precision_init_daemon.py
    stop_remote_owned "$host" snapshot snapshot_daemon.py
    stop_remote_owned "$host" camera server_daemon.py
  done
  record_duration launcher_total "$RUN_START_NS" "$status"
  echo "[services] session-owned daemons released (exit $status)"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$MODE" == --check ]]; then
  echo "[services] file preflight"
  check_started_ns="$(date +%s%N)"
  "$CHECK_PY" demo/precision-insertion/run_live.py --preflight \
    --launch-config "$STAGE/live_board9_gap20_20261013/launch.json" \
    --endpoint-catalog "$STAGE/recipient_gap20_endpoint_catalog_20261013.json" \
    --robot-config "$STAGE/franka_runtime_board9_gap20.json" >/dev/null
  record_duration file_preflight "$check_started_ns"
  echo "[services] file preflight passed; no daemons started"
  exit 0
fi

start_remote_if_missing() {
  local host="$1" kind="$2" pattern="$3" command="$4" port="$5"
  local existing pidfile="/tmp/precision-insertion-$RUN_TAG-$kind.pid"
  existing="$(ssh_pc "$host" "pgrep -f '$pattern' || true")"
  if [[ -n "$existing" ]]; then
    if [[ ! "$existing" =~ ^[0-9]+$ ]]; then
      echo "[services] ambiguous $kind service on $host: $existing" >&2
      return 1
    fi
    echo "[services] $host $kind: using existing PID $existing"
  else
    if ssh_pc "$host" "ss -lnt | grep -q ':$port '"; then
      echo "[services] $host port $port is occupied by another process" >&2
      return 1
    fi
    local wrapped
    printf -v wrapped '%q' "$command"
    ssh_pc "$host" "nohup bash -lc $wrapped > /tmp/precision-insertion-$RUN_TAG-$kind.log 2>&1 < /dev/null & echo \$! > '$pidfile'"
    echo "[services] $host $kind: started"
  fi
  local attempt
  for attempt in {1..30}; do
    if ssh_pc "$host" "ss -lnt | grep -q ':$port '"; then
      return 0
    fi
    sleep 1
  done
  echo "[services] $host $kind did not open port $port" >&2
  return 1
}

for host in "${PCS[@]}"; do
  host_started_ns="$(date +%s%N)"
  home="/home/$host"
  start_remote_if_missing "$host" camera \
    '^[^ ]*python src/camera/server_daemon.py$' \
    "cd $home/paradex && exec env PYTHONPATH=$home/paradex $home/anaconda3/envs/flir_env/bin/python src/camera/server_daemon.py" \
    5480
  start_remote_if_missing "$host" snapshot \
    '[s]napshot_daemon.py' \
    "exec $home/anaconda3/envs/gotrack_cu128/bin/python $home/AutoDex/src/execution/daemon/snapshot_daemon.py" \
    6894
  adapter="$home/precision_init_adapter_20261013_983ba80"
  local_hash="$(sha256sum "$ROOT/demo/precision-insertion/precision_init_daemon.py" | cut -d' ' -f1)"
  remote_hash="$(ssh_pc "$host" "sha256sum '$adapter/precision_init_daemon.py' | cut -d' ' -f1")"
  if [[ "$remote_hash" != "$local_hash" ]]; then
    echo "[services] $host precision adapter differs from this checkout" >&2
    exit 1
  fi
  start_remote_if_missing "$host" precision \
    '[p]recision_init_daemon.py --port-mask 15006 --port-pose 15007 --port-cmd 16893' \
    "exec env AUTODEX_STOCK_ROOT=$home/AutoDex PYTHONPATH=$adapter:$home/AutoDex $home/anaconda3/envs/gotrack_cu128/bin/python $adapter/precision_init_daemon.py --port-mask 15006 --port-pose 15007 --port-cmd 16893" \
    16893
  ssh_pc "$host" "ss -lnt | grep -q ':15006 ' && ss -lnt | grep -q ':15007 '" || {
    echo "[services] $host precision mask/pose ports are not listening" >&2
    exit 1
  }
  record_duration "capture_host_$host" "$host_started_ns"
done

franka_started_ns="$(date +%s%N)"
existing_franka="$(docker ps -q --filter "ancestor=$IMAGE")"
if [[ -n "$existing_franka" ]]; then
  if [[ ! "$existing_franka" =~ ^[a-f0-9]+$ ]]; then
    echo "[services] multiple FR3 daemon containers; refusing to choose" >&2
    exit 1
  fi
  echo "[services] FR3 daemon: using existing container $existing_franka"
else
  FRANKA_OWNED=1
  docker run -d --rm --name "$FRANKA_NAME" --network host \
    --cap-add=SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1:-1 \
    -v /home/robot/paradex:/workspace/paradex "$IMAGE" bash -lc \
    "cd /workspace/paradex/cpp/franka_daemon && if [ ! -x build/franka_daemon ]; then bash build_daemon.sh; fi && exec ./build/franka_daemon $FRANKA_IP --command_port 5555 --state_port 5556" >/dev/null
  echo "[services] FR3 daemon: started $FRANKA_NAME"
fi

for attempt in {1..30}; do
  if [[ "$FRANKA_OWNED" == 1 ]] &&
     [[ "$(docker inspect -f '{{.State.Running}}' "$FRANKA_NAME" 2>/dev/null || true)" != true ]]; then
    echo "[services] FR3 daemon container exited during startup" >&2
    exit 1
  fi
  if ss -lnt | grep -q ':5555 ' && ss -lnt | grep -q ':5556 '; then
    break
  fi
  if [[ "$attempt" == 30 ]]; then
    echo "[services] FR3 daemon did not open command/state ports" >&2
    exit 1
  fi
  sleep 1
done
record_duration franka_daemon "$franka_started_ns"

if [[ "$MODE" == --services-only ]]; then
  echo "[services] all required daemons are available"
  exit 0
fi

echo "[services] running board-9 / 20 mm precision insertion"
app_started_ns="$(date +%s%N)"
"$PY" demo/precision-insertion/run_live.py \
  --launch-config "$STAGE/live_board9_gap20_20261013/launch.json" \
  --endpoint-catalog "$STAGE/recipient_gap20_endpoint_catalog_20261013.json" \
  --integration precision_insertion.franka_integration:factory \
  --robot-config "$STAGE/franka_runtime_board9_gap20.json" \
  --init-port-mask 15006 --init-port-pose 15007 --init-port-cmd 16893 \
  --xy-update-weight 1.0 --enable-robot-motion
record_duration python_live_session "$app_started_ns"
app_timing_recorded=1
