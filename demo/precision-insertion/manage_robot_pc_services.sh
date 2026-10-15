#!/usr/bin/env bash
# Start/check/stop the precision demo daemons independently of run_robot_pc.sh.
# Capture hosts are handled concurrently; borrowed services are never stopped.
set -Eeuo pipefail

IMAGE=franka3/real:bimanual_control_libfranka0.18.0_v2
FRANKA_IP=172.17.1.11
FRANKA_NAME=precision-insertion-franka-managed
PCS=(capture1 capture2 capture3 capture5 capture6)
ACTION=${1:-}
TIMING_FILE=${PRECISION_SERVICE_TIMING_FILE:-/tmp/precision-insertion-services-$(date +%Y%m%d_%H%M%S)-$$.tsv}

if [[ "$ACTION" != start && "$ACTION" != check && "$ACTION" != stop && "$ACTION" != restart ]]; then
  echo "usage: $0 {start|check|stop|restart}" >&2
  exit 2
fi

record_duration() {
  local label=$1 started_ns=$2 ended_ns
  ended_ns=$(date +%s%N)
  printf '%s\t%s\n' "$label" "$(((ended_ns - started_ns) / 1000000))" >> "$TIMING_FILE"
}

ssh_pc() {
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$1" "$2"
}

start_remote_if_missing() {
  local host=$1 kind=$2 pattern=$3 command=$4 port=$5 existing pidfile wrapped attempt
  pidfile="/tmp/precision-insertion-managed-$kind.pid"
  existing=$(ssh_pc "$host" "pgrep -f '$pattern' || true")
  if [[ -n "$existing" ]]; then
    [[ "$existing" =~ ^[0-9]+$ ]] || { echo "ambiguous $kind on $host" >&2; return 1; }
    echo "[services] $host $kind: existing PID $existing"
  else
    ssh_pc "$host" "! ss -lnt | grep -q ':$port '" || {
      echo "port $port is occupied on $host" >&2; return 1;
    }
    printf -v wrapped '%q' "$command"
    ssh_pc "$host" "nohup bash -lc $wrapped > /tmp/precision-insertion-managed-$kind.log 2>&1 < /dev/null & echo \$! > '$pidfile'"
    echo "[services] $host $kind: started"
  fi
  for attempt in {1..30}; do
    if ssh_pc "$host" "ss -lnt | grep -q ':$port '"; then return 0; fi
    sleep 1
  done
  echo "[services] $host $kind did not open port $port" >&2
  return 1
}

check_remote() {
  local host=$1
  ssh_pc "$host" "ss -lnt | grep -q ':5480 ' && ss -lnt | grep -q ':6894 ' && ss -lnt | grep -q ':15006 ' && ss -lnt | grep -q ':15007 ' && ss -lnt | grep -q ':16893 '"
}

stop_remote_owned() {
  local host=$1 kind=$2 needle=$3 pidfile
  pidfile="/tmp/precision-insertion-managed-$kind.pid"
  ssh_pc "$host" "
    if test -f '$pidfile'; then
      pid=\$(cat '$pidfile')
      case \"\$pid\" in *[!0-9]*|'') exit 1;; esac
      if test -r /proc/\$pid/cmdline; then
        cmd=\$(tr '\\000' ' ' < /proc/\$pid/cmdline)
        case \"\$cmd\" in *'$needle'*) kill -TERM \$pid;; *) exit 1;; esac
      fi
      rm -f '$pidfile'
    fi"
}

start_host() {
  local host=$1 started_ns home adapter
  started_ns=$(date +%s%N)
  home="/home/$host"
  start_remote_if_missing "$host" camera \
    '^[^ ]*python src/camera/server_daemon.py$' \
    "cd $home/paradex && exec env PYTHONPATH=$home/paradex $home/anaconda3/envs/flir_env/bin/python src/camera/server_daemon.py" 5480
  start_remote_if_missing "$host" snapshot '[s]napshot_daemon.py' \
    "exec $home/anaconda3/envs/gotrack_cu128/bin/python $home/AutoDex/src/execution/daemon/snapshot_daemon.py" 6894
  adapter="$home/precision_init_adapter_20261013_983ba80"
  start_remote_if_missing "$host" precision \
    '[p]recision_init_daemon.py --port-mask 15006 --port-pose 15007 --port-cmd 16893' \
    "exec env AUTODEX_STOCK_ROOT=$home/AutoDex PYTHONPATH=$adapter:$home/AutoDex $home/anaconda3/envs/gotrack_cu128/bin/python $adapter/precision_init_daemon.py --port-mask 15006 --port-pose 15007 --port-cmd 16893" 16893
  check_remote "$host"
  record_duration "capture_host_$host" "$started_ns"
}

parallel_hosts() {
  local operation=$1 host pid failed=0
  local -a pids=()
  for host in "${PCS[@]}"; do
    (
      if [[ "$operation" == start ]]; then
        start_host "$host"
      elif [[ "$operation" == check ]]; then
        check_remote "$host"
        echo "[services] $host: ready"
      else
        stop_remote_owned "$host" precision precision_init_daemon.py
        stop_remote_owned "$host" snapshot snapshot_daemon.py
        stop_remote_owned "$host" camera server_daemon.py
        echo "[services] $host: managed daemons stopped"
      fi
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

start_franka() {
  local started_ns existing attempt
  started_ns=$(date +%s%N)
  existing=$(docker ps -q --filter "ancestor=$IMAGE")
  if [[ -n "$existing" ]]; then
    [[ "$existing" =~ ^[a-f0-9]+$ ]] || { echo "multiple Franka containers" >&2; return 1; }
    echo "[services] Franka daemon: existing container $existing"
  else
    docker run -d --rm --name "$FRANKA_NAME" --network host \
      --cap-add=SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1:-1 \
      -v /home/robot/paradex:/workspace/paradex "$IMAGE" bash -lc \
      "cd /workspace/paradex/cpp/franka_daemon && if [ ! -x build/franka_daemon ]; then bash build_daemon.sh; fi && exec ./build/franka_daemon $FRANKA_IP --command_port 5555 --state_port 5556" >/dev/null
    echo "[services] Franka daemon: started $FRANKA_NAME"
  fi
  for attempt in {1..30}; do
    if ss -lnt | grep -q ':5555 ' && ss -lnt | grep -q ':5556 '; then
      record_duration franka_daemon "$started_ns"
      return 0
    fi
    sleep 1
  done
  echo "[services] Franka daemon did not open command/state ports" >&2
  return 1
}

stop_managed() {
  parallel_hosts stop
  if docker ps -q --filter "name=^/${FRANKA_NAME}$" | grep -q .; then
    docker stop --timeout 3 "$FRANKA_NAME" >/dev/null
    echo "[services] managed Franka daemon stopped"
  fi
}

printf 'stage\tduration_ms\n' > "$TIMING_FILE"
echo "[services] timing: $TIMING_FILE"
case "$ACTION" in
  start)
    parallel_hosts start
    start_franka
    ;;
  check)
    parallel_hosts check
    ss -lnt | grep -q ':5555 ' && ss -lnt | grep -q ':5556 '
    echo '[services] all endpoints ready'
    ;;
  stop)
    stop_managed
    ;;
  restart)
    stop_managed
    parallel_hosts start
    start_franka
    ;;
esac
