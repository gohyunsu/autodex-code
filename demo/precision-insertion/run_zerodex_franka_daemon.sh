#!/usr/bin/env bash
# Start the existing ParaDex Franka daemon for the ZeroDex FR3.
#
# Run this script with sudo from a dedicated terminal. It only resolves the
# versioned ZeroDex profile and execs the restored daemon; camera services and
# pipeline processes are not started here.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/config/zerodex_capture4.env.sh"

ROBOT_IP=${PRECISION_ZERODEX_FR3_FCI_IP:-172.16.1.11}
COMMAND_PORT=${PRECISION_FRANKA_COMMAND_PORT:-5555}
STATE_PORT=${PRECISION_FRANKA_STATE_PORT:-5556}
DEPS="$PRECISION_STAGE_ROOT/robot_deps"
DAEMON=${PRECISION_FRANKA_DAEMON_BINARY:-$DEPS/restored/home/robot/paradex/cpp/franka_daemon/build/franka_daemon}
LIBRARY_PATH="$DEPS/prefix/usr/lib/x86_64-linux-gnu:$DEPS/prefix/usr/lib"

if [[ ${1:-} == --print ]]; then
  printf 'env LD_LIBRARY_PATH=%q %q %q --command_port %q --state_port %q\n' \
    "$LIBRARY_PATH" "$DAEMON" "$ROBOT_IP" "$COMMAND_PORT" "$STATE_PORT"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "usage: sudo bash $0 [--print]" >&2
  exit 2
fi

exec env LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$DAEMON" "$ROBOT_IP" \
  --command_port "$COMMAND_PORT" --state_port "$STATE_PORT"
