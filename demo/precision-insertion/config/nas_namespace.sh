#!/usr/bin/env bash
# ZeroDex project paths may use local storage, but any ParaDex2 NAS path must
# remain in Hyunsu's namespace. Camera-side ParaDex paths are not passed here.

precision_require_allowed_nas_path() {
  local label=$1
  local value=${2:-}
  [[ -z $value ]] && return 0
  [[ -z ${PRECISION_ALLOWED_NAS_ROOT:-} ]] && return 0

  local mount_root=${PRECISION_NAS_MOUNT_ROOT:-/mnt/paradex2}
  local resolved allowed mount
  resolved=$(realpath -m -- "$value")
  allowed=$(realpath -m -- "$PRECISION_ALLOWED_NAS_ROOT")
  mount=$(realpath -m -- "$mount_root")
  case "$resolved" in
    "$mount"|"$mount"/*)
      case "$resolved" in
        "$allowed"|"$allowed"/*) ;;
        *)
          echo "$label resolves outside the allowed NAS namespace: $resolved" >&2
          return 2
          ;;
      esac
      ;;
  esac
}
