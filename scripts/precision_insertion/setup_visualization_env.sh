#!/usr/bin/env bash
set -euo pipefail

BODEX_PYTHON="${BODEX_PYTHON:-$HOME/miniconda3/envs/autodex_bodex/bin/python}"
VISUALIZATION_VENV="${VISUALIZATION_VENV:-$HOME/.venvs/autodex-viz}"

if [[ ! -x "$BODEX_PYTHON" ]]; then
  echo "missing AutoDex Python: $BODEX_PYTHON" >&2
  exit 1
fi

"$BODEX_PYTHON" -m venv --system-site-packages "$VISUALIZATION_VENV"
"$VISUALIZATION_VENV/bin/python" -m pip install "open3d==0.19.0"

echo "visualization environment: $VISUALIZATION_VENV"
echo "rendering requires a working headless EGL/OpenGL device"
