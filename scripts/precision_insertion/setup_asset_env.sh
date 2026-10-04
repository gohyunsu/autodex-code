#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARADEX_ROOT="${PARADEX_ROOT:-$HOME/paradex}"
ENV_ROOT="${AUTODEX_ASSET_ENV:-$HOME/.venvs/autodex-assets}"

if [[ ! -d "$PARADEX_ROOT/paradex" ]]; then
    echo "ParaDex checkout missing: $PARADEX_ROOT" >&2
    echo "Clone https://github.com/snuvclab/paradex.git first." >&2
    exit 1
fi

python3 -m venv --system-site-packages "$ENV_ROOT"
"$ENV_ROOT/bin/python" -m pip install --no-deps --no-build-isolation -e "$PARADEX_ROOT"
"$ENV_ROOT/bin/python" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"

"$ENV_ROOT/bin/python" - <<'PY'
import numpy
import scipy
import paradex
import autodex

print("asset environment OK")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("paradex", paradex.__file__)
print("autodex", autodex.__file__)
PY

echo "Activate with: source $ENV_ROOT/bin/activate"
