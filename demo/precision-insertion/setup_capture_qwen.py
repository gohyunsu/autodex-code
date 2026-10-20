"""Stage the isolated five-PC Qwen workers and runtime settings.

This prepares files only. It neither starts a service nor connects a camera
or robot. The one worker per capture PC is started by the live runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[2]
STAGE = Path(
    "/home/robot/shared_data/hyunsu/tmp/"
    "precision_insertion_recipient_stage_20261013_b7e2392/"
    "AutoDex/precision_insertion/cylindrical")
WORKER_ROOT = Path(
    "/home/robot/shared_data/hyunsu/tmp/precision_distributed_qwen_20261020")
WORKERS = {
    "24122734": ("capture1", "precision_qwen3_vl_2b"),
    "25305463": ("capture2", "precision_qwen3_vl_2b"),
    "25322638": ("capture3", "precision_qwen3_vl_2b"),
    "25322649": ("capture5", "precision_qwen3_vl_2b"),
    "25305461": ("capture6", "precision_qwen3_vl_2b_cu128"),
}


def main() -> None:
    source = REPO / "demo/precision-insertion/precision_insertion"
    package = WORKER_ROOT / "precision_insertion"
    package.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "local_qwen.py", "distributed_qwen_worker.py"):
        shutil.copy2(source / name, package / name)

    original = STAGE / "franka_runtime_board9_gap20_stock_held.json"
    output = STAGE / "franka_runtime_board9_gap20_distributed.json"
    config = json.loads(original.read_text(encoding="utf-8"))
    config["vlm"].update({
        "mode": "distributed",
        "worker_root": str(WORKER_ROOT),
        "workers": {
            camera: {
                "host": host,
                "python": f"/home/{host}/.venvs/{venv}/bin/python",
                "model_dir": f"/home/{host}/.cache/precision_qwen3_vl_2b",
            }
            for camera, (host, venv) in WORKERS.items()
        },
    })
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(output)
    for camera, (host, _) in WORKERS.items():
        print(f"{host}: {camera}")


if __name__ == "__main__":
    main()
