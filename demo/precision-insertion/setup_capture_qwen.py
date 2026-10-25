"""Stage the isolated five-PC Qwen workers and runtime settings.

This prepares files only. It neither starts a service nor connects a camera
or robot. The one worker per capture PC is started by the live runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[2]
WORKERS = {
    "24122734": ("capture1", "precision_qwen3_vl_2b"),
    "25305463": ("capture2", "precision_qwen3_vl_2b"),
    "25322638": ("capture3", "precision_qwen3_vl_2b"),
    "25322649": ("capture5", "precision_qwen3_vl_2b"),
    "25305461": ("capture6", "precision_qwen3_vl_2b_cu128"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True,
                        help="current FR3 runtime JSON to extend")
    parser.add_argument("--output-config", type=Path, required=True,
                        help="current distributed FR3 runtime JSON to write")
    parser.add_argument("--worker-root", type=Path, required=True,
                        help="shared directory containing capture-PC Qwen worker code")
    args = parser.parse_args()
    source_config = args.source_config.resolve(strict=True)
    output_config = args.output_config.resolve()
    if source_config == output_config:
        parser.error("source and output config paths must differ")
    if not output_config.parent.is_dir():
        parser.error(f"output directory does not exist: {output_config.parent}")

    source = REPO / "demo/precision-insertion/precision_insertion"
    package = args.worker_root / "precision_insertion"
    package.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "local_qwen.py", "distributed_qwen_worker.py"):
        shutil.copy2(source / name, package / name)

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["vlm"].update({
        "mode": "distributed",
        "classification_image_size": [384, 288],
        "worker_root": str(args.worker_root.resolve()),
        "workers": {
            camera: {
                "host": host,
                "python": f"/home/{host}/.venvs/{venv}/bin/python",
                "model_dir": f"/home/{host}/.cache/precision_qwen3_vl_2b",
            }
            for camera, (host, venv) in WORKERS.items()
        },
    })
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(output_config)
    for camera, (host, _) in WORKERS.items():
        print(f"{host}: {camera}")


if __name__ == "__main__":
    main()
