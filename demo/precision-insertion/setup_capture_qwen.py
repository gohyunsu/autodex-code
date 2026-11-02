"""Stage isolated capture-PC Qwen worker code and runtime settings.

This prepares files only. It neither starts a service nor connects a camera
or robot. One model process is started per distinct capture host/GPU; cameras
sharing a host are sent in one batch.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True,
                        help="current FR3 runtime JSON to extend")
    parser.add_argument("--output-config", type=Path, required=True,
                        help="current distributed FR3 runtime JSON to write")
    parser.add_argument("--worker-root", type=Path, required=True,
                        help="shared directory containing capture-PC Qwen worker code")
    parser.add_argument(
        "--workers-json", type=Path,
        help=("optional camera-to-worker mapping; omitted preserves the "
              "original five-PC AutoDex mapping"))
    args = parser.parse_args(argv)
    source_config = args.source_config.resolve(strict=True)
    output_config = args.output_config.resolve()
    if source_config == output_config:
        parser.error("source and output config paths must differ")
    if not output_config.parent.is_dir():
        parser.error(f"output directory does not exist: {output_config.parent}")

    source = REPO / "demo/precision-insertion/precision_insertion"
    package = args.worker_root / "precision_insertion"
    package.mkdir(parents=True, exist_ok=True)
    active_files = ("__init__.py", "local_qwen.py",
                    "distributed_qwen_worker.py")
    for name in active_files:
        shutil.copy2(source / name, package / name)

    workers = ({
        camera: {
            "host": host,
            "python": f"/home/{host}/.venvs/{venv}/bin/python",
            "model_dir": f"/home/{host}/.cache/precision_qwen3_vl_2b",
        }
        for camera, (host, venv) in WORKERS.items()
    } if args.workers_json is None else json.loads(
        args.workers_json.resolve(strict=True).read_text(encoding="utf-8")))
    if not isinstance(workers, dict) or not workers:
        parser.error("workers JSON must be a nonempty camera mapping")

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["vlm"].update({
        "mode": "distributed",
        "classification_image_size": [384, 288],
        "worker_root": str(args.worker_root.resolve()),
        "workers": workers,
    })
    output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    manifest = {
        "schema": "precision_insertion_qwen_worker_stage_v1",
        "scope": "active_worker_files_only_not_model_weights",
        "worker_root": str(args.worker_root.resolve()),
        "worker_files": [
            _file_record(package / name, relative_to=args.worker_root)
            for name in active_files
        ],
        "source_config": _file_record(source_config),
        "output_config": _file_record(output_config),
        "workers_config": (
            _file_record(args.workers_json.resolve(strict=True))
            if args.workers_json is not None else None),
        "workers": workers,
    }
    manifest_path = args.worker_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output_config)
    print(manifest_path)
    for camera, spec in workers.items():
        print(f"{spec.get('host')}: {camera}")


if __name__ == "__main__":
    main()
